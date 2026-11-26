"""Geometry operations shared by joint diffusion and CSP."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone import geometry
from ..backbone.gnn import RadialBasis


class CrystalDiffusion(nn.Module):
    """Shared continuous diffusion on fractional positions and crystal lattices."""

    task: str

    _minimum_atomic_number = 1
    _atomic_number_error = "Atomic numbers must be supported real elements"
    _noise_keys = ("position", "lattice")

    def __init__(
        self, denoiser, position_corruption, lattice_corruption, *,
        loss_fn, cutoff, max_neighbor, max_offset_integer, max_cell_images_per_dim,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.position_corruption = position_corruption
        self.lattice_corruption = lattice_corruption
        self.radial_basis = RadialBasis(denoiser.num_radial, cutoff=cutoff)
        self.max_neighbor = max_neighbor
        self.max_cell_images_per_dim = max_cell_images_per_dim
        self.max_offset_integer = max_offset_integer
        self.loss_fn = loss_fn

    @property
    def atom_types(self):
        return self.denoiser.node_embedding.num_embeddings - 1

    def _check_state(self, state):
        atoms, frac, lattice, counts, batch = (
            state[key] for key in
            ("atomic_numbers", "frac_coords", "lattice", "num_atoms", "batch")
        )
        if counts.ndim != 1 or counts.numel() == 0:
            raise ValueError("num_atoms must have shape [B] with B > 0")
        n, b = atoms.numel(), counts.numel()
        if (atoms.shape != (n,) or frac.shape != (n, 3)
                or lattice.shape != (b, 3, 3) or batch.shape != (n,)):
            raise ValueError("Crystal state shapes must agree with num_atoms")
        if any(x.dtype != torch.long for x in (atoms, counts, batch)):
            raise ValueError("atomic_numbers, num_atoms and batch must be int64")
        if (not frac.is_floating_point() or lattice.dtype != frac.dtype
                or any(x.device != frac.device for x in (atoms, lattice, counts, batch))):
            raise ValueError("Crystal tensors must share device and floating dtype")
        if (torch.any(counts <= 0) or torch.any(batch < 0) or torch.any(batch >= b)
                or not torch.equal(torch.bincount(batch, minlength=b), counts)):
            raise ValueError("batch membership must agree with positive num_atoms")
        if torch.any(atoms < self._minimum_atomic_number) or torch.any(atoms > self.atom_types):
            raise ValueError(self._atomic_number_error)
        if not torch.isfinite(frac).all() or not torch.isfinite(lattice).all():
            raise ValueError("Coordinates and lattices must be finite")

    @staticmethod
    def _check_time(t, state):
        t = torch.as_tensor(t, device=state["lattice"].device,
                            dtype=state["lattice"].dtype)
        if (t.shape != state["num_atoms"].shape or not torch.isfinite(t).all()
                or torch.any(t <= 0) or torch.any(t > 1)):
            raise ValueError("t must have shape [B] and satisfy 0 < t <= 1")
        return t

    @staticmethod
    def _get_noise(noise, name, reference, uniform=False):
        value = noise.get(name)
        if value is None:
            value = torch.rand_like(reference) if uniform else torch.randn_like(reference)
        if (value.shape != reference.shape or value.dtype != reference.dtype
                or value.device != reference.device or not torch.isfinite(value).all()):
            raise ValueError(f"{name} noise must match its reference shape, dtype and device")
        if uniform and (torch.any(value < 0) or torch.any(value >= 1)):
            raise ValueError("atom_uniform must lie in [0, 1)")
        return value

    def _prepare_q_sample(self, clean, t, noise):
        self._check_state(clean)
        if self._minimum_atomic_number == 0 and torch.any(clean["atomic_numbers"] == 0):
            raise ValueError("Clean atomic numbers must not contain MASK")

        frac = clean["frac_coords"]
        t = torch.as_tensor(t, dtype=frac.dtype, device=frac.device)
        if t.ndim == 0:
            t = t.expand(clean["num_atoms"].shape)
        t = self._check_time(t, clean).clone()

        noise = {} if noise is None else noise
        if set(noise) - set(self._noise_keys):
            raise ValueError(f"Unknown noise key. Use {', '.join(self._noise_keys)}")
        return t, noise

    def _q_sample_geometry(self, clean, t, noise, atomic_numbers):
        frac, lattice = clean["frac_coords"], clean["lattice"]
        counts, batch = clean["num_atoms"], clean["batch"]
        position_noise = self._get_noise(noise, "position", frac)
        lattice_noise = self._get_noise(noise, "lattice", lattice)
        frac_t = self.position_corruption.q_sample(
            frac, t, counts, batch, noise=position_noise)
        lattice_t = self.lattice_corruption.q_sample(lattice, t, counts, noise=lattice_noise)

        # Build the noisy state from its core fields so clean geometry stays private.
        noisy = {
            "atomic_numbers": atomic_numbers,
            "frac_coords": frac_t,
            "lattice": lattice_t,
            "num_atoms": counts.clone(),
            "batch": batch.clone(),
        }
        sigma_x, sigma_l = self.noise_scales(noisy, t)
        info = {
            "t_graph": t,
            "coord_target": self.position_corruption.score_target(
                frac, frac_t, t, counts, batch,
                max_offset_integer=self.max_offset_integer),
            "lattice_target": self.lattice_corruption.score_target(lattice, lattice_t, t, counts),
            "sigma_x": sigma_x,
            "sigma_l": sigma_l,
        }
        return noisy, info

    def noise_scales(self, state, t):
        """Scales [N,1] and [B,1,1]; depends only on time and atom counts."""
        t = self._check_time(t, state)
        counts, batch = state["num_atoms"], state["batch"]
        sigma_x = self.position_corruption.scheduler.sigma_std(t, counts)[batch, None]
        sigma_l = self.lattice_corruption.scheduler.std(t, counts)[:, None, None]
        return sigma_x, sigma_l

    def predict(self, noisy, t, return_graph=False):
        """Rebuild geometry from the noisy state and predict scaled scores.

        Keep autograd enabled here for training. Singular cells are reported by
        build_periodic_graph; no projection, clipping or pseudoinverse is applied.
        """
        self._check_state(noisy)
        t = self._check_time(t, noisy)
        edges = geometry.build_periodic_graph(
            noisy["frac_coords"], noisy["lattice"], cutoff=self.radial_basis.cutoff,
            batch=noisy["batch"], max_neighbor=self.max_neighbor,
            max_cell_images_per_dim=self.max_cell_images_per_dim)
        edge_index, edge_image, vectors, distances, num_edges, edge_ptr, edge_batch = edges
        # Rebuild angles from current noisy geometry.
        if getattr(self.denoiser, "use_triplets", False):
            id3_center_b, id3_center_c, cos_phi = geometry.build_triplet_graph(
                edge_index,
                vectors,
                distances,
                num_nodes=noisy["frac_coords"].shape[0],
            )
        else:
            id3_center_b = edge_index.new_empty(0)
            id3_center_c = edge_index.new_empty(0)
            cos_phi = vectors.new_empty(0)
        # GemNet's edge direction points from the neighbor image to the center.
        edge_lattice_cosines = F.cosine_similarity(
            -vectors[:, None, :], noisy["lattice"][edge_batch], dim=-1,
        )
        graph = {
            "edge_index": edge_index, "edge_vectors": vectors,
            "edge_distance": distances, "num_edges": num_edges,
            "edge_ptr": edge_ptr, "edge_batch": edge_batch,
            "radial_features": self.radial_basis(distances),
            "edge_lattice_cosines": edge_lattice_cosines,
            "id3_center_b": id3_center_b,
            "id3_center_c": id3_center_c,
            "cos_phi": cos_phi,
        }
        raw = self.denoiser(atomic_numbers=noisy["atomic_numbers"],
                            t_graph=t, batch=noisy["batch"], **graph)
        prediction = self._format_prediction(raw, noisy)
        if return_graph:
            prediction["graph"] = {**graph, "edge_image": edge_image}
        return prediction

    def _format_prediction(self, raw, noisy):
        return {
            "coord_score_times_std": geometry.cart_to_frac(
                raw["coord_cart"], noisy["lattice"], noisy["batch"]),
            "lattice_score_times_std": raw["lattice_pred"],
        }

    def compute_loss(self, clean, t=None, t_min=1e-5, noise=None):
        """Sample time/noise, predict once, and return (loss, field metrics)."""
        if t is None:
            t_min = float(t_min)
            if not 0.0 < t_min < 1.0:
                raise ValueError("t_min must satisfy 0 < t_min < 1")

            t = torch.rand(
                clean["num_atoms"].shape,
                device=clean["lattice"].device,
                dtype=clean["lattice"].dtype)
            t = t * (1.0 - t_min) + t_min

        noisy, info = self.q_sample(clean, t, noise=noise)
        prediction = self.predict(noisy, info["t_graph"])
        return self.loss_fn(prediction, info, clean["batch"], clean["num_atoms"])

    def scores_from_prediction(self, prediction, noisy, t):
        """Convert an existing prediction to sampling scores without clean data."""
        sigma_x, sigma_l = self.noise_scales(noisy, t)
        return {
            "coord_score": prediction["coord_score_times_std"] / sigma_x,
            "lattice_score": prediction["lattice_score_times_std"] / sigma_l,
        }

    def forward(self, noisy, t):
        return self.predict(noisy, t)
