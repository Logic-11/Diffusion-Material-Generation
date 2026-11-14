import torch
import torch.nn as nn
import torch.nn.functional as F

from . import geometry
from .gnn import RadialBasis
from .loss import JointDiffusionLoss


def mask_atomtype(atomic_numbers, t):
    """Mask concatenated atoms [N_total] with scalar or per-atom probabilities."""
    t = torch.as_tensor(t, dtype=torch.get_default_dtype(), device=atomic_numbers.device)
    if t.ndim == 2 and t.shape[-1] == 1:
        t = t.squeeze(-1)
    mask = torch.bernoulli(t.expand_as(atomic_numbers)).bool()
    return atomic_numbers.masked_fill(mask, 0), mask


def atomtype_ce(logits, atomic_numbers, mask):
    """Mean cross entropy over selected atoms, using clean 1-based atomic numbers."""
    if not torch.any(mask):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], atomic_numbers[mask] - 1)


def add_coord_noise(frac_0, sigma_t, noise=None):
    """frac_0/noise [N_total, 3], sigma_t [N_total, 1] (or a scalar).

    Expand crystal scales with sigma_graph[batch, None] before calling.
    """
    if noise is None:
        noise = torch.randn_like(frac_0)
    return geometry.wrap_frac(frac_0 + sigma_t * noise), noise


def periodic_noise(displace, sigmas):
    """Return the unnormalized wrapped-normal target for each displacement.

    displace has shape [N_total, 3], sigmas [N_total, 1].
    Other broadcastable shapes are also accepted for schedule estimation.
    The scale must be positive; timestep zero is not used for this target.
    """
    displace = torch.as_tensor(displace)
    if not displace.is_floating_point():
        displace = displace.to(torch.get_default_dtype())
    sigmas = torch.as_tensor(sigmas, dtype=displace.dtype, device=displace.device)
    k = torch.arange(-10, 11, dtype=displace.dtype, device=displace.device)
    image_displacements = displace.unsqueeze(-1) + k
    image_sigmas = sigmas.unsqueeze(-1)
    scaled_displacements = image_displacements / image_sigmas
    weights = torch.softmax(-0.5 * scaled_displacements.square(), dim=-1)
    return torch.sum(weights * scaled_displacements / image_sigmas, dim=-1)



def coordinate_target(noise, sigmas, sigmas_norm):
    """Normalized targets: noise [N_total, 3], sigmas/norms [N_total, 1]."""
    target = periodic_noise(sigmas * noise, sigmas)
    return target / torch.sqrt(sigmas_norm)


class JointDiffusion(nn.Module):
    def __init__(self, denoiser, atom_corruption, position_corruption, lattice_corruption, 
                 cutoff=7.0, max_neighbor=50, max_offset_integer=3, loss_fn=None):
        super().__init__()
        if atom_corruption.mask_type != 0:
            raise ValueError("The batch convention requires MASK = 0")
        self.denoiser = denoiser
        self.atom_corruption = atom_corruption
        self.position_corruption = position_corruption
        self.lattice_corruption = lattice_corruption
        self.radial_basis = RadialBasis(denoiser.num_radial, cutoff=cutoff)
        self.max_neighbor = max_neighbor
        # Keep the already verified position target's default, also used upstream.
        self.max_offset_integer = max_offset_integer
        self.loss_fn = JointDiffusionLoss() if loss_fn is None else loss_fn

    @property
    def timesteps(self):
        return self.atom_corruption.scheduler.timesteps

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
        if torch.any(atoms < 0) or torch.any(atoms > self.atom_corruption.atom_types):
            raise ValueError("Atomic numbers must be 0 (MASK) or a supported element")
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

    @torch.no_grad()
    def q_sample(self, clean, k, noise=None):
        """Return (noisy_state, info) for integer k in 1..K, scalar or [B].

        Optional noise keys: atom_uniform [N] in [0,1), position [N,3] IID
        standard Gaussian, lattice [B,3,3] IID standard Gaussian BEFORE
        symmetrization. The input clean dictionary and its tensors are not edited.
        info contains targets, scales and time, and is never passed to Denoiser.
        """
        self._check_state(clean)
        if torch.any(clean["atomic_numbers"] == 0):
            raise ValueError("Clean atomic numbers must not contain MASK")
        frac, lattice = clean["frac_coords"], clean["lattice"]
        counts, batch = clean["num_atoms"], clean["batch"]
        k = torch.as_tensor(k, device=frac.device)
        if k.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("k must contain integer diffusion steps")
        if k.ndim == 0:
            k = k.expand(counts.shape)
        if k.shape != counts.shape or torch.any(k < 1) or torch.any(k > self.timesteps):
            raise ValueError("k must be scalar or [B], with values in 1..K")
        k = k.long().clone()
        t = k.to(frac) / self.timesteps

        noise = {} if noise is None else noise
        if set(noise) - {"atom_uniform", "position", "lattice"}:
            raise ValueError("Unknown noise key; use atom_uniform, position, lattice")
        atom_uniform = self._get_noise(noise, "atom_uniform", frac[:, 0], uniform=True)
        position_noise = self._get_noise(noise, "position", frac)
        lattice_noise = self._get_noise(noise, "lattice", lattice)

        atoms_t, masked = self.atom_corruption.q_sample( clean["atomic_numbers"], t, batch, uniform=atom_uniform )
        frac_t = self.position_corruption.q_sample( frac, t, counts, batch, noise=position_noise )
        lattice_t = self.lattice_corruption.q_sample( lattice, t, counts, noise=lattice_noise )

        # Whitelist the state fields: cached clean edges/geometry cannot leak in.
        noisy = {"atomic_numbers": atoms_t, "frac_coords": frac_t,
                 "lattice": lattice_t, "num_atoms": counts.clone(),
                 "batch": batch.clone()}
        sigma_x, sigma_l = self.noise_scales(noisy, t)
        info = {
            "k_graph": k, "t_graph": t, "masked_atom": masked,
            "atom_target": clean["atomic_numbers"] - 1,
            "coord_target": self.position_corruption.score_target(
                frac, frac_t, t, counts, batch,
                max_offset_integer=self.max_offset_integer),
            "lattice_target": self.lattice_corruption.score_target(
                lattice, lattice_t, t, counts),
            "sigma_x": sigma_x, "sigma_l": sigma_l,
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
        """Rebuild geometry and predict clean-element logits and scaled scores.

        Keep autograd enabled here for training. Singular cells are reported by
        build_periodic_graph; no projection, clipping or pseudoinverse is applied.
        """
        self._check_state(noisy)
        t = self._check_time(t, noisy)
        edges = geometry.build_periodic_graph(
            noisy["frac_coords"], noisy["lattice"], cutoff=self.radial_basis.cutoff,
            batch=noisy["batch"], max_neighbor=self.max_neighbor)
        edge_index, edge_image, vectors, distances, num_edges, edge_ptr, edge_batch = edges
        graph = {
            "edge_index": edge_index, "edge_vectors": vectors,
            "edge_distance": distances, "num_edges": num_edges,
            "edge_ptr": edge_ptr, "edge_batch": edge_batch,
            "radial_features": self.radial_basis(distances),
        }
        raw = self.denoiser(atomic_numbers=noisy["atomic_numbers"],
                            t_graph=t, batch=noisy["batch"], **graph)
        prediction = {
            "atom_logits": raw["atom_logits"],
            "coord_score_times_std": geometry.cart_to_frac(raw["coord_cart"], noisy["lattice"], noisy["batch"]),
            "lattice_score_times_std": raw["lattice_pred"],
        }
        if return_graph:
            prediction["graph"] = {**graph, "edge_image": edge_image}
        return prediction

    def compute_loss(self, clean, k=None, noise=None):
        """Sample time/noise, predict once, and return (loss, field metrics).

        Each crystal draws one uniform integer k in 1..K, shared by all fields.
        Explicit k/noise allow reproducible mathematical and gradient checks.
        q_sample builds constant targets; predict and loss retain autograd.
        The caller owns zero_grad(), backward(), and optimizer.step().
        """
        if k is None:
            k = torch.randint(1, self.timesteps + 1, clean["num_atoms"].shape,
                              device=clean["num_atoms"].device)
        noisy, info = self.q_sample(clean, k, noise=noise)
        prediction = self.predict(noisy, info["t_graph"])
        return self.loss_fn(prediction, info, clean["batch"], clean["num_atoms"])

    def scores_from_prediction(self, prediction, noisy, t):
        """Convert an existing prediction to sampling scores without clean data."""
        sigma_x, sigma_l = self.noise_scales(noisy, t)
        return {
            "atom_logits": prediction["atom_logits"],
            "coord_score": prediction["coord_score_times_std"] / sigma_x,
            "lattice_score": prediction["lattice_score_times_std"] / sigma_l,
        }

    @staticmethod
    def atom_probs_with_mask(atom_logits):
        """Posterior class order is [Z=1,...,118,MASK], with p(clean MASK)=0.

        Input states instead use atomic number 0 for MASK. For a posterior index
        c, use atomic number c+1 for c<118 and atomic number 0 for c=118.
        """
        probs = atom_logits.softmax(dim=-1)
        # return torch.cat((probs, torch.zeros_like(probs[:, :1])), dim=-1)
        return probs

    def forward(self, noisy, t):
        return self.predict(noisy, t)
