"""Joint diffusion of elements, fractional positions, and crystal lattices."""

import torch

from .crystal_diffusion import CrystalDiffusion
from .loss import JointDiffusionLoss


class JointDiffusion(CrystalDiffusion):
    task = "joint"

    _minimum_atomic_number = 0
    _atomic_number_error = "Atomic numbers must be 0 (MASK) or a supported element"
    _noise_keys = ("atom_uniform", "position", "lattice")

    def __init__(self, denoiser, atom_corruption, position_corruption, lattice_corruption,
                 cutoff=7.0, max_neighbor=50, max_offset_integer=3, loss_fn=None,
                 max_cell_images_per_dim=5):
        if atom_corruption.mask_type != 0:
            raise ValueError("The batch convention requires MASK = 0")
        super().__init__(
            denoiser, position_corruption, lattice_corruption,
            loss_fn=JointDiffusionLoss() if loss_fn is None else loss_fn,
            cutoff=cutoff, max_neighbor=max_neighbor, max_offset_integer=max_offset_integer,
            max_cell_images_per_dim=max_cell_images_per_dim,
        )
        self.atom_corruption = atom_corruption

    @property
    def atom_types(self):
        return self.atom_corruption.atom_types

    @property
    def timesteps(self):
        return self.atom_corruption.scheduler.timesteps

    @torch.no_grad()
    def q_sample(self, clean, t, noise=None):
        """Corrupt all three fields for a scalar or per-crystal time.

        Optional noise keys are atom_uniform [N], position [N, 3] and lattice
        [B, 3, 3]. Lattice noise is supplied before symmetrization.
        """
        t, noise = self._prepare_q_sample(clean, t, noise)
        atom_uniform = self._get_noise(
            noise, "atom_uniform", clean["frac_coords"][:, 0], uniform=True)
        atomic_numbers, masked = self.atom_corruption.q_sample(
            clean["atomic_numbers"], t, clean["batch"], uniform=atom_uniform)
        noisy, info = self._q_sample_geometry(clean, t, noise, atomic_numbers)
        info["masked_atom"] = masked
        info["atom_target"] = clean["atomic_numbers"]
        return noisy, info

    def _format_prediction(self, raw, noisy):
        atom_logits = raw["atom_logits"]
        if getattr(self.denoiser, "atom_class_order", "mask_first") == "mask_last":
            # The local D3PM uses atomic numbers as class indices, with MASK at zero.
            atom_logits = torch.cat((atom_logits[:, -1:], atom_logits[:, :-1]), dim=-1)
        return {"atom_logits": atom_logits, **super()._format_prediction(raw, noisy)}

    def scores_from_prediction(self, prediction, noisy, t):
        return {
            "atom_logits": prediction["atom_logits"],
            **super().scores_from_prediction(prediction, noisy, t),
        }

    @staticmethod
    def atom_probs_with_mask(atom_logits):
        """Class order is [MASK,Z=1,...,118], matching input atomic numbers."""
        probs = atom_logits.softmax(dim=-1)
        return probs
