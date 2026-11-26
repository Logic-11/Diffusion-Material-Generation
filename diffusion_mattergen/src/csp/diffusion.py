"""Diffusion of crystal geometry conditioned on fixed atomic numbers."""

import torch

from ..diffusion.crystal_diffusion import CrystalDiffusion
from .loss import CSPLoss


class CSPDiffusion(CrystalDiffusion):
    task = "csp"

    def __init__(
        self, denoiser, position_corruption, lattice_corruption, *,
        cutoff=7.0, max_neighbor=50, max_offset_integer=3, loss_fn=None,
        max_cell_images_per_dim=5,
    ):
        super().__init__(
            denoiser, position_corruption, lattice_corruption,
            loss_fn=CSPLoss() if loss_fn is None else loss_fn,
            cutoff=cutoff, max_neighbor=max_neighbor, max_offset_integer=max_offset_integer,
            max_cell_images_per_dim=max_cell_images_per_dim,
        )
        self.denoiser.output_head.requires_grad_(False)

    @torch.no_grad()
    def q_sample(self, clean, t, noise=None):
        """Corrupt geometry at a scalar or per-crystal time with fixed elements.

        Optional noise keys are position [N, 3] and lattice [B, 3, 3].
        Lattice noise is supplied before symmetrization.
        """
        t, noise = self._prepare_q_sample(clean, t, noise)
        return self._q_sample_geometry(clean, t, noise, clean["atomic_numbers"].clone())
