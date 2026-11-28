"""Continuous score-matching loss for crystal structure prediction."""

import math

from torch import nn

from ..diffusion.loss import geometry_losses_per_crystal


class CSPLoss(nn.Module):
    def __init__(self, coord_weight=0.1, lattice_weight=1.0, reduce="sum"):
        super().__init__()
        if any(not math.isfinite(weight) or weight < 0 for weight in (coord_weight, lattice_weight)):
            raise ValueError("Loss weights must be finite and nonnegative")
        if reduce not in ("sum", "mean"):
            raise ValueError("reduce must be 'sum' or 'mean'")
        self.coord_weight = float(coord_weight)
        self.lattice_weight = float(lattice_weight)
        self.reduce = reduce

    def forward(self, prediction, info, batch, num_atoms):
        coord, lattice = geometry_losses_per_crystal(
            prediction, info, batch, num_atoms, reduce=self.reduce)
        loss = self.coord_weight * coord + self.lattice_weight * lattice
        return loss.mean(), {"loss_coord": coord.mean(), "loss_lattice": lattice.mean()}
