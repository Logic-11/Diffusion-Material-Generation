"""Initialize reverse diffusion from training statistics and field priors."""

import json
from pathlib import Path

import torch

from ...backbone.geometry import wrap_frac
from ..corruption import LatticeCorruption


def sample_n(statistic: str | Path, batch_size: int = 1,
             device: str | torch.device = "cuda") -> torch.Tensor:
    """Sample num_atoms [batch_size] from saved training-set probabilities."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    with open(statistic, 'r', encoding='utf-8') as f:
        data = json.load(f)
    n_dist = data.get('n_dist')
    if not isinstance(n_dist, dict) or not n_dist:
        raise ValueError("statistic must contain a nonempty n_dist dictionary")

    num_atoms = torch.tensor([int(n) for n in n_dist], dtype=torch.long)
    probabilities = torch.tensor(list(n_dist.values()), dtype=torch.float64)
    if torch.any(num_atoms <= 0):
        raise ValueError("n_dist atom counts must be positive integers")
    total = probabilities.sum()
    if (not torch.isfinite(probabilities).all() or torch.any(probabilities < 0)
            or not torch.isfinite(total) or total <= 0):
        raise ValueError("n_dist probabilities must be finite, nonnegative and have positive total")

    probabilities = probabilities / total
    indices = torch.multinomial(probabilities, batch_size, replacement=True)
    return num_atoms[indices].to(device=device)

def sample_geometry_prior(
    num_atoms: torch.Tensor,
    sigma_max: float,
    lat_cor: LatticeCorruption,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw wrapped fractional positions and lattices for the given atom counts."""
    batch_size = num_atoms.shape[0]
    total_num = num_atoms.sum().item()
    batch = torch.repeat_interleave(torch.arange(batch_size, device=device), num_atoms)
    gaussian = torch.randn((total_num, 3), device=device)
    scale = (sigma_max * num_atoms[batch].pow(-1.0 / 3.0)).unsqueeze(-1)
    prior_coord = wrap_frac(scale * gaussian)
    prior_lattice = lat_cor.prior_sample(num_atoms)
    return prior_coord, prior_lattice


def sample_prior(
    num_atoms: torch.Tensor,
    limit_density: float,
    sigma_max: float,
    lat_cor: LatticeCorruption,
    device: str | torch.device = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw the joint prior with every atomic number initially masked."""
    prior_coord, prior_lattice = sample_geometry_prior(num_atoms, sigma_max, lat_cor, device)
    masked_atoms = num_atoms.new_zeros((prior_coord.shape[0],), device=device)
    return masked_atoms, prior_coord, prior_lattice
