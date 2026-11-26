"""Set up fixed-composition crystal sampling using the shared geometry updates."""

from collections.abc import Mapping
from numbers import Integral

import torch

from ..diffusion.crystal_diffusion import CrystalDiffusion
from ..diffusion.sampling.correctors import LangevinCorrector
from ..diffusion.sampling.pc_sampler import Sampler
from ..diffusion.sampling.predictors import LatticePredictor, PositionPredictor
from ..diffusion.sampling.prior import sample_geometry_prior


def build_csp_prior(
    model: CrystalDiffusion,
    composition: Mapping[str, int],
    batch_size: int = 1,
) -> dict[str, torch.Tensor]:
    """Repeat one exact unit-cell composition and draw a geometry prior."""
    from pymatgen.core import Element

    if not isinstance(composition, Mapping) or not composition:
        raise ValueError("composition must map element symbols to positive integer counts")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    species_counts = []
    for symbol, count in composition.items():
        if (not isinstance(symbol, str) or isinstance(count, bool)
                or not isinstance(count, Integral) or count < 1):
            raise ValueError("composition must map element symbols to positive integer counts")
        atomic_number = Element(symbol).Z
        if atomic_number > model.atom_types:
            raise ValueError(f"Element {symbol} is outside the model's element vocabulary")
        species_counts.append((atomic_number, int(count)))
    species = [z for z, count in sorted(species_counts) for _ in range(count)]

    sigma_max = model.position_corruption.scheduler.sigma_max
    device = sigma_max.device
    atomic_numbers = torch.tensor(species, dtype=torch.long, device=device).repeat(batch_size)
    num_atoms = torch.full((batch_size,), len(species), dtype=torch.long, device=device)
    batch = torch.repeat_interleave(num_atoms, output_size=atomic_numbers.numel())
    frac_coords, lattice = sample_geometry_prior(
        num_atoms, sigma_max.item(), model.lattice_corruption, device)
    return {
        "atomic_numbers": atomic_numbers,
        "frac_coords": frac_coords.to(lattice),
        "lattice": lattice,
        "num_atoms": num_atoms,
        "batch": batch,
    }


def build_csp_sampler(model: CrystalDiffusion, *, num_steps: int = 1000) -> Sampler:
    """Build a sampler that updates only fractional coordinates and lattices."""
    return Sampler(
        atom_pred=None,
        pos_pred=PositionPredictor(model.position_corruption),
        lat_pred=LatticePredictor(model.lattice_corruption),
        corrector=LangevinCorrector(),
        num_steps=num_steps,
    )
