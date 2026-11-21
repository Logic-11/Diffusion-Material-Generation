import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if __name__ == "__main__" and not __package__:
    # Allow direct script execution to import src from the project root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.corruption import AtomTypeCorruption, LatticeCorruption, PositionCorruption
from src.denoiser import Denoiser
from src.diffusion import JointDiffusion
from src.loss import JointDiffusionLoss
from src.sampling import (
    AtomTypePredictor, LangevinCorrector, LatticePredictor, PositionPredictor,
    Sampler, sample_n, sample_prior,
)
from src.scheduler import LatticeVPScheduler, MaskScheduler, PositionVEScheduler

if TYPE_CHECKING:
    from pymatgen.core import Structure


def load_model(checkpoint_path: str | Path,
               device: str | torch.device = "cpu") -> JointDiffusion:
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use device='cpu' to load the model.")

    # Load on CPU so the checkpoint's optimizer tensors do not occupy GPU memory.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {"model_config", "diffusion_config", "model_state_dict"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError(
            "Expected a checkpoint from scripts.train.save_model containing "
            "model_config, diffusion_config and model_state_dict."
        )
    model_config = checkpoint["model_config"]
    diffusion_config = checkpoint["diffusion_config"]
    state_dict = checkpoint["model_state_dict"]

    # [GemNet-T triplet] Pre-triplet checkpoints remain loadable as baselines.
    has_triplet_parameters = any(
        ".triplet_interaction." in name for name in state_dict
    )
    if not has_triplet_parameters:
        model_config = {**model_config, "use_triplets": False}

    denoiser = Denoiser(**model_config)
    atom_corruption = AtomTypeCorruption(
        scheduler=MaskScheduler(timesteps=diffusion_config["timesteps"]),
        atom_types=model_config["atom_types"],
        mask_type=diffusion_config["input_mask_type"],
    )
    position_corruption = PositionCorruption(
        scheduler=PositionVEScheduler(**diffusion_config["position"]),
    )
    lattice_corruption = LatticeCorruption(
        scheduler=LatticeVPScheduler(**diffusion_config["lattice"]),
    )
    model = JointDiffusion(
        denoiser=denoiser,
        atom_corruption=atom_corruption,
        position_corruption=position_corruption,
        lattice_corruption=lattice_corruption,
        loss_fn=JointDiffusionLoss(**diffusion_config["loss"]),
        **diffusion_config["graph"],
    )
    atom_class_order = diffusion_config.get("atom_class_order", "mask_last")
    if atom_class_order == "mask_last":
        for name in ("denoiser.output_head.2.weight", "denoiser.output_head.2.bias"):
            value = state_dict[name]
            state_dict[name] = torch.cat((value[-1:], value[:-1]), dim=0)
    elif atom_class_order != "mask_first":
        raise ValueError(f"Unknown atom_class_order: {atom_class_order}")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


@torch.no_grad()
def generate_batch(
    model: JointDiffusion,
    sampler: Sampler,
    statistic_path: str | Path,
    batch_size: int = 1,
) -> list["Structure"]:
    """Generate one batch of crystals using the training run's statistic.json.

    Use a sampler built from this model's corruptions. Structures are returned
    in batch order using the final predictor means and posterior-mode elements.
    """
    sigma_max = model.position_corruption.scheduler.sigma_max
    lattice_corruption = model.lattice_corruption
    num_atoms = sample_n(statistic_path, batch_size=batch_size, device=sigma_max.device)
    atomic_numbers, frac_coords, lattice = sample_prior(
        num_atoms,
        limit_density=lattice_corruption.scheduler.limit_density.item(),
        sigma_max=sigma_max.item(),
        lat_cor=lattice_corruption,
        device=sigma_max.device,
    )

    # Each crystal index is repeated once per atom, in the order used by sample_prior.
    batch = torch.repeat_interleave(num_atoms, output_size=atomic_numbers.numel())
    initial_state = {
        "atomic_numbers": atomic_numbers,
        "frac_coords": frac_coords.to(lattice),
        "lattice": lattice,
        "num_atoms": num_atoms,
        "batch": batch,
    }
    _, mean_state = sampler.sample(model, initial_state)
    return sampler.to_structures(mean_state)


def main(argv: list[str] | None = None) -> None:
    """Generate unconditional crystals in batches and write one CIF per structure."""
    parser = argparse.ArgumentParser(description="Generate crystal structures from a training checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Training checkpoint (.pt).")
    parser.add_argument("--statistic", type=Path,
                        help="Path to statistic.json. Defaults to the file beside the checkpoint.")
    parser.add_argument("-o", "--output", type=Path, default=Path("generated"),
                        help="Directory for CIF files. Defaults to generated.")
    parser.add_argument("-n", "--num-samples", "--num_samples", type=int, default=1,
                        help="Total crystals to generate. Defaults to 1.")
    parser.add_argument("-b", "--batch-size", "--batch_size", type=int, default=1,
                        help="Maximum crystals per batch. Defaults to 1.")
    parser.add_argument("--device", help="Torch device, e.g. cpu or cuda:0. Defaults to CUDA if available, else CPU.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling. Defaults to 42.")
    args = parser.parse_args(argv)

    if args.num_samples < 1 or args.batch_size < 1:
        parser.error("--num-samples and --batch-size must both be positive")
    statistic_path = args.statistic or args.checkpoint.with_name("statistic.json")
    for path in (args.checkpoint, statistic_path):
        if not path.is_file():
            parser.error(f"Expected an input file at {path}")

    args.output.mkdir(parents=True, exist_ok=True)
    paths = [args.output / f"crystal_{i:06d}.cif" for i in range(1, args.num_samples + 1)]
    for path in paths:
        if path.exists():
            parser.error(f"{path} already exists. Choose a different --output directory.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device=device)
    sampler = Sampler(
        atom_pred=AtomTypePredictor(model.atom_corruption),
        pos_pred=PositionPredictor(model.position_corruption),
        lat_pred=LatticePredictor(model.lattice_corruption),
        corrector=LangevinCorrector(),
    )
    torch.manual_seed(args.seed)
    for start in range(0, args.num_samples, args.batch_size):
        batch_paths = paths[start:start + args.batch_size]
        structures = generate_batch(model, sampler, statistic_path, batch_size=len(batch_paths))
        for path, structure in zip(batch_paths, structures, strict=True):
            cif = structure.to(fmt="cif")
            output = path.open("x", encoding="utf-8")
            try:
                with output:
                    output.write(cif)
            except BaseException:
                path.unlink(missing_ok=True)
                raise
        print(f"Saved {start + len(batch_paths)}/{args.num_samples} crystals to {args.output}", flush=True)


if __name__ == "__main__":
    main()
