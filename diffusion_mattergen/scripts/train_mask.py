"""Step 7 joint training entry point (the historical filename is retained).

Run from the project root: python -B -m scripts.train_mask --max-steps 1
Uses standardized data.CrystalDataset NPY caches, not the old SQLite graphs.
"""

import argparse
from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import CrystalDataset, collate_crystals
from src.corruption import AtomTypeCorruption, LatticeCorruption, PositionCorruption
from src.denoiser import Denoiser
from src.diffusion import JointDiffusion
from src.loss import JointDiffusionLoss
from src.scheduler import LatticeVPScheduler, MaskScheduler, PositionVEScheduler


@torch.no_grad()
def training_statistics(dataset):
    """Compute constants once from the ENTIRE supplied training split.

    Density is mean_i(N_i / V_i), not sum(N_i)/sum(V_i). No validation/test
    samples or minibatch estimates enter the lattice schedule or saved priors.
    """
    count_histogram, elements = Counter(), set()
    density_sum, structures = 0.0, 0
    for clean in DataLoader(dataset, batch_size=512, shuffle=False,
                            num_workers=0, collate_fn=collate_crystals):
        counts = clean["num_atoms"]
        volumes = torch.linalg.det(clean["lattice"].double())
        if not torch.isfinite(volumes).all() or torch.any(volumes <= 0):
            raise ValueError("Training cells must have finite positive volumes")
        density_sum += (counts.double() / volumes).sum().item()
        structures += counts.numel()
        count_histogram.update(counts.tolist())
        elements.update(clean["atomic_numbers"].unique().tolist())
    if structures == 0:
        raise ValueError("The training split is empty")
    sizes = sorted(count_histogram)
    return {
        "limit_density": density_sum / structures,
        "num_structures": structures,
        "atomic_numbers": sorted(elements),
        "num_atoms_values": sizes,
        "num_atoms_probabilities": [count_histogram[n] / structures for n in sizes],
    }


def head_gradient_norms(denoiser):
    """L2 norm of each output branch's gradients after TOTAL-loss backward."""
    groups = {"atom": [], "coord": [], "lattice": []}
    for name, parameter in denoiser.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith("output_head."):
            groups["atom"].append(parameter.grad.detach().norm())
        elif name.startswith("EdgeReadlayers."):
            for field in ("coord", "lattice"):
                if f".{field}_" in name:
                    groups[field].append(parameter.grad.detach().norm())
    return {f"grad_{field}": torch.stack(norms).norm().item() if norms else 0.0
            for field, norms in groups.items()}


def train_step(model, optimizer, clean, k=None, noise=None):
    """One update; explicit k/noise are useful for reproducible demonstrations."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = model.compute_loss(clean, k=k, noise=noise)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite joint training loss")
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Non-finite gradient: {name}")
    gradients = head_gradient_norms(model.denoiser)
    optimizer.step()
    return {"loss": loss.detach().item(),
            **{key: value.item() for key, value in metrics.items()}, **gradients}


def main():
    parser = argparse.ArgumentParser(description="Train the three-field MatterGen diffusion loss")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/mp_20")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--max-steps", type=int, help="Stop after this many optimizer updates")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-radial", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path, help="Optional new checkpoint path")
    args = parser.parse_args()
    if (min(args.batch_size, args.epochs, args.hidden_dim, args.num_layers, args.threads) < 1
            or args.num_radial < 2 or args.workers < 0 or not np.isfinite(args.lr) or args.lr <= 0
            or (args.max_steps is not None and args.max_steps < 1)):
        parser.error("Invalid training sizes, step limit, or learning rate")
    if args.checkpoint is not None and args.checkpoint.exists():
        parser.error(f"Checkpoint already exists: {args.checkpoint}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)

    with CrystalDataset(args.data_dir, split="train") as dataset:
        stats = training_statistics(dataset)
        config = {
            "model": {"hidden_dim": args.hidden_dim, "num_radial": args.num_radial,
                      "num_layers": args.num_layers, "atom_types": 118},
            "mask_schedule": {"timesteps": 1000},
            "position_schedule": {"sigma_min": 0.01, "sigma_max": 5.0},
            "lattice_schedule": {"limit_density": stats["limit_density"], "beta_min": 0.1,
                                 "beta_max": 20.0, "limit_var_scaling_constant": 0.25},
            "loss": {"coord_weight": 0.1, "lattice_weight": 1.0,
                     "atom_weight": 1.0, "d3pm_hybrid_lambda": 0.01},
            "graph": {"cutoff": 7.0, "max_neighbor": 50, "max_offset_integer": 3},
            "time_grid": "uniform integer k in 1..K; t=k/K",
            "output_parameterization": "clean-element logits; coordinate/lattice score_times_std",
            "atom_convention": "input MASK=0, elements=1..118; posterior MASK is last",
            "reduction": "feature mean, atom sum per crystal, crystal mean",
            "optimizer": {"lr": args.lr},
            "lr_scheduler": {"factor": 0.6, "patience": 100, "min_lr": 1e-6},
        }
        model = JointDiffusion(
            Denoiser(**config["model"]),
            AtomTypeCorruption(MaskScheduler(**config["mask_schedule"])),
            PositionCorruption(PositionVEScheduler(**config["position_schedule"])),
            LatticeCorruption(LatticeVPScheduler(**config["lattice_schedule"])),
            loss_fn=JointDiffusionLoss(**config["loss"]), **config["graph"],
        ).to(device)
        # Follow the YAML optimizer configuration (Adam, lr=1e-4), which overrides
        # the upstream Lightning class's fallback optimizer.
        optimizer = torch.optim.Adam(model.parameters(), **config["optimizer"])
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **config["lr_scheduler"])
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, collate_fn=collate_crystals)
        print(f"device={device} training_structures={len(dataset)} "
              f"limit_density={stats['limit_density']:.8f}", flush=True)

        steps = 0
        for epoch in range(args.epochs):
            totals, seen = Counter(), 0
            for batch_index, clean in enumerate(loader):
                clean = {key: value.to(device) if torch.is_tensor(value) else value
                         for key, value in clean.items()}
                metrics = train_step(model, optimizer, clean)
                size = clean["num_atoms"].numel()
                for key, value in metrics.items():
                    totals[key] += size * value
                seen += size
                steps += 1
                if args.max_steps is not None and steps >= args.max_steps:
                    break
            partial = batch_index + 1 < len(loader)
            if not partial:
                lr_scheduler.step(totals["loss"] / seen)
            report = " ".join(f"{key}={value / seen:.6g}" for key, value in totals.items())
            print(f"epoch={epoch + 1} steps={steps} crystals={seen} partial={partial} "
                  f"lr={optimizer.param_groups[0]['lr']:.6g} {report}", flush=True)
            if args.max_steps is not None and steps >= args.max_steps:
                break

        if args.checkpoint is not None:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            with args.checkpoint.open("xb") as handle:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "config": config, "training_statistics": stats,
                    "data_dir": str(args.data_dir.resolve()),
                    "standardization": dataset.metadata["standardization"],
                    "seed": args.seed, "steps": steps, "epoch": epoch + 1,
                    "partial_epoch": partial,
                }, handle)
            print(f"checkpoint={args.checkpoint.resolve()}", flush=True)


if __name__ == "__main__":
    main()
