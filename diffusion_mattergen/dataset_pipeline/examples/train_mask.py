"""Minimal integration with diffusion_mattergen's existing atom-type denoiser."""

import argparse
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[1]
MASK_DIFF = PROJECT.parent
sys.path.insert(0, str(MASK_DIFF))
sys.path.insert(0, str(PROJECT))

from crystal_dataset import CrystalDataset, collate_crystals
from src.denoiser import Denoiser
from src.diffusion import atomtype_ce, mask_atomtype


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    with CrystalDataset(args.database, split="train") as dataset:
        if len(dataset) == 0:
            raise ValueError("The training split is empty")
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, collate_fn=collate_crystals,
            generator=torch.Generator().manual_seed(args.seed),
            pin_memory=device.type == "cuda",
        )
        model = Denoiser(hidden_dim=32, num_layers=3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(args.epochs):
            model.train()
            loss_sum, masked_count, steps = 0.0, 0, 0
            for batch in loader:
                batch = {key: value.to(device) if torch.is_tensor(value) else value
                         for key, value in batch.items()}
                # Expand each crystal's time to its atoms using the batch assignment.
                graph_t = torch.randint(1, 101, (batch["num_graphs"],), device=device).float() / 100
                atom_t = graph_t[batch["batch"]]
                masked_atoms, mask = mask_atomtype(batch["atomic_numbers"], atom_t)
                if not mask.any():
                    continue
                logits = model(masked_atoms, batch["edge_index"], batch["edge_vectors"], atom_t[:, None])
                loss = atomtype_ce(logits, batch["atomic_numbers"], mask)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                count = int(mask.sum())
                loss_sum += loss.item() * count
                masked_count += count
                steps += 1
            if (epoch + 1) % 50 == 0 or epoch + 1 == args.epochs:
                if masked_count:
                    print(f"epoch={epoch + 1} loss={loss_sum / masked_count:.6f} masked_atoms={masked_count} steps={steps}")
                else:
                    print(f"epoch={epoch + 1}: no atoms masked, no optimization step")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.lr <= 0:
        parser.error("epochs, batch-size and lr must be positive; workers must be nonnegative")
    train(args)


# Required on Windows when DataLoader uses worker processes.
if __name__ == "__main__":
    main()
