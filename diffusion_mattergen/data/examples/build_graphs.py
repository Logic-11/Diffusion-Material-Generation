"""Load cached crystals and verify every batch with the project's periodic graph builder."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data import CrystalDataset, collate_crystals, inspect_dataset
from src.geometry import build_periodic_graph


@torch.no_grad()
def check_graphs(cache, *, split="all", batch_size=64, workers=0, device="cpu", cutoff=7.0,
                 max_neighbor=50, max_batches=None):
    """Return counts after checking graph membership, geometry and split coverage."""
    metadata = inspect_dataset(cache)
    splits = list(metadata["splits"]) if split == "all" else [split]
    results = {}
    for name in splits:
        started = time.perf_counter()
        counts = {"num_structures": 0, "num_atoms": 0, "num_edges": 0, "num_batches": 0}
        with CrystalDataset(cache, split=name) as dataset:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                collate_fn=collate_crystals)
            for batch in loader:
                batch = {key: value.to(device) if torch.is_tensor(value) else value
                         for key, value in batch.items()}
                edges, images, vectors, distances, num_edges, edge_ptr, edge_batch = build_periodic_graph(
                    batch["frac_coords"], batch["lattice"], cutoff=cutoff,
                    batch=batch["batch"], max_neighbor=max_neighbor,
                )
                i, j = edges
                torch.testing.assert_close(batch["batch"][i], batch["batch"][j])
                torch.testing.assert_close(edge_batch, batch["batch"][i])
                torch.testing.assert_close(num_edges, torch.bincount(edge_batch, minlength=batch["num_graphs"]))
                torch.testing.assert_close(edge_ptr, torch.cat((num_edges.new_zeros(1), num_edges.cumsum(0))))
                displacement = batch["frac_coords"][j] + images - batch["frac_coords"][i]
                expected = torch.einsum("ei,eij->ej", displacement, batch["lattice"][edge_batch])
                torch.testing.assert_close(vectors, expected, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(distances, vectors.norm(dim=-1))
                if (not torch.isfinite(vectors).all() or not torch.isfinite(distances).all()
                        or torch.any(distances <= 0) or torch.any(distances > cutoff)):
                    raise ValueError(f"Invalid graph geometry for {batch['structure_id']}")
                counts["num_structures"] += batch["num_graphs"]
                counts["num_atoms"] += int(batch["num_atoms"].sum())
                counts["num_edges"] += edges.shape[1]
                counts["num_batches"] += 1
                if counts["num_batches"] == 1 or counts["num_batches"] % 50 == 0:
                    print(f"{name}: {counts['num_structures']}/{len(dataset)} crystals checked", file=sys.stderr, flush=True)
                if max_batches is not None and counts["num_batches"] >= max_batches:
                    break
            counts["complete_split"] = counts["num_structures"] == len(dataset)
            if max_batches is None:
                if not counts["complete_split"] or counts["num_atoms"] != metadata["splits"][name]["num_atoms"]:
                    raise ValueError(f"Graph validation did not cover the entire {name} split")
        counts["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        results[name] = counts
    return {"cache": str(Path(cache).resolve()), "batch_size": batch_size, "dtype": "float32",
            "device": str(device), "cutoff_angstrom": cutoff, "max_neighbor": max_neighbor,
            "splits": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cutoff", type=float, default=7.0)
    parser.add_argument("--max-neighbor", type=int, default=50)
    parser.add_argument("--max-batches", type=int, help="optional quick check, per split")
    parser.add_argument("--threads", type=int, default=1, help="CPU threads; small graph operations usually favor one")
    parser.add_argument("--report", type=Path, help="optional new JSON report path")
    args = parser.parse_args()
    if (args.batch_size < 1 or args.workers < 0 or args.threads < 1 or args.max_neighbor < 1
            or (args.max_batches is not None and args.max_batches < 1)):
        parser.error("Counts must be positive; workers may be zero.")
    if args.report is not None and args.report.exists():
        parser.error(f"Report already exists: {args.report}")
    torch.set_num_threads(args.threads)
    result = check_graphs(args.cache, split=args.split, batch_size=args.batch_size, workers=args.workers,
                          device=args.device, cutoff=args.cutoff, max_neighbor=args.max_neighbor,
                          max_batches=args.max_batches)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")


if __name__ == "__main__":  # Required by Windows DataLoader spawn workers.
    main()
