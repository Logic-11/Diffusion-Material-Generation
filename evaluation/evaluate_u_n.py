#!/usr/bin/env python3
"""Evaluate uniqueness and novelty with MatterGen's official structure matcher.

This script intentionally computes only structure-based metrics. It does not
relax structures and therefore never requires a GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from collections.abc import Callable, Iterable

import numpy as np
from ase.io import iread
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm


from structure_matcher import (
    DefaultDisorderedStructureMatcher,
    DefaultOrderedStructureMatcher,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def natural_key(value: str) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def load_extxyz(path: Path, limit: int | None) -> tuple[list[Structure], list[str]]:
    structures, ids = [], []
    for index, atoms in enumerate(iread(path, index=slice(0, limit), format="extxyz")):
        if atoms.cell.rank != 3 or atoms.cell.volume <= 0:
            raise ValueError(f"{path}, frame {index}: a full 3D lattice is required")
        structures.append(AseAtomsAdaptor.get_structure(atoms))
        default_id = (
            f"gen_{index}"
            if path.stem == "generated_crystals"
            else f"{path.stem}_{index}"
        )
        ids.append(str(atoms.info.get("structure_id", default_id)))
    return structures, ids


def load_cif_directory(
    path: Path, limit: int | None
) -> tuple[list[Structure], list[str]]:
    files = sorted(path.glob("*.cif"), key=lambda value: natural_key(value.name))
    if limit is not None:
        files = files[:limit]
    return [Structure.from_file(file) for file in files], [file.stem for file in files]


def load_structures(path: Path, limit: int | None) -> tuple[list[Structure], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Structure path does not exist: {path}")
    if path.is_dir():
        return load_cif_directory(path, limit)
    if path.suffix.lower() in {".extxyz", ".xyz"}:
        return load_extxyz(path, limit)
    if path.suffix.lower() == ".cif":
        return [Structure.from_file(path)], [path.stem]
    raise ValueError(f"Unsupported structure path: {path}")


def load_mp20_cache(
    cache_root: Path, split: str, limit: int | None = None
) -> tuple[list[Structure], list[str]]:
    splits = ("train", "val", "test") if split == "all" else (split,)
    structures, ids = [], []
    for split_name in splits:
        directory = cache_root / split_name
        atomic_numbers = np.load(directory / "atomic_numbers.npy", mmap_mode="r")
        positions = np.load(directory / "pos.npy", mmap_mode="r")
        cells = np.load(directory / "cell.npy", mmap_mode="r")
        counts = np.load(directory / "num_atoms.npy", mmap_mode="r")
        structure_ids = np.load(directory / "structure_id.npy", mmap_mode="r")
        offsets = np.concatenate(([0], np.cumsum(counts)))
        stop = len(counts) if limit is None else min(len(counts), limit - len(structures))
        for index in range(stop):
            start, end = int(offsets[index]), int(offsets[index + 1])
            structures.append(
                Structure(
                    lattice=Lattice(cells[index]),
                    species=atomic_numbers[start:end].astype(int).tolist(),
                    coords=positions[start:end],
                    coords_are_cartesian=False,
                )
            )
            ids.append(f"{split_name}:{structure_ids[index]}")
        if limit is not None and len(structures) >= limit:
            break
    return structures, ids


class ReferenceIndex:
    def __init__(
        self, name: str, size: int,
        entries_for_formula: Callable[[str], Iterable[tuple[str, Structure]]],
    ):
        self.name = name
        self.size = size
        self.entries_for_formula = entries_for_formula


def make_reference(
    structures: list[Structure], ids: list[str], name: str
) -> ReferenceIndex:
    if not structures:
        raise ValueError(f"Empty reference set: {name}")
    groups: dict[str, list[tuple[str, Structure]]] = defaultdict(list)
    for structure, structure_id in zip(structures, ids, strict=True):
        groups[structure.composition.reduced_formula].append((structure_id, structure))
    return ReferenceIndex(name, len(structures), lambda formula: groups.get(formula, ()))


def choose_group_key(structure: Structure) -> str:
    return structure.composition.reduced_formula


def compute_uniqueness(
    structures: list[Structure], matcher
) -> tuple[np.ndarray, list[int | None]]:
    flags = np.zeros(len(structures), dtype=bool)
    duplicate_of: list[int | None] = [None] * len(structures)
    groups: dict[str, list[tuple[int, Structure]]] = defaultdict(list)
    for index, structure in enumerate(structures):
        groups[choose_group_key(structure)].append((index, structure))

    for entries in tqdm(groups.values(), desc="Computing uniqueness"):
        kept: list[tuple[int, Structure]] = []
        for index, structure in entries:
            matched_index = next(
                (
                    kept_index
                    for kept_index, other in kept
                    if matcher.fit(structure, other)
                ),
                None,
            )
            if matched_index is None:
                flags[index] = True
                kept.append((index, structure))
            else:
                duplicate_of[index] = matched_index
    return flags, duplicate_of


def compute_novelty(
    structures: list[Structure],
    reference: ReferenceIndex,
    matcher,
) -> tuple[np.ndarray, list[list[str]]]:
    novel = np.zeros(len(structures), dtype=bool)
    matches: list[list[str]] = [[] for _ in structures]
    for index, structure in enumerate(tqdm(structures, desc="Computing novelty")):
        for reference_id, reference_structure in reference.entries_for_formula(
            choose_group_key(structure)
        ):
            if matcher.fit(structure, reference_structure):
                matches[index].append(reference_id)
        novel[index] = not matches[index]
    return novel, matches


def parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Dataset must use LABEL=PATH syntax")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Both LABEL and PATH are required")
    return label, Path(path).expanduser().resolve()


def main(resources: ExitStack) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        type=parse_dataset,
        help="Structure set in LABEL=PATH form. Repeat for pre/relaxed.",
    )
    parser.add_argument(
        "--reference",
        choices=("mp20-cache", "official-mp2020", "structures", "none"),
        default="mp20-cache",
    )
    parser.add_argument(
        "--mp20-cache",
        type=Path,
        default=(
            PROJECT_ROOT.parent
            / "diffusion_mattergen"
            / "data_cache"
            / "mp_20"
        ),
    )
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="train")
    parser.add_argument("--reference-gz", type=Path)
    parser.add_argument("--reference-path", type=Path, help="Reference CIF, CIF directory or extxyz.")
    parser.add_argument("--ordered-matcher", action="store_true")
    parser.add_argument("--limit", type=int, help="Only evaluate the first N candidates.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "U_N_results",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    if len({label for label, _ in args.dataset}) != len(args.dataset):
        parser.error("Use a different LABEL for each --dataset")
    if (args.reference == "structures") != (args.reference_path is not None):
        parser.error("Use --reference structures together with --reference-path PATH")
    if args.reference_gz and args.reference != "official-mp2020":
        parser.error("--reference-gz requires --reference official-mp2020")

    matcher = (
        DefaultOrderedStructureMatcher()
        if args.ordered_matcher
        else DefaultDisorderedStructureMatcher()
    )

    reference = None
    if args.reference == "mp20-cache":
        if not args.mp20_cache.is_dir():
            parser.error("MP-20 cache not found. Supply --mp20-cache PATH or choose another --reference.")
        reference_structures, reference_ids = load_mp20_cache(
            args.mp20_cache.resolve(), args.split
        )
        reference = make_reference(
            reference_structures, reference_ids, f"MP-20 {args.split} split"
        )
    elif args.reference == "official-mp2020":
        from official_reference import open_official_reference

        reference = ReferenceIndex(*resources.enter_context(open_official_reference(args.reference_gz)))
    elif args.reference == "structures":
        reference = make_reference(*load_structures(args.reference_path, None), str(args.reference_path))

    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "reference": reference.name if reference is not None else "none",
        "reference_size": reference.size if reference is not None else 0,
        "matcher": type(matcher).__name__,
        "datasets": {},
    }
    detailed_rows: list[dict[str, object]] = []

    for label, path in args.dataset:
        structures, ids = load_structures(path, args.limit)
        if not structures:
            parser.error(f"No structures found in {path}")
        unique, duplicate_of = compute_uniqueness(structures, matcher)
        if reference is not None:
            novel, matches = compute_novelty(structures, reference, matcher)
        else:
            novel = None
            matches = [[] for _ in structures]

        summary["datasets"][label] = {
            "path": str(path),
            "count": len(structures),
            "unique_count": int(unique.sum()),
            "unique_fraction": float(unique.mean()),
            "novel_count": None if novel is None else int(novel.sum()),
            "novel_fraction": None if novel is None else float(novel.mean()),
            "novel_unique_count": (
                None if novel is None else int((novel & unique).sum())
            ),
            "novel_unique_fraction": (
                None if novel is None else float((novel & unique).mean())
            ),
            "non_unique_count": int((~unique).sum()),
            "in_reference_count": None if novel is None else int((~novel).sum()),
        }

        for index, (structure, structure_id) in enumerate(zip(structures, ids)):
            detailed_rows.append(
                {
                    "dataset": label,
                    "index": index,
                    "structure_id": structure_id,
                    "reduced_formula": structure.composition.reduced_formula,
                    "is_unique": int(unique[index]),
                    "duplicate_of": (
                        "" if duplicate_of[index] is None else ids[duplicate_of[index]]
                    ),
                    "is_novel": "" if novel is None else int(novel[index]),
                    "is_novel_unique": (
                        "" if novel is None else int(novel[index] and unique[index])
                    ),
                    "reference_matches": ";".join(
                        str(match) for match in matches[index]
                    ),
                }
            )

    with (args.output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)

    with (args.output / "per_structure.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=detailed_rows[0].keys())
        writer.writeheader()
        writer.writerows(detailed_rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved: {args.output / 'summary.json'}")
    print(f"Saved: {args.output / 'per_structure.csv'}")


if __name__ == "__main__":
    with ExitStack() as resources:
        main(resources)
