"""Standardize one CIF, or build and inspect a complete crystal dataset cache."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

from .standardize import DEFAULT_PRIMITIVE_TOLERANCE, standardize_cif


def _write_npz(path: Path, arrays: dict[str, np.ndarray], overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz.tmp", delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            np.savez_compressed(stream, **arrays)
        if overwrite:
            os.replace(temporary, path)
        elif os.name == "nt":
            os.rename(temporary, path)  # On Windows this fails if the destination already exists.
        else:
            os.link(temporary, path)  # Atomic no-clobber publication on POSIX.
    except FileExistsError as exc:
        raise FileExistsError(f"Output already exists: {path}. Choose a new path or pass --overwrite.") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _dataset_main(argv) -> int:
    from .cache import build_dataset, inspect_dataset

    parser = argparse.ArgumentParser(prog="python -m data", description="Build or inspect standardized crystal caches.")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Convert split CSVs or CIF files into NumPy caches")
    build.add_argument("source", type=Path, help="train/val/test CSV directory, one CSV, or CIF file/directory")
    build.add_argument("--output", type=Path, required=True, help="new cache directory (must not exist)")
    build.add_argument("--primitive-tolerance", type=float, default=DEFAULT_PRIMITIVE_TOLERANCE)
    build.add_argument("--skip-invalid", action="store_true", help="skip invalid crystals and record errors.jsonl")
    inspect = commands.add_parser("inspect", help="Read cache counts and preprocessing metadata")
    inspect.add_argument("cache", type=Path)
    args = parser.parse_args(argv)

    def progress(split, count):
        if count == 1 or count % 1000 == 0:
            print(f"{split}: {count} source rows processed", file=sys.stderr, flush=True)

    try:
        if args.command == "build":
            result = build_dataset(args.source, args.output, primitive_tolerance=args.primitive_tolerance,
                                   skip_invalid=args.skip_invalid, progress=progress)
        else:
            result = inspect_dataset(args.cache)
    except (ValueError, OSError, csv.Error) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("build", "inspect"):
        return _dataset_main(argv)
    parser = argparse.ArgumentParser(
        prog="python -m data", description="Standardize one periodic crystal from CIF to NPZ.",
        epilog="Datasets: python -m data build SOURCE --output CACHE; python -m data inspect CACHE.",
    )
    parser.add_argument("cif", type=Path, help="input CIF containing one ordered, fully occupied crystal")
    parser.add_argument("--output", type=Path, required=True, help="destination .npz file")
    parser.add_argument("--primitive-tolerance", type=float, default=DEFAULT_PRIMITIVE_TOLERANCE,
                        help="primitive-cell site tolerance in angstrom (default: %(default)s)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".npz":
        parser.error("--output must name a .npz file.")
    if args.cif.resolve() == args.output.resolve():
        parser.error("Choose an output path different from the input CIF.")
    try:
        arrays = standardize_cif(args.cif, primitive_tolerance=args.primitive_tolerance)
        _write_npz(args.output, arrays, args.overwrite)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")
    volume = np.linalg.det(arrays["lattice"])
    print(f"{args.output}  num_atoms={int(arrays['num_atoms'])}  volume={volume:.5f} A^3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
