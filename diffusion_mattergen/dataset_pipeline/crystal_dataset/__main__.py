import argparse
import json
import sqlite3
import sys

from .build import build_dataset
from .storage import inspect_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build and inspect periodic crystal datasets")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Recursively import CIF files into a new SQLite dataset")
    build.add_argument("source", help="CIF file or directory")
    build.add_argument("output", help="New .sqlite path (must not exist)")
    build.add_argument("--cutoff", type=float, default=5.0, help="Inclusive neighbor cutoff in angstrom (default: 5)")
    build.add_argument("--split-ratios", nargs=3, type=float, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VAL", "TEST"))
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--skip-invalid", action="store_true", help="Skip entire invalid CIFs and record errors")
    inspect = commands.add_parser("inspect", help="Show build configuration and split counts")
    inspect.add_argument("database")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build_dataset(args.source, args.output, cutoff=args.cutoff,
                                   split_ratios=args.split_ratios, seed=args.seed, skip_invalid=args.skip_invalid)
        else:
            result = inspect_dataset(args.database)
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["split_counts"]["val"] == 0 or result["split_counts"]["test"] == 0:
        print("Note: validation/test may be empty for a small dataset; see split_counts.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
