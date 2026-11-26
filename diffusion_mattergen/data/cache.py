"""Stream CIF/CSV datasets into MatterGen-style, split-preserving NumPy caches."""

from contextlib import ExitStack
import csv
from datetime import datetime, timezone
from functools import partial
import hashlib
from importlib.metadata import version
import json
from numbers import Real
import os
from pathlib import Path
import shutil
import tempfile
import warnings

import numpy as np

from .standardize import (
    DEFAULT_PRIMITIVE_TOLERANCE, standardize_cif_text, standardize_mattergen_cif_text,
)


FORMAT = "diffusion_mattergen.crystal_arrays"
FORMAT_VERSION = 1
SPLITS = ("train", "val", "test")
PREPROCESS_MODES = ("strict", "mattergen")
CORE_FILES = {
    "frac_coords": "pos.npy", "lattice": "cell.npy",
    "atomic_numbers": "atomic_numbers.npy", "num_atoms": "num_atoms.npy",
    "structure_id": "structure_id.npy",
}


def _sources(source: Path) -> tuple[Path, dict[str, list[Path]]]:
    if source.is_file():
        if source.suffix.lower() not in (".csv", ".cif"):
            raise ValueError("Input must be a CSV, a CIF, or a directory containing them.")
        split = source.stem.lower() if source.suffix.lower() == ".csv" else "train"
        return source.parent, {split if split in SPLITS else "train": [source]}
    if not source.is_dir():
        raise FileNotFoundError(source)
    csv_files = {p.stem.lower(): p for p in source.iterdir()
                 if p.is_file() and p.suffix.lower() == ".csv"}
    if csv_files:
        unknown = set(csv_files) - set(SPLITS)
        if unknown:
            raise ValueError(f"CSV directories must use train.csv/val.csv/test.csv; unexpected: {sorted(unknown)}")
        return source, {split: [csv_files[split]] for split in SPLITS if split in csv_files}
    files = sorted((p for p in source.rglob("*") if p.is_file() and p.suffix.lower() == ".cif"),
                   key=lambda p: p.relative_to(source).as_posix())
    if not files:
        raise ValueError(f"No CSV or CIF input files found in {source}")
    return source, {"train": files}


def _rows(path: Path, relative: str):
    if path.suffix.lower() == ".cif":
        yield relative, path.read_text(encoding="utf-8-sig"), 1, {}
        return
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        if not {"cif", "material_id"}.issubset(headers) or len(headers) != len(set(headers)):
            raise ValueError(f"{relative}: CSV needs unique columns including cif and material_id.")
        for row_number, row in enumerate(reader, start=1):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{relative}, record {row_number}: CSV field count does not match its header.")
            structure_id = row["material_id"].strip()
            if not structure_id:
                raise ValueError(f"{relative}, record {row_number}: material_id is empty.")
            yield structure_id, row["cif"], row_number, {k: v for k, v in row.items() if k != "cif"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finish_array(raw: Path, output: Path, dtype, shape: tuple) -> None:
    """Copy a sequential spool into a standard NPY without loading it all into RAM."""
    source = np.memmap(raw, dtype=dtype, mode="r", shape=shape)
    target = None
    try:
        target = np.lib.format.open_memmap(output, mode="w+", dtype=dtype, shape=shape)
        for start in range(0, shape[0], 65536):
            target[start:start + 65536] = source[start:start + 65536]
        target.flush()
    finally:
        source._mmap.close()
        if target is not None:
            target._mmap.close()
    raw.unlink()


def build_dataset(
    source: str | Path, output: str | Path, *,
    primitive_tolerance: float = DEFAULT_PRIMITIVE_TOLERANCE,
    preprocess: str = "strict", skip_invalid: bool = False, progress=None,
) -> dict:
    """Standardize all input structures once and preserve input splits and row order.

    A split CSV requires material_id and cif. Other columns are preserved verbatim
    in records.jsonl. A CIF directory is imported recursively into train. Edges are
    deliberately computed later from each batch's current coordinates and lattice.
    Existing output directories are never replaced. progress(split, rows_seen) is
    optional. Invalid crystal rows abort by default; explicitly skipped rows are
    recorded in errors.jsonl. Duplicate IDs and I/O/schema errors always abort.

    The mattergen mode uses pymatgen's default primitive tolerances and saves
    the reduced lattice for polar decomposition when the dataset loads it.
    """
    if preprocess not in PREPROCESS_MODES:
        raise ValueError(f"preprocess must be one of {PREPROCESS_MODES}")
    if (not isinstance(primitive_tolerance, Real) or isinstance(primitive_tolerance, bool)
            or not 0 < primitive_tolerance < np.inf):
        raise ValueError("primitive_tolerance must be a finite positive number in angstrom.")
    mattergen = preprocess == "mattergen"
    if mattergen and primitive_tolerance != DEFAULT_PRIMITIVE_TOLERANCE:
        raise ValueError("A custom primitive_tolerance is only supported with preprocess='strict'.")
    standardize = (standardize_mattergen_cif_text if mattergen else
                   partial(standardize_cif_text, primitive_tolerance=primitive_tolerance))
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; use a new cache directory: {output}")
    root, splits = _sources(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)).resolve()
    # All recursive cleanup below is confined to this newly created sibling directory.
    if temporary.parent != output.parent or temporary == output:
        raise RuntimeError("Unexpected temporary cache location")
    manifest = {
        "format": FORMAT, "format_version": FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(), "source_root": str(root),
        "standardization": {
            "preprocess": preprocess,
            "primitive_tolerance_angstrom": None if mattergen else primitive_tolerance,
            "primitive": True, "niggli": True, "symmetric_lattice": not mattergen,
            "lattice_vectors": "rows", "length_unit": "angstrom",
        },
        "split_policy": "preserve source CSV splits and row order; CIF directories use train",
        "core_files": CORE_FILES, "skip_invalid": bool(skip_invalid),
        "num_structures": 0, "num_atoms": 0, "num_skipped": 0,
        "sources": [], "splits": {},
        "versions": {name: version(name) for name in ("numpy", "pymatgen")},
    }
    seen = {}
    try:
        with (temporary / "errors.jsonl").open("w", encoding="utf-8") as errors:
            for split, paths in splits.items():
                destination = temporary / split
                destination.mkdir()
                ids, counts = [], []
                rows_seen, skipped, warning_rows = 0, 0, 0
                with ExitStack() as stack:
                    streams = {name: stack.enter_context((destination / f"{name}.bin").open("wb"))
                               for name in ("pos", "cell", "atomic_numbers")}
                    records = stack.enter_context((destination / "records.jsonl").open("w", encoding="utf-8"))
                    for path in paths:
                        relative = path.relative_to(root).as_posix()
                        manifest["sources"].append({"path": relative, "split": split, "sha256": _sha256(path)})
                        for structure_id, cif, row_number, fields in _rows(path, relative):
                            rows_seen += 1
                            location = f"{relative}, record {row_number} ({structure_id})"
                            if structure_id in seen:
                                raise ValueError(f"Duplicate structure_id at {location}; first seen at {seen[structure_id]}")
                            seen[structure_id] = location
                            try:
                                with warnings.catch_warnings(record=True) as captured:
                                    warnings.simplefilter("always")
                                    cell = standardize(cif, source_name=location)
                            except ValueError as exc:
                                if not skip_invalid:
                                    raise
                                skipped += 1
                                errors.write(json.dumps({"split": split, "structure_id": structure_id,
                                                         "source": relative, "record": row_number,
                                                         "error": str(exc)}, ensure_ascii=False) + "\n")
                            else:
                                notes = list(dict.fromkeys(str(w.message) for w in captured))
                                warning_rows += bool(notes)
                                cell["frac_coords"].tofile(streams["pos"])
                                cell["lattice"].tofile(streams["cell"])
                                cell["atomic_numbers"].tofile(streams["atomic_numbers"])
                                ids.append(structure_id)
                                counts.append(int(cell["num_atoms"]))
                                records.write(json.dumps({"structure_id": structure_id, "source": relative,
                                                          "record": row_number, "fields": fields,
                                                          "parser_warnings": notes}, ensure_ascii=False) + "\n")
                            if progress is not None:
                                progress(split, rows_seen)
                if not counts:
                    raise ValueError(f"Split {split} contains no valid structures ({rows_seen} source rows).")
                num_structures, num_atoms = len(counts), sum(counts)
                for name, dtype, shape in (("pos", np.float64, (num_atoms, 3)),
                                           ("cell", np.float64, (num_structures, 3, 3)),
                                           ("atomic_numbers", np.int64, (num_atoms,))):
                    _finish_array(destination / f"{name}.bin", destination / f"{name}.npy", dtype, shape)
                np.save(destination / "num_atoms.npy", np.asarray(counts, dtype=np.int64), allow_pickle=False)
                np.save(destination / "structure_id.npy", np.asarray(ids, dtype=np.str_), allow_pickle=False)
                manifest["splits"][split] = {
                    "num_structures": num_structures, "num_atoms": num_atoms,
                    "min_atoms": min(counts), "max_atoms": max(counts),
                    "num_source_rows": rows_seen, "num_skipped": skipped,
                    "rows_with_parser_warnings": warning_rows,
                }
                manifest["num_structures"] += num_structures
                manifest["num_atoms"] += num_atoms
                manifest["num_skipped"] += skipped
        (temporary / "metadata.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"Output appeared during build: {output}")
        os.rename(temporary, output)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def inspect_dataset(path: str | Path) -> dict:
    """Read a completed cache's manifest without loading crystal arrays."""
    with (Path(path) / "metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("format") != FORMAT or metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported crystal cache format/version.")
    return metadata
