"""Parse CIFs once and persist complete periodic radius graphs."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import tempfile
import warnings

import numpy as np

from .storage import SCHEMA, SCHEMA_VERSION, pack_arrays

SPLITS = ("train", "val", "test")


def structure_arrays(structure, cutoff: float) -> dict[str, np.ndarray]:
    """Rows of lattice are basis vectors; edge 0 is center, edge 1 neighbor."""
    from pymatgen.core import Structure

    if len(structure) == 0 or not structure.is_ordered:
        raise ValueError("Only nonempty, fully occupied, ordered structures are supported")
    lattice = np.asarray(structure.lattice.matrix, dtype=np.float64)
    frac = np.asarray(structure.frac_coords, dtype=np.float64)
    numbers = np.asarray(structure.atomic_numbers, dtype=np.int64)
    if not np.isfinite(lattice).all() or not np.isfinite(frac).all():
        raise ValueError("Lattice and coordinates must be finite")
    if abs(np.linalg.det(lattice)) < 1e-8:
        raise ValueError("Lattice is singular or has near-zero volume")
    if not ((numbers >= 1) & (numbers <= 118)).all():
        raise ValueError("Atomic numbers must be in [1, 118]; 0 is reserved for masking")

    frac = np.mod(frac, 1.0)
    wrapped = Structure(lattice, numbers.tolist(), frac)
    centers, neighbors, images, _ = wrapped.get_neighbor_list(
        r=cutoff, numerical_tol=1e-8, exclude_self=True
    )
    centers = np.asarray(centers, dtype=np.int64)
    neighbors = np.asarray(neighbors, dtype=np.int64)
    images = np.rint(images).astype(np.int64).reshape(-1, 3)
    vectors = (frac[neighbors] + images - frac[centers]) @ lattice
    distances = np.linalg.norm(vectors, axis=1)
    if np.any(distances < 1e-8):
        raise ValueError("Distinct sites overlap under periodic boundary conditions")
    # Enforce the documented inclusive cutoff independently of neighbor-search tolerance.
    keep = distances <= cutoff
    centers, neighbors, images = centers[keep], neighbors[keep], images[keep]
    vectors, distances = vectors[keep], distances[keep]
    order = np.lexsort((images[:, 2], images[:, 1], images[:, 0], neighbors, centers))
    return {
        "atomic_numbers": numbers,
        "frac_coords": frac,
        "lattice": lattice,
        "positions": frac @ lattice,
        "edge_index": np.stack((centers[order], neighbors[order])),
        "edge_images": images[order],
        "edge_vectors": vectors[order],
        "edge_distances": distances[order],
    }


def _assign_splits(connection, ratios, seed):
    # All blocks in one CIF, and byte-identical copies of that CIF, stay together.
    groups = [row[0] for row in connection.execute("SELECT DISTINCT sha256 FROM sources ORDER BY sha256")]
    random.Random(seed).shuffle(groups)
    exact = np.asarray(ratios) * len(groups)
    counts = np.floor(exact).astype(int)
    for index in sorted(range(3), key=lambda i: (-(exact[i] - counts[i]), i))[:len(groups) - int(counts.sum())]:
        counts[index] += 1
    if groups and counts[0] == 0:
        donor = int(np.argmax(counts[1:])) + 1
        counts[donor] -= 1
        counts[0] += 1
    start = 0
    for split, count in zip(SPLITS, counts):
        for digest in groups[start:start + count]:
            connection.execute(
                "UPDATE samples SET split = ? WHERE source_path IN "
                "(SELECT source_path FROM sources WHERE sha256 = ?)", (split, digest)
            )
        start += count
    return dict(zip(SPLITS, map(int, counts)))


def build_dataset(
    source: str | Path,
    output: str | Path,
    *,
    cutoff: float = 5.0,
    split_ratios=(0.8, 0.1, 0.1),
    seed: int = 42,
    skip_invalid: bool = False,
) -> dict:
    """Build a new immutable database. Existing outputs are never overwritten.

    A failed CIF aborts the build by default. With skip_invalid=True, the entire
    file is skipped and the reason is recorded, including multi-block failures.
    """
    from pymatgen.io.cif import CifParser

    source, output = Path(source).resolve(), Path(output).resolve()
    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be finite and positive (angstrom)")
    ratios = tuple(float(value) for value in split_ratios)
    if (len(ratios) != 3 or any(not math.isfinite(r) or r < 0 for r in ratios)
            or ratios[0] <= 0 or not math.isclose(sum(ratios), 1.0, abs_tol=1e-8)):
        raise ValueError("split_ratios must be three nonnegative values summing to 1, with train > 0")
    # Normalize accepted floating-point roundoff before integer apportionment.
    ratios = tuple(r / sum(ratios) for r in ratios)
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new dataset path: {output}")
    if source.is_file() and source.suffix.lower() == ".cif":
        root, files = source.parent, [source]
    elif source.is_dir():
        root = source
        files = sorted((p for p in source.rglob("*") if p.is_file() and p.suffix.lower() == ".cif"),
                       key=lambda p: p.relative_to(root).as_posix())
    else:
        raise FileNotFoundError(f"Expected a CIF file or source directory: {source}")
    if not files:
        raise ValueError(f"No .cif files found in {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=output.name + ".", suffix=".building", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    connection = None
    skipped = []
    try:
        connection = sqlite3.connect(temporary)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        connection.executescript(SCHEMA)
        for path in files:
            relative = path.relative_to(root).as_posix()
            connection.execute("SAVEPOINT source_file")
            try:
                content = path.read_bytes()
                cif_text = content.decode("utf-8-sig")
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    parser = CifParser.from_str(cif_text, occupancy_tolerance=1.0, frac_tolerance=0.0)
                    structures = parser.parse_structures(primitive=False, check_occu=True, on_error="raise")
                notes = list(dict.fromkeys([str(w.message) for w in captured] + parser.warnings))
                connection.execute("INSERT INTO sources VALUES (?, ?, ?, ?)", (
                    relative, hashlib.sha256(content).hexdigest(), cif_text, json.dumps(notes, ensure_ascii=False)
                ))
                for index, structure in enumerate(structures):
                    arrays = structure_arrays(structure, cutoff)
                    connection.execute(
                        "INSERT INTO samples (sample_id, source_path, structure_index, formula, "
                        "num_atoms, num_edges, split, arrays) VALUES (?, ?, ?, ?, ?, ?, 'train', ?)",
                        (f"{relative}::{index}", relative, index, structure.composition.reduced_formula,
                         len(structure), arrays["edge_index"].shape[1], pack_arrays(arrays)),
                    )
            except Exception as exc:
                connection.execute("ROLLBACK TO source_file")
                connection.execute("RELEASE source_file")
                # I/O/storage failures must abort, even when bad CIFs may be skipped.
                if not skip_invalid or isinstance(exc, (OSError, sqlite3.Error, MemoryError)):
                    raise ValueError(f"Failed to process {relative}: {exc}") from exc
                skipped.append({"source_path": relative, "error": f"{type(exc).__name__}: {exc}"})
                continue
            connection.execute("RELEASE source_file")

        total = connection.execute("SELECT COUNT(*), SUM(num_atoms), SUM(num_edges) FROM samples").fetchone()
        if total[0] == 0:
            raise ValueError(f"No valid structures were found; skipped files: {skipped}")
        group_counts = _assign_splits(connection, ratios, seed)
        split_counts = {split: 0 for split in SPLITS}
        split_counts.update(dict(connection.execute("SELECT split, COUNT(*) FROM samples GROUP BY split")))
        warning_count = connection.execute("SELECT COUNT(*) FROM sources WHERE parser_warnings != '[]'").fetchone()[0]
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(root),
            "num_discovered_files": len(files),
            "num_source_files": connection.execute("SELECT COUNT(*) FROM sources").fetchone()[0],
            "num_structures": total[0], "num_atoms": total[1], "num_edges": total[2],
            "cutoff_angstrom": cutoff, "split_ratios": list(ratios), "seed": seed,
            "split_counts": split_counts, "split_group_counts": group_counts,
            "split_grouping": "source SHA256 (all structures in each CIF kept together)",
            "primitive": False, "frac_tolerance": 0.0, "occupancy_tolerance": 1.0,
            "skip_invalid": skip_invalid, "skipped_files": skipped,
            "files_with_parser_warnings": warning_count,
            "versions": {name: version(name) for name in ("numpy", "pymatgen")},
        }
        connection.execute("INSERT INTO metadata VALUES ('build', ?)", (json.dumps(metadata, ensure_ascii=False),))
        connection.commit()
        connection.close()
        connection = None
        # Same-directory hard link publishes the completed file atomically and
        # fails if the destination exists, including a simultaneous competing build.
        os.link(temporary, output)
        return metadata
    finally:
        if connection is not None:
            connection.close()
        temporary.unlink(missing_ok=True)
