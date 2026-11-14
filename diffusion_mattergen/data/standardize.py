"""Primitive, Niggli-reduced crystals in the row-vector lattice convention."""

from collections.abc import Iterator
from numbers import Real
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser, str2float


DEFAULT_PRIMITIVE_TOLERANCE = 0.25  # angstrom


def standardize_cell(
    structure: Structure, *, primitive_tolerance: float = DEFAULT_PRIMITIVE_TOLERANCE
) -> dict[str, np.ndarray]:
    """Return one crystal as atomic_numbers [N], frac_coords [N, 3],
    lattice [3, 3] and scalar num_atoms. Multiple crystals are concatenated
    along the atom axis by the training data loader.

    The tolerance is in angstrom. Lattice vectors are rows, so cart = frac @ lattice.
    Only element identity and geometry enter the model representation.
    """
    if not isinstance(structure, Structure):
        raise TypeError("Pass a pymatgen Structure to standardize_cell, or use standardize_cif for a file.")
    if (
        not isinstance(primitive_tolerance, Real)
        or isinstance(primitive_tolerance, bool)
        or not 0 < primitive_tolerance < np.inf
    ):
        raise ValueError("primitive_tolerance must be a finite positive number in angstrom.")
    if not len(structure) or not structure.is_ordered:
        raise ValueError("Expected a nonempty ordered structure with exactly one fully occupied element per site.")
    if not all(structure.lattice.pbc):
        raise ValueError("A crystal must be periodic along all three lattice vectors.")
    numbers = structure.atomic_numbers
    if any(z < 1 or z > 118 for z in numbers):
        raise ValueError("Replace dummy species with elements having atomic numbers from 1 to 118.")

    lattice = structure.lattice.matrix
    frac = structure.frac_coords
    if not np.isfinite(lattice).all() or not np.isfinite(frac).all():
        raise ValueError("Check that all lattice entries and fractional coordinates are finite.")
    volume = abs(np.linalg.det(lattice))
    if not np.isfinite(volume) or volume < 1e-8 or np.linalg.matrix_rank(lattice) < 3:
        raise ValueError("Degenerate lattice: check the three cell lengths, angles and volume.")

    # Work on a fresh, element-only structure so decoration cannot change the primitive cell.
    crystal = Structure(lattice, numbers, frac, to_unit_cell=True)
    try:
        primitive = crystal.get_primitive_structure(tolerance=primitive_tolerance)
        reduced = primitive.get_reduced_structure("niggli")
        basis = np.asarray(reduced.lattice.matrix, dtype=np.float64)
        frac = np.asarray(reduced.frac_coords, dtype=np.float64)
        # Change basis and fractional coordinates together before removing a rigid rotation.
        if np.linalg.det(basis) < 0:
            basis, frac = -basis, -frac
        axes, lengths, _ = np.linalg.svd(basis)
    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        raise ValueError(f"Cell reduction failed. Check the lattice and primitive_tolerance. {exc}") from exc

    # The left polar factor preserves the row-basis metric L @ L.T.
    symmetric = (axes * lengths) @ axes.T
    symmetric = (symmetric + symmetric.T) * 0.5
    wrapped = np.remainder(frac, 1.0)
    wrapped[wrapped == 1.0] = 0.0  # Remainders of tiny negative floats can round to 1.
    return {
        "atomic_numbers": np.asarray(reduced.atomic_numbers, dtype=np.int64),
        "frac_coords": wrapped,
        "lattice": symmetric,
        "num_atoms": np.asarray(len(reduced), dtype=np.int64),
    }


def _cif_blocks(text: str) -> Iterator[str]:
    """Keep even identically named blocks separate before passing them to pymatgen."""
    lines = []
    in_text_field = False
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if not in_text_field and stripped[:5].lower() == "data_":
            if lines:
                yield "".join(lines)
            lines = ["data_" + stripped[5:]]
        elif lines:
            if in_text_field and not line.startswith(";"):
                # Fold text as pymatgen does, before its block splitter sees header-like prose.
                lines[-1] = lines[-1].rstrip("\r\n") + " "
            lines.append(line)
        if line.startswith(";"):
            in_text_field = not in_text_field
    if lines:
        yield "".join(lines)


def _cif_numbers(fields: dict, key: str) -> np.ndarray:
    values = fields.get(key, [])
    if isinstance(values, str):
        values = [values]
    try:
        numbers = np.array([str2float(value) for value in values])
        if "." not in values and np.isfinite(numbers).all():
            return numbers
    except ValueError:
        pass
    raise ValueError(f"Check {key}: each listed value must be a finite number.")


def standardize_cif(
    path: str | Path, *, primitive_tolerance: float = DEFAULT_PRIMITIVE_TOLERANCE
) -> dict[str, np.ndarray]:
    """Read exactly one ordered crystal from a UTF-8 CIF and return its model arrays."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Save {path} as a UTF-8 CIF and retry.") from exc
    return standardize_cif_text(text, primitive_tolerance=primitive_tolerance, source_name=str(path))


def standardize_cif_text(
    text: str, *, primitive_tolerance: float = DEFAULT_PRIMITIVE_TOLERANCE,
    source_name: str = "<CIF text>",
) -> dict[str, np.ndarray]:
    """Standardize a CIF string, including multiline CIF fields stored in CSVs."""
    if not isinstance(text, str):
        raise TypeError("CIF text must be a string.")
    selected = None
    try:
        for block in _cif_blocks(text):
            parser = CifParser.from_str(block, occupancy_tolerance=1.0, frac_tolerance=0.0)
            for fields in parser.as_dict().values():
                if not any(key.startswith("_atom_site_") for key in fields):
                    continue
                if selected is not None:
                    raise ValueError("Expected exactly one crystal structure. Split the CIF into separate files.")
                selected = parser
                # The CIF parser can discard zero occupancy or substitute 1 for an unknown value.
                if np.any(_cif_numbers(fields, "_atom_site_occupancy") != 1.0):
                    raise ValueError("Each site must be fully occupied by one element (occupancy 1).")
                for axis in "xyz":
                    _cif_numbers(fields, f"_atom_site_fract_{axis}")
                for axis in "abc":
                    if np.any(_cif_numbers(fields, f"_cell_length_{axis}") <= 0):
                        raise ValueError(f"Degenerate lattice: _cell_length_{axis} must be positive.")
                for angle in ("alpha", "beta", "gamma"):
                    degrees = _cif_numbers(fields, f"_cell_angle_{angle}")
                    if np.any((degrees <= 0) | (degrees >= 180)):
                        raise ValueError(f"Degenerate lattice: _cell_angle_{angle} must be between 0 and 180 degrees.")
        if selected is None:
            raise ValueError("Expected exactly one crystal structure. Check the atom-site data block.")
        structure, = selected.parse_structures(primitive=False, check_occu=True, on_error="raise")
        return standardize_cell(structure, primitive_tolerance=primitive_tolerance)
    except (ValueError, KeyError, IndexError, TypeError, ArithmeticError, np.linalg.LinAlgError) as exc:
        raise ValueError(f"Cannot standardize {source_name}: {exc}") from exc
