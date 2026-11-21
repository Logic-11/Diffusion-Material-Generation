import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifParser

from data import standardize_cell, standardize_cif
from data.__main__ import main

SAMPLE = Path(__file__).resolve().parents[1] / "Ba2PrCu3O7.cif"


class StandardizeCellTests(unittest.TestCase):
    def setUp(self):
        # An already Niggli-reduced triclinic cell with distinguishable sites across boundaries.
        self.basis = np.array([[3.0, 0, 0], [0.4, 4, 0], [0.3, 0.7, 5]])
        self.frac = np.array([[-0.02, 0.05, 0.90], [0.16, 0.11, 0.99],
                              [1.04, 0.28, 0.90], [0.05, 0.12, 1.15]])
        self.numbers = [6, 8, 14, 16]
        self.cell = Structure(self.basis, self.numbers, self.frac)

    def test_triclinic_geometry_rotation_basis_changes_and_supercell(self):
        angle = 0.63
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        variants = {
            "original": self.cell,
            "rotated": Structure(self.basis @ rotation, self.numbers, self.frac),
            "supercell": self.cell * [2, 1, 2],
        }
        for name, change in {
            "sheared basis": [[1, 1, 0], [0, 1, 0], [0, 0, 1]],
            "left-handed basis": [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
        }.items():
            variants[name] = Structure(change @ self.basis, self.numbers, self.frac @ np.linalg.inv(change))

        # Three independent short bonds give a signed volume which detects an accidental reflection.
        reference_bonds = (self.frac[1:] - self.frac[0] - np.rint(self.frac[1:] - self.frac[0])) @ self.basis
        for name, source in variants.items():
            with self.subTest(name=name):
                out = standardize_cell(source)
                self.assertEqual(int(out["num_atoms"]), 4)
                lattice, frac = out["lattice"], out["frac_coords"]
                self.assertTrue(np.all((frac >= 0) & (frac < 1)))
                np.testing.assert_allclose(lattice, lattice.T)
                self.assertTrue(np.all(np.linalg.eigvalsh(lattice) > 0))
                np.testing.assert_allclose(lattice @ lattice.T, self.basis @ self.basis.T, atol=1e-10)
                self.assertAlmostEqual(np.linalg.det(lattice) / 4, source.volume / len(source))

                order = np.argsort(out["atomic_numbers"])
                np.testing.assert_array_equal(out["atomic_numbers"][order], self.numbers)
                result = Structure(lattice, self.numbers, frac[order])
                np.testing.assert_allclose(result.distance_matrix, self.cell.distance_matrix, atol=1e-10)
                bonds = []
                for j in range(1, 4):
                    _, image = result[0].distance_and_image(result[j])
                    bonds.append((result[j].frac_coords + image - result[0].frac_coords) @ lattice)
                self.assertAlmostEqual(np.linalg.det(bonds), np.linalg.det(reference_bonds))
        np.testing.assert_array_equal(self.cell.frac_coords, self.frac)

    def test_fractional_roundoff_at_boundary(self):
        cell = Structure(Lattice.cubic(4), ["Si"], [[-1e-19, 1, 3]])
        np.testing.assert_array_equal(standardize_cell(cell)["frac_coords"], [[0, 0, 0]])

    def test_tolerance_changes_primitive_site_matching(self):
        crystal = Structure(np.diag([8, 4, 4]), ["Si", "Si"], [[0, 0, 0], [0.51, 0, 0]])
        self.assertEqual(int(standardize_cell(crystal)["num_atoms"]), 1)
        self.assertEqual(int(standardize_cell(crystal, primitive_tolerance=0.001)["num_atoms"]), 2)
        for tolerance in (0, -1, np.nan, np.inf):
            with self.subTest(tolerance=tolerance), self.assertRaisesRegex(ValueError, "primitive_tolerance"):
                standardize_cell(self.cell, primitive_tolerance=tolerance)

    def test_invalid_structure_inputs(self):
        cubic = Lattice.cubic(4)
        cases = [
            (Structure(cubic, [], []), "nonempty"),
            (Structure(cubic, [{"Na": 0.5}], [[0, 0, 0]]), "fully occupied"),
            (Structure(cubic, [{"Na": 0.5, "K": 0.5}], [[0, 0, 0]]), "fully occupied"),
            (Structure(cubic, ["Si"], [[np.nan, 0, 0]]), "finite"),
            (Structure(np.diag([4, 4, 0]), ["Si"], [[0, 0, 0]]), "Degenerate"),
            (Structure(np.diag([4, 4, 1e-12]), ["Si"], [[0, 0, 0]]), "Degenerate"),
            (Structure(Lattice(np.eye(3), pbc=(True, True, False)), ["Si"], [[0, 0, 0]]), "periodic"),
        ]
        for structure, reason in cases:
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                standardize_cell(structure)


class StandardizeCifTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.input = self.directory / "input.cif"
        self.text = SAMPLE.read_text(encoding="utf-8")

    def test_sample_api_and_npz_roundtrip(self):
        result = standardize_cif(SAMPLE)
        self.assertEqual(set(result), {"atomic_numbers", "frac_coords", "lattice", "num_atoms"})
        self.assertEqual(result["atomic_numbers"].dtype, np.int64)
        self.assertEqual(result["frac_coords"].dtype, np.float64)
        self.assertEqual(result["lattice"].dtype, np.float64)
        self.assertEqual((result["num_atoms"].shape, result["num_atoms"].dtype), ((), np.int64))
        self.assertEqual(int(result["num_atoms"]), 13)
        self.assertAlmostEqual(np.linalg.det(result["lattice"]), 184.02153, places=5)
        np.testing.assert_array_equal(np.sort(result["atomic_numbers"]), [8] * 7 + [29] * 3 + [56, 56, 59])
        self.assertEqual(result["frac_coords"].shape, (13, 3))
        structure, = CifParser(SAMPLE, frac_tolerance=0).parse_structures(primitive=False)
        from_memory = standardize_cell(structure)
        output = self.directory / "nested folder" / "sample.npz"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(SAMPLE), "--output", str(output)]), 0)
        with np.load(output, allow_pickle=False) as archive:
            self.assertEqual(set(archive.files), set(result))
            for name in result:
                np.testing.assert_array_equal(archive[name], result[name])
                np.testing.assert_array_equal(from_memory[name], result[name])

    def test_multistructure_files_including_duplicate_headers(self):
        for second in (self.text, self.text.replace("data_Ba2PrCu3O7", "data_second")):
            self.input.write_text(self.text + "\n" + second, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                standardize_cif(self.input)
        self.input.write_text("data_pub\n_publ_section_title 'metadata only'\n" + self.text, encoding="utf-8")
        self.assertEqual(int(standardize_cif(self.input)["num_atoms"]), 13)

    def test_header_like_multiline_text_is_part_of_a_single_structure(self):
        text = self.text.replace("data_Ba2PrCu3O7", "data_Ba2PrCu3O7\n_publ_section_title\n;\ndata_example\n;")
        self.input.write_text(text, encoding="utf-8")
        self.assertEqual(int(standardize_cif(self.input)["num_atoms"]), 13)

    def test_invalid_cif_fields(self):
        for old, new, reason in [
            ("0.82112200  1", "0.82112200  0.5", "fully occupied"),
            ("0.82112200  1", "0.82112200  0", "fully occupied"),
            ("0.82112200  1", "0.82112200  ?", "_atom_site_occupancy"),
            ("0.82112200", "nan", "_atom_site_fract_z"),
            ("0.82112200", ".", "_atom_site_fract_z"),
            ("3.90392300", "0.0", "_cell_length_a"),
        ]:
            with self.subTest(new=new):
                self.input.write_text(self.text.replace(old, new), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, reason):
                    standardize_cif(self.input)
        self.input.write_text("data_pub\n_publ_section_title 'metadata only'\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            standardize_cif(self.input)

    def test_output_protection_and_explicit_overwrite(self):
        output = self.directory / "sample.npz"
        output.write_bytes(b"existing")
        args = [str(SAMPLE), "--output", str(output), "--primitive-tolerance", "0.1"]
        error = io.StringIO()
        with redirect_stderr(error), self.assertRaises(SystemExit):
            main(args)
        self.assertIn("--overwrite", error.getvalue())
        self.assertEqual(output.read_bytes(), b"existing")
        self.input.write_text("data_empty\n", encoding="utf-8")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main([str(self.input), "--output", str(output), "--overwrite"])
        self.assertEqual(output.read_bytes(), b"existing")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main([*args, "--overwrite"]), 0)
        with np.load(output, allow_pickle=False) as archive:
            self.assertEqual(int(archive["num_atoms"]), 13)
        self.assertEqual(set(self.directory.iterdir()), {output, self.input})


if __name__ == "__main__":
    unittest.main()
