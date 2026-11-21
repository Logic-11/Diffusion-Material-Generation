"""Dataset contracts: split preservation, rollback, worker loading and real geometry."""

import csv
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter
import torch
from torch.utils.data import DataLoader

from data import CrystalDataset, build_dataset, collate_crystals, inspect_dataset
from data.__main__ import main
from src.geometry import build_periodic_graph


class DatasetCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "cache"
        self.single = Structure(Lattice.cubic(4.2), ["Si"], [[0, 0, 0]])
        self.pair = Structure(Lattice.cubic(5), ["Cs", "Cl"], [[0, 0, 0], [.5, .5, .5]])

    def csv(self, split, records):
        path = self.source / f"{split}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["material_id", "cif", "band_gap", "note"])
            writer.writeheader()
            for name, structure in records:
                text = str(CifWriter(structure)) if isinstance(structure, Structure) else structure
                writer.writerow({"material_id": name, "cif": text, "band_gap": "1.25", "note": "中文,\nmetadata"})
        return path

    def dataset(self, **kwargs):
        dataset = CrystalDataset(self.output, **kwargs)
        self.addCleanup(dataset.close)
        return dataset

    def test_splits_ids_primitive_geometry_and_metadata_survive_cache(self):
        self.csv("train", [("cscl", self.pair * [2, 1, 1]), ("silicon", self.single)])
        self.csv("val", [("validation", self.pair)])
        self.csv("test", [("test", self.single)])
        info = build_dataset(self.source, self.output)
        self.assertEqual({k: v["num_structures"] for k, v in info["splits"].items()},
                         {"train": 2, "val": 1, "test": 1})
        self.assertEqual(info["num_atoms"], 6)
        dataset = self.dataset(dtype=torch.float64)
        self.assertEqual([sample["structure_id"] for sample in dataset], ["cscl", "silicon"])
        cell = dataset[0]
        self.assertEqual(cell["num_atoms"], 2)
        self.assertAlmostEqual(torch.linalg.det(cell["lattice"]).item(), 125.)
        self.assertTrue(torch.all(torch.linalg.eigvalsh(cell["lattice"]) > 0))
        torch.testing.assert_close(cell["lattice"], cell["lattice"].T)
        structure = Structure(cell["lattice"].numpy(), cell["atomic_numbers"].numpy(), cell["frac_coords"].numpy())
        np.testing.assert_allclose(structure.distance_matrix, self.pair.distance_matrix)
        records = [json.loads(line) for line in (self.output / "train/records.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(records[0]["fields"]["note"], "中文,\nmetadata")
        self.assertEqual(records[0]["fields"]["band_gap"], "1.25")
        self.assertEqual(records[1]["record"], 2)
        self.assertIsInstance(dataset._arrays["frac_coords"], np.memmap)
        self.assertEqual(set(p.name for p in (self.output / "train").iterdir()),
                         {"atomic_numbers.npy", "pos.npy", "cell.npy", "num_atoms.npy", "structure_id.npy", "records.jsonl"})
        self.assertEqual(inspect_dataset(self.output), info)

    def test_variable_batches_match_independent_periodic_neighbor_oracle(self):
        self.csv("train", [("pair", self.pair), ("single", self.single), ("last", self.pair)])
        build_dataset(self.source, self.output)
        dataset = self.dataset(dtype=torch.float64)
        batches = list(DataLoader(dataset, batch_size=2, collate_fn=collate_crystals))
        self.assertEqual([b["num_graphs"] for b in batches], [2, 1])
        batch = batches[0]
        self.assertEqual(batch["ptr"].tolist(), [0, 2, 3])
        self.assertEqual(batch["batch"].tolist(), [0, 0, 1])
        self.assertEqual(batch["lattice"].shape, (2, 3, 3))
        result = build_periodic_graph(batch["frac_coords"], batch["lattice"], 5.1,
                                      batch["batch"], max_neighbor=None)
        edges, images, vectors, distances, counts, ptr, edge_batch = result
        torch.testing.assert_close(batch["batch"][edges[0]], batch["batch"][edges[1]])
        torch.testing.assert_close(edge_batch, batch["batch"][edges[0]])
        self.assertEqual(ptr.tolist(), [0, int(counts[0]), edges.shape[1]])
        for g in range(2):
            sample = dataset[g]
            structure = Structure(sample["lattice"].numpy(), sample["atomic_numbers"].numpy(), sample["frac_coords"].numpy())
            center, neighbor, shifts, lengths = structure.get_neighbor_list(5.1, numerical_tol=0.)
            # With zero tolerance pymatgen may report a tiny roundoff self distance.
            expected = {(int(i), int(j), *map(int, shift)): length
                        for i, j, shift, length in zip(center, neighbor, shifts, lengths)
                        if i != j or np.any(shift != 0)}
            selected = edge_batch == g
            keys = torch.cat(((edges[:, selected] - batch["ptr"][g]).T, images[selected]), dim=1).tolist()
            self.assertEqual(set(map(tuple, keys)), set(expected))
            for key, distance in zip(keys, distances[selected]):
                self.assertAlmostEqual(float(distance), expected[tuple(key)], places=10)
        # Graph construction must remain usable after replacing the clean geometry.
        noisy_frac = (batch["frac_coords"] + .01 * torch.randn_like(batch["frac_coords"])).requires_grad_()
        noisy_lattice = (batch["lattice"] * 1.02).requires_grad_()
        graph = build_periodic_graph(noisy_frac, noisy_lattice, batch=batch["batch"])
        graph[2].square().sum().backward()
        self.assertTrue(torch.isfinite(noisy_frac.grad).all())
        self.assertTrue(torch.isfinite(noisy_lattice.grad).all())

    def test_sample_mutations_pickle_and_spawn_workers_do_not_modify_storage(self):
        self.csv("train", [(f"sample-{i}", self.pair if i % 2 else self.single) for i in range(5)])
        build_dataset(self.source, self.output)
        dataset = self.dataset()
        sample = dataset[0]
        sample["frac_coords"].add_(.2)
        sample["lattice"].zero_()
        sample["atomic_numbers"].zero_()
        self.assertTrue(torch.all(dataset[0]["frac_coords"] == 0))
        self.assertEqual(dataset[0]["atomic_numbers"].tolist(), [14])
        clone = pickle.loads(pickle.dumps(dataset))
        self.addCleanup(clone.close)
        self.assertEqual(clone[-1]["structure_id"], "sample-4")
        loader = DataLoader(dataset, batch_size=2, num_workers=2, multiprocessing_context="spawn",
                            collate_fn=collate_crystals)
        self.assertEqual([name for batch in loader for name in batch["structure_id"]],
                         [f"sample-{i}" for i in range(5)])
        with self.assertRaises(IndexError):
            dataset[5]

    def test_invalid_rows_abort_atomically_or_are_explicitly_logged(self):
        self.csv("train", [("good", self.single), ("bad", "data_empty\n")])
        with self.assertRaisesRegex(ValueError, "record 2"):
            build_dataset(self.source, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".cache.building-*")), [])
        info = build_dataset(self.source, self.output, skip_invalid=True)
        self.assertEqual((info["num_structures"], info["num_skipped"]), (1, 1))
        errors = [json.loads(line) for line in (self.output / "errors.jsonl").read_text().splitlines()]
        self.assertEqual(errors[0]["structure_id"], "bad")
        before = (self.output / "metadata.json").read_bytes()
        with self.assertRaises(FileExistsError):
            build_dataset(self.source, self.output)
        self.assertEqual((self.output / "metadata.json").read_bytes(), before)

    def test_duplicate_ids_cannot_cross_splits_even_when_skipping_invalid(self):
        self.csv("train", [("same-id", self.single)])
        self.csv("test", [("same-id", self.pair)])
        with self.assertRaisesRegex(ValueError, "Duplicate structure_id"):
            build_dataset(self.source, self.output, skip_invalid=True)
        self.assertFalse(self.output.exists())

    def test_csv_schema_empty_split_and_invalid_tolerance_fail_without_cache(self):
        path = self.source / "train.csv"
        path.write_text("material_id,wrong\na,b\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unique columns"):
            build_dataset(self.source, self.output)
        self.csv("train", [])
        with self.assertRaisesRegex(ValueError, "no valid structures"):
            build_dataset(self.source, self.output)
        for tolerance in (0, np.nan, True):
            with self.assertRaisesRegex(ValueError, "primitive_tolerance"):
                build_dataset(self.source, self.output, primitive_tolerance=tolerance)
        self.assertFalse(self.output.exists())

    def test_recursive_cifs_single_csv_cli_and_cache_validation(self):
        for directory in (self.source, self.source / "nested"):
            directory.mkdir(exist_ok=True)
            (directory / "sample.cif").write_text(str(CifWriter(self.single)), encoding="utf-8")
        build_dataset(self.source, self.output)
        dataset = self.dataset()
        self.assertEqual({s["structure_id"] for s in dataset}, {"sample.cif", "nested/sample.cif"})
        dataset.close()
        np.save(self.output / "train/num_atoms.npy", np.array([99, 1], dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            CrystalDataset(self.output)
        csv_path = self.csv("val", [("csv-only", self.pair)])
        second = self.root / "second"
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            self.assertEqual(main(["build", str(csv_path), "--output", str(second)]), 0)
        self.assertEqual(set(json.loads(stdout.getvalue())["splits"]), {"val"})
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["inspect", str(second)]), 0)
        with self.assertRaisesRegex(ValueError, "absent"):
            CrystalDataset(second)

    def test_float32_rounding_keeps_coordinates_in_one_period(self):
        self.csv("train", [("silicon", self.single)])
        build_dataset(self.source, self.output)
        coords = np.array([[1.0 - 1e-12, 0., 0.]])
        np.save(self.output / "train/pos.npy", coords)
        sample = self.dataset()[0]
        self.assertTrue(torch.all((sample["frac_coords"] >= 0) & (sample["frac_coords"] < 1)))
        with self.assertRaisesRegex(ValueError, "empty"):
            collate_crystals([])


if __name__ == "__main__":
    unittest.main()
