from contextlib import closing
from itertools import product
import json
from pathlib import Path
import pickle
import sqlite3
import tempfile
import unittest

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter
import torch
from torch.utils.data import DataLoader

from crystal_dataset import CrystalDataset, build_dataset, collate_crystals, inspect_dataset
from crystal_dataset.build import structure_arrays

REAL_CIF = Path(__file__).resolve().parents[2] / "data" / "Ba2PrCu3O7.cif"


def cif(structure):
    return str(CifWriter(structure))


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "input"
        self.source.mkdir()
        self.output = self.root / "crystals.sqlite"
        self.single = Structure(Lattice.cubic(2), ["Na"], [[0, 0, 0]])
        self.pair = Structure(Lattice.cubic(5), ["Na", "Cl"], [[0, 0, 0], [.5, .5, .5]])

    def write(self, name, text):
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def dataset(self, path=None, **kwargs):
        dataset = CrystalDataset(path or self.output, **kwargs)
        self.addCleanup(dataset.close)
        return dataset

    def test_real_cif_roundtrip_and_provenance(self):
        metadata = build_dataset(REAL_CIF, self.output)
        self.assertEqual(metadata["num_structures"], 1)
        self.assertEqual(metadata["split_counts"], {"train": 1, "val": 0, "test": 0})
        self.assertEqual(inspect_dataset(self.output), metadata)
        dataset = self.dataset(split="train", dtype=torch.float64)
        sample = dataset[0]
        self.assertEqual(sample["num_atoms"], 13)
        self.assertEqual(sorted(sample["atomic_numbers"].tolist()), sorted([56, 56, 59, 29, 29, 29] + [8] * 7))
        torch.testing.assert_close(sample["positions"], sample["frac_coords"] @ sample["lattice"])
        i, j = sample["edge_index"]
        expected = (sample["frac_coords"][j] + sample["edge_images"] - sample["frac_coords"][i]) @ sample["lattice"]
        torch.testing.assert_close(sample["edge_vectors"], expected)
        torch.testing.assert_close(sample["edge_distances"], expected.norm(dim=1))
        self.assertEqual(len(self.dataset(split="val")), 0)
        with self.assertRaises(IndexError):
            dataset[1]
        with closing(sqlite3.connect(self.output)) as connection:
            text, notes = connection.execute("SELECT cif_text, parser_warnings FROM sources").fetchone()
            self.assertEqual(text, REAL_CIF.read_bytes().decode("utf-8-sig"))
            self.assertIsInstance(json.loads(notes), list)

    def test_periodic_self_images_and_multiple_images(self):
        arrays = structure_arrays(self.single, 2.01)
        self.assertEqual(arrays["edge_index"].shape, (2, 6))
        np.testing.assert_array_equal(arrays["edge_index"], np.zeros((2, 6)))
        np.testing.assert_allclose(arrays["edge_distances"], 2)
        self.assertFalse(np.any(np.all(arrays["edge_images"] == 0, axis=1)))
        extended = structure_arrays(self.single, 4.01)
        self.assertTrue(np.any(np.abs(extended["edge_images"]) == 2))

    def test_triclinic_graph_matches_bruteforce(self):
        structure = Structure([[3, 0, 0], [2.8, 1.5, 0], [.5, .3, 4]],
                              ["Na", "Cl"], [[.1, .2, .3], [.8, .6, .9]])
        cutoff = 3.2
        arrays = structure_arrays(structure, cutoff)
        expected = {}
        for i, j in product(range(2), repeat=2):
            for shift in product(range(-4, 5), repeat=3):
                if i == j and shift == (0, 0, 0):
                    continue
                vector = (structure.frac_coords[j] + shift - structure.frac_coords[i]) @ structure.lattice.matrix
                if np.linalg.norm(vector) <= cutoff:
                    expected[(i, j, *shift)] = vector
        actual = {(int(i), int(j), *shift): vector for (i, j), shift, vector in
                  zip(arrays["edge_index"].T, arrays["edge_images"], arrays["edge_vectors"])}
        self.assertEqual(set(actual), set(expected))
        for key in actual:
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-12)
            i, j, x, y, z = key
            np.testing.assert_allclose(actual[(j, i, -x, -y, -z)], -actual[key], atol=1e-12)

    def test_recursive_input_multiblock_and_variable_size_batch(self):
        self.write("a/same.cif", cif(self.single))
        self.write("b/same.CIF", cif(self.pair))
        self.write("multiple.cif", cif(self.single) + "\n" + cif(self.pair))
        build_dataset(self.source, self.output, cutoff=2.01)
        dataset = self.dataset()
        self.assertEqual(len(dataset), 4)
        self.assertEqual(len({dataset[i]["sample_id"] for i in range(4)}), 4)
        batch = collate_crystals([dataset[i] for i in range(4)])
        self.assertEqual(batch["atomic_numbers"].shape, (6,))
        self.assertEqual(batch["frac_coords"].shape, (6, 3))
        self.assertEqual(batch["ptr"].tolist(), [0, 1, 3, 4, 6])
        self.assertEqual(batch["batch"].tolist(), [0, 1, 1, 2, 3, 3])
        self.assertEqual(batch["lattice"].shape, (4, 3, 3))
        i, j = batch["edge_index"]
        torch.testing.assert_close(batch["batch"][i], batch["batch"][j])
        for graph in range(4):
            sample = dataset[graph]
            n0, n1 = batch["ptr"][graph:graph + 2].tolist()
            e0, e1 = batch["edge_ptr"][graph:graph + 2].tolist()
            torch.testing.assert_close(batch["edge_index"][:, e0:e1] - n0, sample["edge_index"])
            for name in ("atomic_numbers", "frac_coords", "positions"):
                torch.testing.assert_close(batch[name][n0:n1], sample[name])
            for name in ("edge_images", "edge_vectors", "edge_distances"):
                torch.testing.assert_close(batch[name][e0:e1], sample[name])
        self.assertEqual(dataset[2]["split"], dataset[3]["split"])

    def test_empty_edges_are_batchable(self):
        self.write("single.cif", cif(self.single))
        build_dataset(self.source, self.output, cutoff=.01)
        sample = self.dataset()[0]
        batch = collate_crystals([sample, sample])
        self.assertEqual(batch["edge_index"].shape, (2, 0))
        self.assertEqual(batch["edge_vectors"].shape, (0, 3))
        self.assertEqual(batch["edge_images"].shape, (0, 3))
        self.assertEqual(batch["ptr"].tolist(), [0, 1, 2])
        self.assertEqual(batch["edge_ptr"].tolist(), [0, 0, 0])
        self.assertEqual(batch["batch"].tolist(), [0, 1])

    def test_invalid_local_edge_index_is_rejected(self):
        self.write("pair.cif", cif(self.pair))
        build_dataset(self.source, self.output, cutoff=4.5)
        sample = self.dataset()[0]
        for indices in (sample["edge_index"] - sample["num_atoms"],
                        sample["edge_index"] + sample["num_atoms"]):
            invalid = {**sample, "edge_index": indices}
            with self.assertRaisesRegex(ValueError, "local atom indices"):
                collate_crystals([invalid])

    def test_splits_reproducible_and_duplicate_sources_grouped(self):
        for i in range(10):
            self.write(f"sample{i}.cif", f"# source {i}\n" + cif(self.single))
        self.write("duplicate.cif", (self.source / "sample0.cif").read_text(encoding="utf-8"))
        metadata = build_dataset(self.source, self.output)
        second = self.root / "second.sqlite"
        build_dataset(self.source, second)
        self.assertEqual(metadata["split_group_counts"], {"train": 8, "val": 1, "test": 1})
        datasets = [self.dataset(self.output), self.dataset(second)]
        maps = [{d[i]["source_path"]: d[i]["split"] for i in range(len(d))} for d in datasets]
        self.assertEqual(maps[0], maps[1])
        self.assertEqual(maps[0]["sample0.cif"], maps[0]["duplicate.cif"])
        split_sets = [{d[i]["sample_id"] for i in range(len(d))} for d in
                      [self.dataset(split=split) for split in ("train", "val", "test")]]
        self.assertEqual(len(set.union(*split_sets)), 11)
        self.assertTrue(all(not a & b for a, b in ((split_sets[0], split_sets[1]), (split_sets[0], split_sets[2]), (split_sets[1], split_sets[2]))))

    def test_invalid_input_atomic_failure_and_file_level_skip(self):
        self.write("valid.cif", cif(self.single))
        disorder = Structure(Lattice.cubic(3), [{"Na": .5}], [[0, 0, 0]])
        self.write("invalid.cif", cif(self.pair) + "\n" + cif(disorder))
        with self.assertRaisesRegex(ValueError, "invalid.cif"):
            build_dataset(self.source, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob("*.building")), [])
        metadata = build_dataset(self.source, self.output, skip_invalid=True)
        self.assertEqual(metadata["num_structures"], 1)
        self.assertEqual(metadata["num_source_files"], 1)
        self.assertEqual(metadata["skipped_files"][0]["source_path"], "invalid.cif")
        with self.assertRaises(FileExistsError):
            build_dataset(self.source, self.output)
        self.assertEqual(inspect_dataset(self.output), metadata)

    def test_bad_arguments_and_empty_sources(self):
        for options in ({"cutoff": 0}, {"cutoff": float("nan")}, {"split_ratios": (0, .5, .5)},
                        {"split_ratios": (.9, .2, .1)}, {"split_ratios": (1,)}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                build_dataset(self.source, self.output, **options)
        with self.assertRaisesRegex(ValueError, "No .cif"):
            build_dataset(self.source, self.output)
        self.write("broken.cif", "not a CIF")
        with self.assertRaises(ValueError):
            build_dataset(self.source, self.output, skip_invalid=True)
        self.assertFalse(self.output.exists())
        with self.assertRaises(ValueError):
            collate_crystals([])

    def test_spawn_workers_and_pickle_after_parent_read(self):
        self.write("one.cif", cif(self.single))
        self.write("two.cif", cif(self.pair))
        build_dataset(self.source, self.output)
        dataset = self.dataset()
        dataset[0]  # Open a connection before spawning/pickling.
        restored = pickle.loads(pickle.dumps(dataset))
        self.addCleanup(restored.close)
        self.assertEqual(restored[0]["sample_id"], dataset[0]["sample_id"])
        loader = DataLoader(dataset, batch_size=2, num_workers=2,
                            multiprocessing_context="spawn", collate_fn=collate_crystals)
        batches = list(loader)
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0]["atomic_numbers"].shape, (3,))
        self.assertEqual(batches[0]["num_atoms"].tolist(), [1, 2])

    def test_current_denoiser_batch_matches_individual_and_backpropagates(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.denoiser import Denoiser
        from src.diffusion import atomtype_ce
        self.write("one.cif", cif(self.single))
        self.write("two.cif", cif(self.pair))
        build_dataset(self.source, self.output, cutoff=4.5)
        dataset = self.dataset()
        samples = [dataset[0], dataset[1]]
        batch = collate_crystals(samples)
        torch.manual_seed(7)
        model = Denoiser(hidden_dim=8, num_layers=2)
        times = torch.tensor([.3, .7])[batch["batch"]][:, None]
        logits = model(batch["atomic_numbers"], batch["edge_index"], batch["edge_vectors"], times)
        self.assertEqual(logits.shape, (3, 118))
        separate = []
        for graph, sample in enumerate(samples):
            n0, n1 = batch["ptr"][graph:graph + 2].tolist()
            separate.append(model(sample["atomic_numbers"], sample["edge_index"], sample["edge_vectors"], times[n0:n1]))
        torch.testing.assert_close(logits, torch.cat(separate), atol=1e-6, rtol=1e-5)
        loss = atomtype_ce(logits, batch["atomic_numbers"], torch.ones(3, dtype=torch.bool))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.node_embedding.weight.grad).all())
        self.assertGreater(model.node_embedding.weight.grad.norm().item(), 0)

    def test_real_single_sample_training_step(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.denoiser import Denoiser
        from src.diffusion import atomtype_ce, mask_atomtype

        build_dataset(REAL_CIF, self.output)
        sample = self.dataset()[0]
        batch = collate_crystals([sample])
        self.assertEqual(batch["frac_coords"].shape, (13, 3))
        self.assertEqual(batch["lattice"].shape, (1, 3, 3))
        self.assertEqual(batch["batch"].tolist(), [0] * 13)
        self.assertEqual(batch["ptr"].tolist(), [0, 13])
        times = torch.ones(batch["num_graphs"])[batch["batch"]]
        masked, mask = mask_atomtype(batch["atomic_numbers"], times)
        self.assertTrue(mask.all())
        model = Denoiser(hidden_dim=8, num_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        logits = model(masked, batch["edge_index"], batch["edge_vectors"], times[:, None])
        self.assertEqual(logits.shape, (13, 118))
        loss = atomtype_ce(logits, batch["atomic_numbers"], mask)
        self.assertTrue(torch.isfinite(loss))
        before = model.output_head[-1].weight.detach().clone()
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.output_head[-1].weight))
        print(f"\nSingle CIF: frac_coords={tuple(batch['frac_coords'].shape)}, "
              f"batch={tuple(batch['batch'].shape)}, logits={tuple(logits.shape)}, loss={loss.item():.6f}")

    def test_standardized_cif_integrates_with_concatenated_batch(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from data import standardize_cell, standardize_cif
        from src.denoiser import Denoiser
        from src.geometry import frac_to_cart

        samples = []
        for graph, cell in enumerate((standardize_cif(REAL_CIF), standardize_cell(self.pair))):
            structure = Structure(cell["lattice"], cell["atomic_numbers"], cell["frac_coords"])
            arrays = structure_arrays(structure, cutoff=4.5)
            sample = {name: torch.from_numpy(value.copy()).float() if value.dtype.kind == "f"
                      else torch.from_numpy(value.copy()).long() for name, value in arrays.items()}
            sample.update(num_atoms=int(cell["num_atoms"]), num_edges=arrays["edge_index"].shape[1],
                          sample_id=str(graph), source_path=str(REAL_CIF), structure_index=0,
                          formula=structure.composition.reduced_formula, split="train")
            samples.append(sample)
        batch = collate_crystals(samples)
        self.assertEqual(batch["num_atoms"].tolist(), [13, 2])
        self.assertEqual(batch["frac_coords"].shape, (15, 3))
        torch.testing.assert_close(frac_to_cart(batch["frac_coords"], batch["lattice"], batch["batch"]),
                                   batch["positions"], atol=1e-6, rtol=1e-5)
        centers, neighbors = batch["edge_index"]
        torch.testing.assert_close(batch["batch"][centers], batch["batch"][neighbors])
        model = Denoiser(hidden_dim=8, num_layers=2)
        times = torch.tensor([.2, .8])[batch["batch"]][:, None]
        logits = model(batch["atomic_numbers"], batch["edge_index"], batch["edge_vectors"], times)
        self.assertEqual(logits.shape, (15, 118))
        self.assertTrue(torch.isfinite(logits).all())

    def test_database_portable_without_source_and_schema_guard(self):
        source = self.write("source.cif", cif(self.single))
        build_dataset(self.source, self.output)
        source.unlink()
        renamed = self.root / "moved.sqlite"
        self.output.rename(renamed)
        self.assertEqual(self.dataset(renamed)[0]["num_atoms"], 1)
        invalid = self.root / "bad.sqlite"
        with closing(sqlite3.connect(invalid)) as connection:
            connection.execute("PRAGMA user_version = 999")
        with self.assertRaisesRegex(ValueError, "schema"):
            CrystalDataset(invalid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
