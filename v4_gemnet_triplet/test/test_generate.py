"""Exercise the generation CLI with fixed structures in place of model sampling."""

import io
import shutil
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pymatgen.core import Lattice, Structure
import torch

from scripts import generate
from scripts.train import save_model, train_step
from src.geometry import build_periodic_graph

ROOT = Path(__file__).resolve().parents[1]


class TestGraphConfigCheckpoint(unittest.TestCase):
    def test_image_cap_survives_training_and_checkpoint_loading(self):
        torch.manual_seed(42)
        model = generate.JointDiffusion(
            generate.Denoiser(hidden_dim=8, num_radial=4, num_layers=1),
            generate.AtomTypeCorruption(generate.MaskScheduler()),
            generate.PositionCorruption(generate.PositionVEScheduler()),
            generate.LatticeCorruption(generate.LatticeVPScheduler(0.05)),
            cutoff=2.0,
            max_neighbor=4,
            max_cell_images_per_dim=2,
            max_angle_neighbors=2,
        )
        state = {
            "atomic_numbers": torch.tensor([8, 14, 6]),
            "frac_coords": torch.tensor(
                [[.1, .1, .1], [.4, .1, .1], [.1, .4, .1]]
            ),
            "lattice": 4 * torch.eye(3)[None],
            "num_atoms": torch.tensor([3]), "batch": torch.tensor([0, 0, 0]),
        }
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        with patch("src.diffusion.geometry.build_periodic_graph", wraps=build_periodic_graph) as build:
            loss, _ = train_step(model, optimizer, state)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertEqual(build.call_args.kwargs["max_cell_images_per_dim"], 2)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

        directory = ROOT / "test" / "_generate_checkpoint_tmp"
        directory.mkdir()
        self.addCleanup(shutil.rmtree, directory, True)
        try:
            path = save_model(0, loss, None, model, optimizer, None, output_dir=directory)
            restored = generate.load_model(path)
            self.assertEqual(restored.max_cell_images_per_dim, 2)
            self.assertEqual(restored.max_angle_neighbors, 2)
            self.assertEqual(restored.denoiser.num_spherical, 7)
            self.assertTrue(restored.denoiser.use_triplets)
            with torch.no_grad():
                expected = model.predict(state, [.5])
                actual = restored.predict(state, [.5])
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name])

            # Existing checkpoints have no cap field and use the safe default.
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            del checkpoint["diffusion_config"]["graph"]["max_cell_images_per_dim"]
            del checkpoint["diffusion_config"]["graph"]["max_angle_neighbors"]
            for key in ("num_spherical", "use_triplets"):
                checkpoint["model_config"].pop(key, None)
            checkpoint["model_state_dict"] = {
                name: value
                for name, value in checkpoint["model_state_dict"].items()
                if ".triplet_interaction." not in name
            }
            legacy_path = Path(directory) / "legacy.pt"
            torch.save(checkpoint, legacy_path)
            self.assertEqual(generate.load_model(legacy_path).max_cell_images_per_dim, 5)
            self.assertFalse(generate.load_model(legacy_path).denoiser.use_triplets)
        finally:
            shutil.rmtree(directory, ignore_errors=True)


class TestGenerationCLI(unittest.TestCase):
    def setUp(self):
        root = ROOT / "test" / "_generate_cli_tmp"
        root.mkdir()
        self.addCleanup(shutil.rmtree, root, True)
        self.checkpoint = root / "best.pt"
        self.checkpoint.touch()
        self.statistic = root / "statistic.json"
        self.statistic.write_text('{"n_dist": {"1": 1.0}}', encoding="utf-8")
        self.output = root / "cifs"
        self.args = ["--checkpoint", str(self.checkpoint), "--output", str(self.output), "--device", "cpu"]

    def test_reuses_sampler_and_writes_short_final_batch(self):
        model = SimpleNamespace(
            atom_corruption=SimpleNamespace(mask_type=0, scheduler=SimpleNamespace(timesteps=4)),
            position_corruption=object(), lattice_corruption=object(),
        )
        crystals = [Structure(Lattice.cubic(4), [z], [[0.25, 0.5, 0.75]]) for z in (6, 8, 11, 14, 17)]
        with (patch.object(generate, "load_model", return_value=model) as load,
              patch.object(generate, "generate_batch", side_effect=[crystals[:2], crystals[2:4], crystals[4:]]) as batch,
              patch.object(generate.torch, "manual_seed") as seed):
            generate.main(self.args + ["--num-samples", "5", "--batch-size", "2", "--seed", "19"])

        load.assert_called_once_with(self.checkpoint, device="cpu")
        seed.assert_called_once_with(19)
        self.assertEqual([call.kwargs["batch_size"] for call in batch.call_args_list], [2, 2, 1])
        sampler = batch.call_args_list[0].args[1]
        for call in batch.call_args_list:
            self.assertEqual(call.args, (model, sampler, self.statistic))
        self.assertIs(sampler.atom_pred.atom_corr, model.atom_corruption)
        self.assertIs(sampler.position_pred.pos_corr, model.position_corruption)
        self.assertIs(sampler.lattice_pred.lat_corr, model.lattice_corruption)

        paths = sorted(self.output.glob("*.cif"))
        self.assertEqual([p.name for p in paths], [f"crystal_{i:06d}.cif" for i in range(1, 6)])
        for path, expected in zip(paths, crystals, strict=True):
            actual = Structure.from_file(path)
            self.assertEqual(actual.atomic_numbers, expected.atomic_numbers)
            self.assertAlmostEqual(actual.volume, expected.volume)

    def test_existing_cif_is_detected_before_loading(self):
        self.output.mkdir()
        existing = self.output / "crystal_000003.cif"
        existing.write_text("original", encoding="utf-8")
        override = self.statistic.rename(self.statistic.with_name("custom.json"))
        with (patch.object(generate, "load_model") as load, redirect_stderr(io.StringIO()) as errors,
              self.assertRaises(SystemExit)):
            generate.main(self.args + ["--num-samples", "3", "--statistic", str(override)])

        load.assert_not_called()
        self.assertIn(str(existing), errors.getvalue())
        self.assertEqual(list(self.output.iterdir()), [existing])
        self.assertEqual(existing.read_text(encoding="utf-8"), "original")
