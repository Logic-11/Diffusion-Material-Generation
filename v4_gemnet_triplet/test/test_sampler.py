"""Integration checks for joint corrector/predictor sampling."""

import unittest
from unittest.mock import patch

import torch
from pymatgen.core import Structure

from src.corruption import AtomTypeCorruption, LatticeCorruption, PositionCorruption
from src.denoiser import Denoiser
from src.diffusion import JointDiffusion
from src.loss import JointDiffusionLoss
from src.sampling import AtomTypePredictor, LangevinCorrector, LatticePredictor, PositionPredictor, Sampler
from src.scheduler import LatticeVPScheduler, MaskScheduler, PositionVEScheduler


class TestSampler(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        atoms = AtomTypeCorruption(MaskScheduler(4), atom_types=2)
        positions = PositionCorruption(PositionVEScheduler(sigma_min=0.05, sigma_max=0.3))
        cells = LatticeCorruption(LatticeVPScheduler(0.05, beta_max=0.4))
        self.model = JointDiffusion(
            Denoiser(hidden_dim=8, num_radial=8, num_layers=1, atom_types=2),
            atoms, positions, cells, cutoff=2.5,
            loss_fn=JointDiffusionLoss(timesteps=4, num_atom_types=2))
        self.sampler = Sampler(AtomTypePredictor(atoms), PositionPredictor(positions),
                               LatticePredictor(cells), LangevinCorrector(ita_max=0.002))
        self.prior = {
            "atomic_numbers": torch.zeros(3, dtype=torch.long),
            "frac_coords": torch.tensor([[0.1, 0.1, 0.3], [0.4, 0.2, 0.7], [0.3, 0.1, 0.3]]),
            "lattice": torch.stack((4 * torch.eye(3), 5 * torch.eye(3))),
            "num_atoms": torch.tensor([2, 1]),
            "batch": torch.tensor([0, 1, 0]),
        }

    def test_terminal_step_uses_corrected_geometry_and_fresh_scores(self):
        original = {name: value.clone() for name, value in self.prior.items()}
        coords, lattice = self.prior["frac_coords"], self.prior["lattice"]
        first = {"atom_logits": torch.tensor([0., 1., 0.]).log().expand(3, -1),
                 "coord_score_times_std": torch.ones_like(coords),
                 "lattice_score_times_std": torch.ones_like(lattice)}
        # At k=1, observing MASK weights the real classes by beta=1/4.
        # These clean logits favor Z=2, but the reverse posterior favors MASK.
        second = {"atom_logits": torch.tensor([0.2, 0.3, 0.5]).log().expand(3, -1),
                  "coord_score_times_std": torch.zeros_like(coords),
                  "lattice_score_times_std": torch.zeros_like(lattice)}
        t = torch.full((2,), 0.25)
        with patch.object(self.model, "predict", side_effect=[first, second]) as predict:
            sample, mean = self.sampler.sample_t(self.model, self.prior, t, torch.zeros_like(t))

        self.assertEqual(predict.call_count, 2)
        corrected, second_t = predict.call_args_list[1].args
        torch.testing.assert_close(second_t, t)
        self.assertFalse(torch.equal(corrected["frac_coords"], coords))
        self.assertFalse(torch.equal(corrected["lattice"], lattice))
        torch.testing.assert_close(corrected["atomic_numbers"], original["atomic_numbers"])
        torch.testing.assert_close(mean["atomic_numbers"], torch.zeros(3, dtype=torch.long))
        torch.testing.assert_close(mean["frac_coords"], corrected["frac_coords"])
        scheduler = self.model.lattice_corruption.scheduler
        mu = scheduler.limit_mean(self.prior["num_atoms"])
        expected_cell = mu + (corrected["lattice"] - mu) / scheduler.mean_coeff(t)[:, None, None]
        torch.testing.assert_close(mean["lattice"], expected_cell)
        for field in ("frac_coords", "lattice"):
            torch.testing.assert_close(sample[field], mean[field], rtol=0, atol=0)
        torch.testing.assert_close(self.prior, original)

    def test_full_chain_follows_samples_across_every_noise_level(self):
        transitions, outputs = [], []
        sample_t = self.sampler.sample_t

        def record_step(model, state, t, s):
            self.assertFalse(model.training)
            self.assertFalse(torch.is_grad_enabled())
            if outputs:
                torch.testing.assert_close(state, outputs[-1][0])
            transitions.append((t.tolist(), s.tolist()))
            outputs.append(sample_t(model, state, t, s))
            return outputs[-1]

        with patch.object(self.sampler, "sample_t", side_effect=record_step):
            final = self.sampler.sample(self.model, self.prior)

        levels = [1., 0.75, 0.5, 0.25, 0.]
        self.assertEqual(transitions, [([t, t], [s, s]) for t, s in zip(levels, levels[1:])])
        self.assertFalse(torch.equal(outputs[0][0]["frac_coords"], outputs[0][1]["frac_coords"]))
        torch.testing.assert_close(final, outputs[-1])
        for state in final:
            torch.testing.assert_close(state["num_atoms"], self.prior["num_atoms"])
            torch.testing.assert_close(state["batch"], self.prior["batch"])
            self.assertTrue(torch.isfinite(state["lattice"]).all())
            self.assertTrue(torch.all((state["frac_coords"] >= 0) & (state["frac_coords"] < 1)))
        self.assertTrue(self.model.training)

    def test_joint_step_rejects_skipping_an_atom_level(self):
        with self.assertRaisesRegex(ValueError, "exactly one reverse atom diffusion step"):
            self.sampler.sample_t(self.model, self.prior, [1., 1.], [0.5, 0.5])

    def test_to_structures_keeps_site_order_and_row_lattices(self):
        coords = torch.tensor([[-0.25, 1., 0.125], [0.1, 0.2, 0.3], [1.5, 0.25, 0.75]],
                              dtype=torch.float64, requires_grad=True)
        lattices = torch.tensor([[[3., 0.5, 0.], [0., 4., 0.2], [0.1, 0., 5.]],
                                 [[6., 0., 1.], [0., 5., 0.], [0., 0., 4.]]], dtype=torch.float64)
        state = {**self.prior, "atomic_numbers": torch.tensor([11, 8, 17]),
                 "frac_coords": coords, "lattice": lattices}
        wrapped = coords.detach().remainder(1)
        structures = self.sampler.to_structures(state)

        self.assertEqual([tuple(s.atomic_numbers) for s in structures], [(11, 17), (8,)])
        for index, structure in enumerate(structures):
            self.assertIsInstance(structure, Structure)
            frac = wrapped[state["batch"] == index]
            torch.testing.assert_close(torch.tensor(structure.lattice.matrix), lattices[index])
            torch.testing.assert_close(torch.tensor(structure.frac_coords), frac)
            torch.testing.assert_close(torch.tensor(structure.cart_coords), frac @ lattices[index])
        self.assertEqual(coords[0, 0].item(), -0.25)

    def test_to_structures_reports_invalid_final_states(self):
        state = {**self.prior, "atomic_numbers": torch.tensor([1, 8, 17])}
        for invalid in (0, 119):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                    ValueError, f"Crystal 1 contains atomic number {invalid}"):
                self.sampler.to_structures({**state, "atomic_numbers": torch.tensor([1, invalid, 17])})
        with self.assertRaisesRegex(ValueError, "batch membership"):
            self.sampler.to_structures({**state, "num_atoms": torch.tensor([1, 2])})
        lattices = state["lattice"].clone()
        lattices[1, 2] = lattices[1, 0]
        with self.assertRaisesRegex(ValueError, "Crystal 1 has a singular lattice"):
            self.sampler.to_structures({**state, "lattice": lattices})


if __name__ == "__main__":
    unittest.main()
