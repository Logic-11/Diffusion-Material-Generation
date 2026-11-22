"""Joint-loss checks updated for continuous time and MASK-first atoms."""

import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from scripts.train import train_step, training_statistics
from src.loss import (
    JointDiffusionLoss,
    _standard_mask_keep_probs,
    absorbing_d3pm_terms,
    aggregate_per_crystal,
    categorical_kl,
    categorical_posterior,
)
from test_joint_diffusion import make_model, make_state

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TestDiscreteLoss(unittest.TestCase):
    def test_generic_posterior_matches_enumerated_bayes_rule(self):
        transition = torch.tensor(
            [[.6, .3, .1], [.2, .5, .3], [.1, .2, .7]],
            dtype=torch.float64,
        )
        previous = torch.tensor(
            [[.8, .1, .1], [.3, .6, .1], [.2, .3, .5]],
            dtype=transition.dtype,
        )
        clean = transition.new_tensor([[.2, .3, .5], [.7, .2, .1]])
        observed = torch.tensor([1, 2])
        actual = categorical_posterior(clean, observed, transition, previous)
        expected = torch.empty_like(clean)
        for row in range(2):
            for previous_class in range(3):
                expected[row, previous_class] = sum(
                    clean[row, clean_class]
                    * previous[clean_class, previous_class]
                    * transition[previous_class, observed[row]]
                    for clean_class in range(3)
                )
        expected /= expected.sum(-1, keepdim=True)
        torch.testing.assert_close(actual, expected)

    def test_mask_first_terms_match_dense_forward_matrices(self):
        steps, classes = 6, 4
        keep_probs = _standard_mask_keep_probs(steps)
        cumulative = torch.eye(classes, dtype=torch.float64)
        for k in range(1, steps + 1):
            beta = 1.0 / (steps - k + 1)
            transition = torch.eye(classes, dtype=torch.float64)
            transition[1:, 1:] *= 1.0 - beta
            transition[1:, 0] = beta

            targets = torch.tensor([1, 2, 3, 1, 2])
            masked = torch.tensor([True, False, True, False, True])
            observed = torch.where(masked, 0, targets)
            if k == steps:
                observed = torch.zeros_like(targets)
            logits = torch.linspace(
                -1.5, 2.5, targets.numel() * classes, dtype=torch.float64
            ).reshape(-1, classes).requires_grad_()
            clean_probs = logits.softmax(-1)
            true_probs = F.one_hot(targets, classes).to(logits)
            true_posterior = categorical_posterior(
                true_probs, observed, transition, cumulative
            )
            model_posterior = categorical_posterior(
                clean_probs, observed, transition, cumulative
            )
            if k == 1:
                reference = -model_posterior.gather(
                    1, targets[:, None]
                ).squeeze(-1).log()
            else:
                reference = categorical_kl(true_posterior, model_posterior)
            vb, ce = absorbing_d3pm_terms(
                logits,
                targets,
                masked,
                torch.full_like(targets, k),
                timesteps=steps,
                keep_probs=keep_probs,
            )
            torch.testing.assert_close(vb, reference, atol=2e-7, rtol=2e-6)
            reference_grad, = torch.autograd.grad(
                (reference + .01 * ce).sum(), logits, retain_graph=True
            )
            fast_grad, = torch.autograd.grad((vb + .01 * ce).sum(), logits)
            torch.testing.assert_close(
                fast_grad, reference_grad, atol=2e-7, rtol=2e-6
            )
            cumulative = cumulative @ transition

    def test_extreme_logits_are_finite_and_backpropagate(self):
        logits = torch.tensor([[1000., -1000.]] * 3, requires_grad=True)
        targets = torch.ones(3, dtype=torch.long)
        vb, ce = absorbing_d3pm_terms(
            logits,
            targets,
            torch.tensor([True, False, True]),
            torch.tensor([1, 2, 6]),
            timesteps=6,
        )
        self.assertTrue(torch.isfinite(vb).all())
        torch.testing.assert_close(ce, torch.full((3,), 2000.))
        (vb + .01 * ce).sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


class TestJointLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.batch = torch.tensor([1, 0, 1, 0, 1])
        self.counts = torch.tensor([2, 3])
        self.prediction = {
            "atom_logits": torch.zeros(
                5, 119, dtype=torch.float64, requires_grad=True
            ),
            "coord_score_times_std": torch.tensor(
                [[1., 2., 3.], [0., 1., 0.], [2., 0., 0.],
                 [-1., 0., 1.], [1., 1., 1.]],
                dtype=torch.float64,
                requires_grad=True,
            ),
            "lattice_score_times_std": torch.stack(
                (torch.arange(1, 10).reshape(3, 3), torch.full((3, 3), 2))
            ).double().requires_grad_(),
        }
        self.info = {
            "coord_target": torch.zeros(5, 3, dtype=torch.float64),
            "lattice_target": torch.zeros(2, 3, 3, dtype=torch.float64),
            "atom_target": torch.ones(5, dtype=torch.long),
            "masked_atom": torch.tensor([True, True, False, False, True]),
            "t_graph": torch.tensor([.001, .004], dtype=torch.float64),
        }

    def test_metrics_use_all_fields_and_expected_weights(self):
        criterion = JointDiffusionLoss(coord_weight=.3, lattice_weight=2, atom_weight=4)
        loss, metrics = criterion(
            self.prediction, self.info, self.batch, self.counts
        )
        expected = (
            .3 * metrics["loss_coord"]
            + 2 * metrics["loss_lattice"]
            + 4 * metrics["loss_atom"]
        )
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(
            metrics["loss_atom"],
            metrics["loss_atom_vb"] + .01 * metrics["loss_atom_ce"],
        )
        loss.backward()
        self.assertTrue(all(torch.isfinite(value.grad).all()
                            for value in self.prediction.values()))

    def test_all_nine_lattice_entries_are_used(self):
        error = torch.zeros(1, 3, 3, dtype=torch.float64)
        error[0, 0, 1] = error[0, 1, 0] = 3
        torch.testing.assert_close(
            aggregate_per_crystal(error.square(), None, 1),
            error.new_tensor([2.]),
        )

    def test_exact_continuous_targets_have_zero_gradient(self):
        for prediction_key, target_key in (
            ("coord_score_times_std", "coord_target"),
            ("lattice_score_times_std", "lattice_target"),
        ):
            self.info[target_key] = self.prediction[prediction_key].detach().clone()
        loss, metrics = JointDiffusionLoss()(
            self.prediction, self.info, self.batch, self.counts
        )
        self.assertEqual(metrics["loss_coord"].item(), 0)
        self.assertEqual(metrics["loss_lattice"].item(), 0)
        loss.backward()
        for key in ("coord_score_times_std", "lattice_score_times_std"):
            self.assertEqual(torch.count_nonzero(self.prediction[key].grad).item(), 0)

    def test_invalid_time_and_shapes_are_rejected(self):
        for invalid in (
            {"t_graph": torch.tensor([-.1, .5])},
            {"t_graph": torch.tensor([1.1, .5])},
            {"atom_target": torch.zeros(5, dtype=torch.long)},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                JointDiffusionLoss()(
                    self.prediction,
                    {**self.info, **invalid},
                    self.batch,
                    self.counts,
                )


class TestTrainingIntegration(unittest.TestCase):
    def test_compute_loss_matches_separate_calls_for_fixed_time(self):
        model, clean = make_model(), make_state()
        times, noise = [.02, .07], {
            "atom_uniform": torch.rand_like(clean["frac_coords"][:, 0]),
            "position": torch.randn_like(clean["frac_coords"]),
            "lattice": torch.randn_like(clean["lattice"]),
        }
        noisy, info = model.q_sample(clean, times, noise)
        expected, _ = model.loss_fn(
            model.predict(noisy, info["t_graph"]),
            info,
            clean["batch"],
            clean["num_atoms"],
        )
        actual, _ = model.compute_loss(clean, times, noise=noise)
        torch.testing.assert_close(actual, expected)

    def test_compute_loss_samples_continuous_time(self):
        model, clean = make_model(), make_state()
        sampled = torch.tensor([.25, .75], dtype=torch.float64)
        with patch("src.diffusion.torch.rand", return_value=sampled) as draw, \
             patch.object(model, "q_sample", wraps=model.q_sample) as corrupt:
            loss, _ = model.compute_loss(clean, t_min=1e-5)
        draw.assert_called_once_with(
            clean["num_atoms"].shape,
            device=clean["lattice"].device,
            dtype=clean["lattice"].dtype,
        )
        torch.testing.assert_close(
            corrupt.call_args.args[1],
            sampled * (1.0 - 1e-5) + 1e-5,
        )
        self.assertTrue(torch.isfinite(loss))

    def test_real_denoiser_optimizer_step(self):
        model, clean = make_model(), make_state()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        heads = (
            model.denoiser.output_head[-1].weight,
            model.denoiser.EdgeReadlayers[0].coord_output.weight,
            model.denoiser.EdgeReadlayers[0].lattice_output.weight,
        )
        before = [head.detach().clone() for head in heads]
        loss, metrics = train_step(model, optimizer, clean)
        self.assertTrue(math.isfinite(loss))
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        for old, parameter in zip(before, heads):
            self.assertFalse(torch.equal(old, parameter))
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_training_statistics_writes_empirical_distribution(self):
        samples = [
            {
                "atomic_numbers": torch.full((n,), z),
                "frac_coords": torch.zeros(n, 3),
                "lattice": torch.eye(3) * side,
                "num_atoms": n,
                "structure_id": str(n),
                "split": "train",
            }
            for n, z, side in ((1, 6, 1.), (3, 8, 2.))
        ]
        directory = ROOT / "test" / "_statistics_tmp"
        directory.mkdir(exist_ok=True)
        try:
            density, n_dist = training_statistics(samples, directory)
            statistic = json.loads(
                (Path(directory) / "statistic.json").read_text(encoding="utf-8")
            )
        finally:
            (directory / "statistic.json").unlink(missing_ok=True)
            directory.rmdir()
        self.assertEqual(density, (1. + 3. / 8.) / 2.)
        self.assertEqual(n_dist, {1: .5, 3: .5})
        self.assertEqual(statistic["n_dist"], {"1": .5, "3": .5})


if __name__ == "__main__":
    unittest.main(verbosity=2)
