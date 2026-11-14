"""Step 7: matrix/analytic references and a real joint optimizer update."""

import math
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from src.loss import (JointDiffusionLoss, absorbing_d3pm_terms, aggregate_per_crystal,
                      categorical_kl, categorical_posterior)
from scripts.train_mask import train_step, training_statistics
from test_joint_diffusion import make_model, make_state


class TestDiscreteLoss(unittest.TestCase):
    def test_generic_posterior_matches_enumerated_bayes_rule(self):
        # Non-absorbing, non-symmetric matrices detect row/column confusion.
        q = torch.tensor([[.6, .3, .1], [.2, .5, .3], [.1, .2, .7]], dtype=torch.float64)
        previous = torch.tensor([[.8, .1, .1], [.3, .6, .1], [.2, .3, .5]], dtype=q.dtype)
        p0 = q.new_tensor([[.2, .3, .5], [.7, .2, .1]])
        observed = torch.tensor([1, 2])
        actual = categorical_posterior(p0, observed, q, previous)
        expected = torch.empty_like(p0)
        for row in range(2):
            for i in range(3):
                expected[row, i] = sum(p0[row, x] * previous[x, i] * q[i, observed[row]]
                                       for x in range(3))
        expected /= expected.sum(-1, keepdim=True)
        torch.testing.assert_close(actual, expected)
        batched = categorical_posterior(p0, observed, q.repeat(2, 1, 1), previous.repeat(2, 1, 1))
        torch.testing.assert_close(actual, batched)

    def test_matrix_vb_and_gradients_equal_simplification_at_every_step(self):
        # Three real elements plus MASK, with Q_bar obtained by matrix products.
        # Includes k=1 NLL, every intermediate KL, terminal k=K, masked/unmasked.
        steps, classes = 7, 4
        previous = torch.eye(classes, dtype=torch.float64)
        for k in range(1, steps + 1):
            beta = 1.0 / (steps - k + 1)
            q = torch.eye(classes, dtype=torch.float64)
            q[:-1, :-1] *= 1 - beta
            q[:-1, -1] = beta
            targets = torch.arange(classes - 1).repeat(2 if k < steps else 1)
            observed = (torch.cat((torch.full((3,), 3), torch.arange(3))) if k < steps
                        else torch.full((3,), 3))
            logits = torch.linspace(-1.5, 2.5, targets.numel() * 3, dtype=q.dtype)
            logits = logits.reshape(-1, 3).requires_grad_()
            p0 = torch.cat((logits.softmax(-1), logits.new_zeros(targets.numel(), 1)), -1)
            q0 = F.one_hot(targets, classes).to(q.dtype)
            true_posterior = categorical_posterior(q0, observed, q, previous)
            model_posterior = categorical_posterior(p0, observed, q, previous)
            torch.testing.assert_close(model_posterior.sum(-1), torch.ones_like(targets, dtype=q.dtype))
            if k == 1:
                reference = -model_posterior.gather(1, targets[:, None]).squeeze(-1).log()
            else:
                reference = categorical_kl(true_posterior, model_posterior)
            vb, ce = absorbing_d3pm_terms(logits, targets, observed == 3,
                                         torch.full_like(targets, k))
            torch.testing.assert_close(vb, reference, atol=1e-12, rtol=1e-12)
            reference_grad, = torch.autograd.grad((reference + .01 * ce).sum(), logits, retain_graph=True)
            fast_grad, = torch.autograd.grad((vb + .01 * ce).sum(), logits)
            torch.testing.assert_close(fast_grad, reference_grad, atol=1e-12, rtol=1e-12)
            previous = previous @ q

    def test_zero_probability_support_and_impossible_observation(self):
        q = torch.tensor([[1., 0., 0.], [.25, 0., .75]], dtype=torch.float64)
        p = torch.tensor([[1., 0., 0.], [.125, .125, .75]], dtype=q.dtype, requires_grad=True)
        kl = categorical_kl(q, p)
        torch.testing.assert_close(kl, q.new_tensor([0., .25 * math.log(2)]))
        kl.sum().backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        infinite = categorical_kl(q[:1], q.new_tensor([[0., 1., 0.]]))
        self.assertTrue(torch.isinf(infinite).all())
        self.assertFalse(torch.isnan(infinite).any())
        with self.assertRaisesRegex(ValueError, "zero probability"):
            categorical_posterior(q[:1], torch.tensor([1]), torch.eye(3, dtype=q.dtype),
                                  torch.eye(3, dtype=q.dtype))

    def test_extreme_logits_use_stable_ce_and_unmasked_auxiliary_gradients(self):
        logits = torch.tensor([[1000., -1000.]] * 3, requires_grad=True)
        vb, ce = absorbing_d3pm_terms(logits, torch.ones(3, dtype=torch.long),
                                     torch.tensor([True, False, True]), torch.tensor([1, 2, 1000]))
        torch.testing.assert_close(ce, torch.full((3,), 2000.))
        torch.testing.assert_close(vb, torch.tensor([2000., 0., 2.]))
        (vb + .01 * ce).sum().backward()
        torch.testing.assert_close(logits.grad, torch.tensor([[1.01, -1.01], [.01, -.01], [.011, -.011]]))


class TestJointLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.batch = torch.tensor([1, 0, 1, 0, 1])
        self.counts = torch.tensor([2, 3])
        self.prediction = {
            "atom_logits": torch.zeros(5, 2, dtype=torch.float64, requires_grad=True),
            "coord_score_times_std": torch.tensor(
                [[1., 2., 3.], [0., 1., 0.], [2., 0., 0.], [-1., 0., 1.], [1., 1., 1.]],
                dtype=torch.float64, requires_grad=True),
            "lattice_score_times_std": torch.stack((torch.arange(1, 10).reshape(3, 3),
                                                    torch.full((3, 3), 2))).double().requires_grad_(),
        }
        self.info = {
            "coord_target": torch.zeros(5, 3, dtype=torch.float64),
            "lattice_target": torch.zeros(2, 3, 3, dtype=torch.float64),
            "atom_target": torch.tensor([0, 1, 0, 1, 0]),
            "masked_atom": torch.tensor([True, True, False, False, True]),
            "k_graph": torch.tensor([1, 4]),
        }

    def test_feature_means_atom_sums_and_crystal_mean_against_hand_values(self):
        loss, metrics = JointDiffusionLoss()(self.prediction, self.info, self.batch, self.counts)
        expected = {"loss_coord": 4., "loss_lattice": (285 / 9 + 4) / 2,
                    "loss_atom_vb": .75 * math.log(2), "loss_atom_ce": 2.5 * math.log(2),
                    "loss_atom": .775 * math.log(2)}
        for key, value in expected.items():
            self.assertAlmostEqual(metrics[key].item(), value, places=12)
            self.assertFalse(metrics[key].requires_grad)
        self.assertAlmostEqual(loss.item(), .1 * expected["loss_coord"] + expected["loss_lattice"]
                               + expected["loss_atom"], places=12)
        loss.backward()
        self.assertTrue(all(torch.isfinite(x.grad).all() for x in self.prediction.values()))

    def test_all_nine_lattice_entries_are_used(self):
        error = torch.zeros(1, 3, 3, dtype=torch.float64)
        error[0, 0, 1] = error[0, 1, 0] = 3
        # Both mirrored entries count: (9 + 9)/9, not a six-entry average.
        torch.testing.assert_close(aggregate_per_crystal(error.square(), None, 1), error.new_tensor([2.]))

    def test_exact_continuous_targets_give_zero_loss_and_zero_gradient(self):
        for name, target in (("coord_score_times_std", "coord_target"),
                             ("lattice_score_times_std", "lattice_target")):
            self.info[target] = self.prediction[name].detach().clone()
        loss, metrics = JointDiffusionLoss()(self.prediction, self.info, self.batch, self.counts)
        self.assertEqual(metrics["loss_coord"].item(), 0)
        self.assertEqual(metrics["loss_lattice"].item(), 0)
        loss.backward()
        for key in ("coord_score_times_std", "lattice_score_times_std"):
            self.assertEqual(torch.count_nonzero(self.prediction[key].grad).item(), 0)

    def test_no_mask_keeps_auxiliary_ce_and_all_fields_trainable(self):
        self.info["masked_atom"].zero_()
        loss, metrics = JointDiffusionLoss()(self.prediction, self.info, self.batch, self.counts)
        self.assertEqual(metrics["loss_atom_vb"].item(), 0)
        torch.testing.assert_close(metrics["loss_atom"], .01 * metrics["loss_atom_ce"])
        loss.backward()
        for tensor in self.prediction.values():
            self.assertGreater(tensor.grad.norm().item(), 0)

    def test_repeating_batch_preserves_loss_and_custom_weights_apply_once(self):
        criterion = JointDiffusionLoss(coord_weight=.3, lattice_weight=2, atom_weight=4,
                                      d3pm_hybrid_lambda=.02)
        loss, metrics = criterion(self.prediction, self.info, self.batch, self.counts)
        expected = .3 * metrics["loss_coord"] + 2 * metrics["loss_lattice"] + 4 * metrics["loss_atom"]
        torch.testing.assert_close(loss.detach(), expected)
        torch.testing.assert_close(metrics["loss_atom"], metrics["loss_atom_vb"] + .02 * metrics["loss_atom_ce"])
        repeated = {key: torch.cat((value, value)) for key, value in self.prediction.items()}
        info = {key: torch.cat((value, value)) for key, value in self.info.items()}
        repeat_loss, _ = criterion(repeated, info, torch.cat((self.batch, self.batch + 2)), self.counts.repeat(2))
        torch.testing.assert_close(loss, repeat_loss)

    def test_bad_steps_and_shapes_are_reported(self):
        for invalid in (torch.tensor([0, 1]), torch.tensor([1., 2.])):
            self.info["k_graph"] = invalid
            with self.assertRaises(ValueError):
                JointDiffusionLoss()(self.prediction, self.info, self.batch, self.counts)
        with self.assertRaises(ValueError):
            JointDiffusionLoss(coord_weight=float("nan"))


class TestTrainingIntegration(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(72)

    def test_compute_loss_matches_separate_calls_with_fixed_time_and_noise(self):
        model, clean = make_model(), make_state()
        noise = {"atom_uniform": torch.rand_like(clean["frac_coords"][:, 0]),
                 "position": torch.randn_like(clean["frac_coords"]),
                 "lattice": torch.randn_like(clean["lattice"])}
        before = {key: value.clone() for key, value in clean.items()}
        noisy, info = model.q_sample(clean, [20, 70], noise)
        expected, _ = model.loss_fn(model.predict(noisy, info["t_graph"]), info,
                                   clean["batch"], clean["num_atoms"])
        actual, _ = model.compute_loss(clean, [20, 70], noise)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(actual.requires_grad)
        for key in clean:
            torch.testing.assert_close(before[key], clean[key])

    def test_compute_loss_samples_one_positive_integer_time_per_crystal(self):
        model, clean = make_model(), make_state()
        with patch("src.diffusion.torch.randint", return_value=torch.tensor([1, 1000])) as draw, \
             patch.object(model, "q_sample", wraps=model.q_sample) as corrupt:
            loss, _ = model.compute_loss(clean)
        draw.assert_called_once_with(1, 1001, torch.Size([2]), device=clean["num_atoms"].device)
        self.assertEqual(corrupt.call_args.args[1].tolist(), [1, 1000])
        self.assertTrue(torch.isfinite(loss))

    def check_optimizer_update(self, dtype, device):
        model, clean = make_model(dtype=dtype, device=device), make_state(dtype=dtype, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        heads = (model.denoiser.output_head[-1].weight,
                 model.denoiser.EdgeReadlayers[0].coord_output.weight,
                 model.denoiser.EdgeReadlayers[0].lattice_output.weight)
        before = [value.detach().clone() for value in heads]
        # Force an entirely unmasked batch: auxiliary CE must still train atoms.
        noise = {"atom_uniform": clean["frac_coords"].new_full((5,), .999),
                 "position": torch.randn_like(clean["frac_coords"]),
                 "lattice": torch.randn_like(clean["lattice"])}
        report = train_step(model, optimizer, clean, k=[20, 70], noise=noise)
        self.assertEqual(report["loss_atom_vb"], 0)
        self.assertTrue(all(math.isfinite(value) for value in report.values()))
        for field in ("atom", "coord", "lattice"):
            self.assertGreater(report[f"grad_{field}"], 0)
        for old, parameter in zip(before, heads):
            self.assertFalse(torch.equal(old, parameter))
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_real_denoiser_adam_update_cpu_float64(self):
        self.check_optimizer_update(torch.float64, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_real_denoiser_adam_update_cuda_float32(self):
        self.check_optimizer_update(torch.float32, "cuda")

    def test_training_statistics_use_mean_density_and_empirical_size_counts(self):
        samples = [{"atomic_numbers": torch.full((n,), z), "frac_coords": torch.zeros(n, 3),
                    "lattice": torch.eye(3) * side, "num_atoms": n,
                    "structure_id": str(n), "split": "train"}
                   for n, z, side in ((1, 6, 1.), (3, 8, 2.))]
        stats = training_statistics(samples)
        self.assertEqual(stats["limit_density"], (1. + 3. / 8.) / 2.)
        self.assertEqual(stats["num_atoms_values"], [1, 3])
        self.assertEqual(stats["num_atoms_probabilities"], [.5, .5])
        self.assertEqual(stats["atomic_numbers"], [6, 8])
        self.assertEqual(stats["num_structures"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
