"""Gaussian conditioning and MatterGen convention checks for the VP predictor."""

import unittest

import torch

from src.corruption import LatticeCorruption
from src.sampling import LatticePredictor
from src.scheduler import LatticeVPScheduler


class TestLatticePredictor(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91)
        self.corruption = LatticeCorruption(LatticeVPScheduler(
            limit_density=0.125, beta_min=0.25, beta_max=2.0,
            limit_var_scaling_constant=0.25)).double()
        self.predictor = LatticePredictor(self.corruption)

    @staticmethod
    def analytical_parameters(t, counts):
        # Evaluate the forward marginal independently of scheduler methods.
        alpha = torch.exp(-0.125 * t - 0.4375 * t.square())
        mu = (counts.to(t) / 0.125).pow(1 / 3)[:, None, None] * torch.eye(3).to(t)
        variance = (1 - alpha.square()) * 0.25 * counts.to(t).pow(2 / 3)
        return alpha, mu, variance

    @staticmethod
    def predictor_noise_reference(draw):
        weights = draw.new_full((3, 3), 2**0.5)
        weights.fill_diagonal_(1.0)
        return draw * weights

    def test_oracle_score_gives_conditional_mean_and_mattergen_noise(self):
        counts = torch.tensor([1, 8, 27])
        t = torch.tensor([0.7, 1.0, 0.3], dtype=torch.float64)
        s = torch.tensor([0.2, 0.6, 0.1], dtype=torch.float64)
        alpha_t, mu, var_t = self.analytical_parameters(t, counts)
        alpha_s, _, var_s = self.analytical_parameters(s, counts)
        clean = mu + self.corruption.symmetric_noise(torch.randn_like(mu))
        marginal_t = mu + alpha_t[:, None, None] * (clean - mu)
        marginal_s = mu + alpha_s[:, None, None] * (clean - mu)
        lattice = marginal_t + self.corruption.symmetric_noise(torch.randn_like(clean))
        score = -(lattice - marginal_t) / var_t[:, None, None]
        raw_noise = t.new_tensor([-1.25, 0.0, 0.75])[:, None, None]

        # Gaussian conditioning: Cov(L_s,L_t|L_0) = var_s * alpha_t / alpha_s.
        covariance = var_s * alpha_t / alpha_s
        gain = covariance / var_t
        expected_mean = marginal_s + gain[:, None, None] * (lattice - marginal_t)
        conditional_var = var_s - covariance.square() / var_t
        expected_sample = expected_mean + conditional_var.sqrt()[:, None, None] * (
            self.predictor_noise_reference(raw_noise))
        sample, mean = self.predictor.lattice_predict(lattice, score, t, s, counts, raw_noise)
        torch.testing.assert_close(mean, expected_mean)
        torch.testing.assert_close(sample, expected_sample)
        torch.testing.assert_close(sample, sample.transpose(-1, -2), atol=0, rtol=0)

    def test_mixed_clean_endpoint_recovers_clean_without_mutating_inputs(self):
        counts = torch.tensor([1, 8])
        t = torch.tensor([0.5, 0.8], dtype=torch.float64)
        s = torch.tensor([0.0, 0.2], dtype=torch.float64)
        alpha_t, mu, var_t = self.analytical_parameters(t, counts)
        clean = mu + 0.2 * torch.eye(3, dtype=t.dtype)
        marginal_t = mu + alpha_t[:, None, None] * (clean - mu)
        lattice = (marginal_t + 0.1).requires_grad_()
        score = (-(lattice.detach() - marginal_t) / var_t[:, None, None]).requires_grad_()
        noise = torch.ones_like(t)[:, None, None]
        inputs = (lattice, score, t, s, counts, noise)
        originals = [value.detach().clone() for value in inputs]
        sample, mean = self.predictor.lattice_predict(*inputs)
        opposite, other_mean = self.predictor.lattice_predict(lattice, score, t, s, counts, -noise)
        torch.testing.assert_close(mean[0], clean[0])
        torch.testing.assert_close(sample[0], mean[0], atol=0, rtol=0)
        torch.testing.assert_close(opposite[0], mean[0], atol=0, rtol=0)
        self.assertFalse(torch.equal(sample[1], opposite[1]))
        torch.testing.assert_close(mean, other_mean, atol=0, rtol=0)
        for value, original in zip(inputs, originals):
            torch.testing.assert_close(value, original, atol=0, rtol=0)
        self.assertFalse(sample.requires_grad)
        self.assertFalse(mean.requires_grad)

    def test_mattergen_clips_mean_coefficients_after_computing_variance(self):
        corruption = LatticeCorruption(LatticeVPScheduler(
            limit_density=1.0, beta_min=20.0, beta_max=20.0,
            limit_var_scaling_constant=1.0)).double()
        predictor = LatticePredictor(corruption)
        counts = torch.tensor([1, 8, 27])
        t = torch.tensor([1.0, 0.2, 1.0], dtype=torch.float64)
        s = torch.tensor([0.1, 0.1, 0.0], dtype=torch.float64)
        mu = counts.double().pow(1 / 3)[:, None, None] * torch.eye(3, dtype=t.dtype)
        lattice, score = mu + 0.5, torch.full_like(mu, 0.125)
        noise = torch.ones_like(t)[:, None, None]
        raw_ratio = torch.exp(-10.0 * (t - s))
        limit_var = counts.double().pow(2 / 3)
        var_t = (1 - torch.exp(-20.0 * t)) * limit_var
        var_s = (1 - torch.exp(-20.0 * s)) * limit_var
        # Closed-form forward transition variance, before the ratio is clipped.
        transition_var = (1 - torch.exp(-20.0 * (t - s))) * limit_var
        clipped_ratio = raw_ratio.clamp(min=0.001)
        expected_mean = mu + (lattice - mu + transition_var[:, None, None] * score) / (
            clipped_ratio[:, None, None])
        expected_sample = expected_mean + (transition_var * var_s / var_t).sqrt()[:, None, None] * (
            self.predictor_noise_reference(noise))
        with self.assertLogs("src.sampling", level="WARNING") as logs:
            sample, mean = predictor.lattice_predict(lattice, score, t, s, counts, noise)
        self.assertIn("0.001", logs.output[0])
        torch.testing.assert_close(mean, expected_mean, atol=1e-10, rtol=1e-12)
        torch.testing.assert_close(sample, expected_sample, atol=1e-10, rtol=1e-12)
        torch.testing.assert_close(sample[-1], mean[-1], atol=0, rtol=0)

    def test_random_step_draws_one_gaussian_per_crystal(self):
        counts = torch.tensor([1, 8, 27, 1])
        t = torch.full((counts.numel(),), 0.8, dtype=torch.float64)
        s = torch.full_like(t, 0.3)
        alpha_t, mu, var_t = self.analytical_parameters(t, counts)
        alpha_s, _, var_s = self.analytical_parameters(s, counts)
        covariance = var_s * alpha_t / alpha_s
        conditional_var = var_s - covariance.square() / var_t
        # Use the same CPU RNG state for the reference and the predictor.
        with torch.random.fork_rng(devices=[]):
            draws = torch.randn((counts.numel(), 1, 1), dtype=t.dtype)
            next_draw = torch.randn((), dtype=t.dtype)
        sample, mean = self.predictor.lattice_predict(mu, torch.zeros_like(mu), t, s, counts)
        torch.testing.assert_close(mean, mu)
        expected_noise = self.predictor_noise_reference(draws)
        torch.testing.assert_close(sample - mean, conditional_var.sqrt()[:, None, None] * expected_noise)
        torch.testing.assert_close(torch.randn((), dtype=t.dtype), next_draw, atol=0, rtol=0)

    def test_predictor_does_not_apply_polar_or_positive_definite_projection(self):
        lattice = -torch.eye(3, dtype=torch.float64)[None]
        sample, mean = self.predictor.lattice_predict(
            lattice, torch.zeros_like(lattice), [0.8], [0.2], torch.tensor([1]),
            lattice.new_zeros((1, 1, 1)))
        self.assertTrue(torch.all(torch.linalg.eigvalsh(mean) < 0))
        torch.testing.assert_close(sample, mean, atol=0, rtol=0)

    def test_default_schedule_small_steps_preserve_float32_and_device(self):
        predictor = LatticePredictor(LatticeCorruption(LatticeVPScheduler(0.05)))
        lattice = torch.eye(3, dtype=torch.float32).repeat(2, 1, 1)
        sample, mean = predictor.lattice_predict(
            lattice, torch.ones_like(lattice), [0.001, 1.0], [0.0, 0.999],
            torch.tensor([1, 8]))
        for value in (sample, mean):
            self.assertTrue(torch.isfinite(value).all())
            self.assertEqual(value.dtype, lattice.dtype)
            self.assertEqual(value.device, lattice.device)
        torch.testing.assert_close(sample[0], mean[0], atol=0, rtol=0)

    def test_invalid_shapes_times_counts_and_noise_are_rejected(self):
        valid = dict(lattice=torch.eye(3, dtype=torch.float64)[None],
                     score=torch.ones(1, 3, 3, dtype=torch.float64),
                     t=[0.5], s=[0.2], num_atoms=torch.tensor([1]))
        invalid = [dict(lattice=torch.eye(3, dtype=torch.float64)),
                   dict(lattice=torch.ones(1, 3, 3, dtype=torch.long)),
                   dict(score=torch.ones(1, 3, 3, dtype=torch.float32)),
                   dict(score=torch.full((1, 3, 3), float('nan'), dtype=torch.float64)),
                   dict(t=[0.0]), dict(t=[1.1]), dict(t=0.5),
                   dict(s=[0.5]), dict(s=[0.6]), dict(s=[-0.1]), dict(s=[float('nan')]),
                   dict(num_atoms=torch.tensor([0])), dict(num_atoms=torch.tensor([1, 2])),
                   dict(num_atoms=torch.tensor([1.0])),
                   dict(noise=torch.zeros(1, 3, 3, dtype=torch.float64)),
                   dict(noise=torch.zeros(2, 1, 1, dtype=torch.float64)),
                   dict(noise=torch.zeros(1, 1, 1, dtype=torch.float32)),
                   dict(noise=torch.full((1, 1, 1), float('inf'), dtype=torch.float64))]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.predictor.lattice_predict(**(valid | changes))


if __name__ == "__main__":
    unittest.main(verbosity=2)
