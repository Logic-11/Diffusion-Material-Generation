"""Step 5: analytical and distributional checks of the lattice marginal."""

import unittest

import torch

from src.corruption import LatticeCorruption
from src.scheduler import LatticeVPScheduler


class TestLatticeCorruption(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(51)
        self.corruption = LatticeCorruption(LatticeVPScheduler(0.05)).double()
        self.counts = torch.tensor([8, 64])
        self.clean = torch.tensor([
            [[4.0, 0.3, 0.1], [0.3, 5.0, 0.2], [0.1, 0.2, 6.0]],
            [[7.0, 0.2, 0.4], [0.2, 8.0, 0.3], [0.4, 0.3, 9.0]],
        ], dtype=torch.float64)

    def analytical_parameters(self, t):
        # Evaluate the plan's formula independently of scheduler methods.
        rho = float(self.corruption.scheduler.limit_density)
        n = self.counts.double()
        mu = (n / rho).pow(1 / 3)[:, None, None] * torch.eye(3)
        a = torch.exp(-0.5 * (0.1 * t + 0.5 * (20.0 - 0.1) * t.square()))
        mean = mu + a[:, None, None] * (self.clean - mu)
        variance = (1 - a.square()) * 0.25 * n.pow(2 / 3)
        return mean, variance

    def test_marginal_mean_variance_symmetry_and_six_independent_entries(self):
        samples = 80_000
        t = torch.tensor([0.2, 0.8], dtype=torch.float64)
        mean, variance = self.analytical_parameters(t)
        draws = self.corruption.q_sample(
            self.clean.repeat(samples, 1, 1), t.repeat(samples),
            self.counts.repeat(samples)).reshape(samples, 2, 3, 3)
        torch.testing.assert_close(draws, draws.transpose(-1, -2), atol=0, rtol=0)
        residual = (draws - mean) / variance.sqrt()[None, :, None, None]
        self.assertLess(residual.mean(0).abs().max().item(), 0.018)
        self.assertLess((residual.var(0) - 1).abs().max().item(), 0.025)
        rows, cols = torch.triu_indices(3, 3)
        for graph in range(2):
            six = residual[:, graph, rows, cols]
            covariance = torch.cov(six.T)
            self.assertLess((covariance - torch.eye(6)).abs().max().item(), 0.025)

    def test_scaled_score_matches_the_used_symmetric_noise(self):
        t = torch.tensor([1e-8, 0.7], dtype=torch.float64)
        raw = torch.randn_like(self.clean)
        noisy = self.corruption.q_sample(self.clean, t, self.counts, raw)
        target = self.corruption.score_target(self.clean, noisy, t, self.counts)
        expected = raw.clone()
        for i in range(3):
            for j in range(i + 1, 3):
                expected[:, i, j] = expected[:, j, i] = (raw[:, i, j] + raw[:, j, i]) / 2**0.5
        torch.testing.assert_close(target, -expected, atol=1e-10, rtol=1e-9)

    def test_zero_time_is_clean_but_has_no_score_target(self):
        t = torch.zeros(2, dtype=torch.float64)
        noisy = self.corruption.q_sample(self.clean, t, self.counts)
        torch.testing.assert_close(noisy, self.clean, atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, "t > 0"):
            self.corruption.score_target(self.clean, noisy, t, self.counts)

    def test_prior_returns_samples_with_atom_count_scaling(self):
        samples = 80_000
        draws = self.corruption.prior_sample(self.counts.repeat(samples))
        self.assertIsInstance(draws, torch.Tensor)
        draws = draws.reshape(samples, 2, 3, 3)
        rho = float(self.corruption.scheduler.limit_density)
        mu = (self.counts.double() / rho).pow(1 / 3)[:, None, None] * torch.eye(3)
        variance = 0.25 * self.counts.double().pow(2 / 3)
        residual = (draws - mu) / variance.sqrt()[None, :, None, None]
        self.assertLess(residual.mean(0).abs().max().item(), 0.018)
        self.assertLess((residual.var(0) - 1).abs().max().item(), 0.025)
        torch.testing.assert_close(mu[1], 2 * mu[0])
        torch.testing.assert_close(variance[1], 4 * variance[0])

    def test_no_positive_definite_projection_is_applied(self):
        raw = -100 * torch.eye(3, dtype=torch.float64).repeat(2, 1, 1)
        noisy = self.corruption.q_sample(
            self.clean, torch.ones(2, dtype=torch.float64), self.counts, raw)
        self.assertTrue(torch.all(torch.linalg.eigvalsh(noisy) < 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
