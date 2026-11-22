"""Analytical and distributional checks of MatterGen's wrapped VE predictor."""

import unittest

import torch

from src.corruption import PositionCorruption
from src.sampling import PositionPredictor
from src.scheduler import PositionVEScheduler


class TestPositionPredictor(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(81)
        self.corruption = PositionCorruption(
            PositionVEScheduler(sigma_min=0.25, sigma_max=0.5)).double()
        self.predictor = PositionPredictor(self.corruption)

    def test_hand_calculated_step_wraps_sample_and_mean(self):
        # n=1, t=1, s=1/2: sigma_t=1/2, sigma_s=sqrt(1/8).
        # Thus the mean increment is score/8 and the random increment noise/4.
        coord = torch.tensor([[0.95, 0.05, 0.25]], dtype=torch.float64)
        score = torch.tensor([[2., -2., 0.]], dtype=torch.float64)
        noise = torch.tensor([[1., -1., 0.]], dtype=torch.float64)
        sample, mean = self.predictor.position_predict(
            coord, score, [1.], [0.5], torch.tensor([1]), torch.tensor([0]), noise)
        torch.testing.assert_close(mean, coord.new_tensor([[0.2, 0.8, 0.25]]))
        torch.testing.assert_close(sample, coord.new_tensor([[0.45, 0.55, 0.25]]))

    def test_oracle_score_recovers_gaussian_conditional_for_mixed_sizes(self):
        # Independent Gaussian-conditioning reference, with interleaved membership.
        counts = torch.tensor([1, 8])
        batch = torch.tensor([1, 0, 1, 1, 1, 1, 1, 1, 1])
        t = torch.tensor([0.8, 1.], dtype=torch.float64)
        s = torch.tensor([0.2, 0.5], dtype=torch.float64)
        clean = torch.linspace(0.25, 0.7, 27, dtype=torch.float64).reshape(9, 3)
        coord = clean + torch.linspace(-0.03, 0.03, 27).double().reshape(9, 3)
        noise = torch.linspace(-1., 1., 27, dtype=torch.float64).reshape(9, 3)
        var_t = (0.25 * 2.0**t / counts.double().pow(1 / 3)).square()
        var_s = (0.25 * 2.0**s / counts.double().pow(1 / 3)).square()
        score = -(coord - clean) / var_t[batch, None]

        # Cov(x_s,x_t)=var_s, so the Gaussian conditioning gain is var_s/var_t.
        gain = var_s / var_t
        expected_mean = clean + gain[batch, None] * (coord - clean)
        conditional_var = var_s - gain * var_s
        expected_sample = expected_mean + conditional_var.sqrt()[batch, None] * noise
        sample, mean = self.predictor.position_predict(coord, score, t, s, counts, batch, noise)
        torch.testing.assert_close(mean, expected_mean.remainder(1.))
        torch.testing.assert_close(sample, expected_sample.remainder(1.))

    def test_mixed_clean_endpoint_changes_mean_and_removes_noise(self):
        coord = torch.full((3, 3), 0.4, dtype=torch.float64, requires_grad=True)
        score = torch.ones_like(coord, requires_grad=True)
        t = torch.tensor([0.5, 0.8], dtype=torch.float64)
        s = torch.tensor([0., 0.2], dtype=torch.float64)
        counts, batch = torch.tensor([1, 2]), torch.tensor([0, 1, 1])
        noise = torch.ones_like(coord)
        inputs = (coord, score, t, s, counts, batch, noise)
        before = [value.detach().clone() for value in inputs]

        sample, mean = self.predictor.position_predict(*inputs)
        opposite, other_mean = self.predictor.position_predict(
            coord, score, t, s, counts, batch, -noise)
        # sigma_t**2=1/8 for the terminal crystal; subtracting sigma_min**2 is wrong.
        torch.testing.assert_close(mean[0], coord.new_full((3,), 0.525))
        torch.testing.assert_close(sample[0], mean[0], atol=0, rtol=0)
        torch.testing.assert_close(opposite[0], mean[0], atol=0, rtol=0)
        self.assertFalse(torch.equal(sample[1:], opposite[1:]))
        torch.testing.assert_close(mean, other_mean, atol=0, rtol=0)
        for value, original in zip(inputs, before):
            torch.testing.assert_close(value, original, atol=0, rtol=0)
        self.assertFalse(sample.requires_grad)
        self.assertFalse(mean.requires_grad)

    def test_integer_translations_leave_periodic_results_unchanged(self):
        coord = torch.tensor([[0.1, 0.6, 0.9]], dtype=torch.float64)
        score, noise = torch.randn_like(coord), torch.randn_like(coord)
        args = (score, [0.9], [0.3], torch.tensor([1]), torch.tensor([0]), noise)
        original = self.predictor.position_predict(coord, *args)
        shifted = self.predictor.position_predict(coord + coord.new_tensor([[2., -1., 3.]]), *args)
        for result, expected in zip(shifted, original):
            torch.testing.assert_close(result, expected)
            self.assertTrue(torch.all((result >= 0) & (result < 1)))

    def test_random_draws_have_conditional_variance_and_independent_atom_noise(self):
        # Many two-atom crystals; small variance makes wrapping negligibly likely.
        predictor = PositionPredictor(PositionCorruption(
            PositionVEScheduler(sigma_min=0.0625, sigma_max=0.125)).double())
        b = 8192
        counts = torch.full((b,), 2, dtype=torch.long)
        batch = torch.arange(b).repeat_interleave(2)
        coord = torch.full((2 * b, 3), 0.5, dtype=torch.float64)
        t, s = coord.new_ones(b), coord.new_full((b,), 0.5)
        sample, mean = predictor.position_predict(
            coord, torch.zeros_like(coord), t, s, counts, batch)
        torch.testing.assert_close(mean, coord, atol=0, rtol=0)
        # For this schedule and times, reverse_std = sigma_max * n^(-1/3) / 2.
        expected_std = 0.125 * 2**(-1 / 3) / 2
        residual = ((sample - mean) / expected_std).reshape(b, 6)
        self.assertLess(residual.mean(0).abs().max().item(), 0.04)
        self.assertLess((torch.cov(residual.T) - torch.eye(6)).abs().max().item(), 0.06)

    def test_float32_zero_noise_preserves_dtype_and_device(self):
        predictor = PositionPredictor(self.corruption.float())
        coord = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
        sample, mean = predictor.position_predict(
            coord, torch.ones_like(coord), [0.8], [0.4],
            torch.tensor([1]), torch.tensor([0]), torch.zeros_like(coord))
        torch.testing.assert_close(sample, mean, atol=0, rtol=0)
        self.assertEqual(sample.dtype, coord.dtype)
        self.assertEqual(sample.device, coord.device)

    def test_invalid_times_membership_and_noise_are_rejected(self):
        valid = dict(coord=torch.zeros((1, 3), dtype=torch.float64),
                     score=torch.ones((1, 3), dtype=torch.float64),
                     t=[0.5], s=[0.2], num_atoms=torch.tensor([1]), batch=torch.tensor([0]))
        invalid = [dict(s=[0.6]), dict(s=[-0.1]), dict(t=[0.]), dict(t=[1.1]),
                   dict(t=[float('nan')]), dict(s=0.2), dict(num_atoms=torch.tensor([2])),
                   dict(batch=torch.tensor([1])), dict(noise=torch.zeros((1, 1))),
                   dict(noise=torch.zeros((1, 3), dtype=torch.float32)),
                   dict(noise=torch.full((1, 3), float('nan'), dtype=torch.float64))]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.predictor.position_predict(**(valid | changes))


if __name__ == "__main__":
    unittest.main(verbosity=2)
