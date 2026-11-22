"""Independent Bayes-rule and sampling checks for the absorbing D3PM predictor."""

import unittest

import torch

from src.corruption import AtomTypeCorruption
from src.sampling import AtomTypePredictor
from src.scheduler import MaskScheduler


class TestAtomTypePredictor(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(101)
        self.steps = 5
        self.predictor = self.make_predictor(self.steps)

    @staticmethod
    def make_predictor(steps, atom_types=2):
        return AtomTypePredictor(AtomTypeCorruption(
            MaskScheduler(timesteps=steps), atom_types=atom_types))

    def test_posterior_matches_dense_forward_matrices_at_every_step(self):
        # Build row-stochastic matrices and their products independently of the
        # optimized training/sampling helper. Include all current classes and
        # nonzero clean MASK predictions, including impossible observations at K.
        clean_probs = torch.tensor([[.1, .6, .3], [.2, .3, .5], [.25, .5, .25]],
                                   dtype=torch.float64)
        observed = torch.arange(3)
        batch = torch.zeros(3, dtype=torch.long)
        previous = torch.eye(3, dtype=clean_probs.dtype)
        for k in range(1, self.steps + 1):
            beta = torch.tensor(float(self.steps - k + 1), dtype=torch.float32).reciprocal()
            transition = torch.eye(3, dtype=clean_probs.dtype)
            transition[1:, 1:] *= 1 - beta.double()
            transition[1:, 0] = beta.double()
            marginal = clean_probs @ previous
            likelihood = transition[:, observed].T
            weights = (marginal + 1e-20) * (likelihood + 1e-20)
            expected = weights / weights.sum(-1, keepdim=True)
            actual = self.predictor.posterior_logits(
                observed, clean_probs.log(), [(k - 1) / (self.steps - 1)], batch).softmax(-1)
            with self.subTest(k=k):
                # Upstream rounds the cumulative keep probability to float32.
                torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
            previous = previous @ transition

    def test_zero_clean_mask_mass_gives_one_over_k_unmasking(self):
        # Double-precision times just below boundaries must not be rounded up
        # to another step by a float32 conversion in the predictor.
        t = torch.tensor([0., .249999999999, .25, .499999999999, .5, .999999999999, 1.],
                         dtype=torch.float64)
        k = torch.tensor([1, 1, 2, 2, 3, 4, 5], dtype=t.dtype)
        clean = torch.tensor([[0., .7, .3]], dtype=t.dtype).repeat(t.numel(), 1)
        posterior = self.predictor.posterior_logits(
            torch.zeros(t.numel(), dtype=torch.long), clean.log(), t,
            torch.arange(t.numel())).softmax(-1)
        expected = torch.stack((1 - 1 / k, .7 / k, .3 / k), dim=-1)
        torch.testing.assert_close(posterior, expected, atol=2e-7, rtol=2e-6)

    def test_learned_mask_mass_is_retained_at_last_step_without_mutation(self):
        predictor = self.make_predictor(4)
        numbers = torch.tensor([0])
        logits = torch.tensor([[.2, .5, .3]], dtype=torch.float64).log().requires_grad_()
        t = torch.tensor([0.], dtype=torch.float64, requires_grad=True)
        batch = torch.tensor([0])
        inputs = (numbers, logits, t, batch)
        originals = [value.detach().clone() for value in inputs]
        reverse_logits = predictor.posterior_logits(*inputs)
        sample, mode = predictor.atom_type_predict(*inputs)
        # At k=1, Q_bar_0=I and beta=1/4. Conditioning a MASK weights pi by
        # [1, 1/4, 1/4]; this is neither raw pi nor a forced real-element draw.
        torch.testing.assert_close(reverse_logits.softmax(-1), logits.new_tensor([[.5, .3125, .1875]]))
        self.assertEqual(mode.item(), 0)
        self.assertEqual(logits.argmax(-1).item(), 1)
        self.assertEqual(sample.dtype, torch.long)
        self.assertFalse(reverse_logits.requires_grad)
        for value, original in zip(inputs, originals):
            torch.testing.assert_close(value, original, atol=0, rtol=0)

    def test_revealed_elements_keep_their_type_for_supported_predictions(self):
        numbers = torch.tensor([1, 2])
        logits = torch.tensor([[.3, .6, .1], [.2, .3, .5]], dtype=torch.float64).log()
        args = (numbers, logits, [.25, .75], torch.tensor([0, 1]))
        probs = self.predictor.posterior_logits(*args).softmax(-1)
        torch.testing.assert_close(probs, torch.eye(3, dtype=logits.dtype)[numbers], atol=1e-15, rtol=0)
        sample, mode = self.predictor.atom_type_predict(*args)
        torch.testing.assert_close(sample, numbers, atol=0, rtol=0)
        torch.testing.assert_close(mode, numbers, atol=0, rtol=0)

    def test_epsilon_is_added_to_each_bayes_factor_before_multiplication(self):
        # Impossible exact conditioning event: all predicted clean mass is MASK,
        # but the observed state is element 1. MatterGen smooths both factors.
        logits = torch.tensor([[0., -float('inf'), -float('inf')]], dtype=torch.float64)
        actual = self.predictor.posterior_logits(torch.tensor([1]), logits, [.25], torch.tensor([0]))
        epsilon = 1e-20
        weights = logits.new_tensor([[(1 + epsilon) * epsilon,
                                     epsilon * (.75 + epsilon), epsilon**2]])
        torch.testing.assert_close(actual, weights.log(), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(actual.softmax(-1), weights / weights.sum(-1, keepdim=True))
        self.assertTrue(torch.isfinite(actual).all())

    def test_interleaved_batch_uses_each_crystals_time(self):
        batch = torch.tensor([2, 0, 2, 1, 1])
        t = torch.tensor([0., 1., .5], dtype=torch.float64)
        numbers = torch.zeros(batch.numel(), dtype=torch.long)
        clean = torch.tensor([[.1, .6, .3]], dtype=t.dtype).repeat(batch.numel(), 1)
        k = torch.tensor([3., 1., 3., 5., 5.], dtype=t.dtype)
        # Closed form for the standard schedule, including learned clean MASK.
        mask_weight = .1 + .9 * (k - 1) / self.steps
        weights = torch.stack((mask_weight, torch.full_like(k, .6 / self.steps),
                               torch.full_like(k, .3 / self.steps)), dim=-1)
        expected = weights / weights.sum(-1, keepdim=True)
        actual = self.predictor.posterior_logits(numbers, clean.log(), t, batch).softmax(-1)
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
        sample, mode = self.predictor.atom_type_predict(numbers, clean.log(), t, batch)
        torch.testing.assert_close(mode, expected.argmax(-1), atol=0, rtol=0)
        self.assertEqual(sample.shape, numbers.shape)
        self.assertEqual(sample.device, numbers.device)

    def test_categorical_samples_follow_the_reverse_distribution(self):
        draws = 32768
        numbers = torch.zeros(draws, dtype=torch.long)
        clean = torch.tensor([[.1, .6, .3]]).expand(draws, -1)
        sample, mode = self.predictor.atom_type_predict(
            numbers, clean.log(), [.5], torch.zeros_like(numbers))
        # k=3: r=[.46,.36,.18], likelihood=[1,1/3,1/3].
        expected = torch.tensor([.71875, .1875, .09375])
        frequency = torch.bincount(sample, minlength=3).float() / draws
        self.assertLess((frequency - expected).abs().max().item(), .008)
        self.assertTrue(torch.all(mode == 0))
        self.assertTrue(torch.any(sample != mode))

    def test_low_precision_logits_are_promoted_before_probabilities(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                logits = torch.tensor([[.1, .7, .2]]).log().to(dtype)
                args = (torch.tensor([0]), logits,
                        torch.tensor([.499999999999], dtype=torch.float64), torch.tensor([0]))
                posterior = self.predictor.posterior_logits(*args)
                work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
                clean = logits.to(work_dtype).softmax(-1)
                # k=2 (not 3), regardless of logit precision.
                real_weights = .2 * clean[:, 1:]
                weights = torch.cat((1 - .8 * clean[:, 1:].sum(-1, keepdim=True), real_weights), dim=-1)
                expected = weights / weights.sum(-1, keepdim=True)
                self.assertEqual(posterior.dtype, work_dtype)
                torch.testing.assert_close(posterior.softmax(-1), expected, atol=2e-7, rtol=2e-6)
                for value in self.predictor.atom_type_predict(*args):
                    self.assertEqual(value.dtype, torch.long)
                    self.assertEqual(value.device, logits.device)

    def test_single_step_and_last_real_class_need_no_offset(self):
        predictor = self.make_predictor(1, atom_types=118)
        numbers = torch.zeros(3, dtype=torch.long)
        logits = torch.full((3, 119), -float('inf'))
        logits[:, 118] = 0
        sample, mode = predictor.atom_type_predict(numbers, logits, [0., .5, 1.], torch.arange(3))
        expected = torch.full_like(numbers, 118)
        torch.testing.assert_close(sample, expected, atol=0, rtol=0)
        torch.testing.assert_close(mode, expected, atol=0, rtol=0)

    def test_invalid_inputs_and_mask_encoding_are_rejected(self):
        valid = dict(atomic_numbers=torch.tensor([0, 2]), atom_logits=torch.zeros(2, 3),
                     t=[.5], batch=torch.tensor([0, 0]))
        invalid = [dict(atomic_numbers=torch.tensor([0])),
                   dict(atomic_numbers=torch.tensor([0., 2.])),
                   dict(atomic_numbers=torch.tensor([-1, 0])),
                   dict(atomic_numbers=torch.tensor([0, 3])),
                   dict(atom_logits=torch.zeros(2, 4)), dict(atom_logits=torch.zeros(0, 3)),
                   dict(atom_logits=torch.zeros(3)), dict(atom_logits=torch.zeros(2, 3, dtype=torch.long)),
                   dict(atom_logits=torch.full((2, 3), float('nan'))),
                   dict(atom_logits=torch.full((2, 3), float('inf'))),
                   dict(atom_logits=torch.tensor([[0., 1., 2.], [-float('inf')] * 3])),
                   dict(t=.5), dict(t=[]), dict(t=[0]), dict(t=[-.1]), dict(t=[1.1]),
                   dict(t=[float('nan')]), dict(t=[float('inf')]),
                   dict(batch=torch.tensor([0])), dict(batch=torch.tensor([0., 0.])),
                   dict(batch=torch.tensor([-1, 0])), dict(batch=torch.tensor([0, 1]))]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.predictor.atom_type_predict(**(valid | changes))
        with self.assertRaises(ValueError):
            AtomTypePredictor(AtomTypeCorruption(MaskScheduler(), mask_type=1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
