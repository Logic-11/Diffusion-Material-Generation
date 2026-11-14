"""Regression checks for the DiffCSP noise schedules and coordinate target."""

from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.diffusion import add_coord_noise, atomtype_ce, coordinate_target, mask_atomtype, periodic_noise
from src.scheduler import BetaScheduler, ExpScheduler


class TestDiffusionSchedules(unittest.TestCase):
    def test_small_schedules_and_registered_buffers(self):
        coord = ExpScheduler(3)
        lattice = BetaScheduler(3)
        torch.testing.assert_close(coord.sigmas, torch.tensor([0.0, 0.005, 0.05, 0.5]))
        torch.testing.assert_close(
            lattice.betas,
            torch.tensor([0.0, 0.257116, 0.668255, 0.9999]),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(lattice.alphas_prod, torch.cumprod(1 - lattice.betas, dim=0))
        self.assertEqual(
            set(lattice.state_dict()),
            {"betas", "alphas", "alphas_prod", "sigmas_square", "sigmas"},
        )
        self.assertEqual(coord.to(dtype=torch.float64).sigmas.dtype, torch.float64)

    def test_full_cosine_schedule(self):
        lattice = BetaScheduler(1000)
        self.assertTrue(torch.isfinite(lattice.sigmas).all())
        self.assertAlmostEqual(lattice.betas[-1].item(), 0.9999, places=6)
        self.assertLess(lattice.alphas_prod[-1].item(), 1e-8)

    def test_batched_noise_and_periodic_target(self):
        schedule = ExpScheduler(3)
        graph_steps = torch.tensor([2, 3])
        batch = torch.tensor([0, 0, 1])
        sigmas = schedule.sigmas[graph_steps][batch, None]
        frac = torch.tensor([[.99, .2, .3], [.40, .25, .75], [.01, .4, .6]])
        noise = torch.tensor([[2., 0, 0], [0, 0, 0], [-1., 0, 0]])
        noisy, returned = add_coord_noise(frac, sigmas, noise)
        torch.testing.assert_close(noisy[:, 0], torch.tensor([.09, .4, .51]))
        torch.testing.assert_close(returned, noise)
        target = periodic_noise(sigmas * returned, sigmas)
        self.assertEqual(target.shape, frac.shape)
        self.assertTrue(torch.isfinite(target).all())
        norms = schedule.sigmas_norm[graph_steps][batch, None]
        normalized = coordinate_target(returned, sigmas, norms)
        torch.testing.assert_close(normalized, target / norms.sqrt())

    def test_mask_and_cross_entropy_for_concatenated_atoms(self):
        atoms = torch.tensor([11, 17, 8])
        batch = torch.tensor([0, 0, 1])
        atom_t = torch.tensor([1., 0.])[batch]
        masked, selected = mask_atomtype(atoms, atom_t)
        torch.testing.assert_close(masked, torch.tensor([0, 0, 8]))
        torch.testing.assert_close(selected, torch.tensor([True, True, False]))
        logits = torch.randn(3, 118, requires_grad=True)
        loss = atomtype_ce(logits, atoms, selected)
        expected = torch.nn.functional.cross_entropy(logits[selected], atoms[selected] - 1)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(torch.count_nonzero(logits.grad[~selected]).item(), 0)

    def test_no_mask_returns_differentiable_scalar_zero(self):
        logits = torch.randn(4, 118, requires_grad=True)
        loss = atomtype_ce(logits, torch.tensor([8, 17, 11, 6]), torch.zeros(4, dtype=torch.bool))
        self.assertEqual(loss.shape, ())
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(torch.count_nonzero(logits.grad).item(), 0)


class TestPeriodicNoise(unittest.TestCase):
    def test_each_coordinate_has_its_own_weighted_average(self):
        displace = torch.tensor([[0.0, 0.02, 0.98], [0.2, -0.02, 0.01]])
        sigmas = torch.tensor([[0.1], [0.005]])
        expected = torch.tensor([[0.0, 2.0, -2.0], [8000.0, -800.0, 400.0]])
        torch.testing.assert_close(periodic_noise(displace, sigmas), expected, rtol=1e-5, atol=3e-4)

    def test_periodicity_and_scalar_input(self):
        displace = torch.tensor([-0.1, 0.02, 0.15], dtype=torch.float64)
        torch.testing.assert_close(
            periodic_noise(displace, 0.1),
            periodic_noise(displace + 1, 0.1),
            atol=1e-10,
            rtol=1e-10,
        )
        self.assertAlmostEqual(periodic_noise(0.02, 0.1).item(), 2.0, places=5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_input(self):
        displace = torch.tensor([[0.02, 0.98, 0.2]], device="cuda")
        target = periodic_noise(displace, torch.tensor([[0.1]], device="cuda"))
        self.assertEqual(target.device.type, "cuda")
        self.assertTrue(torch.isfinite(target).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
