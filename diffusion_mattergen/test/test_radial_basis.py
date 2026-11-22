"""Numerical and autograd checks for Gaussian radial features."""

from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.geometry import build_periodic_graph
from src.gnn import RadialBasis


class TestRadialBasis(unittest.TestCase):
    def test_values_match_gaussians_and_polynomial_envelope(self):
        basis = RadialBasis(num_radial=5, cutoff=4.0).double()
        distances = torch.tensor([0.0, 0.4, 1.0, 2.0, 3.7, 4.0, 4.1], dtype=torch.float64)
        centers = torch.arange(5, dtype=distances.dtype)
        gaussians = torch.exp(-0.5 * (distances[:, None] - centers).square())
        scaled = distances / 4.0
        envelope = 1.0 - 21.0 * scaled**5 + 35.0 * scaled**6 - 15.0 * scaled**7
        envelope = torch.where(distances < 4.0, envelope, torch.zeros_like(envelope))
        expected = envelope[:, None] * gaussians

        features = basis(distances)
        self.assertEqual(features.shape, (7, 5))
        torch.testing.assert_close(features, expected, atol=1e-13, rtol=1e-11)
        self.assertEqual(features[0, 0].item(), 1.0)
        self.assertEqual(torch.count_nonzero(features[-2:]).item(), 0)

    def test_cutoff_stays_nonnegative_and_finite(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                basis = RadialBasis().to(dtype=dtype)
                cutoff = torch.tensor(7.0, dtype=dtype)
                below = torch.nextafter(cutoff, torch.zeros_like(cutoff))
                above = torch.nextafter(cutoff, torch.full_like(cutoff, float("inf")))
                distances = torch.cat((
                    torch.linspace(0.0, 7.0, 1001, dtype=dtype),
                    torch.stack((below, cutoff, above)),
                    torch.tensor([torch.finfo(dtype).max], dtype=dtype),
                ))
                features = basis(distances)
                self.assertTrue(torch.isfinite(features).all().item())
                self.assertTrue((features >= 0).all().item())
                self.assertEqual(torch.count_nonzero(features[distances >= 7.0]).item(), 0)
                self.assertLess(features[-4].max().item(), 1e-15)

    def test_first_and_second_derivatives_vanish_at_cutoff(self):
        basis = RadialBasis(num_radial=5, cutoff=4.0).double()
        distances = torch.tensor([0.0, 3.999, 4.0, 4.1], dtype=torch.float64, requires_grad=True)
        features = basis(distances)
        first, = torch.autograd.grad(features.sum(), distances, create_graph=True)
        second, = torch.autograd.grad(first.sum(), distances)
        self.assertTrue(torch.isfinite(first).all().item())
        self.assertTrue(torch.isfinite(second).all().item())
        torch.testing.assert_close(first[2:], torch.zeros_like(first[2:]), atol=0, rtol=0)
        torch.testing.assert_close(second[2:], torch.zeros_like(second[2:]), atol=0, rtol=0)

    def test_autograd_matches_finite_differences(self):
        basis = RadialBasis(num_radial=8, cutoff=4.0).double()
        distances = torch.tensor([0.3, 1.1, 2.4, 3.6], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(basis, (distances,)))
        self.assertTrue(torch.autograd.gradgradcheck(basis, (distances,)))

    def test_empty_distances_preserve_shape_and_autograd(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                basis = RadialBasis().to(dtype=dtype)
                distances = torch.empty(0, dtype=dtype, requires_grad=True)
                features = basis(distances)
                self.assertEqual(features.shape, (0, 32))
                self.assertEqual(features.dtype, dtype)
                features.sum().backward()
                self.assertIsNotNone(distances.grad)
                self.assertEqual(distances.grad.shape, (0,))

    def test_buffers_follow_dtype_and_are_saved(self):
        basis = RadialBasis(num_radial=5, cutoff=4.0).double()
        self.assertEqual(list(basis.parameters()), [])
        self.assertEqual(set(dict(basis.named_buffers())), {"centers", "width"})
        self.assertTrue(all(buffer.dtype == torch.float64 for buffer in basis.buffers()))
        restored = RadialBasis(num_radial=5, cutoff=4.0).double()
        restored.load_state_dict(basis.state_dict())
        distances = torch.tensor([0.0, 1.2, 3.9, 4.0], dtype=torch.float64)
        torch.testing.assert_close(restored(distances), basis(distances))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_values_gradients_and_device_validation(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                cpu_basis = RadialBasis().to(dtype=dtype)
                gpu_basis = RadialBasis().to(device="cuda", dtype=dtype)
                gpu_basis.load_state_dict(cpu_basis.state_dict())
                cpu_distances = torch.tensor([0.0, 0.4, 2.1, 6.9, 7.0, 7.1], dtype=dtype, requires_grad=True)
                gpu_distances = cpu_distances.detach().cuda().requires_grad_(True)
                cpu_features = cpu_basis(cpu_distances)
                gpu_features = gpu_basis(gpu_distances)
                torch.testing.assert_close(gpu_features.cpu(), cpu_features)
                cpu_features.sum().backward()
                gpu_features.sum().backward()
                torch.testing.assert_close(gpu_distances.grad.cpu(), cpu_distances.grad)
                self.assertTrue(all(buffer.device.type == "cuda" for buffer in gpu_basis.buffers()))
                with self.assertRaisesRegex(ValueError, "dtype and device"):
                    cpu_basis(gpu_distances)

    def test_gradients_reach_periodic_coordinates_and_lattice(self):
        basis = RadialBasis(num_radial=8, cutoff=1.5).double()
        frac = torch.tensor([[0.875, 0.25, 0.25], [0.125, 0.25, 0.25]],
                            dtype=torch.float64, requires_grad=True)
        lattice = torch.diag(torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64)).requires_grad_(True)

        def encode_graph(coordinates, cell):
            graph = build_periodic_graph(coordinates, cell, cutoff=1.5, max_neighbor=None)
            return basis(graph[3])

        self.assertTrue(torch.autograd.gradcheck(encode_graph, (frac, lattice)))
        encode_graph(frac, lattice).square().sum().backward()
        for gradient in (frac.grad, lattice.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all().item())
            self.assertGreater(gradient.norm().item(), 0.0)

    def test_invalid_parameters_and_distances(self):
        for num_radial in (True, 0, 1, -1, 2.5, "32"):
            with self.subTest(num_radial=num_radial), self.assertRaisesRegex(ValueError, "num_radial"):
                RadialBasis(num_radial=num_radial)
        for cutoff in (True, 0.0, -1.0, float("nan"), float("inf"), "7"):
            with self.subTest(cutoff=cutoff), self.assertRaisesRegex(ValueError, "cutoff"):
                RadialBasis(cutoff=cutoff)

        basis = RadialBasis()
        invalid_distances = (
            torch.zeros(2, 1),
            torch.tensor([1, 2]),
            torch.tensor([1.0 + 0.0j]),
            torch.tensor([-0.1]),
            torch.tensor([float("nan")]),
            torch.tensor([float("inf")]),
            torch.ones(2, dtype=torch.float64),
        )
        for distances in invalid_distances:
            with self.subTest(distances=distances), self.assertRaises(ValueError):
                basis(distances)


if __name__ == "__main__":
    unittest.main(verbosity=2)
