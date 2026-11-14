"""Regression checks for concatenated crystal geometry and graph isolation."""

from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.denoiser import Denoiser
from src.geometry import build_neighbor_graph, cart_to_frac, frac_to_cart, periodic_displace
from src.gnn import build_graph_fc


class TestBatchedGeometry(unittest.TestCase):
    def test_distinct_lattices_and_variable_atom_counts(self):
        frac = torch.tensor([[.98, 0, 0], [.02, 0, 0], [.4, .2, .1]])
        lattice = torch.stack((torch.eye(3) * 10, torch.eye(3) * 20))
        batch = torch.tensor([0, 0, 1])
        cart = frac_to_cart(frac, lattice, batch)
        torch.testing.assert_close(cart_to_frac(cart, lattice, batch), frac)
        torch.testing.assert_close(cart[2], torch.tensor([8., 4., 2.]))
        displace = periodic_displace(frac[:2], lattice[0])
        self.assertEqual(displace.shape, (2, 2, 3))
        torch.testing.assert_close(displace[0, 1, 0], torch.tensor(.4))
        indices, vectors, distances = build_neighbor_graph(frac, lattice, 1, batch=batch)
        self.assertEqual(indices.tolist(), [[0, 1], [1, 0]])
        torch.testing.assert_close(distances, torch.tensor([.4, .4]))
        torch.testing.assert_close(vectors.norm(dim=-1), distances)

    def test_interleaved_batch_still_has_no_cross_crystal_edges(self):
        frac = torch.tensor([[.98, 0, 0], [.4, .2, .1], [.02, 0, 0]])
        batch = torch.tensor([0, 1, 0])
        lattice = torch.stack((torch.eye(3) * 10, torch.eye(3) * 20))
        indices, vectors, distances = build_graph_fc(frac, lattice, batch, torch.tensor([2, 1]))
        self.assertEqual(indices.tolist(), [[0, 2], [2, 0]])
        torch.testing.assert_close(batch[indices[0]], batch[indices[1]])
        torch.testing.assert_close(vectors.norm(dim=-1), distances)

    def test_single_atom_has_no_fractional_image_self_pair(self):
        indices, vectors, distances = build_graph_fc(
            torch.zeros(1, 3), torch.eye(3)[None], torch.tensor([0]), torch.tensor([1]),
        )
        self.assertEqual(indices.shape, (2, 0))
        self.assertEqual(vectors.shape, (0, 3))
        self.assertEqual(distances.shape, (0,))

    def test_multiple_lattices_require_atom_assignment(self):
        with self.assertRaisesRegex(ValueError, "batch"):
            frac_to_cart(torch.zeros(3, 3), torch.eye(3).repeat(2, 1, 1))
        with self.assertRaisesRegex(ValueError, "num_atoms"):
            build_graph_fc(torch.zeros(3, 3), torch.eye(3).repeat(2, 1, 1),
                           torch.tensor([0, 0, 1]), torch.tensor([1, 2]))


class TestGraphIsolation(unittest.TestCase):
    def test_other_crystal_does_not_receive_input_gradients(self):
        torch.manual_seed(11)
        model = Denoiser(hidden_dim=8, num_layers=3)
        atoms = torch.tensor([11, 17, 8, 14, 14])
        indices = torch.tensor([[0, 2, 1, 1, 3, 4], [1, 1, 0, 2, 4, 3]])
        vectors = torch.tensor([[1., 0, 0], [1, -1, -1], [-1, 0, 0],
                                [-1, 1, 1], [0, 0, 1.5], [0, 0, -1.5]], requires_grad=True)
        times = torch.tensor([[.2], [.2], [.2], [.8], [.8]], requires_grad=True)
        logits = model(atoms, indices, vectors, times)
        self.assertEqual(logits.shape, (5, 118))
        logits[:3].square().mean().backward()
        for tensor, split in ((vectors, 4), (times, 3)):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad[:split].norm().item(), 0)
            self.assertEqual(torch.count_nonzero(tensor.grad[split:]).item(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
