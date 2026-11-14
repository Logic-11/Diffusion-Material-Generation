"""Periodic graph regressions, independent of the dataset and denoiser."""

from pathlib import Path
import sys
import unittest

from pymatgen.core import Structure
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.geometry import build_periodic_graph


def edge_keys(index, images):
    return [tuple(row) for row in torch.cat((index.T, images), dim=1).tolist()]


class TestPeriodicGraph(unittest.TestCase):
    def test_cubic_cutoff_includes_second_image_and_excludes_origin(self):
        frac = torch.zeros(1, 3)
        index, images, vectors, distances, counts, ptr = build_periodic_graph(
            frac, 2 * torch.eye(3), cutoff=4, max_neighbor=None
        )
        self.assertEqual(counts.tolist(), [32])
        self.assertEqual(ptr.tolist(), [0, 32])
        self.assertEqual(len(set(edge_keys(index, images))), 32)
        self.assertTrue((index == 0).all())
        self.assertFalse((images == 0).all(dim=1).any())
        self.assertEqual((distances == 4).sum().item(), 6)
        self.assertIn([2, 0, 0], images.tolist())
        torch.testing.assert_close(vectors, 2 * images.to(frac))
        torch.testing.assert_close(distances, vectors.norm(dim=1))

    def test_returned_lengths_are_inclusive_cutoffs(self):
        cell = torch.tensor([[1.323, -.296, .416], [-.713, 1.951, .166], [.381, .521, 2.122]],
                            dtype=torch.float64)
        frac = cell.new_tensor([[.1, .2, .3], [.8, .6, .4], [.21, .93, .63]])
        full = build_periodic_graph(frac, cell, 2, max_neighbor=None)
        for radius in full[3].unique().tolist():
            within = full[3] <= radius
            graph = build_periodic_graph(frac, cell, radius, max_neighbor=None)
            self.assertCountEqual(edge_keys(*graph[:2]), edge_keys(full[0][:, within], full[1][within]))

    def test_skew_cell_and_unwrapped_sites_match_pymatgen(self):
        lattice = torch.tensor([[2., 0., 0.], [5.9, 1., 0.], [.3, .2, 4.]], dtype=torch.float64)
        frac = lattice.new_tensor([[.10, .20, .30], [.80, .60, .40]])
        shifts = torch.tensor([[2, -1, 3], [-1, 4, 0]])
        cutoff = 2.12
        structure = Structure(lattice.numpy(), ["Na", "Cl"], frac.numpy())
        centers, neighbors, offsets, lengths = structure.get_neighbor_list(cutoff, numerical_tol=0.)
        expected = {}
        for i, j, offset, length in zip(centers, neighbors, offsets, lengths):
            image = torch.as_tensor(offset, dtype=torch.long)
            key = (int(i), int(j), *(image + shifts[i] - shifts[j]).tolist())
            expected[key] = ((frac[j] - frac[i] + image) @ lattice, length)

        index, images, vectors, distances, counts, _ = build_periodic_graph(
            frac + shifts, lattice, cutoff, max_neighbor=None
        )
        keys = edge_keys(index, images)
        self.assertEqual(counts.item(), len(expected))
        self.assertCountEqual(keys, expected.keys())
        self.assertIn((0, 0, 6, -2, 0), keys)
        for k, key in enumerate(keys):
            vector, distance = expected[key]
            torch.testing.assert_close(vectors[k], vector)
            self.assertAlmostEqual(distances[k].item(), distance, places=12)

    def test_interleaved_batch_and_empty_crystals(self):
        frac = torch.tensor([[.98, 0, 0], [0, 0, 0], [.02, 0, 0], [.5, .5, .5]])
        batch = torch.tensor([2, 0, 2, 1])
        lattice = torch.stack([scale * torch.eye(3) for scale in (2, 10, 10, 10)])
        index, images, vectors, distances, counts, ptr = build_periodic_graph(
            frac, lattice, 2, batch, max_neighbor=1
        )
        self.assertEqual(counts.tolist(), [6, 0, 2, 0])
        self.assertEqual(ptr.tolist(), [0, 6, 6, 8, 8])
        i, j = index
        self.assertEqual(batch[i].tolist(), [0] * 6 + [2] * 2)
        torch.testing.assert_close(batch[i], batch[j])
        expected = torch.einsum("ei,eij->ej", frac[j] - frac[i] + images, lattice[batch[i]])
        torch.testing.assert_close(vectors, expected)
        torch.testing.assert_close(distances, vectors.norm(dim=1))

    def test_empty_edge_shapes(self):
        for frac, lattice, counts in (
            (torch.zeros(1, 3), 10 * torch.eye(3), [0]),
            (torch.empty(0, 3), torch.empty(0, 3, 3), []),
        ):
            with self.subTest(counts=counts):
                index, images, vectors, distances, num_edges, ptr = build_periodic_graph(frac, lattice, 1)
                self.assertEqual([t.shape for t in (index, images, vectors, distances)],
                                 [(2, 0), (0, 3), (0, 3), (0,)])
                self.assertTrue(all(t.dtype == torch.long for t in (index, images, num_edges, ptr)))
                self.assertEqual(vectors.dtype, frac.dtype)
                self.assertEqual(distances.dtype, frac.dtype)
                self.assertEqual(num_edges.tolist(), counts)
                self.assertEqual(ptr.tolist(), [0] * (len(counts) + 1))

    def test_neighbor_limit_keeps_complete_shell(self):
        frac = torch.zeros(1, 3, dtype=torch.float64)
        for k, count in ((1, 6), (7, 18), (50, 32)):
            with self.subTest(max_neighbor=k):
                graph = build_periodic_graph(frac, 2 * torch.eye(3).to(frac), 4, max_neighbor=k)
                self.assertEqual(graph[4].tolist(), [count])
                self.assertEqual(graph[5].tolist(), [0, count])
        lattice = torch.diag(frac.new_tensor([2., 2. + 5e-7, 4.]))
        graph = build_periodic_graph(frac, lattice, 3, max_neighbor=1)
        self.assertCountEqual(graph[1].tolist(), [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])

    def test_neighbor_limit_keeps_reverse_and_drops_unselected_pair(self):
        frac = torch.tensor([[0, 0, 0], [.1, 0, 0], [.3, 0, 0]])
        index, images, _, distances, counts, ptr = build_periodic_graph(
            frac, 10 * torch.eye(3), cutoff=3.1, max_neighbor=1
        )
        self.assertCountEqual(index.T.tolist(), [[0, 1], [1, 0], [1, 2], [2, 1]])
        self.assertTrue((images == 0).all())
        torch.testing.assert_close(distances.sort().values, torch.tensor([1., 1., 2., 2.]))
        self.assertEqual(counts.tolist(), [4])
        self.assertEqual(ptr.tolist(), [0, 4])

    def check_gradients(self, device):
        frac = torch.tensor([[.875, .25, .25], [.125, .25, .25]], device=device, requires_grad=True)
        cell = torch.diag(frac.new_tensor([4., 5., 6.])).requires_grad_()
        graph = build_periodic_graph(frac, cell, 1.1, max_neighbor=1)
        self.assertTrue(all(t.device == frac.device for t in graph))
        self.assertCountEqual(edge_keys(*graph[:2]), [(0, 1, 1, 0, 0), (1, 0, -1, 0, 0)])
        graph[3].sum().backward()
        torch.testing.assert_close(frac.grad, frac.new_tensor([[-8, 0, 0], [8, 0, 0]]))
        torch.testing.assert_close(cell.grad, torch.diag(frac.new_tensor([.5, 0, 0])))

    def test_cpu_gradients(self):
        self.check_gradients("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_graph_and_gradients(self):
        self.check_gradients("cuda")

    def test_invalid_parameters(self):
        frac, lattice = torch.zeros(1, 3), torch.eye(3)
        for cutoff in (0, -1, float("inf"), float("nan")):
            with self.subTest(cutoff=cutoff), self.assertRaisesRegex(ValueError, "cutoff"):
                build_periodic_graph(frac, lattice, cutoff)
        for limit in (0, -1, 1.5, True):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "max_neighbor"):
                build_periodic_graph(frac, lattice, max_neighbor=limit)
        with self.assertRaisesRegex(ValueError, "nonsingular"):
            build_periodic_graph(frac, torch.zeros(3, 3))
        with self.assertRaisesRegex(ValueError, "finite"):
            build_periodic_graph(frac + float("nan"), lattice)


if __name__ == "__main__":
    unittest.main(verbosity=2)
