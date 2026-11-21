"""Periodic graph regressions, independent of the dataset and denoiser."""

from pathlib import Path
from itertools import product
import sys
import unittest
from unittest.mock import patch

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
        index, images, vectors, distances, counts, ptr, _ = build_periodic_graph(
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

    def test_radius_excludes_overlapping_sites_and_includes_cutoff(self):
        cell = torch.eye(3, dtype=torch.float64)
        frac = cell.new_zeros((6, 3))
        frac[:, 0] = cell.new_tensor([0., 0., .005, .01, .02, .25])
        index, images, _, distances, _, _, _ = build_periodic_graph(
            frac, cell, cutoff=.02, max_neighbor=None)
        self.assertCountEqual(index.T.tolist(), [[0, 4], [1, 4], [2, 4], [4, 0], [4, 1], [4, 2]])
        self.assertTrue((images == 0).all())
        self.assertTrue(((distances > .01) & (distances <= .02)).all())

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

        # This reference intentionally needs images beyond the default cap.
        index, images, vectors, distances, counts, _, _ = build_periodic_graph(
            frac + shifts, lattice, cutoff, max_neighbor=None, max_cell_images_per_dim=7
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
        index, images, vectors, distances, counts, ptr, _ = build_periodic_graph(
            frac, lattice, 2, batch, max_neighbor=6
        )
        self.assertEqual(counts.tolist(), [6, 0, 2, 0])
        self.assertEqual(ptr.tolist(), [0, 6, 6, 8, 8])
        i, j = index
        self.assertEqual(batch[i].tolist(), [0] * 6 + [2] * 2)
        torch.testing.assert_close(batch[i], batch[j])
        expected = torch.einsum("ei,eij->ej", frac[j] - frac[i] + images, lattice[batch[i]])
        torch.testing.assert_close(vectors, expected)
        torch.testing.assert_close(distances, vectors.norm(dim=1))
        for start, stop in zip(ptr[:-1].tolist(), ptr[1:].tolist()):
            middle = (start + stop) // 2
            torch.testing.assert_close(index[:, middle:stop], index[:, start:middle].flip([0]))
            torch.testing.assert_close(images[middle:stop], -images[start:middle])
            torch.testing.assert_close(vectors[middle:stop], -vectors[start:middle])

    def test_empty_edge_shapes(self):
        for frac, lattice, counts in (
            (torch.zeros(1, 3), 10 * torch.eye(3), [0]),
            (torch.empty(0, 3), torch.empty(0, 3, 3), []),
        ):
            with self.subTest(counts=counts):
                index, images, vectors, distances, num_edges, ptr, _ = build_periodic_graph(frac, lattice, 1)
                self.assertEqual([t.shape for t in (index, images, vectors, distances)],
                                 [(2, 0), (0, 3), (0, 3), (0,)])
                self.assertTrue(all(t.dtype == torch.long for t in (index, images, num_edges, ptr)))
                self.assertEqual(vectors.dtype, frac.dtype)
                self.assertEqual(distances.dtype, frac.dtype)
                self.assertEqual(num_edges.tolist(), counts)
                self.assertEqual(ptr.tolist(), [0] * (len(counts) + 1))

    def test_neighbor_limit_cuts_tied_shell_before_mirroring(self):
        frac = torch.zeros(1, 3, dtype=torch.float64)
        cell = 2 * torch.eye(3).to(frac)
        images = [shift for shift in product(range(-2, 3), repeat=3)
                  if 0 < sum(c * c for c in shift) <= 4]
        squared = (2 * frac.new_tensor(images)).square().sum(dim=1)
        ranked = squared[None, :].sort(dim=1).indices[0]
        for k in (1, 7, 50):
            with self.subTest(max_neighbor=k):
                selected = set(ranked[:k].tolist())
                half = [s for i, s in enumerate(images) if i in selected and s < (0, 0, 0)]
                expected = half + [tuple(-c for c in s) for s in half]
                graph = build_periodic_graph(frac, cell, cutoff=4, max_neighbor=k)
                self.assertEqual([tuple(s) for s in graph[1].tolist()], expected)
                self.assertEqual(graph[5].tolist(), [0, len(expected)])

        lattice = torch.diag(frac.new_tensor([2., 2. + 5e-7, 4.]))
        graph = build_periodic_graph(frac, lattice, 3, max_neighbor=2)
        self.assertEqual(graph[1].tolist(), [[-1, 0, 0], [1, 0, 0]])

    def test_one_sided_neighbor_survives_only_in_the_canonical_direction(self):
        for middle, expected in ((.1, [[1, 0], [2, 1], [0, 1], [1, 2]]),
                                 (.2, [[2, 1], [1, 2]])):
            with self.subTest(middle=middle):
                frac = torch.tensor([[0, 0, 0], [middle, 0, 0], [.3, 0, 0]])
                index, images, _, distances, counts, ptr, _ = build_periodic_graph(
                    frac, 10 * torch.eye(3), cutoff=3.1, max_neighbor=1)
                self.assertEqual(index.T.tolist(), expected)
                self.assertTrue((images == 0).all())
                torch.testing.assert_close(distances, (frac[index[1]] - frac[index[0]]).norm(dim=1) * 10)
                self.assertEqual(counts.tolist(), [len(expected)])
                self.assertEqual(ptr.tolist(), [0, len(expected)])

    def check_bounded_image_search(self, device):
        frac = torch.zeros(1, 3, dtype=torch.float64, device=device)
        arange, cartesian_prod = torch.arange, torch.cartesian_prod

        def bounded_axis(start, stop, **kwargs):
            # Fail before allocating if the OOM regression is reintroduced.
            self.assertGreater(stop - start, 0)
            self.assertLessEqual(stop - start, 11)
            return arange(start, stop, **kwargs)

        def bounded_product(*axes):
            self.assertLessEqual(axes[0].numel() * axes[1].numel() * axes[2].numel(), 1331)
            return cartesian_prod(*axes)

        cells = (
            # All images within the search cap fall below the minimum distance.
            (1e-20 * torch.eye(3, dtype=frac.dtype, device=device), False),
            (frac.new_tensor([[4., 4., 4.], [4., 4.0000001, 4.], [4., 4., 4.0000001]]), True),
        )
        for cell, has_edges in cells:
            with (self.subTest(device=device, cell=cell.tolist()),
                  patch("src.geometry.torch.arange", side_effect=bounded_axis),
                  patch("src.geometry.torch.cartesian_prod", side_effect=bounded_product)):
                graph = build_periodic_graph(frac, cell, max_neighbor=None)
            self.assertEqual(graph[1].shape[0] > 0, has_edges)
            self.assertTrue((graph[1].abs() <= 5).all())
            self.assertTrue(torch.isfinite(graph[2]).all())
            self.assertTrue(torch.isfinite(graph[3]).all())
            self.assertCountEqual(graph[1].tolist(), (-graph[1]).tolist())

    def test_near_singular_image_search_is_bounded_before_allocation(self):
        self.check_bounded_image_search("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_near_singular_image_search_is_bounded_before_allocation(self):
        self.check_bounded_image_search("cuda")

    def test_configured_image_cap_matches_finite_radius_search(self):
        frac = torch.zeros(1, 3, dtype=torch.float64)
        for cap in (1, 2, 5):
            with self.subTest(cap=cap):
                graph = build_periodic_graph(
                    frac, torch.eye(3).to(frac), cutoff=3.1, max_neighbor=None,
                    max_cell_images_per_dim=cap)
                expected = [shift for shift in product(range(-cap, cap + 1), repeat=3)
                            if 0 < sum(x * x for x in shift) <= 3.1 ** 2]
                self.assertCountEqual([tuple(shift) for shift in graph[1].tolist()], expected)
                torch.testing.assert_close(graph[2], graph[1].to(frac))

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
        for limit in (None, 0, -1, 1.5, True):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "max_cell_images_per_dim"):
                build_periodic_graph(frac, lattice, max_cell_images_per_dim=limit)
        with self.assertRaisesRegex(ValueError, "nonsingular"):
            build_periodic_graph(frac, torch.zeros(3, 3))
        with self.assertRaisesRegex(ValueError, "finite"):
            build_periodic_graph(frac + float("nan"), lattice)


if __name__ == "__main__":
    unittest.main(verbosity=2)
