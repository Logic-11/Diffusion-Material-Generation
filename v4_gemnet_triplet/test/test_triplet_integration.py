"""GemNet-T triplet integration checks for the continuous-time interface."""

import math
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.corruption import AtomTypeCorruption, LatticeCorruption, PositionCorruption
from src.denoiser import Denoiser
from src.diffusion import JointDiffusion
from src.geometry import build_triplet_graph
from src.gnn import RadialBasis
from src.scheduler import LatticeVPScheduler, MaskScheduler, PositionVEScheduler


class TestTripletGeometry(unittest.TestCase):
    def test_shared_center_angles_and_periodic_edge_ids(self):
        edge_index = torch.tensor([[0, 0, 0], [1, 2, 1]], dtype=torch.long)
        vectors = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        distances = vectors.norm(dim=-1)
        id3_b, id3_c, cos_phi = build_triplet_graph(
            edge_index, vectors, distances, num_nodes=3
        )
        self.assertEqual(id3_b.numel(), 6)
        observed = {
            (int(source), int(target)): float(value)
            for source, target, value in zip(id3_c, id3_b, cos_phi)
        }
        expected = {
            (0, 1): 0.0,
            (0, 2): -1.0,
            (1, 0): 0.0,
            (1, 2): 0.0,
            (2, 0): -1.0,
            (2, 1): 0.0,
        }
        self.assertEqual(set(observed), set(expected))
        for key, value in expected.items():
            self.assertAlmostEqual(observed[key], value, places=12)

    def test_angle_neighbor_cap_and_empty_graph(self):
        edge_index = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.long)
        vectors = torch.tensor(
            [[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        distances = vectors.norm(dim=-1)
        id3_b, id3_c, cos_phi = build_triplet_graph(
            edge_index,
            vectors,
            distances,
            num_nodes=4,
            max_angle_neighbors=2,
        )
        self.assertEqual(id3_b.numel(), 2)
        self.assertEqual(set(id3_b.tolist()), {1, 2})
        self.assertEqual(set(id3_c.tolist()), {1, 2})
        self.assertEqual(cos_phi.shape, (2,))

        empty = torch.empty((2, 0), dtype=torch.long)
        empty_vectors = torch.empty((0, 3), dtype=torch.float64)
        empty_b, empty_c, empty_cos = build_triplet_graph(
            empty, empty_vectors, empty_vectors.norm(dim=-1), num_nodes=1
        )
        self.assertEqual(empty_b.shape, (0,))
        self.assertEqual(empty_c.shape, (0,))
        self.assertEqual(empty_cos.shape, (0,))

    def test_cosine_gradient_reaches_edge_vectors(self):
        vectors = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        edge_index = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.long)
        distances = vectors.norm(dim=-1)
        _, _, cos_phi = build_triplet_graph(
            edge_index, vectors, distances, num_nodes=4
        )
        cos_phi.sum().backward()
        self.assertIsNotNone(vectors.grad)
        self.assertTrue(torch.isfinite(vectors.grad).all())
        self.assertGreater(vectors.grad.norm().item(), 0.0)


def make_edge_graph(dtype=torch.float64):
    edge_index = torch.tensor(
        [[0, 0, 1, 1], [1, 2, 0, 2]], dtype=torch.long
    )
    vectors = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
         [-1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=dtype,
    )
    distances = vectors.norm(dim=-1)
    id3_b, id3_c, cos_phi = build_triplet_graph(
        edge_index, vectors, distances, num_nodes=3
    )
    return edge_index, vectors, distances, id3_b, id3_c, cos_phi


class TestTripletDenoiser(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)

    @staticmethod
    def make_model(use_triplets=True):
        return Denoiser(
            hidden_dim=8,
            num_radial=8,
            num_layers=1,
            atom_types=118,
            num_spherical=4,
            use_triplets=use_triplets,
        ).double()

    @staticmethod
    def make_inputs():
        edge_index, vectors, distances, id3_b, id3_c, cos_phi = make_edge_graph()
        radial = RadialBasis(num_radial=8, cutoff=3.0).double()
        return {
            "atomic_numbers": torch.tensor([8, 14, 6]),
            "radial_features": radial(distances),
            "edge_vectors": vectors,
            "edge_distance": distances,
            "edge_index": edge_index,
            "t_graph": torch.tensor([0.4], dtype=torch.float64),
            "batch": torch.zeros(3, dtype=torch.long),
            "num_edges": torch.tensor([4]),
            "edge_ptr": torch.tensor([0, 4]),
            "edge_batch": torch.zeros(4, dtype=torch.long),
            "id3_center_b": id3_b,
            "id3_center_c": id3_c,
            "cos_phi": cos_phi,
        }

    def test_rotation_invariance_and_triplet_gradients(self):
        model = self.make_model().train()
        inputs = self.make_inputs()
        output = model(**inputs)
        self.assertEqual(output["atom_logits"].shape, (3, 119))
        self.assertEqual(output["coord_cart"].shape, (3, 3))
        self.assertEqual(output["lattice_pred"].shape, (1, 3, 3))

        rotation = torch.tensor(
            [[0.5, -math.sqrt(3) / 2, 0.0],
             [math.sqrt(3) / 2, 0.5, 0.0],
             [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        rotated_inputs = dict(inputs)
        rotated_inputs["edge_vectors"] = inputs["edge_vectors"] @ rotation.T
        rotated = model(**rotated_inputs)
        torch.testing.assert_close(
            output["atom_logits"], rotated["atom_logits"],
            atol=1e-7, rtol=1e-6,
        )
        torch.testing.assert_close(
            output["coord_cart"] @ rotation.T,
            rotated["coord_cart"],
            atol=1e-7, rtol=1e-6,
        )

        model.zero_grad(set_to_none=True)
        sum(value.square().mean() for value in output.values()).backward()
        triplet_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if ".triplet_interaction." in name
        ]
        self.assertTrue(triplet_parameters)
        for parameter in triplet_parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.norm().item(), 0.0)

    def test_disabled_triplets_require_no_angle_inputs(self):
        inputs = self.make_inputs()
        for key in ("id3_center_b", "id3_center_c", "cos_phi"):
            inputs.pop(key)
        enabled = self.make_model(use_triplets=True)
        with self.assertRaisesRegex(ValueError, "triplet"):
            enabled(**inputs)
        disabled = self.make_model(use_triplets=False)
        output = disabled(**inputs)
        self.assertEqual(output["atom_logits"].shape, (3, 119))


class TestTripletDiffusion(unittest.TestCase):
    @staticmethod
    def make_state():
        return {
            "atomic_numbers": torch.tensor([8, 14, 6, 8, 14]),
            "frac_coords": torch.tensor(
                [[.1, .1, .1], [.3, .15, .1], [.2, .1, .1],
                 [.4, .25, .15], [.25, .45, .3]],
                dtype=torch.float64,
            ),
            "lattice": torch.tensor(
                [[[4., .4, .1], [.4, 4.8, .2], [.1, .2, 5.2]],
                 [[5., .2, .5], [.2, 4.5, .3], [.5, .3, 5.8]]],
                dtype=torch.float64,
            ),
            "num_atoms": torch.tensor([2, 3]),
            "batch": torch.tensor([0, 0, 1, 1, 1]),
        }

    def make_model(self, use_triplets=True):
        denoiser = Denoiser(
            hidden_dim=8,
            num_radial=8,
            num_layers=1,
            atom_types=118,
            num_spherical=4,
            use_triplets=use_triplets,
        ).double()
        return JointDiffusion(
            denoiser,
            AtomTypeCorruption(MaskScheduler()),
            PositionCorruption(PositionVEScheduler()),
            LatticeCorruption(LatticeVPScheduler(0.05)),
            cutoff=2.4,
            max_neighbor=10,
            max_cell_images_per_dim=3,
            max_angle_neighbors=4,
        ).double()

    def test_predict_rebuilds_triplets_from_noisy_geometry(self):
        torch.manual_seed(23)
        model, state = self.make_model(), self.make_state()
        state["frac_coords"].requires_grad_()
        state["lattice"].requires_grad_()
        prediction = model.predict(
            state, torch.tensor([.3, .7], dtype=torch.float64), return_graph=True
        )
        graph = prediction["graph"]
        id3_b, id3_c = graph["id3_center_b"], graph["id3_center_c"]
        self.assertGreater(id3_b.numel(), 0)
        self.assertEqual(graph["cos_phi"].shape, id3_b.shape)
        torch.testing.assert_close(
            graph["edge_index"][0, id3_b],
            graph["edge_index"][0, id3_c],
        )
        objective = (
            prediction["atom_logits"].square().mean()
            + prediction["coord_score_times_std"].square().mean()
            + prediction["lattice_score_times_std"].square().mean()
        )
        objective.backward()
        for value in (state["frac_coords"], state["lattice"]):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.norm().item(), 0.0)

    def test_joint_loss_trains_triplet_parameters(self):
        torch.manual_seed(29)
        model, clean = self.make_model(), self.make_state()
        model.zero_grad(set_to_none=True)
        loss, metrics = model.compute_loss(clean, t=torch.tensor([.02, .07]))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(value) for value in metrics.values()))
        loss.backward()
        triplet_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if ".triplet_interaction." in name
        ]
        self.assertTrue(triplet_parameters)
        for parameter in triplet_parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.norm().item(), 0.0)

    def test_ablation_skips_triplet_graph(self):
        model, state = self.make_model(use_triplets=False), self.make_state()
        prediction = model.predict(
            state, torch.tensor([.3, .7], dtype=torch.float64), return_graph=True
        )
        self.assertEqual(prediction["graph"]["id3_center_b"].shape, (0,))
        self.assertEqual(prediction["graph"]["cos_phi"].shape, (0,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
