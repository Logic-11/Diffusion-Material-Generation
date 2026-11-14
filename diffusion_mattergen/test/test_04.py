"""Tests for Denoiser's concatenated graphs and per-atom type logits.

Run from the project root after activating the environment:
    conda activate diffusion
    python -B test/test_04.py

Supply per-atom t as [N_total, 1]. No vector output is required here. With an atom-type loss, vector_mlp parameters are
expected to have grad=None because their outputs are not used.
"""

import math
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.denoiser import Denoiser
from src.gnn import MPBlock


class TestDenoiser(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.atomic_numbers = torch.tensor([11, 17, 8], dtype=torch.long)
        # Row 0 is the center; row 1 is its neighbor.
        self.edge_index = torch.tensor(
            [[0, 2, 1, 1], [1, 1, 0, 2]], dtype=torch.long
        )
        self.positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]
        )
        self.edge_vectors = self.make_vectors(self.positions, self.edge_index)
        self.t = torch.full((3, 1), 0.2)

    @staticmethod
    def make_vectors(positions, edge_index):
        return positions[edge_index[1]] - positions[edge_index[0]]

    @staticmethod
    def make_model(hidden_dim=8, num_layers=3):
        return Denoiser(hidden_dim=hidden_dim, edge_dim=1, num_layers=num_layers)

    def assert_output(self, output, num_nodes):
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(tuple(output.shape), (num_nodes, 118))
        self.assertTrue(torch.isfinite(output).all().item())

    def test_01_layers_are_registered_and_independent(self):
        model = self.make_model()
        self.assertIsInstance(model.MPlayers, nn.ModuleList)
        self.assertEqual(len(model.MPlayers), 3)
        for layer in model.MPlayers:
            self.assertIsInstance(layer, MPBlock)
        self.assertEqual(len({id(layer) for layer in model.MPlayers}), 3)
        weight_addresses = {
            layer.edge_mlp[0].weight.data_ptr() for layer in model.MPlayers
        }
        self.assertEqual(len(weight_addresses), 3)
        names = dict(model.named_parameters())
        for i in range(3):
            self.assertIn(f"MPlayers.{i}.edge_mlp.0.weight", names)

    def test_02_forward_shapes_and_mask_token(self):
        # Each layer receives scalar geometric features [E_total, 1].
        for hidden_dim, num_layers in ((8, 3), (64, 1)):
            with self.subTest(hidden_dim=hidden_dim, num_layers=num_layers):
                model = self.make_model(hidden_dim, num_layers).eval()
                with torch.no_grad():
                    output = model(
                        self.atomic_numbers, self.edge_index,
                        self.edge_vectors, self.t,
                    )
                    self.assert_output(output, 3)
                    masked = self.atomic_numbers.clone()
                    masked[0] = 0
                    self.assert_output(
                        model(masked, self.edge_index, self.edge_vectors, self.t), 3
                    )

    def test_03_two_graphs_match_separate_calls(self):
        model = self.make_model().eval()
        atoms_b = torch.tensor([14, 14], dtype=torch.long)
        index_b = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        positions_b = torch.tensor([[0., 0, 0], [0, 0, 1.5]])
        vectors_b = self.make_vectors(positions_b, index_b)
        t_b = torch.full((2, 1), 0.8)
        atoms = torch.cat((self.atomic_numbers, atoms_b))
        indices = torch.cat((self.edge_index, index_b + 3), dim=1)
        vectors = torch.cat((self.edge_vectors, vectors_b))
        times = torch.cat((self.t, t_b))
        with torch.no_grad():
            output_a = model(self.atomic_numbers, self.edge_index, self.edge_vectors, self.t)
            output_b = model(atoms_b, index_b, vectors_b, t_b)
            combined = model(atoms, indices, vectors, times)
        self.assert_output(combined, 5)
        torch.testing.assert_close(combined, torch.cat((output_a, output_b)), atol=1e-6, rtol=1e-5)

    def test_04_scalar_rotation_and_translation_invariance(self):
        model = self.make_model().eval()
        rotation = torch.tensor([
            [0.5, -math.sqrt(3) / 2, 0.0],
            [math.sqrt(3) / 2, 0.5, 0.0],
            [0.0, 0.0, 1.0],
        ])
        rotated = self.make_vectors(
            self.positions @ rotation.T, self.edge_index
        )
        shifted = self.make_vectors(
            self.positions + torch.tensor([1.2, -2.3, 0.4]), self.edge_index
        )
        with torch.no_grad():
            original = model(
                self.atomic_numbers, self.edge_index, self.edge_vectors, self.t,
            )
            for vectors in (rotated, shifted):
                output = model(
                    self.atomic_numbers, self.edge_index, vectors, self.t,
                )
                # node_output has scalar channels, not x/y/z components.
                torch.testing.assert_close(
                    output, original, atol=1e-6, rtol=1e-5
                )

    def test_05_empty_graph_and_single_atom(self):
        model = self.make_model().eval()
        empty_index = torch.empty((2, 0), dtype=torch.long)
        empty_vectors = torch.empty((0, 3))
        with torch.no_grad():
            for num_nodes in (3, 1):
                with self.subTest(num_nodes=num_nodes):
                    output = model(
                        self.atomic_numbers[:num_nodes], empty_index,
                        empty_vectors, self.t[:num_nodes],
                    )
                    self.assert_output(output, num_nodes)

    def test_06_backward_connectivity_and_finite_gradients(self):
        model = self.make_model().train()
        vectors = self.edge_vectors.clone().requires_grad_(True)
        times = self.t.clone().requires_grad_(True)

        # Clear old gradients before building a fresh computation graph.
        model.zero_grad(set_to_none=True)
        output = model(self.atomic_numbers, self.edge_index, vectors, times)
        loss = nn.functional.cross_entropy(output, self.atomic_numbers - 1)
        self.assertTrue(loss.requires_grad)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()  # Do not put this forward pass inside torch.no_grad().

        gradient_norms = {}
        unused_vector_parameters = []
        for name, parameter in model.named_parameters():
            if ".vector_mlp." in name:
                # Denoiser currently discards node_vectors. A scalar loss
                # therefore has no path to these parameters.
                self.assertIsNone(parameter.grad, name)
                unused_vector_parameters.append(name)
                continue

            self.assertIsNotNone(parameter.grad, f"Missing gradient: {name}")
            self.assertTrue(
                torch.isfinite(parameter.grad).all().item(),
                f"Non-finite gradient: {name}",
            )
            parts = name.split(".")
            group = ".".join(parts[:3]) if parts[0] == "MPlayers" else parts[0]
            gradient_norms[group] = gradient_norms.get(group, 0.0) + (
                parameter.grad.detach().double().norm().item()
            )

        expected_groups = {"node_embedding", "time_embedding", "output_head"}
        for i in range(model.num_layers):
            expected_groups.update({f"MPlayers.{i}.edge_mlp", f"MPlayers.{i}.node_mlp"})
        self.assertEqual(set(gradient_norms), expected_groups)
        for group, norm in gradient_norms.items():
            self.assertGreater(norm, 0.0, f"All gradients are zero: {group}")
        for name, tensor in (("edge_vectors", vectors), ("t", times)):
            self.assertIsNotNone(tensor.grad, f"Missing input gradient: {name}")
            self.assertTrue(torch.isfinite(tensor.grad).all().item(), name)
            self.assertGreater(tensor.grad.norm().item(), 0.0, name)
        self.assertEqual(len(unused_vector_parameters), 4 * model.num_layers)

        print(f"\nGradient check: loss={loss.item():.6f}", flush=True)
        for group, norm in gradient_norms.items():
            print(f"  {group}: grad_norm_sum={norm:.3e}", flush=True)
        print(
            f"  Expected unused vector_mlp parameters: {len(unused_vector_parameters)}",
            flush=True,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
