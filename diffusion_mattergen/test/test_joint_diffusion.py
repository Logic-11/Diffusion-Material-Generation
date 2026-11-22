"""Step 6 integration checks, without defining step 7's training loss."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.corruption import AtomTypeCorruption, LatticeCorruption, PositionCorruption
from src.denoiser import Denoiser
from src.diffusion import JointDiffusion
from src.scheduler import LatticeVPScheduler, MaskScheduler, PositionVEScheduler


def make_state(dtype=torch.float64, device="cpu"):
    return {
        "atomic_numbers": torch.tensor([8, 14, 6, 8, 14], device=device),
        "frac_coords": torch.tensor([
            [.1, .1, .1], [.3, .15, .1],
            [.2, .1, .1], [.4, .25, .15], [.25, .45, .3],
        ], dtype=dtype, device=device),
        "lattice": torch.tensor([
            [[4., .4, .1], [.4, 4.8, .2], [.1, .2, 5.2]],
            [[5., .2, .5], [.2, 4.5, .3], [.5, .3, 5.8]],
        ], dtype=dtype, device=device),
        "num_atoms": torch.tensor([2, 3], device=device),
        "batch": torch.tensor([0, 0, 1, 1, 1], device=device),
    }


def make_model(denoiser=None, dtype=torch.float64, device="cpu", cutoff=2.4):
    denoiser = denoiser if denoiser is not None else Denoiser(
        hidden_dim=8, num_radial=8, num_layers=1)
    return JointDiffusion(
        denoiser, AtomTypeCorruption(MaskScheduler()),
        PositionCorruption(PositionVEScheduler()),
        LatticeCorruption(LatticeVPScheduler(0.05)), cutoff=cutoff,
    ).to(dtype=dtype, device=device)


class RecordingDenoiser(nn.Module):
    num_radial = 8

    def forward(self, **kwargs):
        self.inputs = kwargs
        n = kwargs["atomic_numbers"].numel()
        b = kwargs["t_graph"].numel()
        reference = kwargs["radial_features"]
        return {
            "atom_logits": reference.new_zeros(n, 119),
            "coord_cart": reference.new_tensor([1., 2., -1.]).expand(n, 3),
            "lattice_pred": reference.new_full((b, 3, 3), 2.0),
        }


class TestJointDiffusion(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(61)

    def test_joint_noise_is_reproducible_shared_in_time_and_does_not_edit_clean(self):
        clean, model = make_state(), make_model()
        clean["edge_index"] = torch.tensor([[999], [999]])  # deliberately stale
        before = {key: value.clone() for key, value in clean.items()}
        noise = {
            "atom_uniform": clean["frac_coords"].new_tensor([.1, .9, .1, .9, .99]),
            "position": torch.randn_like(clean["frac_coords"]),
            "lattice": torch.randn_like(clean["lattice"]),
        }
        with patch.object(model.atom_corruption, "q_sample", wraps=model.atom_corruption.q_sample) as atoms, \
             patch.object(model.position_corruption, "q_sample", wraps=model.position_corruption.q_sample) as positions, \
             patch.object(model.lattice_corruption, "q_sample", wraps=model.lattice_corruption.q_sample) as cells:
            noisy, info = model.q_sample(clean, [.25, 1.0], noise)
        for call in (atoms, positions, cells):
            torch.testing.assert_close(call.call_args.args[1], info["t_graph"])
        self.assertEqual(noisy["atomic_numbers"].tolist(), [0, 14, 0, 0, 0])
        torch.testing.assert_close(info["atom_target"], clean["atomic_numbers"])
        self.assertEqual(info["coord_target"].shape, (5, 3))
        self.assertEqual(info["lattice_target"].shape, (2, 3, 3))
        self.assertEqual(info["sigma_x"].shape, (5, 1))
        self.assertEqual(info["sigma_l"].shape, (2, 1, 1))
        torch.testing.assert_close(
            info["lattice_target"], -model.lattice_corruption.symmetric_noise(noise["lattice"]))
        self.assertEqual(set(noisy), {"atomic_numbers", "frac_coords", "lattice", "num_atoms", "batch"})
        repeated, _ = model.q_sample(clean, [.25, 1.0], noise)
        for key in noisy:
            torch.testing.assert_close(noisy[key], repeated[key], atol=0, rtol=0)
            self.assertNotEqual(noisy[key].data_ptr(), clean[key].data_ptr())
        for key in before:
            torch.testing.assert_close(clean[key], before[key], atol=0, rtol=0)
        self.assertTrue(torch.all((noisy["frac_coords"] >= 0) & (noisy["frac_coords"] < 1)))

    def test_continuous_time_grid_is_consistent_with_atom_scheduler(self):
        model = make_model()
        k = torch.arange(1, model.timesteps + 1)
        for dtype in (torch.float32, torch.float64):
            t = k.to(dtype) / model.timesteps
            torch.testing.assert_close(model.atom_corruption.scheduler.step_from_time(t), k)
        _, scalar_info = model.q_sample(make_state(), .5)
        torch.testing.assert_close(
            scalar_info["t_graph"],
            torch.tensor([.5, .5], dtype=torch.float64),
        )
        for invalid in (0, 1.1, [1.0], [1.0, 2.0], [1, 2, 3]):
            with self.subTest(t=invalid), self.assertRaises(ValueError):
                model.q_sample(make_state(), invalid)

    def test_non_cubic_conversion_uses_the_current_lattice_and_shared_network_input(self):
        recorder = RecordingDenoiser()
        model, state = make_model(recorder), make_state()
        # Deliberately nonsymmetric for this algebra test to detect a transpose bug.
        # Physical clean/noisy states elsewhere use symmetric lattices.
        state["lattice"][0, 0, 1] = 0.8
        state["atomic_numbers"][0] = 0
        times = state["lattice"].new_tensor([.3, .7])
        pred = model.predict(state, times, return_graph=True)
        cart = state["frac_coords"].new_tensor([1., 2., -1.]).expand(5, 3)
        recovered = torch.einsum(
            "ni,nij->nj", pred["coord_score_times_std"], state["lattice"][state["batch"]])
        torch.testing.assert_close(recovered, cart)
        torch.testing.assert_close(pred["lattice_score_times_std"], torch.full_like(state["lattice"], 2))
        self.assertIs(recorder.inputs["atomic_numbers"], state["atomic_numbers"])
        self.assertIs(recorder.inputs["batch"], state["batch"])
        torch.testing.assert_close(recorder.inputs["t_graph"], times)
        graph = pred["graph"]
        torch.testing.assert_close(
            state["batch"][graph["edge_index"][0]], state["batch"][graph["edge_index"][1]])
        torch.testing.assert_close(graph["edge_ptr"][1:] - graph["edge_ptr"][:-1], graph["num_edges"])

    def test_graph_vectors_distances_and_radial_features_are_recomputed(self):
        model = make_model(cutoff=1.5)
        state = {
            "atomic_numbers": torch.tensor([0, 8]),
            "frac_coords": torch.tensor([[0., 0., 0.], [.1, 0., 0.]], dtype=torch.float64),
            "lattice": 10 * torch.eye(3, dtype=torch.float64)[None],
            "num_atoms": torch.tensor([2]), "batch": torch.tensor([0, 0]),
            "edge_index": torch.tensor([[999], [999]]),
        }
        first = model.predict(state, [.2], return_graph=True)["graph"]
        self.assertEqual(first["edge_index"].shape[1], 2)
        changed = {**state, "frac_coords": state["frac_coords"] * 1.1}
        second = model.predict(changed, [.2], return_graph=True)["graph"]
        torch.testing.assert_close(second["edge_distance"], 1.1 * first["edge_distance"])
        torch.testing.assert_close(second["edge_vectors"], 1.1 * first["edge_vectors"])
        self.assertFalse(torch.equal(second["radial_features"], first["radial_features"]))
        for changed in ({**state, "frac_coords": state["frac_coords"] * 3},
                        {**state, "lattice": state["lattice"] * 2}):
            graph = model.predict(changed, [.2], return_graph=True)["graph"]
            self.assertEqual(graph["edge_index"].shape, (2, 0))
            self.assertEqual(graph["radial_features"].shape, (0, 8))

    def test_score_conversion_needs_no_clean_state_and_mask_is_first(self):
        model, noisy = make_model(RecordingDenoiser()), make_state()
        t = noisy["lattice"].new_tensor([.25, .75])
        pred = model.predict(noisy, t)
        scores = model.scores_from_prediction(pred, noisy, t)
        sigma_x, sigma_l = model.noise_scales(noisy, t)
        torch.testing.assert_close(scores["coord_score"] * sigma_x, pred["coord_score_times_std"])
        torch.testing.assert_close(scores["lattice_score"] * sigma_l, pred["lattice_score_times_std"])
        probs = model.atom_probs_with_mask(pred["atom_logits"])
        self.assertEqual(probs.shape, (5, 119))
        torch.testing.assert_close(probs.sum(-1), torch.ones(5, dtype=probs.dtype))
        self.assertGreater(torch.count_nonzero(probs[:, 0]).item(), 0)

    def test_actual_denoiser_gradients_reach_all_three_heads(self):
        model, clean = make_model(), make_state()
        noisy, info = model.q_sample(clean, [.02, .07])
        pred = model.predict(noisy, info["t_graph"])
        # An autograd smoke objective, NOT the step 7 diffusion training loss.
        objective = sum(value.square().mean() for value in pred.values())
        objective.backward()
        heads = (
            model.denoiser.output_head[-1].weight,
            model.denoiser.EdgeReadlayers[0].coord_output.weight,
            model.denoiser.EdgeReadlayers[0].lattice_output.weight,
        )
        for parameter in heads:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.norm().item(), 0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_singular_lattice_and_zero_time_are_reported(self):
        model, state = make_model(), make_state()
        state["lattice"][0].zero_()
        with self.assertRaisesRegex(ValueError, "nonsingular"):
            model.predict(state, [.2, .3])
        with self.assertRaisesRegex(ValueError, "0 < t"):
            model.predict(make_state(), [0., .2])

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_float32_joint_path(self):
        model = make_model(dtype=torch.float32, device="cuda")
        clean = make_state(dtype=torch.float32, device="cuda")
        noisy, info = model.q_sample(clean, [.02, .07])
        prediction = model.predict(noisy, info["t_graph"])
        for tensor in prediction.values():
            self.assertEqual(tensor.device.type, "cuda")
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertTrue(torch.isfinite(tensor).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
