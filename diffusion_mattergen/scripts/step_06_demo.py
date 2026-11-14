"""Run from the project root: python -B -m scripts.step_06_demo.

This demonstrates step 6 on two synthetic symmetric positive-definite cells.
It does not implement the step 7 loss or the later generation loop.
"""

import torch

from src.corruption import AtomTypeCorruption, LatticeCorruption, PositionCorruption
from src.denoiser import Denoiser
from src.diffusion import JointDiffusion
from src.scheduler import LatticeVPScheduler, MaskScheduler, PositionVEScheduler


def main():
    torch.manual_seed(6)
    clean = {
        "atomic_numbers": torch.tensor([8, 14, 6, 8, 14]),
        "frac_coords": torch.tensor([
            [.1, .1, .1], [.3, .15, .1],
            [.2, .1, .1], [.4, .25, .15], [.25, .45, .3],
        ]),
        "lattice": torch.tensor([
            [[4., .4, .1], [.4, 4.8, .2], [.1, .2, 5.2]],
            [[5., .2, .5], [.2, 4.5, .3], [.5, .3, 5.8]],
        ]),
        "num_atoms": torch.tensor([2, 3]),
        "batch": torch.tensor([0, 0, 1, 1, 1]),
    }
    # Only for this synthetic two-crystal demonstration. In training, compute
    # rho once over the entire training split; never recompute it per mini-batch.
    density = (clean["num_atoms"] / torch.linalg.det(clean["lattice"])).mean().item()
    model = JointDiffusion(
        denoiser=Denoiser(hidden_dim=16, num_radial=16, num_layers=1),
        atom_corruption=AtomTypeCorruption(MaskScheduler(timesteps=1000)),
        position_corruption=PositionCorruption(PositionVEScheduler()),
        lattice_corruption=LatticeCorruption(LatticeVPScheduler(limit_density=density)),
    ).to(clean["lattice"])

    # One integer time per crystal; the three fields share that crystal's time.
    k = torch.tensor([50, 250])
    noise = {
        "atom_uniform": torch.rand(5),
        "position": torch.randn(5, 3),
        "lattice": torch.randn(2, 3, 3),  # IID, not already symmetric
    }
    noisy, info = model.q_sample(clean, k, noise=noise)
    prediction = model.predict(noisy, info["t_graph"], return_graph=True)
    scores = model.scores_from_prediction(prediction, noisy, info["t_graph"])
    atom_probs = model.atom_probs_with_mask(prediction["atom_logits"])

    print("t_graph:", info["t_graph"].tolist())
    for key in ("atomic_numbers", "frac_coords", "lattice"):
        print("noisy", key, tuple(noisy[key].shape))
    for key in ("atom_target", "coord_target", "lattice_target"):
        print("target", key, tuple(info[key].shape))
    for key in ("atom_logits", "coord_score_times_std", "lattice_score_times_std"):
        print("prediction", key, tuple(prediction[key].shape))
        assert torch.isfinite(prediction[key]).all()
    print("edges per crystal:", prediction["graph"]["num_edges"].tolist())
    print("posterior class probabilities:", tuple(atom_probs.shape))
    print("clean MASK probability:", atom_probs[:, -1].tolist())
    assert torch.isfinite(scores["coord_score"]).all()
    assert torch.isfinite(scores["lattice_score"]).all()


if __name__ == "__main__":
    main()
