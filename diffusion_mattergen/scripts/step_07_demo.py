"""A readable joint loss -> backward -> Adam example, without dataset files.

From the project root: python -B -m scripts.step_07_demo
"""

import torch

from src.corruption import AtomTypeCorruption, LatticeCorruption, PositionCorruption
from src.denoiser import Denoiser
from src.diffusion import JointDiffusion
from src.loss import JointDiffusionLoss
from src.scheduler import LatticeVPScheduler, MaskScheduler, PositionVEScheduler
from scripts.train_mask import head_gradient_norms


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
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
    # These two synthetic crystals constitute this demonstration's ENTIRE data.
    # Real training computes rho once over the complete training split.
    rho = (clean["num_atoms"] / torch.linalg.det(clean["lattice"])).mean().item()
    model = JointDiffusion(
        Denoiser(hidden_dim=16, num_radial=16, num_layers=1),
        AtomTypeCorruption(MaskScheduler(timesteps=1000)),
        PositionCorruption(PositionVEScheduler()),
        LatticeCorruption(LatticeVPScheduler(limit_density=rho)),
        loss_fn=JointDiffusionLoss(),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    k = torch.tensor([50, 250])
    noise = {
        "atom_uniform": torch.tensor([.01, .99, .01, .99, .99]),
        "position": torch.randn_like(clean["frac_coords"]),
        "lattice": torch.randn_like(clean["lattice"]),
    }

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = model.compute_loss(clean, k=k, noise=noise)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = head_gradient_norms(model.denoiser)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(value > 0 for value in gradients.values())
    optimizer.step()

    print("k_graph:", k.tolist())
    print(f"loss={loss.item():.8f}")
    for name, value in metrics.items():
        print(f"{name}={value.item():.8f}")
    for name, value in gradients.items():
        print(f"{name}={value:.8f}")
    print("Completed one joint Adam update.")


if __name__ == "__main__":
    main()
