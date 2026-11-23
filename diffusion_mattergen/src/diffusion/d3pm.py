"""Shared probability calculations for absorbing-state element diffusion."""

from numbers import Integral

import torch
from torch.nn import functional as F


def _standard_mask_keep_probs(timesteps):
    """Match MaskDiffusion._create_state, including its float32 betas/state."""
    if isinstance(timesteps, bool) or not isinstance(timesteps, Integral) or timesteps < 1:
        raise ValueError("timesteps must be a positive integer")
    beta = (timesteps - torch.arange(timesteps, dtype=torch.float32)).reciprocal()
    beta = torch.cat((beta.new_zeros(1), beta)).double()
    keep = (1 - beta).cumprod(0)
    keep[-1] = 0.0
    return keep.float()


def _absorbing_posterior_logits(clean_probs, noisy_classes, k_atom, keep_probs,
                                timesteps, epsilon=1e-20):
    """Upstream MaskDiffusion's efficient posterior, without dense matrices.

    k is the one-based forward step. Upstream sample_and_compute_posterior_q
    instead receives t=k-1. Both the marginal and transition likelihood get
    epsilon BEFORE their logarithms, rather than after their product.
    """
    previous = k_atom - 1
    keep = keep_probs.to(device=clean_probs.device)[previous, None]
    real_probs = keep * clean_probs[:, 1:]
    marginal = torch.cat((1.0 - real_probs.sum(-1, keepdim=True), real_probs), dim=-1)
    marginal = torch.where(previous[:, None] == 0, clean_probs, marginal)

    observed = F.one_hot(noisy_classes, clean_probs.shape[-1]).to(clean_probs)
    # The schedule is evaluated on integer t, giving float32 beta upstream.
    beta = (timesteps - previous).float().reciprocal()[:, None]
    likelihood_real = (1.0 - beta) * observed[:, 1:] + beta * observed[:, :1]
    likelihood = torch.cat((observed[:, :1], likelihood_real), dim=-1)
    return (marginal + epsilon).log() + (likelihood + epsilon).log()
