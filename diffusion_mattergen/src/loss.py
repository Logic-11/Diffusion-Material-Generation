"""Joint loss using MatterGen's log-space absorbing D3PM training objective.

The final atom-logit column is a LEARNED clean MASK class. With the project's
118 real elements this requires 119 logits; upstream's 100 elements use 101.
The coordinate/lattice targets and state dictionaries retain the local API.
"""

import math
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical, kl_divergence


def categorical_posterior(clean_probs, noisy_classes, transition, cumulative_previous):
    """Bayes posterior over the previous state, using ROW-stochastic matrices.

    clean_probs: [N,C], noisy_classes: [N] zero-based class indices.
    transition: Q_k, cumulative_previous: Q_bar_{k-1}, each [C,C] or [N,C,C].
    Class order for atoms is [Z=1,...,C-1,MASK], unlike input MASK=0.
    This exact probability-space reference is independent of the training
    implementation below, which follows upstream's epsilon-smoothed logits.
    An impossible conditioning event raises explicitly.
    """
    if clean_probs.ndim != 2 or not clean_probs.is_floating_point():
        raise ValueError("clean_probs must be floating [N,C]")
    n, c = clean_probs.shape
    if (noisy_classes.shape != (n,) or noisy_classes.dtype != torch.long
            or noisy_classes.device != clean_probs.device
            or torch.any(noisy_classes < 0) or torch.any(noisy_classes >= c)):
        raise ValueError("noisy_classes must be int64 [N] in 0..C-1 on the same device")
    for value in (clean_probs, transition, cumulative_previous):
        if (value.device != clean_probs.device or value.dtype != clean_probs.dtype
                or not torch.isfinite(value).all() or torch.any(value < 0)
                or not torch.allclose(value.sum(-1), torch.ones_like(value.sum(-1)),
                                      atol=1e-6, rtol=1e-5)):
            raise ValueError("Probabilities must be finite, nonnegative and row-normalized")
    for matrix in (transition, cumulative_previous):
        if matrix.shape not in ((c, c), (n, c, c)):
            raise ValueError("Transition matrices must have shape [C,C] or [N,C,C]")
    previous = torch.matmul(clean_probs[:, None, :], cumulative_previous).squeeze(1)
    # Q_k[i, observed j], as a likelihood for each possible previous state i.
    likelihood = transition.expand(n, c, c).gather(
        2, noisy_classes[:, None, None].expand(n, c, 1)).squeeze(-1)
    unnormalized = previous * likelihood
    evidence = unnormalized.sum(-1, keepdim=True)
    if torch.any(evidence <= 0):
        raise ValueError("The observed class has zero probability under this forward process")
    return unnormalized / evidence


def categorical_kl(true_probs, model_probs):
    """KL(q || p) per row, with exact zero-mass terms contributing zero.

    If q>0 but p=0 the result is +inf, as mathematically required. Avoid taking
    log(0) on the zero-q support, rather than evaluating 0 * (log(0)-log(0)).
    This probability-space helper is only a mathematical reference. Training
    below uses Categorical(logits=...) with upstream's posterior smoothing.
    """
    if true_probs.shape != model_probs.shape or true_probs.ndim != 2:
        raise ValueError("Both categorical distributions must have shape [N,C]")
    support = true_probs > 0
    safe_q = torch.where(support, true_probs, torch.ones_like(true_probs))
    safe_p = torch.where(support, model_probs, torch.ones_like(model_probs))
    return (true_probs * (safe_q.log() - safe_p.log())).sum(-1)


def aggregate_per_crystal(values, batch, num_crystals, reduce="sum"):
    """Mean features, then sum/mean atoms, as in aggregate_per_sample upstream.

    With batch=None, the first axis already indexes crystals (e.g. lattice).
    Returns [B]; averaging over crystals belongs to the joint loss below.
    """
    if reduce not in ("sum", "mean"):
        raise ValueError("reduce must be 'sum' or 'mean'")
    if values.ndim < 1 or values.shape[0] == 0 or num_crystals < 1:
        raise ValueError("Cannot aggregate an empty batch")
    per_row = values.reshape(values.shape[0], -1).mean(dim=-1)
    if batch is None:
        if per_row.shape != (num_crystals,):
            raise ValueError("Unbatched values must have one row per crystal")
        return per_row
    if (batch.shape != per_row.shape or batch.dtype != torch.long
            or batch.device != values.device or torch.any(batch < 0)
            or torch.any(batch >= num_crystals)):
        raise ValueError("batch must assign each row to a valid crystal")
    result = per_row.new_zeros(num_crystals).index_add(0, batch, per_row)
    if reduce == "mean":
        result = result / torch.bincount(batch, minlength=num_crystals).clamp_min(1).to(result)
    return result


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
    real_probs = keep * clean_probs[:, :-1]
    marginal = torch.cat((real_probs, 1.0 - real_probs.sum(-1, keepdim=True)), dim=-1)
    marginal = torch.where(previous[:, None] == 0, clean_probs, marginal)

    observed = F.one_hot(noisy_classes, clean_probs.shape[-1]).to(clean_probs)
    # The schedule is evaluated on integer t, giving float32 beta upstream.
    beta = (timesteps - previous).float().reciprocal()[:, None]
    likelihood_real = (1.0 - beta) * observed[:, :-1] + beta * observed[:, -1:]
    likelihood = torch.cat((likelihood_real, observed[:, -1:]), dim=-1)
    return (marginal + epsilon).log() + (likelihood + epsilon).log()


def absorbing_d3pm_terms(atom_logits, atom_target, masked_atom, k_atom,
                         timesteps=1000, keep_probs=None):
    """Return per-atom (KL / first-step reconstruction NLL, auxiliary CE).

    atom_logits includes real elements AND a learned final MASK logit.
    atom_target uses zero-based real-element labels; MASK is not a clean label.
    k_atom is one-based (upstream's zero-based loss index is k_atom-1).
    This is the log_space=True, predict_x0=True, step_size=1 official path;
    it does not assume zero predicted clean MASK probability.
    """
    if atom_logits.ndim != 2 or not atom_logits.is_floating_point():
        raise ValueError("atom_logits must be floating [N,C_real+1], with MASK last")
    n, c = atom_logits.shape
    if (atom_target.shape != (n,) or atom_target.dtype != torch.long
            or torch.any(atom_target < 0) or torch.any(atom_target >= c - 1)):
        raise ValueError("atom_target must be zero-based int64 element labels [N]")
    if masked_atom.shape != (n,) or masked_atom.dtype != torch.bool:
        raise ValueError("masked_atom must be bool [N]")
    if isinstance(timesteps, bool) or not isinstance(timesteps, Integral) or timesteps < 1:
        raise ValueError("timesteps must be a positive integer")
    if (k_atom.shape != (n,) or k_atom.dtype != torch.long
            or torch.any(k_atom < 1) or torch.any(k_atom > timesteps)):
        raise ValueError("k_atom must be int64 diffusion steps [N] in 1..timesteps")
    if any(value.device != atom_logits.device for value in (atom_target, masked_atom, k_atom)):
        raise ValueError("All atom-loss inputs must share a device")
    if c < 2 or n == 0 or not torch.isfinite(atom_logits).all():
        raise ValueError("Atom logits must be finite with at least one real class and MASK")
    # Upstream's D3PM probability calculations use float32, not half precision.
    logits = atom_logits.float() if atom_logits.dtype in (torch.float16, torch.bfloat16) else atom_logits
    if keep_probs is None:
        keep_probs = _standard_mask_keep_probs(timesteps)
    if keep_probs.shape != (timesteps + 1,):
        raise ValueError("keep_probs must include all times 0..timesteps")
    noisy_classes = torch.where(masked_atom, c - 1, atom_target)
    true_clean = F.one_hot(atom_target, c).to(logits)
    true_logits = _absorbing_posterior_logits(
        true_clean, noisy_classes, k_atom, keep_probs, timesteps)
    model_logits = _absorbing_posterior_logits(
        logits.softmax(dim=-1), noisy_classes, k_atom, keep_probs, timesteps)
    kl = kl_divergence(Categorical(logits=true_logits), Categorical(logits=model_logits))
    reconstruction_nll = F.cross_entropy(model_logits, atom_target, reduction="none")
    vb = torch.where(k_atom == 1, reconstruction_nll, kl)
    auxiliary_ce = F.cross_entropy(logits, atom_target, reduction="none")
    return vb, auxiliary_ce


class JointDiffusionLoss(nn.Module):
    """MatterGen loss with the project's dictionary and metric names.

    info['t_graph'] is the same time used for the noisy state and model call.
    D3PM uses floor(t*(K-1)) as its zero-based loss index, exactly upstream.
    Real-element count is configurable: local default 118; upstream default 100.
    Metrics retain autograd, matching SummedFieldLoss; detach when logging.
    """

    def __init__(self, coord_weight=0.1, lattice_weight=1.0, atom_weight=1.0,
                 d3pm_hybrid_lambda=0.01, timesteps=1000, num_atom_types=118,
                 reduce="sum"):
        super().__init__()
        settings = (coord_weight, lattice_weight, atom_weight, d3pm_hybrid_lambda)
        if any(not math.isfinite(x) or x < 0 for x in settings):
            raise ValueError("Loss weights must be finite and nonnegative")
        self.coord_weight = float(coord_weight)
        self.lattice_weight = float(lattice_weight)
        self.atom_weight = float(atom_weight)
        self.d3pm_hybrid_lambda = float(d3pm_hybrid_lambda)
        if (isinstance(num_atom_types, bool) or not isinstance(num_atom_types, Integral)
                or num_atom_types < 1):
            raise ValueError("num_atom_types must be a positive integer")
        if reduce not in ("sum", "mean"):
            raise ValueError("reduce must be 'sum' or 'mean'")
        self.timesteps = int(timesteps)
        self.num_atom_types = int(num_atom_types)
        self.reduce = reduce
        self.register_buffer("_keep_probs", _standard_mask_keep_probs(timesteps), persistent=False)

    def forward(self, prediction, info, batch, num_atoms):
        b, n = num_atoms.numel(), batch.numel()
        if (num_atoms.shape != (b,) or num_atoms.dtype != torch.long or b == 0
                or torch.any(num_atoms <= 0) or batch.dtype != torch.long
                or batch.shape != (n,) or torch.any(batch < 0) or torch.any(batch >= b)
                or not torch.equal(torch.bincount(batch, minlength=b), num_atoms)):
            raise ValueError("batch membership must agree with positive num_atoms [B]")
        coord = prediction["coord_score_times_std"]
        lattice = prediction["lattice_score_times_std"]
        if (coord.shape != (n, 3) or info["coord_target"].shape != coord.shape
                or lattice.shape != (b, 3, 3)
                or info["lattice_target"].shape != lattice.shape):
            raise ValueError("Coordinate and lattice predictions/targets must be [N,3] and [B,3,3]")
        if prediction["atom_logits"].shape != (n, self.num_atom_types + 1):
            raise ValueError(
                f"atom_logits must have shape [{n}, {self.num_atom_types + 1}]: "
                f"{self.num_atom_types} real elements plus a learned final MASK logit. "
                "Update Denoiser.output_head; do not append a constant MASK column.")
        if "t_graph" not in info:
            raise ValueError("info must include t_graph shared by corruption and prediction")
        t = info["t_graph"]
        if (t.shape != (b,) or not t.is_floating_point() or not torch.isfinite(t).all()
                or torch.any(t < 0) or torch.any(t > 1) or t.device != batch.device):
            raise ValueError("t_graph must be floating [B] in [0,1] on the batch device")
        # field_loss.d3pm_loss passes this zero-based index to
        # compute_kl_reverse_process, which conditions on the next noisy state.
        k_graph = (t * (self.timesteps - 1)).long() + 1
        if "k_graph" in info and (info["k_graph"].dtype != torch.long
                or not torch.equal(info["k_graph"], k_graph)):
            raise ValueError("k_graph disagrees with floor(t_graph*(timesteps-1))+1")

        vb, ce = absorbing_d3pm_terms(prediction["atom_logits"], info["atom_target"],
                                     info["masked_atom"], k_graph[batch],
                                     timesteps=self.timesteps, keep_probs=self._keep_probs)
        coord_per_crystal = aggregate_per_crystal(
            (coord - info["coord_target"]).square(), batch, b, reduce=self.reduce)
        lattice_per_crystal = aggregate_per_crystal(
            (lattice - info["lattice_target"]).square(), None, b)
        vb_per_crystal = aggregate_per_crystal(vb, batch, b, reduce=self.reduce)
        ce_per_crystal = aggregate_per_crystal(ce, batch, b, reduce=self.reduce)
        atom_per_crystal = vb_per_crystal + self.d3pm_hybrid_lambda * ce_per_crystal
        total_per_crystal = (self.coord_weight * coord_per_crystal
                             + self.lattice_weight * lattice_per_crystal
                             + self.atom_weight * atom_per_crystal)
        metrics = {
            "loss_coord": coord_per_crystal.mean(),
            "loss_lattice": lattice_per_crystal.mean(),
            "loss_atom": atom_per_crystal.mean(),
            "loss_atom_vb": vb_per_crystal.mean(),
            "loss_atom_ce": ce_per_crystal.mean(),
        }
        return total_per_crystal.mean(), metrics
