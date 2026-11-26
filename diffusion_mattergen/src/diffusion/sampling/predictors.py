"""Reverse diffusion updates for elements, fractional coordinates, and lattices."""

import logging
from collections.abc import Iterable
from numbers import Integral

import torch

from ...backbone.geometry import wrap_frac
from ..d3pm import _absorbing_posterior_logits, _standard_mask_keep_probs

logger = logging.getLogger(__name__)


class AtomTypePredictor:
    def __init__(self, atom_corr, *, allowed_atomic_numbers: Iterable[int] | None = None):
        if atom_corr.mask_type != 0:
            raise ValueError("AtomTypePredictor requires class order [MASK=0, Z=1, ...]")
        self.atom_corr = atom_corr
        # Use the same rounded cumulative schedule as the official D3PM and
        # training loss, rather than the algebraic (K-k)/K approximation.
        self._keep_probs = _standard_mask_keep_probs(atom_corr.scheduler.timesteps)
        self._allowed_mask = None
        if allowed_atomic_numbers is not None:
            numbers = list(allowed_atomic_numbers)
            if not numbers or any(
                isinstance(z, bool) or not isinstance(z, Integral)
                or not 1 <= z <= atom_corr.atom_types for z in numbers
            ):
                raise ValueError("allowed_atomic_numbers must contain supported real elements")
            self._allowed_mask = torch.zeros(atom_corr.atom_types + 1, dtype=torch.bool)
            self._allowed_mask[numbers] = True

    @torch.no_grad()
    def posterior_logits(self, atomic_numbers, atom_logits, t, batch):
        """Return unnormalized reverse logits [M, C], with MASK in column 0.

        atomic_numbers: int64 [M], including MASK=0; atom_logits: float [M, C]
        predicting clean types, with C=atom_corr.atom_types+1; t: float [B] in
        [0,1]; batch: int64 [M]. All atom tensors share a device. Negative
        infinity may exclude classes, but each row needs a finite allowed logit.

        This is one discrete reverse step k -> k-1, where
        k=floor(t*(K-1))+1. At t=0 this selects the last reverse step k=1.
        An element whitelist constrains the predicted clean classes. The
        absorbing-state posterior can still retain MASK at intermediate steps.

        For row-stochastic Q_k and clean prediction pi=softmax(atom_logits):
        logits[j] = log((pi @ Q_bar_{k-1})[j] + 1e-20)
                    + log(Q_k[j, current_type] + 1e-20).
        The training helper evaluates these terms without dense C-by-C matrices.
        Half-precision logits are promoted to float32 before probabilities.
        """
        if (atom_logits.ndim != 2 or not atom_logits.is_floating_point()
                or atom_logits.shape[0] == 0
                or atom_logits.shape[1] != self.atom_corr.atom_types + 1):
            raise ValueError("atom_logits must be floating [M, atom_types+1], with MASK first")

        logits = (atom_logits.float() if atom_logits.dtype in (torch.float16, torch.bfloat16)
                  else atom_logits)
        if self._allowed_mask is not None:
            allowed_mask = self._allowed_mask.to(device=logits.device)
            self._allowed_mask = allowed_mask
            logits = logits.masked_fill(~allowed_mask, -torch.inf)
        if (torch.isnan(atom_logits).any() or torch.isposinf(atom_logits).any()
                or not torch.isfinite(logits).any(dim=-1).all()):
            raise ValueError("atom_logits need a finite allowed entry per row and no NaN/+inf")
        num_atoms, num_classes = atom_logits.shape
        if (atomic_numbers.shape != (num_atoms,) or atomic_numbers.dtype != torch.long
                or atomic_numbers.device != atom_logits.device
                or torch.any(atomic_numbers < 0) or torch.any(atomic_numbers >= num_classes)):
            raise ValueError("atomic_numbers must be int64 [M] in 0..atom_types on the logits device")

        # Preserve the time tensor's precision so categorical step boundaries
        # do not depend on the neural network's logit dtype.
        t = torch.as_tensor(t, device=atom_logits.device)
        if (t.ndim != 1 or t.numel() == 0 or not t.is_floating_point()
                or not torch.isfinite(t).all() or torch.any(t < 0) or torch.any(t > 1)):
            raise ValueError("t must be floating [B] with finite values in [0,1]")
        if (batch.shape != (num_atoms,) or batch.dtype != torch.long
                or batch.device != atom_logits.device
                or torch.any(batch < 0) or torch.any(batch >= t.numel())):
            raise ValueError("batch must be int64 [M] indexing the graph times on the logits device")

        k_atom = self.atom_corr.scheduler.step_from_time(t)[batch]
        return _absorbing_posterior_logits(
            logits.softmax(dim=-1), atomic_numbers, k_atom, self._keep_probs,
            self.atom_corr.scheduler.timesteps)

    @torch.no_grad()
    def atom_type_predict(self, atomic_numbers, atom_logits, t, batch):
        """Return (categorical sample, posterior mode), both int64 [M].

        Pass prediction['atom_logits'] from the shared current-state model call.
        The second output is an argmax class, not an average atomic number.
        Class indices already equal atomic numbers in this project's MASK-first
        representation, so neither output needs an offset or permutation.
        """
        logits = self.posterior_logits(atomic_numbers, atom_logits, t, batch)
        posterior = torch.distributions.Categorical(logits=logits)
        return posterior.sample(), posterior.probs.argmax(dim=-1)


class PositionPredictor:
    def __init__(self, pos_corr):
        self.pos_corr = pos_corr

    @torch.no_grad()
    def position_predict(self, coord, score, t, s, num_atoms, batch, noise=None):
        """Wrapped VE ancestral step from time t to s; return (sample, mean).

        coord/score/noise: [M, 3]; t/s/num_atoms: [B]; batch: int64 [M].
        score is the fractional score from scores_from_prediction, already
        divided by the marginal standard deviation. 

        At s=0, sigma_s is set to zero BEFORE computing either coefficient.
        Both returned tensors are wrapped into [0, 1).
        """
        if coord.ndim != 2 or coord.shape[1] != 3 or not coord.is_floating_point():
            raise ValueError("coord must be a floating tensor of shape [M, 3]")
        if (score.shape != coord.shape or score.dtype != coord.dtype
                or score.device != coord.device):
            raise ValueError("score must match coord's shape, dtype and device")
        if not torch.isfinite(coord).all() or not torch.isfinite(score).all():
            raise ValueError("coord and score must be finite")
        if (num_atoms.ndim != 1 or num_atoms.numel() == 0
                or num_atoms.dtype != torch.long or num_atoms.device != coord.device
                or torch.any(num_atoms <= 0)):
            raise ValueError("num_atoms must be positive int64 [B] on the coordinate device")
        if (batch.shape != (coord.shape[0],) or batch.dtype != torch.long
                or batch.device != coord.device or torch.any(batch < 0)
                or torch.any(batch >= num_atoms.numel())
                or not torch.equal(torch.bincount(batch, minlength=num_atoms.numel()), num_atoms)):
            raise ValueError("batch must assign exactly num_atoms atoms to each crystal")

        t = torch.as_tensor(t, dtype=coord.dtype, device=coord.device)
        s = torch.as_tensor(s, dtype=coord.dtype, device=coord.device)
        if (t.shape != num_atoms.shape or s.shape != num_atoms.shape
                or not torch.isfinite(t).all() or not torch.isfinite(s).all()
                or torch.any(t > 1) or torch.any(s < 0) or torch.any(s >= t)):
            raise ValueError("t and s must have shape [B] and satisfy 0 <= s < t <= 1")

        sigma_t = self.pos_corr.scheduler.sigma_std(t, num_atoms)
        sigma_s = self.pos_corr.scheduler.sigma_std(s, num_atoms)
        sigma_s = torch.where(s == 0, torch.zeros_like(sigma_s), sigma_s)
        delta = sigma_t.square() - sigma_s.square()
        if (not torch.isfinite(sigma_t).all() or not torch.isfinite(sigma_s).all()
                or not torch.isfinite(delta).all() or torch.any(sigma_t <= 0)
                or torch.any(sigma_s < 0) or torch.any(delta < 0)):
            raise ValueError("Position scheduler must give finite scales with 0 <= sigma_s <= sigma_t and sigma_t > 0")
        reverse_std = delta.sqrt() * (sigma_s / sigma_t)

        if noise is None:
            noise = torch.randn_like(coord)
        elif (noise.shape != coord.shape or noise.dtype != coord.dtype
                or noise.device != coord.device or not torch.isfinite(noise).all()):
            raise ValueError("noise must be finite and match coord's shape, dtype and device")

        # Broadcast crystal coefficients to atoms, retaining independent XYZ noise.
        mean = coord + delta[batch, None] * score
        sample = mean + reverse_std[batch, None] * noise
        return wrap_frac(sample), wrap_frac(mean)


class LatticePredictor:
    def __init__(self, lat_corr):
        self.lat_corr = lat_corr

    @torch.no_grad()
    def lattice_predict(self, lattice, score, t, s, num_atoms, noise=None):
        """Return (sample, mean) for a shifted VP ancestral step.

        lattice and score have shape [B, 3, 3]. The score is already divided by
        the marginal standard deviation. t, s and num_atoms have shape [B],
        with 0 <= s < t <= 1. Optional noise has shape [B, 1, 1] and supplies
        one standard normal draw per crystal, as in MatterGen.
        """
        if (lattice.ndim != 3 or lattice.shape[1:] != (3, 3)
                or lattice.shape[0] == 0 or not lattice.is_floating_point()):
            raise ValueError("lattice must be a nonempty floating tensor of shape [B, 3, 3]")
        if (score.shape != lattice.shape or score.dtype != lattice.dtype
                or score.device != lattice.device):
            raise ValueError("score must match lattice's shape, dtype and device")
        if not torch.isfinite(lattice).all() or not torch.isfinite(score).all():
            raise ValueError("lattice and score must be finite")
        if (num_atoms.shape != (lattice.shape[0],) or num_atoms.dtype != torch.long
                or num_atoms.device != lattice.device or torch.any(num_atoms <= 0)):
            raise ValueError("num_atoms must be positive int64 [B] on the lattice device")

        t = torch.as_tensor(t, dtype=lattice.dtype, device=lattice.device)
        s = torch.as_tensor(s, dtype=lattice.dtype, device=lattice.device)
        if (t.shape != num_atoms.shape or s.shape != num_atoms.shape
                or not torch.isfinite(t).all() or not torch.isfinite(s).all()
                or torch.any(t > 1) or torch.any(s < 0) or torch.any(s >= t)):
            raise ValueError("t and s must have shape [B] and satisfy 0 <= s < t <= 1")

        scheduler = self.lat_corr.scheduler
        alpha_t = scheduler.mean_coeff(t)
        alpha_s = scheduler.mean_coeff(s)
        sigma_t = scheduler.std(t, num_atoms)
        sigma_s = scheduler.std(s, num_atoms)
        is_time_zero = s == 0
        sigma_s = torch.where(is_time_zero, torch.zeros_like(sigma_s), sigma_s)
        limit_mean = scheduler.limit_mean(num_atoms).to(lattice)
        if (not torch.isfinite(alpha_t).all() or not torch.isfinite(alpha_s).all()
                or not torch.isfinite(sigma_t).all() or not torch.isfinite(sigma_s).all()
                or not torch.isfinite(limit_mean).all() or torch.any(alpha_t < 0)
                or torch.any(alpha_s <= 0) or torch.any(sigma_t <= 0)
                or torch.any(sigma_s < 0)):
            raise ValueError("Lattice scheduler returned invalid mean coefficients or noise scales")

        alpha_ratio = alpha_t / alpha_s
        forward_var = sigma_t.square() - sigma_s.square() * alpha_ratio.square()
        if (not torch.isfinite(alpha_ratio).all() or not torch.isfinite(forward_var).all()
                or torch.any(forward_var < 0)):
            raise ValueError("Lattice scheduler must give a finite nonnegative transition variance")
        reverse_std = forward_var.sqrt() * (sigma_s / sigma_t)
        reverse_std = torch.where(is_time_zero, torch.zeros_like(reverse_std), reverse_std)

        # Match MatterGen: compute the transition variance BEFORE clipping r.
        if torch.any(alpha_ratio < 0.001):
            logger.warning("Clipping lattice alpha_t / alpha_s to 0.001; consider smaller time steps.")
            alpha_ratio = alpha_ratio.clamp(min=0.001, max=1.0)
        x_coeff = alpha_ratio.reciprocal()[:, None, None]
        score_coeff = (forward_var / alpha_ratio)[:, None, None]

        if noise is None:
            noise = torch.randn_like(x_coeff)
        elif (noise.shape != x_coeff.shape or noise.dtype != lattice.dtype
                or noise.device != lattice.device or not torch.isfinite(noise).all()):
            raise ValueError("noise must be finite with shape [B, 1, 1] and match lattice's dtype and device")
        symmetric_noise = self.lat_corr.symmetric_noise(noise)
        mean = x_coeff * lattice + score_coeff * score + (1.0 - x_coeff) * limit_mean
        sample = mean + reverse_std[:, None, None] * symmetric_noise
        return sample, mean
