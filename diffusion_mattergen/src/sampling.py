import json
import logging
from collections.abc import Mapping
from pathlib import Path

import math
import torch

from .geometry import wrap_frac, polar_decomposition
from .corruption import AtomTypeCorruption, LatticeCorruption, PositionCorruption
from .denoiser import Denoiser
from .diffusion import JointDiffusion
from .loss import JointDiffusionLoss, _absorbing_posterior_logits, _standard_mask_keep_probs
from .scheduler import LatticeVPScheduler, MaskScheduler, PositionVEScheduler

logger = logging.getLogger(__name__)

def sample_n(statistic: str | Path, batch_size: int = 1,
             device: str | torch.device = "cuda") -> torch.Tensor:
    """Sample num_atoms [batch_size] from saved training-set probabilities."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    with open(statistic, 'r', encoding='utf-8') as f:
        data = json.load(f)
    n_dist = data.get('n_dist')
    if not isinstance(n_dist, dict) or not n_dist:
        raise ValueError("statistic must contain a nonempty n_dist dictionary")

    num_atoms = torch.tensor([int(n) for n in n_dist], dtype=torch.long)
    probabilities = torch.tensor(list(n_dist.values()), dtype=torch.float64)
    if torch.any(num_atoms <= 0):
        raise ValueError("n_dist atom counts must be positive integers")
    total = probabilities.sum()
    if (not torch.isfinite(probabilities).all() or torch.any(probabilities < 0)
            or not torch.isfinite(total) or total <= 0):
        raise ValueError("n_dist probabilities must be finite, nonnegative and have positive total")

    probabilities = probabilities / total
    indices = torch.multinomial(probabilities, batch_size, replacement=True)
    return num_atoms[indices].to(device=device)

def sample_prior(num_atoms: torch.tensor, limit_density: float, sigma_max: float,
                 lat_cor: LatticeCorruption, 
                 device: str | torch.device = 'cuda'):
    batch_size = num_atoms.shape[0]
    total_num = torch.sum(num_atoms).item()
    batch = torch.repeat_interleave(torch.arange(batch_size, device=device), num_atoms)
    masked_atoms = num_atoms.new_zeros((total_num), device=device)
    gaussian = torch.randn((total_num, 3), device=device)
    scale = (sigma_max * num_atoms[batch].pow(-1.0/3.0)).unsqueeze(-1)
    prior_coord = wrap_frac(scale * gaussian)
    prior_lattice = lat_cor.prior_sample(num_atoms)
    return masked_atoms, prior_coord, prior_lattice

class LangevinCorrector:
    def __init__(self, rx=0.4, rl=0.2, ita_max=1e6):
        self.rx = rx
        self.rl = rl
        self.ita_max = torch.tensor(ita_max)
        
    def coord_correct(self, coord, score, batch, batch_size, noise=None):
        # Sum squared components per crystal before taking its L2 norm.
        score_graph = score.new_zeros((batch_size,))
        score_graph.scatter_add_(dim=0, index=batch, src=score.square().sum(dim=-1))
        score_norm = score_graph.sqrt().mean()
        if noise is None:
            noise = torch.randn_like(score)
        if noise.shape != score.shape:
            raise RuntimeError(f"Shape of noise {noise.size()} must be the same as score {score.size()}.")
        noise_graph = score.new_zeros((batch_size,))
        noise_graph.scatter_add_(dim=0, index=batch, src=noise.square().sum(dim=-1))
        noise_norm = noise_graph.sqrt().mean()

        max_step = self.ita_max.to(score)
        if score_norm == 0:
            ita_x = max_step
        else:
            ita_x = torch.minimum(2 * torch.square(self.rx * noise_norm / score_norm), max_step)
        corrected = coord + ita_x * score + torch.sqrt(2*ita_x) * noise
        return wrap_frac(corrected)

    def lattice_correct(self, lattice, score, t, s, 
                        corruption, noise=None):
        '''s is the next sample timestep of t
           corruption is LatticeCorruption'''
        score_graph = torch.linalg.norm(score, dim=(1, 2), ord='fro')
        score_norm = torch.mean(score_graph)
        if noise is None:
            noise = torch.randn_like(score)
        if noise.shape != score.shape:
            raise RuntimeError(f"Shape of noise {noise.size()} must be the same as score {score.size()}.")
        sym_noise = corruption.symmetric_noise(noise)
        noise_graph = torch.linalg.norm(sym_noise, dim=(1, 2), ord='fro')
        noise_norm = torch.mean(noise_graph)
        coeff_t = corruption.scheduler.mean_coeff(t)
        coeff_s = corruption.scheduler.mean_coeff(s)
        gamma = torch.square(coeff_t / coeff_s)[:, None, None]

        max_step = self.ita_max.to(score)
        if score_norm == 0:
            ita_l = max_step
        else:
            ita_l = torch.minimum(2 * torch.square(self.rl * noise_norm / score_norm) * gamma, max_step)
        corrected = lattice + ita_l * score + torch.sqrt(2*ita_l) * sym_noise
        return polar_decomposition(corrected)

class AtomTypePredictor:
    def __init__(self, atom_corr):
        if atom_corr.mask_type != 0:
            raise ValueError("AtomTypePredictor requires class order [MASK=0, Z=1, ...]")
        self.atom_corr = atom_corr
        # Use the same rounded cumulative schedule as the official D3PM and
        # training loss, rather than the algebraic (K-k)/K approximation.
        self._keep_probs = _standard_mask_keep_probs(atom_corr.scheduler.timesteps)

    @torch.no_grad()
    def posterior_logits(self, atomic_numbers, atom_logits, t, batch):
        """Return unnormalized reverse logits [M, C], with MASK in column 0.

        atomic_numbers: int64 [M], including MASK=0; atom_logits: float [M, C]
        predicting clean types, with C=atom_corr.atom_types+1; t: float [B] in
        [0,1]; batch: int64 [M]. All atom tensors share a device. Negative
        infinity may exclude classes, but each logit row needs a finite entry.

        This is one discrete reverse step k -> k-1, where
        k=floor(t*(K-1))+1. In particular t=0 selects the LAST reverse step
        k=1, not an identity update. Like MatterGen's default predictor, this
        method does not skip discrete steps or force the final MASK mass to 0.

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
        if (torch.isnan(atom_logits).any() or torch.isposinf(atom_logits).any()
                or not torch.isfinite(atom_logits).any(dim=-1).all()):
            raise ValueError("atom_logits may contain -inf, but each row needs a finite entry and no NaN/+inf")
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

        logits = (atom_logits.float() if atom_logits.dtype in (torch.float16, torch.bfloat16)
                  else atom_logits)
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


class Sampler:
    def __init__(self, atom_pred, pos_pred, lat_pred, corrector):
        self.atom_pred = atom_pred
        self.position_pred = pos_pred
        self.lattice_pred = lat_pred
        self.corrector = corrector

    @torch.no_grad()
    def sample_t(self, model, noisy, t, s):
        """Take one joint reverse step from t to s, with one time per crystal.

        Return (sample_state, mean_state). For atomic numbers the second state
        contains the posterior mode. Both states share the input batch metadata.
        """
        num_atoms, batch = noisy["num_atoms"], noisy["batch"]
        lattice = noisy["lattice"]
        t = torch.as_tensor(t, device=lattice.device)
        s = torch.as_tensor(s, device=lattice.device)
        if not t.is_floating_point():
            t = t.to(lattice.dtype)
        if (t.shape != num_atoms.shape or s.shape != num_atoms.shape
                or not torch.all((0 <= s) & (s < t) & (t <= 1))):
            raise ValueError("t and s must have shape [B] and satisfy 0 <= s < t <= 1")
        step_from_time = self.atom_pred.atom_corr.scheduler.step_from_time
        # s=0 is the clean endpoint; step_from_time itself numbers noisy levels from 1.
        step_s = torch.where(s == 0, 0, step_from_time(s))
        if not torch.all(step_from_time(t) == step_s + 1):
            raise ValueError("t and s must span exactly one reverse atom diffusion step")

        continuous_t, continuous_s = t.to(lattice), s.to(lattice)
        scores = self._predict_scores(model, noisy, continuous_t)
        corrected_coords = self.corrector.coord_correct(
            noisy["frac_coords"], scores["coord_score"], batch, num_atoms.numel())
        corrected_lattice = self.corrector.lattice_correct(
            lattice, scores["lattice_score"], continuous_t, continuous_s, self.lattice_pred.lat_corr)
        corrected = {**noisy, "frac_coords": corrected_coords, "lattice": corrected_lattice}

        # All three predictors use the same corrected geometry and new prediction at t.
        scores = self._predict_scores(model, corrected, continuous_t)
        sample_state, mean_state = corrected.copy(), corrected.copy()
        sample_state["atomic_numbers"], mean_state["atomic_numbers"] = self.atom_pred.atom_type_predict(
            corrected["atomic_numbers"], scores["atom_logits"], t, batch)
        sample_state["frac_coords"], mean_state["frac_coords"] = self.position_pred.position_predict(
            corrected_coords, scores["coord_score"], continuous_t, continuous_s, num_atoms, batch)
        sample_state["lattice"], mean_state["lattice"] = self.lattice_pred.lattice_predict(
            corrected_lattice, scores["lattice_score"], continuous_t, continuous_s, num_atoms)
        return sample_state, mean_state

    @torch.no_grad()
    def sample(self, model, initial_state):
        """Denoise a batch of prior crystals from t=1 to 0 and return (sample, mean).

        Each step continues from the sampled state. The final mean contains
        the continuous predictor means and the most probable atom types.
        Use ``to_structures(mean_state)`` to build the final crystals.
        """
        steps = model.timesteps
        if steps < 1 or steps != self.atom_pred.atom_corr.scheduler.timesteps:
            raise ValueError("model and atom predictor must share a positive number of diffusion steps")
        batch_shape = initial_state["num_atoms"].shape
        # Keep the discrete time grid precise. Continuous updates use the lattice dtype.
        times = torch.arange(steps, -1, -1, dtype=torch.float64,
                             device=initial_state["lattice"].device) / steps
        state = initial_state
        was_training = isinstance(model, torch.nn.Module) and model.training
        if was_training:
            model.eval()
        try:
            for index in range(steps):
                t = times[index].expand(batch_shape)
                s = times[index + 1].expand(batch_shape)
                state, mean_state = self.sample_t(model, state, t, s)
        finally:
            if was_training:
                model.train()
        return state, mean_state

    @staticmethod
    def to_structures(state: Mapping[str, torch.Tensor]):
        """Return a list of pymatgen Structures in batch order from one final state.

        Pass mean_state from sample for posterior-mode elements, or sample_state
        for sampled elements. Lattice vectors are rows in angstroms and positions
        are fractional. Site order within each crystal follows the input.
        """
        from pymatgen.core import Structure

        numbers = state["atomic_numbers"].detach().cpu()
        batch = state["batch"].detach().cpu()
        counts = state["num_atoms"].detach().cpu()
        coords = state["frac_coords"].detach().to(device="cpu", dtype=torch.float64)
        lattices = state["lattice"].detach().to(device="cpu", dtype=torch.float64)

        if any(field.dtype != torch.long for field in (numbers, batch, counts)):
            raise ValueError("atomic_numbers, batch and num_atoms must be int64")
        if (counts.ndim != 1 or counts.numel() == 0 or numbers.ndim != 1
                or batch.shape != numbers.shape or coords.shape != (numbers.numel(), 3)
                or lattices.shape != (counts.numel(), 3, 3)):
            raise ValueError("Expected one lattice per crystal and one fractional coordinate per atom")
        if (torch.any(counts <= 0) or torch.any(batch < 0) or torch.any(batch >= counts.numel())
                or not torch.equal(torch.bincount(batch, minlength=counts.numel()), counts)):
            raise ValueError("batch membership must agree with positive num_atoms")

        structures = []
        for index, lattice in enumerate(lattices):
            sites = batch == index
            species, frac = numbers[sites], coords[sites]
            invalid = species[(species < 1) | (species > 118)]
            if invalid.numel():
                raise ValueError(f"Crystal {index} contains atomic number {invalid[0].item()}. "
                                 "Expected a real element in 1..118 (0 is MASK).")
            if not torch.isfinite(frac).all() or not torch.isfinite(lattice).all():
                raise ValueError(f"Crystal {index} has non-finite coordinates or lattice")
            if torch.linalg.matrix_rank(lattice) < 3:
                raise ValueError(f"Crystal {index} has a singular lattice")
            structures.append(Structure(lattice.numpy(), species.tolist(), frac.numpy(),
                                        coords_are_cartesian=False, to_unit_cell=True))
        return structures

    @torch.no_grad()
    def _predict_scores(self, model, state, t):
        predict = getattr(model, "predict", None)
        scores_from_prediction = getattr(model, "scores_from_prediction", None)
        if not callable(predict) or not callable(scores_from_prediction):
            raise TypeError("model must provide callable predict and scores_from_prediction methods")
        for field in ("atomic_numbers", "frac_coords", "lattice", "num_atoms", "batch"):
            if field not in state:
                raise KeyError(f"state is missing required field {field!r}")
            if not torch.is_tensor(state[field]):
                raise TypeError(f"state[{field!r}] must be a torch.Tensor")

        prediction = predict(state, t)
        return scores_from_prediction(prediction, state, t)
