import torch
import torch.nn as nn
from .geometry import wrap_frac

class AtomTypeCorruption(nn.Module):
    def __init__(self, scheduler, atom_types=118, mask_type=0):
        super().__init__()
        self.scheduler = scheduler
        self.atom_types = atom_types
        self.mask_type = mask_type

    def marginal_prob(self, clean_type, t_graph, atom_batch):
        '''clean_type: (N,)   t_graph: (B,)   atom_batch:(N,)
           Return: 
           type_prob: (N, atom_types+1), only [:, mask_type]
           or [:, clean_type] will have non-zero values'''
        step_graph = self.scheduler.step_from_time(t_graph)
        mask_prob = self.scheduler.mask_probs[step_graph][atom_batch]
        type_prob = mask_prob.new_zeros((clean_type.shape[0], self.atom_types+1))   # (N, atom_types+1)
        type_prob[:, self.mask_type] = mask_prob
        type_prob.scatter_(dim=1, index=clean_type[:,None], src=(1.0-mask_prob[:, None]))
        return type_prob

    def q_sample(self, clean_type, t_graph, atom_batch, uniform=None):
        step_graph = self.scheduler.step_from_time(t_graph)
        mask_prob = self.scheduler.mask_probs[step_graph][atom_batch]
        if uniform is None:
            uniform = torch.rand_like(mask_prob)
        masked_atom = uniform < mask_prob
        noisy_type = clean_type.masked_fill(masked_atom, self.mask_type)
        return noisy_type, masked_atom

class PositionCorruption(nn.Module):
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler

    def marginal_prob(self, clean_position, t_graph, num_atoms, atom_batch):
        '''Return: mean and std-variance of q(Xt|X0, n)'''
        mean = clean_position
        std = self.scheduler.sigma_std(t_graph, num_atoms)[atom_batch, None]
        return mean, std

    def q_sample(self, clean_position, t_graph, num_atoms, atom_batch, noise=None):
        mean, std = self.marginal_prob(clean_position, t_graph, num_atoms, atom_batch)
        if noise is None:
            noise = torch.randn_like(clean_position)
        noisy_position = wrap_frac(mean + std * noise)
        return noisy_position

    @torch.no_grad()
    def score_target(
        self,
        clean_position,
        noisy_position,
        t_graph,
        num_atoms,
        atom_batch,
        max_offset_integer=3,
    ):
        """Return std times the wrapped-normal score, with shape [N, 3]."""
        mean, std = self.marginal_prob(clean_position, t_graph, num_atoms, atom_batch)

        # Fractional coordinates have identity period vectors.
        axis = torch.arange(
            -max_offset_integer,
            max_offset_integer + 1,
            dtype=clean_position.dtype,
            device=clean_position.device)
        grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="xy"), dim=-1)
        offsets = grid.reshape(-1, 3)  # [343, 3] when M=3

        displacement = (noisy_position - mean)[:, None, :] + offsets[None]
        variance = std.square()  # [N, 1]
        log_weights = -displacement.square().sum(dim=-1) / (2.0 * variance)
        weights = torch.softmax(log_weights, dim=-1)

        score = -(weights[..., None] * displacement).sum(dim=1) / variance
        return score * std

class LatticeCorruption(nn.Module):
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler

    def marginal_prob(self, clean_lattice, t_graph, num_atoms):
        """Return the mean [B, 3, 3] and standard deviation [B, 1, 1]."""
        mean_coeff = self.scheduler.mean_coeff(t_graph)[:, None, None]
        limit_mean = self.scheduler.limit_mean(num_atoms)
        mean_t = mean_coeff * clean_lattice + (1.0 - mean_coeff) * limit_mean
        std_t = self.scheduler.std(t_graph, num_atoms)[:, None, None]
        return mean_t, std_t

    @staticmethod
    def symmetric_noise(noise):
        """Symmetrize IID noise [B, 3, 3], preserving each entry's variance."""
        eye = torch.eye(3, dtype=noise.dtype, device=noise.device)
        symmetric = (noise + noise.transpose(-1, -2)) / (2.0 ** 0.5)
        return (1.0 - eye) * symmetric + eye * noise

    def q_sample(self, clean_lattice, t_graph, num_atoms, noise=None):
        """Sample the symmetric marginal; optional noise is raw IID Gaussian."""
        if noise is None:
            noise = torch.randn_like(clean_lattice)
        sym_noise = self.symmetric_noise(noise)
        mean, std = self.marginal_prob(clean_lattice, t_graph, num_atoms)
        noisy_lattice = mean + std * sym_noise
        return noisy_lattice

    def prior_sample(self, num_atoms, noise=None):
        lim_mean = self.scheduler.limit_mean(num_atoms)
        lim_var = self.scheduler.limit_var(num_atoms)[:, None, None]
        if noise is None:
            noise = torch.randn_like(lim_mean)
        sym_noise = self.symmetric_noise(noise)
        prior_lattice = lim_mean + torch.sqrt(lim_var) * sym_noise
        return prior_lattice

    @torch.no_grad()
    def score_target(self, clean_lattice, noisy_lattice, t_graph, num_atoms):
        """Return score_times_std = -epsilon_symmetric, only for t > 0."""
        mean, std = self.marginal_prob(clean_lattice, t_graph, num_atoms)
        if not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("Lattice score_target requires a finite positive std (t > 0)")
        return -(noisy_lattice - mean) / std
