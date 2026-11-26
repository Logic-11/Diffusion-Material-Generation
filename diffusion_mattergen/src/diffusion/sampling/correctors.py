"""Langevin updates for fractional coordinates and lattices."""

import torch

from ...backbone.geometry import polar_decomposition, wrap_frac


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
