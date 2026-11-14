import torch
import torch.nn as nn
import math

from .diffusion import periodic_noise

def beta_schedule(timesteps:int, s=0.008):
    t = torch.arange(0, timesteps+1, dtype=torch.float32)
    f_t = torch.square(torch.cos((t / timesteps + s) / (1 + s) * math.pi / 2))
    f_t = f_t / f_t[0]
    beta_t = 1 - (f_t[1:] / f_t[:-1])
    beta_t = torch.clip(beta_t, 0.0001, 0.9999)
    betas = torch.cat([torch.zeros_like(beta_t[:1]), beta_t], dim=0)
    return betas

@torch.no_grad()
def sigma_norm(sigmas, num_samples=10000, chunk_size=32):
    """Estimate the mean squared periodic target for each positive sigma."""
    if sigmas.ndim != 1 or sigmas.numel() == 0:
        raise ValueError("sigmas must be a nonempty one-dimensional tensor")
    if not torch.isfinite(sigmas).all() or torch.any(sigmas <= 0):
        raise ValueError("sigmas must be finite and positive")
    if num_samples < 1 or chunk_size < 1:
        raise ValueError("num_samples and chunk_size must be positive")

    noise = torch.randn(num_samples, sigmas.numel(), dtype=sigmas.dtype, device=sigmas.device)
    norms = []
    for start in range(0, sigmas.numel(), chunk_size):
        sigma_chunk = sigmas[start:start + chunk_size]
        displacement = (noise[:, start:start + chunk_size] * sigma_chunk).remainder(1.0)
        target = periodic_noise(displacement, sigma_chunk)
        norms.append(target.square().mean(dim=0))
    return torch.cat(norms, dim=0)


class BetaScheduler(nn.Module):
    def __init__(self, timesteps):
        super().__init__()
        betas = beta_schedule(timesteps)
        alphas = 1 - betas
        alphas_prod = torch.cumprod(alphas, dim=0)

        sigmas_square_t = betas[1:] * (1 - alphas_prod[:-1]) / (1 - alphas_prod[1:])
        sigmas_square = torch.cat([torch.zeros_like(sigmas_square_t[:1]), sigmas_square_t], dim=0)
        sigmas = torch.sqrt(sigmas_square)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_prod', alphas_prod)
        self.register_buffer('sigmas_square', sigmas_square)
        self.register_buffer('sigmas', sigmas)

class ExpScheduler(nn.Module):
    def __init__(self, timesteps, sigma_min=0.005, sigma_max=0.5,
                 norm_samples=10000, norm_chunk_size=32):
        super().__init__()
        if timesteps < 2:
            raise ValueError("timesteps must be at least 2")
        t = torch.arange(1, timesteps+1, dtype=torch.float32)
        sigma_t = sigma_min * torch.pow(sigma_max/sigma_min, (t-1)/(timesteps-1))
        sigmas = torch.cat([torch.zeros_like(sigma_t[:1]), sigma_t], dim=0)
        norms = sigma_norm(sigma_t, num_samples=norm_samples, chunk_size=norm_chunk_size)
        sigmas_norm = torch.cat([torch.ones_like(norms[:1]), norms], dim=0)

        self.register_buffer('sigmas', sigmas)
        self.register_buffer('sigmas_norm', sigmas_norm)


# ---------------------------------------------------
# Schedulers from MatterGen

class MaskScheduler(nn.Module):
    def __init__(self, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        k = torch.arange(timesteps + 1, dtype=torch.float64)
        betas = torch.zeros_like(k)
        betas[1:] = 1.0 / (timesteps - k[1:] + 1.0)
        keep_probs = (timesteps - k) / timesteps
        mask_probs = k / timesteps
        # reverse process
        unmask_probs = torch.zeros_like(k)
        unmask_probs[1:] = 1.0 / k[1:]

        self.register_buffer("betas", betas.float())
        self.register_buffer("keep_probs", keep_probs.float())
        self.register_buffer("mask_probs", mask_probs.float())
        self.register_buffer("unmask_probs", unmask_probs.float())

    def step_from_time(self, t):
        '''t:[0, 1] -> k:{1, ..., K}'''
        return ((self.timesteps - 1) * t).long() + 1

class PositionVEScheduler(nn.Module):
    def __init__(self, 
                 sigma_min=0.01, 
                 sigma_max=5.0):
        super().__init__()
        self.register_buffer("sigma_min", torch.tensor(float(sigma_min)))
        self.register_buffer("sigma_max", torch.tensor(float(sigma_max)))

    def sigma_base(self, t):
        log_ratio = torch.log(self.sigma_max / self.sigma_min)
        return self.sigma_min * torch.exp(t * log_ratio)

    def sigma_std(self, t, num_atoms):
        return self.sigma_base(t) * num_atoms.to(t).pow(-1.0 / 3.0)

    def diffusion_coeff(self, t, num_atoms):
        log_ratio = torch.log(self.sigma_max / self.sigma_min)
        return self.sigma_std(t, num_atoms) * torch.sqrt(2.0 * log_ratio)

class LatticeVPScheduler(nn.Module):
    def __init__(self,
                limit_density,
                beta_min=0.1,
                beta_max=20.0,
                limit_var_scaling_constant=0.25):
        super().__init__()

        if not (
            limit_density > 0
            and beta_min > 0
            and beta_max >= beta_min
            and limit_var_scaling_constant > 0
        ):
            raise ValueError("Invalid lattice schedule parameters")

        self.register_buffer("limit_density", torch.tensor(float(limit_density)))
        self.register_buffer("beta_min", torch.tensor(float(beta_min)))
        self.register_buffer("beta_max", torch.tensor(float(beta_max)))
        self.register_buffer("limit_var_scaling_constant",torch.tensor(float(limit_var_scaling_constant)))

    def beta(self, t):
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def log_mean_coeff(self, t):
        return (-0.5 * self.beta_min * t
                -0.25 * (self.beta_max - self.beta_min) * t.square())

    def mean_coeff(self, t):
        return self.log_mean_coeff(t).exp()

    def limit_mean(self, num_atoms):
        n = num_atoms.to(self.limit_density)
        side = (n / self.limit_density).pow(1.0 / 3.0)
        eye = torch.eye(3, dtype=n.dtype, device=n.device)
        return side[:, None, None] * eye

    def limit_var(self, num_atoms):
        n = num_atoms.to(self.limit_var_scaling_constant)
        return self.limit_var_scaling_constant * n.pow(2.0 / 3.0)

    def std(self, t, num_atoms):
        variance_factor = -torch.expm1(2.0 * self.log_mean_coeff(t))
        return (variance_factor * self.limit_var(num_atoms).to(t)).sqrt()

    def diffusion_coeff(self, t, num_atoms):
        return (self.beta(t) * self.limit_var(num_atoms).to(t)).sqrt()
