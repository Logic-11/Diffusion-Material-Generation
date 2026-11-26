"""Configuration shared by the training entry point and checkpoint helpers."""

import math
from functools import cache
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf


CONFIG_FILE = Path(__file__).resolve().parents[2] / "scripts" / "conf" / "train.yaml"
GRADIENT_CLIPPERS = {
    'value': torch.nn.utils.clip_grad_value_,
    'norm': torch.nn.utils.clip_grad_norm_,
}


@cache
def default_trainer() -> DictConfig:
    """Defaults for helpers called directly, outside a Hydra run."""
    return OmegaConf.load(CONFIG_FILE).trainer


def validate_config(cfg: DictConfig) -> None:
    trainer = cfg.trainer
    devices = trainer.get('devices', 1)
    if type(devices) is not int or devices < 1:
        raise ValueError("trainer.devices must be a positive integer")
    if devices > 1:
        if torch.device(trainer.device).type != 'cuda':
            raise ValueError("trainer.devices greater than one requires a CUDA device")
        for name, batch_size in (
            ('data.batchsize', cfg.data.batchsize),
            ('dataloader.val_batch_size', cfg.dataloader.val_batch_size),
        ):
            if type(batch_size) is not int or batch_size < devices:
                raise ValueError(f"{name} must be an integer at least as large as trainer.devices")
    if type(trainer.max_epochs) is not int or trainer.max_epochs < 0:
        raise ValueError("trainer.max_epochs must be a nonnegative integer")
    max_steps = trainer.max_steps
    if max_steps is not None and (type(max_steps) is not int or max_steps < 0):
        raise ValueError("trainer.max_steps must be a nonnegative integer or null")
    if trainer.gradient_clip_algorithm not in GRADIENT_CLIPPERS:
        raise ValueError("trainer.gradient_clip_algorithm must be 'value' or 'norm'")
    clip_val = trainer.gradient_clip_val
    if clip_val is not None and (not math.isfinite(clip_val) or clip_val < 0):
        raise ValueError("trainer.gradient_clip_val must be finite and nonnegative or null")
    if not 0 < trainer.t_min < 1:
        raise ValueError("trainer.t_min must be between 0 and 1 (exclusive)")

    resume = cfg.resume
    if type(resume.enabled) is not bool:
        raise ValueError("resume.enabled must be a boolean")
    if resume.checkpoint_path is not None and (
        not isinstance(resume.checkpoint_path, str) or not resume.checkpoint_path.strip()
    ):
        raise ValueError("resume.checkpoint_path must be a nonempty path or null")
    interval = resume.save_every_n_steps
    if interval is not None and (type(interval) is not int or interval < 1):
        raise ValueError("resume.save_every_n_steps must be a positive integer or null")

    model = cfg.model
    atom_corruption = model.get('atom_corruption')
    init_checkpoint = cfg.get('init_checkpoint')
    if init_checkpoint is not None:
        if not isinstance(init_checkpoint, str) or not init_checkpoint.strip():
            raise ValueError("init_checkpoint must be a nonempty path or null")
        if atom_corruption is not None:
            raise ValueError("init_checkpoint requires model=csp")
        if resume.enabled:
            raise ValueError("Choose either init_checkpoint or resume.enabled")

    if atom_corruption is not None:
        if model.loss_fn.timesteps != atom_corruption.scheduler.timesteps:
            raise ValueError("The atom scheduler and D3PM loss must use the same number of timesteps")
        atom_types = model.denoiser.atom_types
        if atom_corruption.atom_types != atom_types or model.loss_fn.num_atom_types != atom_types:
            raise ValueError("The denoiser, atom corruption and loss must agree on the number of elements")
    position = model.position_corruption.scheduler
    if not (0 < position.sigma_min <= position.sigma_max and math.isfinite(position.sigma_max)):
        raise ValueError("Position VE requires finite sigmas with 0 < sigma_min <= sigma_max")
