"""Save and restore the training state and the model metadata used for sampling."""

import logging
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from pickle import UnpicklingError

import numpy as np
import torch
from hydra.utils import to_absolute_path

from src.diffusion.checkpoint import checkpoint_task, read_model_checkpoint
from src.diffusion.crystal_diffusion import CrystalDiffusion
from src.diffusion.joint_diffusion import JointDiffusion

from .distributed import TrainingProcess


logger = logging.getLogger(__name__)


@dataclass
class EpochProgress:
    loader_rng_state: torch.Tensor
    batches_completed: int = 0
    loss_sum: float = 0.0
    num_crystals: int = 0

    @property
    def mean_loss(self) -> float:
        return self.loss_sum / self.num_crystals


@dataclass
class TrainingState:
    start_epoch: int = 0
    global_step: int = 0
    best_val_loss: float = math.inf
    best_checkpoint_path: str | None = None
    epoch_progress: EpochProgress | None = None


def load_joint_denoiser_weights(model: CrystalDiffusion, checkpoint_path: str | Path) -> None:
    """Initialize the denoiser from a joint checkpoint before creating the optimizer."""
    checkpoint = read_model_checkpoint(checkpoint_path)
    if checkpoint_task(checkpoint) != 'joint':
        raise ValueError("init_checkpoint must point to a joint diffusion checkpoint")
    prefix = 'denoiser.'
    weights = {
        key.removeprefix(prefix): value
        for key, value in checkpoint['model_state_dict'].items()
        if key.startswith(prefix)
    }
    model.denoiser.load_state_dict(weights, strict=True)


def _build_checkpoint(epoch, epoch_loss, val_loss, model: CrystalDiffusion, optimizer, lr_scheduler,
                      *, configs, global_step, partial_epoch, t_min, training_state):
    position = model.position_corruption.scheduler
    lattice = model.lattice_corruption.scheduler
    loss_fn = model.loss_fn
    loss = {
        name: getattr(loss_fn, name)
        for name in ('coord_weight', 'lattice_weight', 'reduce')
    }
    diffusion_config = {
        'atom_class_order': getattr(model.denoiser, 'atom_class_order', 'mask_first'),
        'time_sampling': f'uniform time in [{t_min:g}, 1)',
        't_min': t_min,
        'position': {
            name: getattr(position, name).item()
            for name in ('sigma_min', 'sigma_max')
        },
        'lattice': {
            name: getattr(lattice, name).item()
            for name in ('limit_density', 'beta_min', 'beta_max', 'limit_var_scaling_constant')
        },
        'loss': loss,
        'graph': {
            'cutoff': model.radial_basis.cutoff,
            'max_neighbor': model.max_neighbor,
            'max_cell_images_per_dim': model.max_cell_images_per_dim,
            'max_offset_integer': model.max_offset_integer,
        },
        'output_parameterization': {
            'coordinates': 'fractional score_times_std',
            'lattice': 'score_times_std',
        },
    }
    if isinstance(model, JointDiffusion):
        loss.update({
            name: getattr(loss_fn, name)
            for name in ('atom_weight', 'd3pm_hybrid_lambda', 'timesteps', 'num_atom_types')
        })
        diffusion_config.update({
            'timesteps': model.timesteps,
            'input_mask_type': model.atom_corruption.mask_type,
            'clean_mask_probability': 'learned in the first column of the joint diffusion output',
        })
        diffusion_config['output_parameterization']['atoms'] = 'clean-class logits; MASK followed by real elements'

    return {
        'task': model.task,
        'epoch': epoch,
        'epoch_loss': float(epoch_loss),
        'val_loss': None if val_loss is None else float(val_loss),
        'partial_epoch': partial_epoch,
        'global_step': global_step,
        'training_state': training_state,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': (
            None if lr_scheduler is None else lr_scheduler.state_dict()
        ),
        'configs': {} if configs is None else dict(configs),
        'model_config': {
            'hidden_dim': model.denoiser.hidden_dim,
            'num_radial': model.denoiser.num_radial,
            'num_layers': model.denoiser.num_layers,
            'atom_types': model.atom_types,
            'num_spherical': model.denoiser.num_spherical,
            'use_triplets': model.denoiser.use_triplets,
        },
        'diffusion_config': diffusion_config,
    }


def _write_checkpoint(checkpoint, output_dir, checkpoint_type):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f'{checkpoint_type}.pt')
    # Close the temporary file before replacing it (required on Windows).
    with tempfile.NamedTemporaryFile(dir=output_dir, prefix='.checkpoint-',
                                     suffix='.tmp', delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, checkpoint_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return checkpoint_path


def save_training_checkpoint(
    epoch, progress: EpochProgress, state: TrainingState, *, model: CrystalDiffusion,
    optimizer, lr_scheduler, output_dir: Path, configs, loader_signature, device,
    t_min, partial_epoch, val_loss=None, epoch_loss=None, process: TrainingProcess | None = None,
) -> str | None:
    if process is None:
        process = TrainingProcess(device)
    if epoch_loss is None:
        loss_sum, num_crystals = process.sum_loss(progress.loss_sum, progress.num_crystals)
        epoch_loss = loss_sum / num_crystals

    epoch_progress = None
    if partial_epoch:
        epoch_progress = {
            'batches_completed': progress.batches_completed,
            'loss_sum': progress.loss_sum,
            'num_crystals': progress.num_crystals,
            'loader_rng_state': progress.loader_rng_state,
            'loader_signature': loader_signature,
        }

    local_state = {
        'epoch_progress': epoch_progress,
        'rng_state': _capture_rng_state(device),
    }
    rank_states = process.gather_rank_states(local_state)
    if not process.is_primary:
        return None

    improved = val_loss is not None and val_loss < state.best_val_loss
    best_loss = val_loss if improved else state.best_val_loss
    best_path = str(output_dir / 'best.pt') if improved else state.best_checkpoint_path
    training_state = {
        'epoch_progress': epoch_progress,
        'best_val_loss': best_loss,
        'best_checkpoint_path': best_path,
        'rng_state': local_state['rng_state'],
    }
    if rank_states is not None:
        training_state['rank_states'] = rank_states

    checkpoint = _build_checkpoint(
        epoch, epoch_loss, val_loss, model, optimizer, lr_scheduler,
        configs=configs, global_step=state.global_step, partial_epoch=partial_epoch, t_min=t_min,
        training_state=training_state,
    )
    if improved:
        _write_checkpoint(checkpoint, output_dir, 'best')
        state.best_val_loss = best_loss
        state.best_checkpoint_path = best_path
    return _write_checkpoint(checkpoint, output_dir, 'latest')


def read_resume_checkpoint(resume, output_dir):
    if not resume.enabled:
        return None, None
    path = (output_dir / 'latest.pt' if resume.checkpoint_path is None
            else Path(to_absolute_path(resume.checkpoint_path)))
    if not path.is_file():
        raise FileNotFoundError(
            f"Resume checkpoint not found at {path}. "
            "Set resume.checkpoint_path or reuse the run's data.output."
        )

    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    required = {
        'epoch', 'global_step', 'model_state_dict', 'optimizer_state_dict',
        'lr_scheduler_state_dict', 'diffusion_config',
    }
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError(f"Expected a training checkpoint from scripts/train.py: {path}")
    if any(type(checkpoint[key]) is not int or checkpoint[key] < 0 for key in ('epoch', 'global_step')):
        raise ValueError(f"Resume checkpoint must contain a nonnegative epoch and global step: {path}")
    return path, checkpoint


def _restore_training_objects(checkpoint, cfg, model, optimizer, lr_scheduler):
    task = checkpoint_task(checkpoint)
    if task != model.task:
        hint = " Use init_checkpoint to initialize a CSP model from joint weights." if model.task == 'csp' else ""
        raise ValueError(f"Cannot resume a {model.task} run from a {task} checkpoint.{hint}")
    scheduler_state = checkpoint['lr_scheduler_state_dict']
    if (scheduler_state is None) != (lr_scheduler is None):
        raise ValueError("Resume with the learning-rate scheduler configuration used by the checkpoint")
    saved_config = checkpoint.get('configs', {})
    for section in ('optimizer', 'lr_scheduler'):
        previous = saved_config.get(section)
        current = cfg.get(section)
        if (previous and current and previous.get('_target_')
                and previous['_target_'] != current.get('_target_')):
            raise ValueError(f"The configured {section} must match the resume checkpoint")

    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    if lr_scheduler is not None:
        previous_trainer = saved_config.get('trainer', {})
        # Pre-Hydra checkpoints saved their training-loss settings under 'training'.
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and (
            previous_trainer.get('lr_scheduler_monitor') == 'loss_train'
            or ('trainer' not in saved_config and 'training' in saved_config)
        ):
            scheduler_state = {
                **scheduler_state,
                'best': scheduler_state['mode_worse'],
                'num_bad_epochs': 0,
                'cooldown_counter': 0,
            }
        lr_scheduler.load_state_dict(scheduler_state)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def _capture_rng_state(device):
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        'python': random.getstate(),
        # Use primitive types so the checkpoint also loads with weights_only=True.
        'numpy': (algorithm, keys.tolist(), int(position), int(has_gauss), float(cached_gaussian)),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
    }


def _saved_rank_state(saved_state, rank, world_size):
    if saved_state is None:
        return None
    rank_states = saved_state.get('rank_states')
    if rank_states is None:
        return saved_state if world_size == 1 else None
    return rank_states[rank] if len(rank_states) == world_size else None


def restore_rng_state(checkpoint, device, *, rank=0, world_size=1):
    rank_state = _saved_rank_state(checkpoint.get('training_state'), rank, world_size)
    if rank_state is None:
        return
    state = rank_state['rng_state']
    random.setstate(state['python'])
    algorithm, keys, position, has_gauss, cached_gaussian = state['numpy']
    np.random.set_state((algorithm, np.asarray(keys, dtype=np.uint32), position,
                         has_gauss, cached_gaussian))
    torch.set_rng_state(state['torch'])
    if device.type == 'cuda' and state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'], device)


def _restore_best_checkpoint(checkpoint, path):
    state = checkpoint.get('training_state')
    if state is not None:
        best_path = state['best_checkpoint_path']
        if best_path is not None and not Path(best_path).is_file():
            sibling = path.with_name('best.pt')
            best_path = str(sibling) if sibling.is_file() else None
        return state['best_val_loss'], best_path

    val_loss = checkpoint.get('val_loss')
    best_loss = math.inf if val_loss is None else val_loss
    if path.name == 'best.pt':
        return best_loss, str(path)
    sibling = path.with_name('best.pt')
    if sibling.is_file():
        try:
            best = torch.load(sibling, map_location='cpu', weights_only=True)
        except (OSError, RuntimeError, EOFError, UnpicklingError) as exc:
            logger.warning("Unable to read earlier best checkpoint %s: %s", sibling, exc)
        else:
            if (isinstance(best, dict) and type(best.get('global_step')) is int
                    and best['global_step'] <= checkpoint['global_step']
                    and best.get('val_loss') is not None and best['val_loss'] <= best_loss):
                return best['val_loss'], str(sibling)
    return best_loss, None


def restore_training_state(
    checkpoint, path, cfg, model, optimizer, lr_scheduler, *,
    loader_signature, num_batches, rank=0, world_size=1,
) -> TrainingState:
    _restore_training_objects(checkpoint, cfg, model, optimizer, lr_scheduler)
    partial_epoch = checkpoint.get('partial_epoch', False)
    best_loss, best_path = _restore_best_checkpoint(checkpoint, path)
    saved_state = checkpoint.get('training_state')
    rank_state = _saved_rank_state(saved_state, rank, world_size)
    if rank_state is None and partial_epoch and (saved_state is not None or world_size > 1):
        raise ValueError("Resume a partial epoch with the same number of devices used to save the checkpoint")

    progress = None
    if saved_state is None and rank == 0:
        logger.warning("Checkpoint has no saved random state. Continuing with the configured seed.")
        if partial_epoch:
            logger.warning(
                "Checkpoint has no batch cursor. Restarting the interrupted epoch with the saved optimizer state."
            )
    elif rank_state is None and rank == 0:
        logger.warning("GPU count changed. Continuing from the completed epoch with the configured seed.")
    elif partial_epoch and rank_state is not None:
        saved_progress = rank_state['epoch_progress']
        if (
            saved_progress is None
            or saved_progress['loader_signature'] != loader_signature
            or not 0 < saved_progress['batches_completed'] < num_batches
            or saved_progress['num_crystals'] != (
                saved_progress['batches_completed'] * loader_signature['batch_size']
            )
        ):
            raise ValueError(
                "A partial epoch requires its saved batch cursor, cache, batch size and sampler settings"
            )
        progress = EpochProgress(
            loader_rng_state=saved_progress['loader_rng_state'],
            batches_completed=saved_progress['batches_completed'],
            loss_sum=saved_progress['loss_sum'],
            num_crystals=saved_progress['num_crystals'],
        )

    return TrainingState(
        start_epoch=checkpoint['epoch'] + (0 if partial_epoch else 1),
        global_step=checkpoint['global_step'],
        best_val_loss=best_loss,
        best_checkpoint_path=best_path,
        epoch_progress=progress,
    )
