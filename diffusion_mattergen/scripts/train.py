import hashlib
import json
import logging
import math
import os
import random
import sys
import tempfile
from contextlib import ExitStack
from functools import cache
from pathlib import Path
from pickle import UnpicklingError

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import CrystalDataset, collate_crystals
from src.diffusion import JointDiffusion

logger = logging.getLogger(__name__)
CONFIG_FILE = Path(__file__).resolve().parent / "conf" / "train.yaml"
GRADIENT_CLIPPERS = {
    'value': torch.nn.utils.clip_grad_value_,
    'norm': torch.nn.utils.clip_grad_norm_,
}


@cache
def _default_trainer() -> DictConfig:
    """Defaults for helpers called directly, outside a Hydra run."""
    return OmegaConf.load(CONFIG_FILE).trainer


def training_statistics(dataset, output_dir, *, limit_density=None):
    num_crystals = len(dataset)
    if not num_crystals:
        raise ValueError("The training split is empty")
    density_sum = 0.0
    n_dist = {}
    for sample in dataset:
        n = int(sample['num_atoms'])
        if limit_density is None:
            volume = np.linalg.det(sample['lattice'].double()).item()
            density_sum += n / volume
        n_dist[n] = n_dist.get(n, 0) + 1
    density = density_sum / num_crystals if limit_density is None else limit_density
    if not math.isfinite(density) or density <= 0:
        raise ValueError("The lattice limit_density must be finite and positive")
    n_dist = {n: count / num_crystals for n, count in sorted(n_dist.items())}
    statistic = {
        'limit_density': density,
        'n_dist': n_dist
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'statistic.json'), "w", encoding="utf-8") as f:
        json.dump(statistic, f, indent=2)
    return density, n_dist


def load_data(cache, bs, split="train", **loader_options):
    if split not in {'train', 'val', 'test'}:
        raise ValueError("Split type must be train, val or test")
    dataset = CrystalDataset(cache, split=split)
    loader_options.setdefault('shuffle', split == 'train')
    return DataLoader(dataset, batch_size=bs, collate_fn=collate_crystals, **loader_options)


def check_gradients(model):
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            print(f"{name}: grad=None")
            continue
        grad = parameter.grad.detach()
        if not torch.isfinite(grad).all():
            raise RuntimeError(f"Gradients contain NaN or Inf: {name}")


def train_step(model: JointDiffusion, optimizer, batch, *, trainer: DictConfig | None = None):
    trainer = _default_trainer() if trainer is None else trainer
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = model.compute_loss(batch, t_min=trainer.t_min)
    if not torch.isfinite(loss).item():
        raise RuntimeError(f"Training loss={loss.detach().item()} is not a finite value")
    loss.backward()
    if trainer.check_gradients:
        check_gradients(model)
    if trainer.gradient_clip_val:
        clip = GRADIENT_CLIPPERS[trainer.gradient_clip_algorithm]
        clip(model.parameters(), trainer.gradient_clip_val)
    optimizer.step()

    logged_metrics = {name: value.detach().item() for name, value in metrics.items()}
    return loss.detach().item(), logged_metrics


@torch.no_grad()
def evaluate(model: JointDiffusion, val_loader, device, *, t_min=None):
    if t_min is None:
        t_min = _default_trainer().t_min
    model.eval()
    loss = 0.0
    num_crystals = 0
    for batch in tqdm(val_loader,
                      desc="Validation",
                      unit="batch",
                      dynamic_ncols=True,
                      leave=False):
        batch = { key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in batch.items() }
        batch_size = batch['lattice'].shape[0]
        batch_loss, _ = model.compute_loss(batch, t_min=t_min)
        if not torch.isfinite(batch_loss).item():
            raise RuntimeError(f"Validation loss={batch_loss.detach().item()} is not a finite value")
        loss += batch_size * batch_loss.item()
        num_crystals += batch_size

    return loss / num_crystals


def save_model(epoch, epoch_loss, val_loss, model: JointDiffusion, optimizer, lr_scheduler,
               *, output_dir='./output', configs=None, global_step=None, type='latest',
               partial_epoch=False, t_min=None, training_state=None):
    position = model.position_corruption.scheduler
    lattice = model.lattice_corruption.scheduler
    loss_fn = model.loss_fn
    if t_min is None:
        t_min = _default_trainer().t_min
    checkpoint = {
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
            'atom_types': model.atom_corruption.atom_types,
            'num_spherical': model.denoiser.num_spherical,
            'use_triplets': model.denoiser.use_triplets,
        },
        'diffusion_config': {
            'timesteps': model.timesteps,
            'input_mask_type': model.atom_corruption.mask_type,
            'atom_class_order': getattr(model.denoiser, 'atom_class_order', 'mask_first'),
            'clean_mask_probability': 'learned in the first column of the joint diffusion output',
            'time_sampling': f'uniform time in [{t_min:g}, 1)',
            't_min': t_min,
            'position': {
                name: getattr(position, name).item()
                for name in ('sigma_min', 'sigma_max')
            },
            'lattice': {
                name: getattr(lattice, name).item()
                for name in ('limit_density', 'beta_min', 'beta_max',
                             'limit_var_scaling_constant')
            },
            'loss': {
                name: getattr(loss_fn, name)
                for name in ('coord_weight', 'lattice_weight', 'atom_weight',
                             'd3pm_hybrid_lambda', 'reduce', 'timesteps',
                             'num_atom_types')
            },
            'graph': {
                'cutoff': model.radial_basis.cutoff,
                'max_neighbor': model.max_neighbor,
                'max_cell_images_per_dim': model.max_cell_images_per_dim,
                'max_offset_integer': model.max_offset_integer,
            },
            'output_parameterization': {
                'atoms': 'clean-class logits; MASK followed by real elements',
                'coordinates': 'fractional score_times_std',
                'lattice': 'score_times_std',
            },
        },
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f'{type}.pt')
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


def _read_resume_checkpoint(resume, output_dir):
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


def _restore_rng_state(state, device):
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


def _validate_config(cfg: DictConfig) -> None:
    trainer = cfg.trainer
    if type(trainer.max_epochs) is not int or trainer.max_epochs < 0:
        raise ValueError("trainer.max_epochs must be a nonnegative integer")
    max_steps = trainer.max_steps
    if max_steps is not None and (type(max_steps) is not int or max_steps < 0):
        raise ValueError("trainer.max_steps must be a nonnegative integer or null")
    if type(trainer.check_val_every_n_epoch) is not int or trainer.check_val_every_n_epoch < 1:
        raise ValueError("trainer.check_val_every_n_epoch must be a positive integer")
    if trainer.lr_scheduler_monitor not in ('loss_train', 'loss_val'):
        raise ValueError("trainer.lr_scheduler_monitor must be 'loss_train' or 'loss_val'")
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
    if model.loss_fn.timesteps != model.atom_corruption.scheduler.timesteps:
        raise ValueError("The atom scheduler and D3PM loss must use the same number of timesteps")
    atom_types = model.denoiser.atom_types
    if model.atom_corruption.atom_types != atom_types or model.loss_fn.num_atom_types != atom_types:
        raise ValueError("The denoiser, atom corruption and loss must agree on the number of elements")
    position = model.position_corruption.scheduler
    if not (0 < position.sigma_min <= position.sigma_max and math.isfinite(position.sigma_max)):
        raise ValueError("Position VE requires finite sigmas with 0 < sigma_min <= sigma_max")


@hydra.main(version_base="1.3", config_path=str(CONFIG_FILE.parent), config_name=CONFIG_FILE.stem)
def main(cfg: DictConfig) -> None:
    _validate_config(cfg)
    trainer = cfg.trainer
    random.seed(trainer.seed)
    np.random.seed(trainer.seed)
    torch.manual_seed(trainer.seed)

    device = torch.device(trainer.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print('CUDA is unavailable. Using CPU.')
        device = torch.device('cpu')

    data = cfg.data
    loader_cfg = cfg.dataloader
    cache_dir = to_absolute_path(data.input)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    resume_path, checkpoint = _read_resume_checkpoint(cfg.resume, output_dir)
    with ExitStack() as datasets:
        loaders = {}
        for split, batch_size in (('train', data.batchsize), ('val', loader_cfg.val_batch_size)):
            loader = load_data(
                cache_dir, batch_size, split, num_workers=loader_cfg.num_workers[split],
                shuffle=loader_cfg.shuffle[split], pin_memory=loader_cfg.pin_memory,
            )
            datasets.callback(loader.dataset.close)
            loaders[split] = loader
        train_loader, val_loader = loaders['train'], loaders['val']
        num_batches = len(train_loader)
        cache_metadata = json.dumps(train_loader.dataset.metadata, sort_keys=True).encode('utf-8')
        loader_signature = {
            'cache_metadata_sha256': hashlib.sha256(cache_metadata).hexdigest(),
            'batch_size': train_loader.batch_size,
            'shuffle': loader_cfg.shuffle.train,
        }

        lattice = cfg.model.lattice_corruption.scheduler
        if checkpoint is None:
            density, _ = training_statistics(train_loader.dataset, output_dir, limit_density=lattice.limit_density)
        else:
            density = checkpoint['diffusion_config']['lattice']['limit_density']
        lattice.limit_density = density
        diffusion_model = instantiate(cfg.model).to(device)
        optimizer = instantiate(cfg.optimizer)(diffusion_model.parameters())
        lr_scheduler = instantiate(cfg.lr_scheduler)(optimizer) if cfg.lr_scheduler is not None else None

        start_epoch = step = 0
        best_loss, best_path = math.inf, None
        latest_path = resume_path
        epoch_progress = None
        if checkpoint is not None:
            _restore_training_objects(checkpoint, cfg, diffusion_model, optimizer, lr_scheduler)
            step = checkpoint['global_step']
            partial_epoch = checkpoint.get('partial_epoch', False)
            start_epoch = checkpoint['epoch'] + (0 if partial_epoch else 1)
            best_loss, best_path = _restore_best_checkpoint(checkpoint, resume_path)
            training_state = checkpoint.get('training_state')
            if training_state is None:
                logger.warning("Checkpoint has no saved random state. Continuing with the configured seed.")
                if partial_epoch:
                    logger.warning(
                        "Checkpoint has no batch cursor. Restarting the interrupted epoch with the saved optimizer state."
                    )
            elif partial_epoch:
                epoch_progress = training_state['epoch_progress']
                if (epoch_progress is None or epoch_progress['loader_signature'] != loader_signature
                        or not 0 < epoch_progress['batches_completed'] < num_batches
                        or epoch_progress['num_crystals'] != epoch_progress['batches_completed'] * data.batchsize):
                    raise ValueError(
                        "A partial epoch requires its saved batch cursor, cache, batch size and shuffle setting"
                    )

            training_statistics(train_loader.dataset, output_dir, limit_density=density)
            if training_state is not None:
                _restore_rng_state(training_state['rng_state'], device)
            logger.info("Resuming %s at epoch %d, global_step=%d, batches_completed=%d",
                        resume_path, start_epoch + 1, step,
                        0 if epoch_progress is None else epoch_progress['batches_completed'])
            if start_epoch >= trainer.max_epochs or (trainer.max_steps is not None and step >= trainer.max_steps):
                logger.info("The checkpoint has reached a training limit. Increase the limit to continue training.")
            del checkpoint

        # Keep checkpoint metadata loadable with torch.load(weights_only=True).
        run_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        run_config['data']['output'] = str(output_dir)
        run_config['data']['input'] = cache_dir
        run_config['trainer']['device'] = str(device)
        if resume_path is not None:
            run_config['resume']['checkpoint_path'] = str(resume_path)
        save_interval = cfg.resume.save_every_n_steps

        def save_checkpoint(epoch, loss_sum, num_crystals, processed_batches, loader_rng_state,
                            *, val_loss=None, save_best=False):
            partial_epoch = processed_batches < num_batches
            partial_progress = None
            if partial_epoch:
                partial_progress = {
                    'batches_completed': processed_batches,
                    'loss_sum': loss_sum,
                    'num_crystals': num_crystals,
                    'loader_rng_state': loader_rng_state,
                    'loader_signature': loader_signature,
                }
            training_state = {
                'epoch_progress': partial_progress,
                'best_val_loss': best_loss,
                'best_checkpoint_path': best_path,
                'rng_state': _capture_rng_state(device),
            }
            args = (epoch, loss_sum / num_crystals, val_loss, diffusion_model, optimizer, lr_scheduler)
            options = dict(output_dir=output_dir, configs=run_config, global_step=step,
                           partial_epoch=partial_epoch, t_min=trainer.t_min, training_state=training_state)
            if save_best:
                save_model(*args, type='best', **options)
            return save_model(*args, **options)

        for epoch in range(start_epoch, trainer.max_epochs):
            val_loss = None
            if epoch_progress is None:
                loss_sum, num_crystals, processed_batches = 0.0, 0, 0
                loader_rng_state = torch.get_rng_state()
            else:
                loss_sum = epoch_progress['loss_sum']
                num_crystals = epoch_progress['num_crystals']
                processed_batches = epoch_progress['batches_completed']
                loader_rng_state = epoch_progress['loader_rng_state']
                epoch_progress = None

            total_batches = num_batches - processed_batches
            if trainer.max_steps is not None:
                remaining_steps = max(0, trainer.max_steps - step)
                total_batches = min(total_batches, remaining_steps)
            if total_batches == 0:
                break

            if processed_batches:
                # Rebuild the original loader order without consuming the restored training RNG.
                with torch.random.fork_rng(devices=[]):
                    torch.set_rng_state(loader_rng_state)
                    batches = iter(train_loader)
                    for _ in range(processed_batches):
                        next(batches)
            else:
                batches = iter(train_loader)

            epoch_lr = optimizer.param_groups[0]['lr']
            with tqdm(total=processed_batches + total_batches,
                      initial=processed_batches,
                      desc=f"Train epoch {epoch + 1}",
                      unit="batch",
                      dynamic_ncols=True) as pbar:
                for batch in batches:
                    batch = { key: value.to(device) if torch.is_tensor(value) else value
                            for key, value in batch.items() }
                    batch_size = batch['lattice'].shape[0]

                    batch_loss, batch_metrics = train_step(diffusion_model, optimizer, batch, trainer=trainer)
                    loss_sum += batch_size * batch_loss
                    num_crystals += batch_size
                    step += 1
                    processed_batches += 1

                    pbar.set_postfix(
                        loss=f"{batch_loss:.4f}",
                        avg=f"{loss_sum / num_crystals:.4f}",
                        X=f"{batch_metrics['loss_coord']:.3f}",
                        L=f"{batch_metrics['loss_lattice']:.3f}",
                        A=f"{batch_metrics['loss_atom']:.3f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                        refresh=False)
                    pbar.update(1)

                    if trainer.max_steps is not None and step >= trainer.max_steps:
                        break
                    if save_interval is not None and step % save_interval == 0 and processed_batches < num_batches:
                        latest_path = save_checkpoint(epoch, loss_sum, num_crystals, processed_batches, loader_rng_state)
            del batches

            epoch_loss = loss_sum / num_crystals
            partial_epoch = processed_batches < num_batches
            if not partial_epoch:
                if (epoch + 1) % trainer.check_val_every_n_epoch == 0:
                    val_loss = evaluate(diffusion_model, val_loader, device, t_min=trainer.t_min)
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    metric = epoch_loss if trainer.lr_scheduler_monitor == 'loss_train' else val_loss
                    if metric is not None:
                        lr_scheduler.step(metric)
                elif lr_scheduler is not None:
                    lr_scheduler.step()

            logger.info(
                "Train epoch %d%s | global_step=%d | loss_train=%.6f | loss_val=%s | lr=%.3e",
                epoch + 1,
                " (partial)" if partial_epoch else "",
                step,
                epoch_loss,
                f"{val_loss:.6f}" if val_loss is not None else "N/A",
                epoch_lr,
            )
            improved = val_loss is not None and val_loss < best_loss
            if improved:
                best_loss = val_loss
                best_path = str(output_dir / 'best.pt')
            latest_path = save_checkpoint(epoch, loss_sum, num_crystals, processed_batches, loader_rng_state,
                                          val_loss=val_loss, save_best=improved)

            if trainer.max_steps is not None and step >= trainer.max_steps:
                break

        print(f"checkpoints:\n latest: {latest_path} \n best: {best_path}")
        print("training completed")


if __name__ == '__main__':
    main()
