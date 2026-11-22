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
               partial_epoch=False, t_min=None):
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
        },
        'diffusion_config': {
            'timesteps': model.timesteps,
            'input_mask_type': model.atom_corruption.mask_type,
            'atom_class_order': 'mask_first',
            'clean_mask_probability': 'learned; first logit column',
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

        lattice = cfg.model.lattice_corruption.scheduler
        density, _ = training_statistics(train_loader.dataset, output_dir, limit_density=lattice.limit_density)
        lattice.limit_density = density
        diffusion_model = instantiate(cfg.model).to(device)
        optimizer = instantiate(cfg.optimizer)(diffusion_model.parameters())
        lr_scheduler = instantiate(cfg.lr_scheduler)(optimizer) if cfg.lr_scheduler is not None else None

        # Keep checkpoint metadata loadable with torch.load(weights_only=True).
        run_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        run_config['data']['output'] = str(output_dir)
        run_config['data']['input'] = cache_dir
        run_config['trainer']['device'] = str(device)
        step = 0
        best_loss = math.inf
        latest_path = best_path = None
        for epoch in range(trainer.max_epochs):
            epoch_loss = 0.0
            val_loss = None
            num_crystals = 0
            processed_batch = 0

            total_batches = len(train_loader)
            if trainer.max_steps is not None:
                remaining_steps = max(0, trainer.max_steps - step)
                total_batches = min(total_batches, remaining_steps)
            if total_batches == 0:
                break

            epoch_lr = optimizer.param_groups[0]['lr']
            with tqdm(total=total_batches,
                      desc=f"Train epoch {epoch + 1}",
                      unit="batch",
                      dynamic_ncols=True) as pbar:
                for batch in train_loader:
                    batch = { key: value.to(device) if torch.is_tensor(value) else value
                            for key, value in batch.items() }
                    batch_size = batch['lattice'].shape[0]

                    batch_loss, batch_metrics = train_step(diffusion_model, optimizer, batch, trainer=trainer)
                    epoch_loss += batch_size * batch_loss
                    num_crystals += batch_size
                    step += 1
                    processed_batch += 1

                    pbar.set_postfix(
                        loss=f"{batch_loss:.4f}",
                        avg=f"{epoch_loss / num_crystals:.4f}",
                        X=f"{batch_metrics['loss_coord']:.3f}",
                        L=f"{batch_metrics['loss_lattice']:.3f}",
                        A=f"{batch_metrics['loss_atom']:.3f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                        refresh=False)
                    pbar.update(1)

                    if trainer.max_steps is not None and step >= trainer.max_steps:
                        break

            epoch_loss = epoch_loss / num_crystals
            partial_epoch = processed_batch < len(train_loader)
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
            latest_path = save_model(
                epoch, epoch_loss, val_loss, diffusion_model, optimizer, lr_scheduler,
                output_dir=output_dir, configs=run_config, global_step=step,
                partial_epoch=partial_epoch, t_min=trainer.t_min,
            )
            if val_loss is not None and val_loss < best_loss:
                best_loss = val_loss
                best_path = save_model(
                    epoch, epoch_loss, val_loss, diffusion_model, optimizer, lr_scheduler,
                    output_dir=output_dir, configs=run_config, global_step=step,
                    type='best', t_min=trainer.t_min,
                )

            if trainer.max_steps is not None and step >= trainer.max_steps:
                break

        print(f"checkpoints:\n latest: {latest_path} \n best: {best_path}")
        print("training completed")


if __name__ == '__main__':
    main()
