import logging
import math
import os
import random
import sys
from functools import partial
from pathlib import Path

if __name__ == "__main__" and sys.platform == "win32":
    # NumPy's MKL and PyTorch can load conflicting OpenMP runtimes on Windows.
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.diffusion.joint_diffusion import JointDiffusion
from src.training.checkpoint import (
    EpochProgress,
    TrainingState,
    read_resume_checkpoint,
    restore_rng_state,
    restore_training_state,
    save_training_checkpoint,
)
from src.training.config import CONFIG_FILE, GRADIENT_CLIPPERS, default_trainer, validate_config
from src.training.data import (
    iter_training_batches,
    load_data as load_data,
    loader_signature,
    training_loaders,
    training_statistics,
)
from src.training.distributed import TrainingProcess, launch_training, training_loss


logger = logging.getLogger(__name__)


def _move_batch_to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def check_gradients(model, *, report_missing=True):
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            if report_missing:
                print(f"{name}: grad=None")
            continue
        grad = parameter.grad.detach()
        if not torch.isfinite(grad).all():
            raise RuntimeError(f"Gradients contain NaN or Inf: {name}")


def train_step(
    model: JointDiffusion, optimizer, batch, *, trainer: DictConfig | None = None,
    process: TrainingProcess | None = None, loss_fn=None,
):
    trainer = default_trainer() if trainer is None else trainer
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if loss_fn is None:
        loss_fn = model.compute_loss
    loss, metrics = loss_fn(batch, t_min=trainer.t_min)
    finite = torch.isfinite(loss)
    if not (finite.item() if process is None else process.all_finite(finite)):
        if process is not None and process.distributed:
            raise RuntimeError("Training loss is not finite on at least one GPU")
        raise RuntimeError(f"Training loss={loss.detach().item()} is not a finite value")
    loss.backward()
    if process is not None:
        process.synchronize_gradients(model)
    if trainer.check_gradients:
        check_gradients(model, report_missing=process is None or process.is_primary)
    if trainer.gradient_clip_val:
        clip = GRADIENT_CLIPPERS[trainer.gradient_clip_algorithm]
        clip(model.parameters(), trainer.gradient_clip_val)
    optimizer.step()

    logged_metrics = {name: value.detach().item() for name, value in metrics.items()}
    return loss.detach().item(), logged_metrics


@torch.no_grad()
def evaluate(model: JointDiffusion, val_loader, device, *, t_min=None, process: TrainingProcess | None = None):
    if t_min is None:
        t_min = default_trainer().t_min
    if process is None:
        process = TrainingProcess(device)
    model.eval()
    loss = 0.0
    num_crystals = 0
    for batch in tqdm(val_loader,
                      desc="Validation (rank 0)" if process.distributed else "Validation",
                      unit="batch",
                      dynamic_ncols=True,
                      leave=False,
                      disable=not process.is_primary):
        batch = _move_batch_to_device(batch, device)
        batch_size = batch['lattice'].shape[0]
        batch_loss, _ = model.compute_loss(batch, t_min=t_min)
        batch_loss = batch_loss.item()
        if not math.isfinite(batch_loss):
            loss = batch_loss
            break
        loss += batch_size * batch_loss
        num_crystals += batch_size

    loss, num_crystals = process.sum_loss(loss, num_crystals)
    if not math.isfinite(loss):
        raise RuntimeError(f"Validation loss={loss} is not a finite value")
    if not num_crystals:
        raise ValueError("The validation split is empty")
    return loss / num_crystals


def fit(
    model: JointDiffusion, optimizer, lr_scheduler, train_loader, val_loader, *,
    trainer: DictConfig, device: torch.device, state: TrainingState, save_checkpoint,
    latest_path=None, save_interval=None, process: TrainingProcess | None = None, loss_fn=None,
) -> None:
    if process is None:
        process = TrainingProcess(device)
    num_batches = len(train_loader)
    max_steps = trainer.max_steps
    for epoch in range(state.start_epoch, trainer.max_epochs):
        progress = state.epoch_progress or EpochProgress(torch.get_rng_state())
        state.epoch_progress = None

        total_batches = num_batches - progress.batches_completed
        if max_steps is not None:
            total_batches = min(total_batches, max(0, max_steps - state.global_step))
        if total_batches == 0:
            break

        batches = iter_training_batches(
            train_loader, progress.batches_completed, progress.loader_rng_state, epoch=epoch,
        )
        epoch_lr = optimizer.param_groups[0]['lr']
        description = f"Train epoch {epoch + 1}" + (" (rank 0)" if process.distributed else "")
        with tqdm(total=progress.batches_completed + total_batches,
                  initial=progress.batches_completed,
                  desc=description,
                  unit="batch",
                  dynamic_ncols=True,
                  disable=not process.is_primary) as pbar:
            for batch in batches:
                batch = _move_batch_to_device(batch, device)
                batch_size = batch['lattice'].shape[0]
                batch_loss, batch_metrics = train_step(
                    model, optimizer, batch, trainer=trainer, process=process, loss_fn=loss_fn,
                )
                progress.loss_sum += batch_size * batch_loss
                progress.num_crystals += batch_size
                progress.batches_completed += 1
                state.global_step += 1

                pbar.set_postfix(
                    loss=f"{batch_loss:.4f}",
                    avg=f"{progress.mean_loss:.4f}",
                    X=f"{batch_metrics['loss_coord']:.3f}",
                    L=f"{batch_metrics['loss_lattice']:.3f}",
                    A=f"{batch_metrics['loss_atom']:.3f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                    refresh=False,
                )
                pbar.update(1)

                if max_steps is not None and state.global_step >= max_steps:
                    break
                if (save_interval is not None and state.global_step % save_interval == 0
                        and progress.batches_completed < num_batches):
                    latest_path = save_checkpoint(epoch, progress, state, partial_epoch=True)
        del batches

        loss_sum, num_crystals = process.sum_loss(progress.loss_sum, progress.num_crystals)
        epoch_loss = loss_sum / num_crystals
        val_loss = None
        partial_epoch = progress.batches_completed < num_batches
        if not partial_epoch:
            if (epoch + 1) % trainer.check_val_every_n_epoch == 0:
                val_loss = evaluate(model, val_loader, device, t_min=trainer.t_min, process=process)
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric = epoch_loss if trainer.lr_scheduler_monitor == 'loss_train' else val_loss
                if metric is not None:
                    lr_scheduler.step(metric)
            elif lr_scheduler is not None:
                lr_scheduler.step()

        if process.is_primary:
            logger.info(
                "Train epoch %d%s | global_step=%d | loss_train=%.6f | loss_val=%s | lr=%.3e",
                epoch + 1,
                " (partial)" if partial_epoch else "",
                state.global_step,
                epoch_loss,
                f"{val_loss:.6f}" if val_loss is not None else "N/A",
                epoch_lr,
            )
        checkpoint_metrics = {'epoch_loss': epoch_loss} if process.distributed else {}
        latest_path = save_checkpoint(
            epoch, progress, state, partial_epoch=partial_epoch, val_loss=val_loss,
            **checkpoint_metrics,
        )
        if max_steps is not None and state.global_step >= max_steps:
            break

    if process.is_primary:
        print(f"checkpoints:\n latest: {latest_path} \n best: {state.best_checkpoint_path}")
        print("training completed")


def run_training(cfg: DictConfig, *, output_dir: Path) -> None:
    validate_config(cfg)
    trainer = cfg.trainer
    device = torch.device(trainer.device)
    world_size = trainer.get('devices', 1)
    if world_size == 1:
        if device.type == 'cuda' and not torch.cuda.is_available():
            print('CUDA is unavailable. Using CPU.')
            device = torch.device('cpu')
        _run_training(cfg, output_dir=output_dir, process=TrainingProcess(device))
        return

    launch_training(_run_training, cfg, output_dir, device, world_size)


def _run_training(cfg: DictConfig, *, output_dir: Path, process: TrainingProcess) -> None:
    trainer = cfg.trainer
    device = process.device
    seed = trainer.seed + process.rank
    random.seed(seed)
    np.random.seed(seed % (2**32) if process.rank else seed)
    torch.manual_seed(seed)

    cache_dir = to_absolute_path(cfg.data.input)
    resume_path, checkpoint = read_resume_checkpoint(cfg.resume, output_dir)
    with training_loaders(
        cache_dir, cfg.data, cfg.dataloader,
        rank=process.rank, world_size=process.world_size, seed=trainer.seed,
    ) as (train_loader, val_loader):
        if process.distributed and process.is_primary:
            logger.info(
                "Using %d GPUs with %s | train batch=%d/GPU | validation batch=%d/GPU",
                process.world_size, process.backend, train_loader.batch_size, val_loader.batch_size,
            )
        signature = loader_signature(train_loader, cfg.dataloader.shuffle.train)
        lattice = cfg.model.lattice_corruption.scheduler
        if checkpoint is None:
            density = None
            if process.is_primary:
                density, _ = training_statistics(
                    train_loader.dataset, output_dir, limit_density=lattice.limit_density,
                )
            density = process.broadcast_float(density)
        else:
            density = checkpoint['diffusion_config']['lattice']['limit_density']
        lattice.limit_density = density
        model = instantiate(cfg.model).to(device)
        optimizer = instantiate(cfg.optimizer)(model.parameters())
        lr_scheduler = instantiate(cfg.lr_scheduler)(optimizer) if cfg.lr_scheduler is not None else None

        state = TrainingState()
        if checkpoint is not None:
            state = restore_training_state(
                checkpoint, resume_path, cfg, model, optimizer, lr_scheduler,
                loader_signature=signature, num_batches=len(train_loader),
                rank=process.rank, world_size=process.world_size,
            )
            if process.is_primary:
                training_statistics(train_loader.dataset, output_dir, limit_density=density)
                logger.info(
                    "Resuming %s at epoch %d, global_step=%d, batches_completed=%d",
                    resume_path, state.start_epoch + 1, state.global_step,
                    0 if state.epoch_progress is None else state.epoch_progress.batches_completed,
                )
                if state.start_epoch >= trainer.max_epochs or (
                    trainer.max_steps is not None and state.global_step >= trainer.max_steps
                ):
                    logger.info("The checkpoint has reached a training limit. Increase the limit to continue training.")

        loss_fn = training_loss(model, process)
        if checkpoint is not None:
            restore_rng_state(checkpoint, device, rank=process.rank, world_size=process.world_size)
            del checkpoint

        # Keep checkpoint metadata loadable with torch.load(weights_only=True).
        run_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        run_config['data']['output'] = str(output_dir)
        run_config['data']['input'] = cache_dir
        run_config['trainer']['device'] = str(device)
        if resume_path is not None:
            run_config['resume']['checkpoint_path'] = str(resume_path)

        save_checkpoint = partial(
            save_training_checkpoint, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
            output_dir=output_dir, configs=run_config, loader_signature=signature,
            device=device, t_min=trainer.t_min, process=process,
        )
        fit(
            model, optimizer, lr_scheduler, train_loader, val_loader,
            trainer=trainer, device=device, state=state, save_checkpoint=save_checkpoint,
            latest_path=resume_path, save_interval=cfg.resume.save_every_n_steps,
            process=process, loss_fn=loss_fn,
        )


@hydra.main(version_base="1.3", config_path=str(CONFIG_FILE.parent), config_name=CONFIG_FILE.stem)
def main(cfg: DictConfig) -> None:
    run_training(cfg, output_dir=Path(HydraConfig.get().runtime.output_dir))


if __name__ == '__main__':
    main()
