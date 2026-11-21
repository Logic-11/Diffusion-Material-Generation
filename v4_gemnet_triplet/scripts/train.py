import argparse
import os
import random
import tempfile
import json
from tqdm.auto import tqdm
from pathlib import Path

import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

# data loading package
from torch.utils.data import DataLoader
from data import CrystalDataset, collate_crystals

import src.geometry as geometry
from src.denoiser import Denoiser
from src.scheduler import MaskScheduler, PositionVEScheduler, LatticeVPScheduler
from src.corruption import AtomTypeCorruption, PositionCorruption, LatticeCorruption
from src.diffusion import JointDiffusion
from src.loss import JointDiffusionLoss

def training_statistics(dataset, output_dir):
    num_crystals = len(dataset)
    density = 0.0
    n_dist = {}
    for sample in dataset:
        volume = np.linalg.det(sample['lattice'].double()).item()
        n = int(sample['num_atoms'])
        density += n / volume
        n_dist[n] = n_dist.get(n, 0) + 1
    density = density / num_crystals
    n_dist = { n: count / num_crystals for n, count in sorted(n_dist.items())}
    statistic = {
        'limit_density': density,
        'n_dist': n_dist
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'statistic.json'), "w", encoding="utf-8") as f:
        json.dump(statistic, f, indent=2)
    return density, n_dist

def load_data(cache, bs, split="train"):
    if split not in {'train', 'val', 'test'}:
        raise ValueError("Split type must be train, val or test")
    dataset = CrystalDataset(cache, split=split)
    loader = DataLoader(dataset, batch_size=bs, shuffle=(split=='train'),
                        num_workers=0, collate_fn=collate_crystals)
    return loader

def check_gradients(model):
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        if not torch.isfinite(grad).all():
            raise RuntimeError(f"Gradients contain NaN or Inf: {name}")
        grad_norm = grad.norm().item()
        max_abs = grad.abs().max().item()
        # print( f"{name}: "
        #        f"grad_norm={grad_norm:.3e}, "
        #        f"max_abs={max_abs:.3e}")


def train_step(model:JointDiffusion, optimizer, batch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = model.compute_loss(batch)
    if not torch.isfinite(loss).item():
        raise RuntimeError(f"Training loss={loss.detach().item()} is not a finite value")
    loss.backward()
    check_gradients(model)
    torch.nn.utils.clip_grad_value_(model.parameters(),clip_value=0.5)
    optimizer.step()

    logged_metrics = {name: value.detach().item() for name, value in metrics.items()}
    return loss.detach().item(), logged_metrics

@torch.no_grad()
def evaluate(model:JointDiffusion, val_loader, device):
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
        batch_loss, batch_metrics = model.compute_loss(batch)
        if not torch.isfinite(batch_loss).item():
                raise RuntimeError(f"Validation loss={batch_loss.detach().item()} is not a finite value")
        loss += batch_size * batch_loss.item()
        num_crystals += batch_size
    
    return loss / num_crystals


def save_model(epoch, epoch_loss, val_loss, model:JointDiffusion, optimizer, lr_scheduler,
               *, output_dir='./output', configs=None, global_step=None, type='latest',
               partial_epoch=False):
    position = model.position_corruption.scheduler
    lattice = model.lattice_corruption.scheduler
    loss_fn = model.loss_fn
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
            'num_spherical': model.denoiser.num_spherical,
            'use_triplets': model.denoiser.use_triplets,
        },
        'diffusion_config': {
            'timesteps': model.timesteps,
            'input_mask_type': model.atom_corruption.mask_type,
            'atom_class_order': 'mask_first',
            'clean_mask_probability': 'learned; first logit column',
            'time_sampling': 'uniform time in [1e-5, 1)',
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
                'max_angle_neighbors': model.max_angle_neighbors,
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



def main():
    parser = argparse.ArgumentParser(description='training configs')
    parser.add_argument('--cache', '--data_cache', help='cached crystal data', required=True)
    parser.add_argument('-o', '--output', default='./output')
    parser.add_argument('-b', '--batch-size', '--batch_size', type=int, default=2)
    parser.add_argument('-e', '--epochs', '--epoch', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('--wd', '--weight_decay', type=float, default=1e-2)
    parser.add_argument('--num-spherical', type=int, default=7)
    parser.add_argument('--max-angle-neighbors', type=int, default=16)
    parser.add_argument('--disable-triplets', action='store_true')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == 'cuda':
        if torch.cuda.is_available(): device = torch.device('cuda')
        else:
            print('Cuda is not available, automatically change device to cpu.')
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    loader = load_data(args.cache, args.batch_size, split='train')
    val_loader = load_data(args.cache, args.batch_size, split='val')
    limit_density, n_dist = training_statistics(loader.dataset, args.output)

    configs = {
        'hidden_dim': 64,
        'num_radial': 32,
        'num_layers': 3,
        'atom_types': 118,
        'timesteps': 1000,
        'cut_off': 7.0,
        'max_neighbor': 50,
        'max_cell_images_per_dim': 5,
        'max_angle_neighbors': args.max_angle_neighbors,
        'num_spherical': args.num_spherical,
        'use_triplets': not args.disable_triplets,
    }
    if configs['num_spherical'] < 2 or configs['max_angle_neighbors'] < 1:
        parser.error('--num-spherical must be >= 2 and --max-angle-neighbors must be >= 1')
    denoiser = Denoiser(
        configs['hidden_dim'],
        configs['num_radial'],
        configs['num_layers'],
        configs['atom_types'],
        num_spherical=configs['num_spherical'],
        use_triplets=configs['use_triplets'],
    )
    element_scheduler = MaskScheduler(configs['timesteps'])
    position_scheduler = PositionVEScheduler()
    lattice_scheduler = LatticeVPScheduler(limit_density=limit_density)
    element_corruption = AtomTypeCorruption(scheduler=element_scheduler, atom_types=configs['atom_types'])
    position_corruption = PositionCorruption(scheduler=position_scheduler)
    lattice_corruption = LatticeCorruption(scheduler=lattice_scheduler)
    diffusion_loss = JointDiffusionLoss(timesteps=configs['timesteps'], num_atom_types=configs['atom_types'])
    diffusion_model = JointDiffusion(denoiser, element_corruption, position_corruption, lattice_corruption, 
                               cutoff=configs['cut_off'], max_neighbor=configs['max_neighbor'], 
                               max_cell_images_per_dim=configs['max_cell_images_per_dim'],
                               max_angle_neighbors=configs['max_angle_neighbors'],
                               loss_fn=diffusion_loss).to(device)

    # optimizer = optim.AdamW(diffusion_model.parameters(), lr=args.lr, weight_decay=args.wd)
    optimizer = optim.Adam(diffusion_model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=100, min_lr=1e-6)

    
    try:
        step = 0
        best_loss = math.inf
        latest_path = best_path = None
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            val_loss = None
            num_crystals = 0
            processed_batch = 0

            total_batches = len(loader)
            if args.max_steps is not None:
                remaining_steps = max(0, args.max_steps - step)
                total_batches = min(total_batches, remaining_steps)
            if total_batches == 0:
                break

            with tqdm(total=total_batches, 
                      desc=f"Train epoch {epoch+1}/{args.epochs}", 
                      unit="batch", 
                      dynamic_ncols=True) as pbar:
                for batch in loader:
                    batch = { key: value.to(device) if torch.is_tensor(value) else value
                            for key, value in batch.items() }
                    batch_size = batch['lattice'].shape[0]

                    batch_loss, batch_metrics = train_step(diffusion_model, optimizer, batch)
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

                    if (args.max_steps is not None) and (step >= args.max_steps): break

            epoch_loss = epoch_loss / num_crystals
            partial_epoch = processed_batch < len(loader)
            if not partial_epoch:
                lr_scheduler.step(epoch_loss)
                if (epoch + 1) % 5 == 0:
                    val_loss = evaluate(diffusion_model, val_loader, device)

            latest_path = save_model(
                epoch, epoch_loss, val_loss, diffusion_model, optimizer, lr_scheduler,
                output_dir=args.output, configs={**configs, 'training': vars(args)},
                global_step=step, type='latest', partial_epoch=partial_epoch )
            if val_loss is not None and val_loss < best_loss:
                best_loss = val_loss
                best_path = save_model(
                                epoch, epoch_loss, val_loss, diffusion_model, optimizer, lr_scheduler,
                                output_dir=args.output, configs={**configs, 'training': vars(args)},
                                global_step=step, type='best' )

            if (args.max_steps is not None) and (step >= args.max_steps): break

        print(f"checkpoints:\n latest: {latest_path} \n best: {best_path}")
        print("training completed")

    finally:
        loader.dataset.close()
        val_loader.dataset.close()

if __name__ == '__main__':
    main()
