"""Training data loaders and the statistics used during sampling."""

import hashlib
import json
import math
import os
from contextlib import ExitStack, contextmanager

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, Subset

from data import CrystalDataset, collate_crystals


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


def load_data(cache, bs, split="train", *, rank=0, world_size=1, seed=0, **loader_options):
    if split not in {'train', 'val', 'test'}:
        raise ValueError("Split type must be train, val or test")
    dataset = CrystalDataset(cache, split=split)
    shuffle = loader_options.setdefault('shuffle', split == 'train')
    if world_size > 1:
        if split == 'train':
            loader_options['sampler'] = DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed,
            )
            loader_options['shuffle'] = False
        else:
            # Evaluation shards can have different lengths and need no padding.
            dataset = Subset(dataset, range(rank, len(dataset), world_size))
            loader_options['shuffle'] = shuffle and bool(len(dataset))
        if loader_options.get('num_workers', 0):
            loader_options.setdefault('multiprocessing_context', 'spawn')
    return DataLoader(dataset, batch_size=bs, collate_fn=collate_crystals, **loader_options)


@contextmanager
def training_loaders(cache_dir, data_cfg, loader_cfg, *, rank=0, world_size=1, seed=0):
    with ExitStack() as datasets:
        loaders = []
        for split, global_batch_size in (('train', data_cfg.batchsize), ('val', loader_cfg.val_batch_size)):
            batch_size = global_batch_size if world_size == 1 else global_batch_size // world_size
            loader = load_data(
                cache_dir, batch_size, split, rank=rank, world_size=world_size, seed=seed,
                num_workers=loader_cfg.num_workers[split],
                shuffle=loader_cfg.shuffle[split], pin_memory=loader_cfg.pin_memory,
            )
            dataset = loader.dataset
            if isinstance(dataset, Subset):
                dataset = dataset.dataset
            datasets.callback(dataset.close)
            loaders.append(loader)
        yield tuple(loaders)


def loader_signature(train_loader, shuffle):
    cache_metadata = json.dumps(train_loader.dataset.metadata, sort_keys=True).encode('utf-8')
    signature = {
        'cache_metadata_sha256': hashlib.sha256(cache_metadata).hexdigest(),
        'batch_size': train_loader.batch_size,
        'shuffle': shuffle,
    }
    sampler = train_loader.sampler
    if isinstance(sampler, DistributedSampler):
        signature['world_size'] = sampler.num_replicas
        if sampler.shuffle:
            signature['sampler_seed'] = sampler.seed
    return signature


def iter_training_batches(train_loader, batches_completed, loader_rng_state, *, epoch=0):
    if isinstance(train_loader.sampler, DistributedSampler):
        train_loader.sampler.set_epoch(epoch)
    if not batches_completed:
        return iter(train_loader)

    # Rebuild the original loader order without consuming the restored training RNG.
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(loader_rng_state)
        batches = iter(train_loader)
        for _ in range(batches_completed):
            next(batches)
    return batches
