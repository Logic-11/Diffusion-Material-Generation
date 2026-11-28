"""Local multi-GPU launch and synchronization."""

from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import configure_log
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class TrainingProcess:
    device: torch.device
    rank: int = 0
    world_size: int = 1
    backend: str | None = None

    @property
    def distributed(self):
        return self.world_size > 1

    @property
    def is_primary(self):
        return self.rank == 0

    @property
    def collective_device(self):
        return self.device if self.backend == "nccl" else torch.device("cpu")

    def sum_loss(self, loss_sum, num_crystals):
        if not self.distributed:
            return loss_sum, num_crystals
        totals = torch.tensor(
            [loss_sum, num_crystals], dtype=torch.float64, device=self.collective_device,
        )
        dist.all_reduce(totals)
        total_loss, total_crystals = totals.tolist()
        return total_loss, int(total_crystals)

    def broadcast_float(self, value):
        if not self.distributed:
            return value
        tensor = torch.tensor(
            0.0 if value is None else value, dtype=torch.float64, device=self.collective_device,
        )
        dist.broadcast(tensor, src=0)
        return tensor.item()

    def all_finite(self, finite):
        if self.distributed:
            finite = finite.to(device=self.collective_device, dtype=torch.int32)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        return bool(finite.item())

    def gather_rank_states(self, state):
        if not self.distributed:
            return None
        states = [None] * self.world_size
        dist.all_gather_object(states, state)
        return states

    @torch.no_grad()
    def synchronize_gradients(self, model):
        if self.backend != "gloo":
            return

        # Windows Gloo builds can lack CUDA all-reduce. Stage only the gradients
        # on CPU, with zeros for parameters unused locally and None if unused everywhere.
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        used = torch.tensor(
            [parameter.grad is not None for parameter in parameters], dtype=torch.int32, device="cpu",
        )
        dist.all_reduce(used, op=dist.ReduceOp.MAX)
        groups = defaultdict(list)
        for parameter, has_grad in zip(parameters, used.tolist(), strict=True):
            if has_grad:
                groups[parameter.dtype].append(parameter)

        for group in groups.values():
            flattened = torch.cat([
                parameter.grad.reshape(-1) if parameter.grad is not None
                else parameter.new_zeros(parameter.numel())
                for parameter in group
            ]).cpu()
            dist.all_reduce(flattened)
            flattened.div_(self.world_size)
            gradients = flattened.to(self.device).split([parameter.numel() for parameter in group])
            for parameter, gradient in zip(group, gradients, strict=True):
                parameter.grad = gradient.view_as(parameter)


class _TrainingLoss(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch, *, t_min):
        loss, metrics = self.model.compute_loss(batch, t_min=t_min)
        return loss, {name: value.detach() for name, value in metrics.items()}


def training_loss(model, process: TrainingProcess):
    if process.backend == "nccl":
        return DistributedDataParallel(
            _TrainingLoss(model), device_ids=[process.device.index], find_unused_parameters=True,
        )
    if process.backend == "gloo":
        with torch.no_grad():
            for tensor in chain(model.parameters(), model.buffers()):
                cpu_tensor = tensor.detach().cpu().contiguous()
                dist.broadcast(cpu_tensor, src=0)
                tensor.copy_(cpu_tensor)
    return model.compute_loss


def resolve_training_devices(trainer: DictConfig) -> tuple[torch.device, ...]:
    device = torch.device(trainer.device)
    gpu_ids = trainer.get('gpu_ids')
    if gpu_ids is None:
        num_devices = trainer.get('devices', 1)
        if num_devices == 1:
            if device.type == 'cuda' and not torch.cuda.is_available():
                print('CUDA is unavailable. Using CPU.')
                device = torch.device('cpu')
            return (device,)
        first_gpu_id = device.index if device.index is not None else 0
        highest_gpu_id = first_gpu_id + num_devices - 1
        gpu_ids = range(first_gpu_id, highest_gpu_id + 1)
    else:
        highest_gpu_id = max(gpu_ids)

    available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if highest_gpu_id >= available:
        raise ValueError(f"Requested cuda:{highest_gpu_id}, but found {available} visible CUDA GPUs")
    return tuple(torch.device('cuda', gpu_id) for gpu_id in gpu_ids)


def _run_worker(rank, worker, cfg, output_dir, devices, backend, rendezvous_path, job_logging):
    if rank == 0 and job_logging is not None:
        configure_log(*job_logging)

    device = devices[rank]
    world_size = len(devices)
    torch.cuda.set_device(device)
    store = dist.FileStore(rendezvous_path, world_size)
    dist.init_process_group(backend=backend, store=store, rank=rank, world_size=world_size)
    try:
        process = TrainingProcess(device, rank, world_size, backend)
        worker(cfg, output_dir=output_dir, process=process)
    finally:
        dist.destroy_process_group()


def launch_training(worker, cfg, output_dir, devices):
    if not dist.is_available():
        raise RuntimeError("Multi-GPU training requires a PyTorch build with torch.distributed")
    if dist.is_nccl_available():
        backend = "nccl"
    elif dist.is_gloo_available():
        backend = "gloo"
    else:
        raise RuntimeError("Multi-GPU training requires the NCCL or Gloo backend")

    # Resolve Hydra paths and interpolations in the process that owns the run.
    worker_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    worker_cfg.data.input = to_absolute_path(worker_cfg.data.input)
    if worker_cfg.resume.enabled and worker_cfg.resume.checkpoint_path is not None:
        worker_cfg.resume.checkpoint_path = to_absolute_path(worker_cfg.resume.checkpoint_path)
    output_dir = output_dir.resolve()

    job_logging = None
    if HydraConfig.initialized():
        hydra_cfg = HydraConfig.get()
        log_cfg = hydra_cfg.job_logging
        if log_cfg is not None:
            log_cfg = OmegaConf.create(OmegaConf.to_container(log_cfg, resolve=True))
        verbose = hydra_cfg.verbose
        if OmegaConf.is_list(verbose):
            verbose = OmegaConf.create(OmegaConf.to_container(verbose, resolve=True))
        job_logging = (log_cfg, verbose)

    with TemporaryDirectory(prefix="mattergen-ddp-") as directory:
        rendezvous_path = str(Path(directory) / "rendezvous")
        mp.spawn(
            _run_worker,
            args=(worker, worker_cfg, output_dir, devices, backend, rendezvous_path, job_logging),
            nprocs=len(devices),
        )
