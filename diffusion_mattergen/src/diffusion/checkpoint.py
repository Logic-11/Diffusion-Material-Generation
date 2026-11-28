"""Read the model metadata and weights shared by training and generation."""

from pathlib import Path

import torch

from ..backbone.denoiser import Denoiser


def checkpoint_task(checkpoint: dict) -> str:
    """Read the task, defaulting to joint for older checkpoints."""
    task = checkpoint.get("task", "joint")
    if task not in ("joint", "csp"):
        raise ValueError(f"Unknown checkpoint task {task!r}")
    return task


def read_model_checkpoint(checkpoint_path: str | Path) -> dict:
    # Keep optimizer tensors off the GPU when reading weights for inference or initialization.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {"model_config", "diffusion_config", "model_state_dict"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError(
            "Expected a training checkpoint containing "
            "model_config, diffusion_config and model_state_dict."
        )
    if "denoiser.output_head.2.weight" in checkpoint["model_state_dict"]:
        raise ValueError(
            "This checkpoint uses the previous denoiser architecture. "
            "Use a checkpoint from the GemNet-T denoiser."
        )
    atom_class_order = checkpoint["diffusion_config"].get("atom_class_order", Denoiser.atom_class_order)
    if atom_class_order != Denoiser.atom_class_order:
        raise ValueError(f"Expected MASK-last denoiser weights, got {atom_class_order!r}")
    return checkpoint
