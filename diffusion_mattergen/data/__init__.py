"""Crystal standardization, dataset caches and periodic-graph batching."""

from .standardize import standardize_cell, standardize_cif, standardize_cif_text
from .cache import build_dataset, inspect_dataset

__all__ = ["standardize_cell", "standardize_cif", "standardize_cif_text",
           "build_dataset", "inspect_dataset", "CrystalDataset", "collate_crystals"]


def __getattr__(name):
    # Single-crystal standardization and cache building do not require PyTorch.
    if name in ("CrystalDataset", "collate_crystals"):
        from .dataset import CrystalDataset, collate_crystals
        return {"CrystalDataset": CrystalDataset, "collate_crystals": collate_crystals}[name]
    raise AttributeError(name)
