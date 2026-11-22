"""Memory-mapped clean crystals and variable-size batches for build_periodic_graph."""

import operator
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.geometry import polar_decomposition

from .cache import CORE_FILES, SPLITS, inspect_dataset


class CrystalDataset(Dataset):
    """Load one cached split; each DataLoader process opens its own read-only maps.

    Float arrays are stored as float64 and returned as float32 by default. Samples
    own writable tensor copies, so noise/transforms cannot modify the cached data.
    Reduced lattices saved in MatterGen mode are symmetrized after conversion.
    The original CSVs and pymatgen parsing are not needed during sample loading.
    """

    def __init__(self, path: str | Path, split: str = "train", dtype=torch.float32):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        self.path = str(Path(path).resolve())
        self.split, self.dtype = split, dtype
        self.metadata = inspect_dataset(self.path)
        if split not in self.metadata["splits"]:
            raise ValueError(f"Split {split!r} is absent; available: {list(self.metadata['splits'])}")
        self._length = self.metadata["splits"][split]["num_structures"]
        self._cached_lattice_is_symmetric = self.metadata.get("standardization", {}).get(
            "symmetric_lattice", True,
        )
        self._arrays, self._offsets, self._pid = {}, None, None
        self._reader()

    def __len__(self):
        return self._length

    def _reader(self):
        if self._pid != os.getpid():
            self.close()
            try:
                directory = Path(self.path) / self.split
                for name, filename in CORE_FILES.items():
                    self._arrays[name] = np.load(directory / filename, mmap_mode="r", allow_pickle=False)
                counts = self._arrays["num_atoms"]
                if counts.shape != (len(self),) or counts.dtype != np.int64 or np.any(counts <= 0):
                    raise ValueError("Invalid num_atoms in crystal cache")
                total = int(counts.sum())
                expected = {"atomic_numbers": ((total,), np.dtype("int64")),
                            "frac_coords": ((total, 3), np.dtype("float64")),
                            "lattice": ((len(self), 3, 3), np.dtype("float64"))}
                for name, (shape, dtype) in expected.items():
                    if self._arrays[name].shape != shape or self._arrays[name].dtype != dtype:
                        raise ValueError(f"Invalid {name} shape/dtype in crystal cache")
                ids = self._arrays["structure_id"]
                if ids.shape != (len(self),) or ids.dtype.kind != "U":
                    raise ValueError("Invalid structure_id in crystal cache")
                if total != self.metadata["splits"][self.split]["num_atoms"]:
                    raise ValueError("Cache atom count disagrees with metadata")
                self._offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
                self._pid = os.getpid()
            except Exception:
                self.close()
                raise
        return self._arrays

    def __getitem__(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        arrays = self._reader()
        start, stop = self._offsets[index:index + 2]
        frac = torch.from_numpy(arrays["frac_coords"][start:stop].copy()).to(self.dtype)
        # A float64 coordinate just below one can round to one in float32.
        frac.remainder_(1.0)
        lattice = torch.from_numpy(arrays["lattice"][index].copy()).to(self.dtype)
        if not self._cached_lattice_is_symmetric:
            lattice = polar_decomposition(lattice)
        return {
            "atomic_numbers": torch.from_numpy(arrays["atomic_numbers"][start:stop].copy()),
            "frac_coords": frac,
            "lattice": lattice,
            "num_atoms": int(stop - start), "structure_id": str(arrays["structure_id"][index]),
            "split": self.split,
        }

    def close(self):
        for array in self._arrays.values():
            array._mmap.close()
        self._arrays, self._offsets, self._pid = {}, None, None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_arrays={}, _offsets=None, _pid=None)
        return state

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def collate_crystals(samples: list[dict]) -> dict:
    """Concatenate atoms, stack cells, and supply atom-to-crystal indices; no padding.

    Call build_periodic_graph(batch['frac_coords'], batch['lattice'],
                              batch=batch['batch']) after moving tensors to a device
    and, when applicable, corrupting the coordinates/lattices.
    """
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    reference = samples[0]["frac_coords"]
    for sample in samples:
        n = sample["num_atoms"]
        if (n <= 0 or sample["atomic_numbers"].shape != (n,)
                or sample["frac_coords"].shape != (n, 3) or sample["lattice"].shape != (3, 3)):
            raise ValueError("Sample shapes do not agree with num_atoms")
        if (sample["atomic_numbers"].dtype != torch.long
                or any(sample[key].device != reference.device for key in ("atomic_numbers", "frac_coords", "lattice"))
                or any(sample[key].dtype != reference.dtype for key in ("frac_coords", "lattice"))):
            raise ValueError("Samples must share floating dtype/device and use int64 atomic numbers")
    counts = torch.tensor([sample["num_atoms"] for sample in samples], dtype=torch.long, device=reference.device)
    return {
        "atomic_numbers": torch.cat([sample["atomic_numbers"] for sample in samples]),
        "frac_coords": torch.cat([sample["frac_coords"] for sample in samples]),
        "lattice": torch.stack([sample["lattice"] for sample in samples]),
        "num_atoms": counts, "ptr": torch.cat((counts.new_zeros(1), counts.cumsum(0))),
        "batch": torch.repeat_interleave(torch.arange(len(samples), device=reference.device), counts),
        "num_graphs": len(samples), "structure_id": [sample["structure_id"] for sample in samples],
        "split": [sample["split"] for sample in samples],
    }
