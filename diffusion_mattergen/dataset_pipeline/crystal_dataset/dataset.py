"""Lazy per-process SQLite readers and disjoint-graph PyTorch batching."""

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .storage import connect_readonly, inspect_dataset, unpack_arrays


class CrystalDataset(Dataset):
    """Map-style dataset; only row IDs are held in memory, not all structures.

    Connections are opened lazily per process, supporting both Windows spawn
    and Unix fork DataLoader workers. Keep the database unchanged during use.
    """

    def __init__(self, path: str | Path, split: str | None = None, dtype=torch.float32):
        if split not in (None, "train", "val", "test"):
            raise ValueError("split must be None, 'train', 'val', or 'test'")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        self.path = str(Path(path).resolve())
        self.dtype = dtype
        self.split = split
        self.metadata = inspect_dataset(self.path)
        connection = connect_readonly(self.path)
        try:
            query = "SELECT id FROM samples"
            parameters = ()
            if split is not None:
                query += " WHERE split = ?"
                parameters = (split,)
            self._ids = tuple(row[0] for row in connection.execute(query + " ORDER BY id", parameters))
        finally:
            connection.close()
        self._connection = None
        self._pid = None

    def __len__(self):
        return len(self._ids)

    def _reader(self):
        if self._connection is None or self._pid != os.getpid():
            self.close()
            self._connection = connect_readonly(self.path)
            self._pid = os.getpid()
        return self._connection

    def __getitem__(self, index):
        row = self._reader().execute("SELECT * FROM samples WHERE id = ?", (self._ids[index],)).fetchone()
        if row is None:
            raise RuntimeError("Dataset changed while open; rebuild to a different path")
        sample = {}
        for name, array in unpack_arrays(row["arrays"]).items():
            tensor = torch.from_numpy(array)
            sample[name] = tensor.to(self.dtype) if tensor.is_floating_point() else tensor.long()
        for name in ("sample_id", "source_path", "structure_index", "formula", "num_atoms", "num_edges", "split"):
            sample[name] = row[name]
        return sample

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_connection"] = None
        state["_pid"] = None
        return state

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def collate_crystals(samples: list[dict]) -> dict:
    """Concatenate variable-size crystals and offset their local edge indices.

    Atom tensors are [N_total, ...], edge tensors [E_total, ...], and lattices
    [B, 3, 3]. batch maps atoms to crystals; ptr/edge_ptr delimit each sample.
    """
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    device = samples[0]["atomic_numbers"].device
    num_atoms = torch.tensor([sample["num_atoms"] for sample in samples], dtype=torch.long, device=device)
    num_edges = torch.tensor([sample["num_edges"] for sample in samples], dtype=torch.long, device=device)
    ptr = torch.cat((num_atoms.new_zeros(1), num_atoms.cumsum(0)))
    edge_ptr = torch.cat((num_edges.new_zeros(1), num_edges.cumsum(0)))
    for sample in samples:
        indices = sample["edge_index"]
        if torch.any(indices < 0) or torch.any(indices >= sample["num_atoms"]):
            raise ValueError("edge_index must contain local atom indices within each crystal")
    batch = {
        name: torch.cat([sample[name] for sample in samples], dim=0)
        for name in ("atomic_numbers", "frac_coords", "positions", "edge_images", "edge_vectors", "edge_distances")
    }
    batch.update({
        "edge_index": torch.cat([sample["edge_index"] + ptr[i] for i, sample in enumerate(samples)], dim=1),
        "lattice": torch.stack([sample["lattice"] for sample in samples]),
        "batch": torch.repeat_interleave(torch.arange(len(samples), device=device), num_atoms),
        "ptr": ptr, "edge_ptr": edge_ptr,
        "num_atoms": num_atoms, "num_edges": num_edges, "num_graphs": len(samples),
    })
    for name in ("sample_id", "source_path", "structure_index", "formula", "split"):
        batch[name] = [sample[name] for sample in samples]
    return batch
