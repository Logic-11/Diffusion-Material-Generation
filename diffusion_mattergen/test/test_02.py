import torch
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.geometry as geometry

L = torch.tensor([[10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=torch.float32)
F = torch.tensor([[0.98, 0, 0], [0.02, 0, 0], [0.40, 0, 0.6], [0.55, 0, 0.3]], dtype=torch.float32)

edge_index, edge_vec, edge_dist = geometry.build_neighbor_graph(F, L, cutoff=5)
assert edge_index.shape[0] == 2
torch.testing.assert_close(edge_vec.norm(dim=-1), edge_dist)
print(edge_index)
print(edge_vec)
print(edge_dist)

