import math
import torch
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src.gnn as gnn

edge_index = torch.tensor([[0, 2, 1, 1], [1, 1, 0, 2]], dtype=torch.long)
edge_features = torch.tensor([[1], [4], [1], [4]], dtype=torch.float32)
X = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 1]], dtype=torch.float32)

R = torch.tensor([[0.5, -0.5*math.sqrt(3), 0], [0.5*math.sqrt(3), 0.5, 0], [0, 0, 1]], dtype=torch.float32)
X_rotated = X @ R.T

hidden_dim = 64
node_features = torch.randn(3, hidden_dim)

model = gnn.MPBlock(hidden_dim=hidden_dim)

def calc_model(_X):
    centers = _X[edge_index[0]]
    neighbors = _X[edge_index[1]]
    edge_vectors = neighbors - centers
    return model(node_features, edge_features, edge_vectors, edge_index)

node, vec, edge = calc_model(X)
node_rot, vec_rot, edge_rot = calc_model(X_rotated)
torch.testing.assert_close(node, node_rot)
torch.testing.assert_close(vec @ R.T, vec_rot)
print(torch.allclose(node, node_rot), torch.allclose(vec @ R.T, vec_rot))
