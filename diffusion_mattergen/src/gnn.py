import math
from numbers import Integral, Real

import torch
from torch import nn

from .geometry import build_neighbor_graph


class RadialBasis(nn.Module):
    """Expand Cartesian distances in angstroms into fixed Gaussian features.

    Forward accepts floating distances [E] and returns features [E, num_radial].
    Distances must be finite and nonnegative. Values at or beyond cutoff
    produce zero features. Empty inputs return [0, num_radial].
    """

    def __init__(self, num_radial=32, cutoff=7.0):
        super().__init__()
        if isinstance(num_radial, bool) or not isinstance(num_radial, Integral) or num_radial < 2:
            raise ValueError("num_radial must be an integer of at least two")
        if isinstance(cutoff, bool) or not isinstance(cutoff, Real) or not math.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("cutoff must be finite and positive")

        self.num_radial = int(num_radial)
        self.cutoff = float(cutoff)
        self.register_buffer("centers", torch.linspace(0.0, 1.0, self.num_radial))
        self.register_buffer("width", torch.tensor(1.0 / (self.num_radial - 1)))

    def forward(self, distances):
        if distances.ndim != 1 or not distances.is_floating_point():
            raise ValueError("distances must be a floating tensor with shape [E]")
        if distances.device != self.centers.device or distances.dtype != self.centers.dtype:
            raise ValueError("distances and RadialBasis buffers must have the same dtype and device; use .to(...)")
        if not torch.isfinite(distances).all() or torch.any(distances < 0):
            raise ValueError("distances must be finite and nonnegative")

        # Clamp before evaluating the basis to keep distant edges numerically safe.
        scaled = (distances / self.cutoff).clamp(max=1.0)
        offsets = (scaled[:, None] - self.centers) / self.width
        gaussians = torch.exp(-0.5 * offsets.square())

        # Factor 1 - 21*x**5 + 35*x**6 - 15*x**7 to avoid cancellation near x=1.
        polynomial = 1.0 + scaled * (3.0 + scaled * (6.0 + scaled * (10.0 + 15.0 * scaled)))
        envelope = (1.0 - scaled).pow(3) * polynomial
        return envelope[:, None] * gaussians


class EdgeReadout(nn.Module):
    """Read independent coordinate and lattice coefficients from periodic edges.

    Inputs are edge_features [E,H] and radial_features [E,R]. Each returned
    coefficient has shape [E,1] and can take either sign.
    """

    def __init__(self, hidden_dim=64, num_radial=32):
        super().__init__()
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.lattice_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coord_radial = nn.Linear(num_radial, hidden_dim, bias=False)
        self.lattice_radial = nn.Linear(num_radial, hidden_dim, bias=False)
        self.coord_output = nn.Linear(hidden_dim, 1, bias=False)
        self.lattice_output = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, edge_features, radial_features):
        coord_features = self.coord_mlp(edge_features) * self.coord_radial(radial_features)
        lattice_features = self.lattice_mlp(edge_features) * self.lattice_radial(radial_features)
        return self.coord_output(coord_features), self.lattice_output(lattice_features)


class MPBlock(nn.Module):
    def __init__(self, hidden_dim=64, num_radial=32):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim + num_radial, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), 
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.weight_mlp = nn.Linear(num_radial, hidden_dim, bias=False)
        self.edge_feedback_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def edge_update(self, node_features, edge_features, radial_features, edge_index):
        centers = node_features[edge_index[0]]
        neighbors = node_features[edge_index[1]]
        return self.edge_mlp(torch.cat((centers, neighbors, edge_features, radial_features), dim=-1))

    def node_update(self, node_features, agg):
        return self.node_mlp(torch.cat((node_features, agg), dim=-1))

    def edge_feedback(self, node_features, edge_message, edge_index):
        centers = node_features[edge_index[0]]
        neighbors = node_features[edge_index[1]]
        return self.edge_feedback_mlp( torch.cat([centers, neighbors, edge_message], dim=-1) )

    @staticmethod
    def agg_messages(messages, centers, num_nodes, method='sum'):
        """Mean incoming messages, retaining isolated nodes and empty graphs."""
        total = messages.new_zeros((num_nodes, messages.shape[-1]))
        total = total.index_add(0, centers, messages)
        if method=='sum':
            return total
        elif method=='mean':
            degree = torch.bincount(centers, minlength=num_nodes).to(messages.dtype)[:, None]
            return total / degree.clamp_min(1)
        else:
            raise ValueError("method must be 'sum' or 'mean'")

    def forward(self, 
                node_features, 
                edge_features, 
                radial_features, 
                edge_index):
        """Nodes [N_total, H], edges [E_total, D], indices [2, E_total]."""
        edge_features_res = edge_features
        node_features_res = node_features
        edge_message = self.edge_update(node_features, edge_features, radial_features, edge_index) + edge_features_res
        message_weights = self.weight_mlp(radial_features)
        weighted_edge = edge_message * message_weights
        agg = self.agg_messages(weighted_edge, edge_index[0], node_features.shape[0], method='sum')
        node_features = self.node_update(node_features, agg) + node_features_res
        edge_features = edge_message + self.edge_feedback(node_features, edge_message, edge_index)
        return node_features, edge_features
    
