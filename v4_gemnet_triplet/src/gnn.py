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


class CosineBasis(nn.Module):
    """[GemNet-T triplet] Zero-order real spherical-harmonic angle basis.

    Y_l0(cos(phi)) is proportional to P_l(cos(phi)). The unnormalised
    Legendre polynomials below span the same learned angular function space.
    """

    def __init__(self, num_spherical=7):
        super().__init__()
        if (isinstance(num_spherical, bool)
                or not isinstance(num_spherical, Integral)
                or num_spherical < 2):
            raise ValueError("num_spherical must be an integer of at least two")
        self.num_spherical = int(num_spherical)

    def forward(self, cos_phi):
        if cos_phi.ndim != 1 or not cos_phi.is_floating_point():
            raise ValueError("cos_phi must be a floating [T] tensor")
        if (not torch.isfinite(cos_phi).all()
                or torch.any(cos_phi < -1.0) or torch.any(cos_phi > 1.0)):
            raise ValueError("cos_phi must be finite and lie in [-1,1]")
        polynomials = [torch.ones_like(cos_phi), cos_phi]
        for degree in range(2, self.num_spherical):
            polynomials.append(
                (
                    (2 * degree - 1) * cos_phi * polynomials[-1]
                    - (degree - 1) * polynomials[-2]
                ) / degree
            )
        return torch.stack(polynomials[:self.num_spherical], dim=-1)


class TripletInteraction(nn.Module):
    """[GemNet-T triplet] One-hop angular edge-to-edge message passing."""

    def __init__(self, hidden_dim=64, num_radial=32, num_spherical=7):
        super().__init__()
        self.cosine_basis = CosineBasis(num_spherical)
        self.radial_mlp = nn.Linear(num_radial, hidden_dim, bias=False)
        self.neighbor_mlp = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.angle_mlp = nn.Linear(num_spherical, hidden_dim, bias=False)
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        edge_features,
        radial_features,
        id3_center_b,
        id3_center_c,
        cos_phi,
    ):
        if edge_features.shape[0] == 0 or id3_center_b.numel() == 0:
            return edge_features.new_zeros(edge_features.shape)
        neighbor_message = (
            self.neighbor_mlp(edge_features[id3_center_b])
            * self.radial_mlp(radial_features[id3_center_b])
        )
        angle_message = self.angle_mlp(self.cosine_basis(cos_phi))
        triplet_message = neighbor_message * angle_message
        aggregated = edge_features.new_zeros(edge_features.shape)
        aggregated = aggregated.index_add(0, id3_center_c, triplet_message)
        return self.output_mlp(aggregated)


class MPBlock(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        num_radial=32,
        num_spherical=7,
        use_triplets=False,
    ):
        super().__init__()
        # [GemNet-T triplet] Optional one-hop three-body interaction.
        self.use_triplets = use_triplets
        self.triplet_interaction = (
            TripletInteraction(hidden_dim, num_radial, num_spherical)
            if use_triplets else None
        )
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
                edge_index,
                id3_center_b=None,
                id3_center_c=None,
                cos_phi=None):
        """Nodes [N_total, H], edges [E_total, D], indices [2, E_total]."""
        edge_features_res = edge_features
        node_features_res = node_features
        edge_message = self.edge_update(
            node_features, edge_features, radial_features, edge_index
        )
        if self.use_triplets:
            if any(value is None for value in
                   (id3_center_b, id3_center_c, cos_phi)):
                raise ValueError(
                    "triplet edge ids and cos_phi are required when use_triplets=True"
                )
            edge_message = edge_message + self.triplet_interaction(
                edge_features,
                radial_features,
                id3_center_b,
                id3_center_c,
                cos_phi,
            )
        edge_message = edge_message + edge_features_res
        message_weights = self.weight_mlp(radial_features)
        weighted_edge = edge_message * message_weights
        agg = self.agg_messages(weighted_edge, edge_index[0], node_features.shape[0], method='sum')
        node_features = self.node_update(node_features, agg) + node_features_res
        edge_features = edge_message + self.edge_feedback(node_features, edge_message, edge_index)
        return node_features, edge_features
    
