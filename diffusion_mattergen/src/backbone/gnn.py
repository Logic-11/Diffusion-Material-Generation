import math
from numbers import Integral, Real

import torch
from torch import nn
from torch.nn import functional as F


BASIS_CHANNELS = 16
_TRIPLET_MESSAGE_CHANNELS = 64
_SKIP_SCALE = 1.0 / math.sqrt(2.0)

# Default GemNet-T calibration from MatterGen's packaged gemnet-dT.json.
_TRIPLET_RADIAL_SCALES = (18.873615264892578, 16.10817527770996, 15.01930046081543)
_TRIPLET_SUM_SCALES = (7.996850490570068, 7.614634037017822, 7.025179862976074)
_NODE_SUM_SCALES = (1.220463752746582, 0.9690994620323181, 0.8903237581253052)
_COORD_READOUT_SCALES = (
    16.161039352416992, 13.54678726196289, 12.754337310791016, 13.484951972961426,
)


def _stage_scale(scales, index):
    return scales[index] if index < len(scales) else 1.0


@torch.no_grad()
def _he_orthogonal_init(weight):
    """Match GemNet's variance standardization for linear and bilinear weights."""
    nn.init.orthogonal_(weight)
    axes = (0, 1) if weight.ndim == 3 else (1,)
    fan_in = math.prod(weight.shape[axis] for axis in axes)
    if fan_in > 1:
        variance, mean = torch.var_mean(weight, dim=axes, keepdim=True)
        weight.sub_(mean).mul_(torch.rsqrt((variance + 1e-6) * fan_in))


class Dense(nn.Linear):
    def __init__(self, in_features, out_features, *, activated=False):
        super().__init__(in_features, out_features, bias=False)
        self.activated = activated

    def reset_parameters(self):
        _he_orthogonal_init(self.weight)

    def forward(self, features):
        projected = super().forward(features)
        return F.silu(projected) * (1.0 / 0.6) if self.activated else projected


class ResidualLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            Dense(hidden_dim, hidden_dim, activated=True),
            Dense(hidden_dim, hidden_dim, activated=True),
        )

    def forward(self, features):
        return (features + self.layers(features)) * _SKIP_SCALE


class EdgeEmbedding(nn.Module):
    def __init__(self, hidden_dim, edge_dim):
        super().__init__()
        self.dense = Dense(2 * hidden_dim + edge_dim, hidden_dim, activated=True)

    def forward(self, node_features, edge_features, edge_index):
        center, neighbor = edge_index
        # GemNet concatenates the source (neighbor) before the target (center).
        features = torch.cat((node_features[neighbor], node_features[center], edge_features), dim=-1)
        return self.dense(features)


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
    """Read coordinate and lattice edge coefficients from separate radial projections."""

    def __init__(self, hidden_dim=64, *, stage_index=0):
        super().__init__()
        self.coord_mlp = nn.Sequential(
            Dense(hidden_dim, hidden_dim, activated=True),
            *(ResidualLayer(hidden_dim) for _ in range(3)),
        )
        self.lattice_mlp = nn.Sequential(
            Dense(hidden_dim, hidden_dim, activated=True),
            Dense(hidden_dim, hidden_dim),
        )
        self.coord_radial = Dense(BASIS_CHANNELS, hidden_dim)
        self.lattice_radial = Dense(BASIS_CHANNELS, hidden_dim)
        self.coord_output = Dense(hidden_dim, 1)
        self.lattice_output = Dense(hidden_dim, 1)
        self.register_buffer(
            "coord_scale", torch.tensor(_stage_scale(_COORD_READOUT_SCALES, stage_index)),
        )

    def forward(self, edge_features, coord_rbf, lattice_rbf):
        coord_features = self.coord_mlp(edge_features) * self.coord_radial(coord_rbf)
        lattice_features = self.lattice_mlp(edge_features) * self.lattice_radial(lattice_rbf)
        return (self.coord_output(coord_features * self.coord_scale),
                self.lattice_output(lattice_features))


class CosineBasis(nn.Module):
    """Normalized real spherical harmonics Y_l^0 evaluated at cos(phi)."""

    def __init__(self, num_spherical=7):
        super().__init__()
        if (isinstance(num_spherical, bool)
                or not isinstance(num_spherical, Integral)
                or num_spherical < 2):
            raise ValueError("num_spherical must be an integer of at least two")
        self.num_spherical = int(num_spherical)
        degrees = torch.arange(self.num_spherical, dtype=torch.float64)
        normalization = ((2 * degrees + 1) / (4 * math.pi)).sqrt()
        self.register_buffer("normalization", normalization, persistent=False)

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
        return torch.stack(polynomials, dim=-1) * self.normalization.to(cos_phi)


class TripletBasis(nn.Module):
    """Shared projections for the sender RBF and receiver circular basis."""

    def __init__(self, num_radial, num_spherical):
        super().__init__()
        self.cosine_basis = CosineBasis(num_spherical)
        self.sender_radial = Dense(num_radial, BASIS_CHANNELS)
        self.receiver_radial = nn.Linear(
            num_radial, BASIS_CHANNELS * self.cosine_basis.num_spherical, bias=False,
        )
        # Initialize in GemNet's [harmonic, radial, basis] layout before flattening.
        weight = self.receiver_radial.weight.new_empty(
            (self.cosine_basis.num_spherical, num_radial, BASIS_CHANNELS),
        )
        _he_orthogonal_init(weight)
        with torch.no_grad():
            self.receiver_radial.weight.copy_(weight.permute(2, 0, 1).flatten(0, 1))

    def forward(self, radial_features, cos_phi):
        sender_radial = self.sender_radial(radial_features)
        # Each harmonic has its own receiver-distance projection.
        receiver_radial = self.receiver_radial(radial_features).unflatten(
            -1, (BASIS_CHANNELS, self.cosine_basis.num_spherical),
        )
        harmonics = self.cosine_basis(cos_phi).to(sender_radial)
        return sender_radial, receiver_radial, harmonics


class TripletInteraction(nn.Module):
    """GemNet-T distance-angle interaction with reverse-edge exchange."""

    def __init__(self, hidden_dim, *, radial_scale=1.0, sum_scale=1.0):
        super().__init__()
        self.sender_edge = Dense(hidden_dim, hidden_dim, activated=True)
        self.radial_gate = Dense(BASIS_CHANNELS, hidden_dim)
        self.down_projection = Dense(hidden_dim, _TRIPLET_MESSAGE_CHANNELS, activated=True)
        self.bilinear = nn.Linear(
            BASIS_CHANNELS * _TRIPLET_MESSAGE_CHANNELS,
            _TRIPLET_MESSAGE_CHANNELS,
            bias=False,
        )
        # GemNet's unflattened order is [message, basis, output].
        weight = self.bilinear.weight.new_empty(
            (_TRIPLET_MESSAGE_CHANNELS, BASIS_CHANNELS, _TRIPLET_MESSAGE_CHANNELS),
        )
        _he_orthogonal_init(weight)
        with torch.no_grad():
            self.bilinear.weight.copy_(weight.permute(2, 1, 0).flatten(1))
        self.up_projection = Dense(_TRIPLET_MESSAGE_CHANNELS, hidden_dim, activated=True)
        self.reverse_up_projection = Dense(_TRIPLET_MESSAGE_CHANNELS, hidden_dim, activated=True)
        self.register_buffer("radial_scale", torch.tensor(radial_scale))
        self.register_buffer("sum_scale", torch.tensor(sum_scale))

    def forward(self, edge_features, id3_center_b, id3_center_c, triplet_basis, id_swap):
        if id3_center_b.numel() == 0:
            return edge_features.new_zeros(edge_features.shape)
        sender_radial, receiver_radial, harmonics = triplet_basis
        sender_edges = self.sender_edge(edge_features)
        gated = sender_edges * self.radial_gate(sender_radial) * self.radial_scale
        sender_messages = self.down_projection(gated)
        triplet_messages = sender_messages[id3_center_b]

        # Sum sender messages against each harmonic at the receiving edge.
        # This preserves the GemNet-T contraction without a padded neighbor axis.
        angular_sum = torch.stack(
            [
                sender_messages.new_zeros(sender_messages.shape).index_add_(
                    0, id3_center_c, triplet_messages * harmonic[:, None],
                )
                for harmonic in harmonics.unbind(dim=1)
            ],
            dim=1,
        )
        # [E, 16, S] @ [E, S, 64]. The bilinear layer mixes all 16 x 64 pairs.
        combined = torch.bmm(receiver_radial, angular_sum)
        message = self.bilinear(combined.flatten(1)) * self.sum_scale
        direct = self.up_projection(message)
        reverse = self.reverse_up_projection(message)
        return (direct + reverse[id_swap]) * _SKIP_SCALE


class MPBlock(nn.Module):
    """GemNet-T edge update, atom aggregation, and atom-to-edge feedback."""

    def __init__(self, hidden_dim=64, *, use_triplets=False, block_index=0):
        super().__init__()
        self.edge_skip = Dense(hidden_dim, hidden_dim, activated=True)
        self.triplet_interaction = (
            TripletInteraction(
                hidden_dim,
                radial_scale=_stage_scale(_TRIPLET_RADIAL_SCALES, block_index),
                sum_scale=_stage_scale(_TRIPLET_SUM_SCALES, block_index),
            ) if use_triplets else None
        )
        self.before_skip = ResidualLayer(hidden_dim)
        self.after_skip = nn.Sequential(ResidualLayer(hidden_dim), ResidualLayer(hidden_dim))
        self.node_radial = Dense(BASIS_CHANNELS, hidden_dim)
        self.node_layers = nn.Sequential(
            Dense(hidden_dim, hidden_dim, activated=True),
            *(ResidualLayer(hidden_dim) for _ in range(3)),
        )
        self.register_buffer(
            "node_sum_scale", torch.tensor(_stage_scale(_NODE_SUM_SCALES, block_index)),
        )
        self.edge_feedback = EdgeEmbedding(hidden_dim, hidden_dim)
        self.feedback_residual = ResidualLayer(hidden_dim)

    @staticmethod
    def agg_messages(messages, centers, num_nodes, method="sum"):
        """Aggregate incoming messages, retaining isolated nodes and empty graphs."""
        total = messages.new_zeros((num_nodes, messages.shape[-1]))
        total = total.index_add(0, centers, messages)
        if method == "sum":
            return total
        if method == "mean":
            degree = torch.bincount(centers, minlength=num_nodes).to(messages.dtype)[:, None]
            return total / degree.clamp_min(1)
        raise ValueError("method must be 'sum' or 'mean'")

    def forward(self, node_features, edge_features, node_rbf, edge_index, *,
                id3_center_b=None, id3_center_c=None, triplet_basis=None, id_swap=None):
        edge_update = self.edge_skip(edge_features)
        if self.triplet_interaction is not None:
            if any(value is None for value in
                   (id3_center_b, id3_center_c, triplet_basis, id_swap)):
                raise ValueError(
                    "triplet edge IDs, projected basis and id_swap are required when use_triplets=True"
                )
            triplet_update = self.triplet_interaction(
                edge_features, id3_center_b, id3_center_c, triplet_basis, id_swap,
            )
            edge_update = (edge_update + triplet_update) * _SKIP_SCALE
        edge_update = self.before_skip(edge_update)
        edge_features = self.after_skip((edge_features + edge_update) * _SKIP_SCALE)

        gated_edges = edge_features * self.node_radial(node_rbf)
        incoming = self.agg_messages(gated_edges, edge_index[0], node_features.shape[0])
        node_update = self.node_layers(incoming * self.node_sum_scale)
        node_features = (node_features + node_update) * _SKIP_SCALE

        feedback = self.edge_feedback(node_features, edge_features, edge_index)
        edge_features = (edge_features + self.feedback_residual(feedback)) * _SKIP_SCALE
        return node_features, edge_features
