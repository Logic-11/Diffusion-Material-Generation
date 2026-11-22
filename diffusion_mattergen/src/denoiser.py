import math

import torch
from torch import nn

from .geometry import reverse_edge_ids
from .gnn import BASIS_CHANNELS, Dense, EdgeEmbedding, EdgeReadout, MPBlock, TripletBasis


class TimeEmbedding(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        div_term = torch.exp( torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim) )
        self.register_buffer("div_term", div_term)

    def forward(self, t_graph, batch):
        angles = t_graph[:, None] * self.div_term
        embedding = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)
        return embedding[:, :self.hidden_dim][batch]


class Denoiser(nn.Module):
    atom_class_order = "mask_last"

    def __init__(
        self,
        hidden_dim=64,
        num_radial=32,
        num_layers=3,
        atom_types=118,
        num_spherical=7,
        use_triplets=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_radial = num_radial
        self.num_layers = num_layers
        self.num_spherical = num_spherical
        self.use_triplets = use_triplets
        self.triplet_basis = (
            TripletBasis(num_radial, num_spherical) if use_triplets and num_layers > 0 else None
        )

        self.node_embedding = nn.Embedding(atom_types + 1, hidden_dim)
        nn.init.uniform_(self.node_embedding.weight, -math.sqrt(3.0), math.sqrt(3.0))
        self.edge_embedding = EdgeEmbedding(hidden_dim, num_radial)
        self.angle_edge_emb = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embedding = TimeEmbedding(hidden_dim)
        self.atom_latent_emb = nn.Linear(2 * hidden_dim, hidden_dim)

        self.node_radial = Dense(num_radial, BASIS_CHANNELS) if num_layers > 0 else None
        self.coord_radial = Dense(num_radial, BASIS_CHANNELS)
        self.lattice_radial = Dense(num_radial, BASIS_CHANNELS)
        self.MPlayers = nn.ModuleList(
            MPBlock(hidden_dim, use_triplets=use_triplets, block_index=index)
            for index in range(num_layers)
        )
        self.EdgeReadlayers = nn.ModuleList(
            EdgeReadout(hidden_dim, stage_index=index) for index in range(num_layers + 1)
        )
        self.output_head = nn.Linear(hidden_dim, atom_types + 1)

    @staticmethod
    def unit_vec(vec, dist, batch=None, num_graph=None):
        valid = dist > 0
        valid = valid.unsqueeze(-1)
        dist_safe = torch.where(valid, dist.unsqueeze(-1), 1)
        unit = vec / dist_safe
        if batch is None:
            return torch.where(valid, unit, 0)
        else:
            valid_count = torch.bincount(batch, weights=valid.squeeze(-1).float(), minlength=num_graph)
            return torch.where(valid, unit, 0), valid_count

    def forward(self, atomic_numbers, radial_features, edge_vectors, edge_distance,
                edge_index, t_graph, batch, num_edges, edge_ptr, edge_batch,
                id3_center_b=None, id3_center_c=None, cos_phi=None, *, edge_lattice_cosines):
        # Convert the diffusion state's MASK=0 and atomic numbers to GemNet's embedding rows.
        atom_indices = torch.where(
            atomic_numbers == 0, self.node_embedding.num_embeddings - 1, atomic_numbers - 1,
        )
        atom_features = self.node_embedding(atom_indices)
        time_features = self.time_embedding(t_graph, batch).to(dtype=atom_features.dtype)
        node_features = self.atom_latent_emb(torch.cat((atom_features, time_features), dim=-1))
        edge_features = self.edge_embedding(node_features, radial_features, edge_index)
        edge_features = self.angle_edge_emb(
            torch.cat((edge_features, edge_lattice_cosines.to(edge_features)), dim=-1)
        )

        triplet_basis = None
        id_swap = None
        if self.triplet_basis is not None:
            if cos_phi is None:
                raise ValueError("cos_phi is required when use_triplets=True")
            triplet_basis = self.triplet_basis(radial_features, cos_phi)
            id_swap = reverse_edge_ids(num_edges, edge_ptr, edge_batch)

        node_rbf = self.node_radial(radial_features) if self.node_radial is not None else None
        coord_rbf = self.coord_radial(radial_features)
        lattice_rbf = self.lattice_radial(radial_features)
        coord_scores, lattice_scores = self.EdgeReadlayers[0](edge_features, coord_rbf, lattice_rbf)
        for stage_index, block in enumerate(self.MPlayers, start=1):
            node_features, edge_features = block(
                node_features,
                edge_features,
                node_rbf,
                edge_index,
                id3_center_b=id3_center_b,
                id3_center_c=id3_center_c,
                triplet_basis=triplet_basis,
                id_swap=id_swap,
            )
            coord_update, lattice_update = self.EdgeReadlayers[stage_index](
                edge_features, coord_rbf, lattice_rbf,
            )
            coord_scores = coord_scores + coord_update
            lattice_scores = lattice_scores + lattice_update

        # GemNet's unit vectors point from the neighbor image to the center.
        unit_vectors, valid_edge_counts = self.unit_vec(
            -edge_vectors, edge_distance, edge_batch, num_graph=t_graph.shape[0],
        )
        coord_cart = MPBlock.agg_messages(
            coord_scores * unit_vectors, edge_index[0], atomic_numbers.shape[0],
        )
        outer_products = unit_vectors[:, :, None] @ unit_vectors[:, None, :]
        lattice_message = lattice_scores.unsqueeze(-1) * outer_products
        lattice_pred = lattice_message.new_zeros((num_edges.shape[0], 3, 3))
        lattice_pred = lattice_pred.index_add(0, edge_batch, lattice_message)
        edge_normalizer = valid_edge_counts.clamp_min(1).to(lattice_message.dtype)
        lattice_pred = lattice_pred / edge_normalizer[:, None, None]

        return {
            "final_node": node_features,
            "final_edge": edge_features,
            "atom_logits": self.output_head(node_features),
            "coord_cart": coord_cart,
            "lattice_pred": lattice_pred,
        }
