import torch
from torch import nn
from .gnn import MPBlock, EdgeReadout

agg_method = MPBlock.agg_messages

class TimeEmbedding(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(1, hidden_dim),
                                      nn.SiLU(),
                                      nn.Linear(hidden_dim, hidden_dim))
    def forward(self, t_graph, batch):
        '''inputs: t_graph: (B,), batch: (N,)
           output: t_node_embed: (N, H)'''
        t_graph_embed = self.time_mlp(t_graph.unsqueeze(-1))
        t_node_embed = t_graph_embed[batch]
        return t_node_embed

class Denoiser(nn.Module):
    def __init__(self, hidden_dim=64, num_radial=32, num_layers=3, atom_types=118):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_radial = num_radial
        self.num_layers = num_layers

        self.node_embedding = nn.Embedding(atom_types+1, hidden_dim)
        self.edge_init_mlp = nn.Sequential(nn.Linear(2*hidden_dim+num_radial, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.time_embedding = TimeEmbedding(hidden_dim)

        self.num_layers = num_layers
        self.MPlayers = nn.ModuleList([MPBlock(hidden_dim, num_radial) for i in range(num_layers)])
        self.EdgeReadlayers = nn.ModuleList([EdgeReadout(hidden_dim, num_radial) for i in range(num_layers+1)])

        self.output_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, atom_types+1))

    def edge_init(self, node_features, radial_features, edge_index):
        centers = node_features[edge_index[0]]
        neighbors = node_features[edge_index[1]]
        return self.edge_init_mlp(torch.cat((centers, neighbors, radial_features), dim=-1))

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

    def forward(self, 
                atomic_numbers, 
                radial_features, 
                edge_vectors, 
                edge_distance, 
                edge_index, 
                t_graph, 
                batch,
                num_edges, 
                edge_ptr, 
                edge_batch):

        # initialize
        node_features = self.node_embedding(atomic_numbers) + self.time_embedding(t_graph, batch)
        edge_features = self.edge_init(node_features, radial_features, edge_index)
        alphas = edge_features.new_zeros((self.num_layers+1, edge_features.shape[0], 1))
        betas  = edge_features.new_zeros((self.num_layers+1, edge_features.shape[0], 1))

        for layer in range(self.num_layers):
            alphas[layer], betas[layer] = self.EdgeReadlayers[layer](edge_features, radial_features)
            node_features, edge_features = self.MPlayers[layer](node_features, edge_features, radial_features, edge_index)
        alphas[self.num_layers], betas[self.num_layers] = self.EdgeReadlayers[self.num_layers](edge_features, radial_features)
        alpha = torch.sum(alphas, dim=0)   # (E, 1)
        beta = torch.sum(betas, dim=0)     # (E, 1)

        unit_vectors, valid_edge_counts = self.unit_vec(edge_vectors, edge_distance, edge_batch, num_graph=t_graph.shape[0])
        coord_message = alpha * unit_vectors
        coord_cart = agg_method(coord_message, edge_index[0], atomic_numbers.shape[0], method='sum')
        wedge_prod = unit_vectors[:, :, None] @ unit_vectors[:, None, :]   # (E, 3, 3)
        lattice_message = beta.unsqueeze(-1) * wedge_prod
        lattice_pred = lattice_message.new_zeros((num_edges.shape[0], 3, 3))
        lattice_pred = lattice_pred.index_add(0, edge_batch, lattice_message)
        edge_normalizer = valid_edge_counts.clamp_min(1).to(lattice_message.dtype)
        lattice_pred = lattice_pred / edge_normalizer[:, None, None] 

        atom_logits = self.output_head(node_features)
        return {'final_node': node_features, 
                'final_edge': edge_features, 
                'atom_logits': atom_logits, 
                'coord_cart': coord_cart, 
                'lattice_pred': lattice_pred}
