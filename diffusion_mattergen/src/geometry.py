import math
from numbers import Integral

import torch


def _validate_batch(batch, num_nodes, num_graphs, device):
    if batch.shape != (num_nodes,) or batch.dtype != torch.long or batch.device != device:
        raise ValueError("batch must be an int64 [N_total] tensor on the coordinate device")
    if torch.any(batch < 0) or torch.any(batch >= num_graphs):
        raise ValueError("batch contains a crystal index outside the lattice array")


def _transform_coords(coords, lattices, batch):
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Coordinates must have shape [N_total, 3]")
    if lattices.shape == (3, 3):
        return coords @ lattices
    if lattices.ndim != 3 or lattices.shape[1:] != (3, 3):
        raise ValueError("lattices must have shape [3, 3] or [B, 3, 3]")
    if batch is None:
        if lattices.shape[0] != 1:
            raise ValueError("batch is required when passing multiple lattices")
        return coords @ lattices[0]
    _validate_batch(batch, coords.shape[0], lattices.shape[0], coords.device)
    return torch.einsum("ni,nij->nj", coords, lattices[batch])


def frac_to_cart(frac, lattices, batch=None):
    """Convert [N_total, 3] coordinates using one lattice or [B, 3, 3] plus batch."""
    return _transform_coords(frac, lattices, batch)


def cart_to_frac(cart, lattices, batch=None):
    """Inverse conversion for concatenated coordinates and their crystal lattices."""
    return _transform_coords(cart, torch.linalg.inv(lattices), batch)


def wrap_frac(frac):
    return frac - torch.floor(frac)


def periodic_displace(fracs, lattice):
    """Pair displacements [N, N, 3] for one crystal, using rounded fractional images.

    Component-wise rounding is retained from the original helper. Complete
    periodic radius graphs, including multiple images, are built by the dataset.
    """
    if fracs.ndim != 2 or fracs.shape[1] != 3 or lattice.shape != (3, 3):
        raise ValueError("Expected coordinates [N, 3] and one lattice [3, 3]")
    frac_displace = fracs[None, :, :] - fracs[:, None, :]
    frac_displace = frac_displace - torch.round(frac_displace)
    return frac_displace @ lattice


def periodic_distance(cart_displace):
    return torch.linalg.norm(cart_displace, dim=-1)


def build_neighbor_graph(frac, lattice, cutoff, batch=None):
    """Return concatenated indices [2, E], vectors [E, 3], distances [E].

    Pass frac [N_total, 3], lattice [B, 3, 3] and batch [N_total] for multiple
    crystals. A single crystal also accepts lattice [3, 3] without batch.
    Only within-crystal pairs are evaluated; self pairs are excluded.
    This helper uses rounded fractional images, not a complete periodic radius graph.
    """
    if frac.ndim != 2 or frac.shape[1] != 3:
        raise ValueError("frac must have shape [N_total, 3]")
    if lattice.shape == (3, 3):
        lattice = lattice[None]
    if lattice.ndim != 3 or lattice.shape[1:] != (3, 3):
        raise ValueError("lattice must have shape [3, 3] or [B, 3, 3]")
    if batch is None:
        if lattice.shape[0] != 1:
            raise ValueError("batch is required when passing multiple lattices")
        batch = torch.zeros(frac.shape[0], dtype=torch.long, device=frac.device)
    _validate_batch(batch, frac.shape[0], lattice.shape[0], frac.device)

    edge_indices, edge_vectors, edge_distances = [], [], []
    for graph in range(lattice.shape[0]):
        nodes = (batch == graph).nonzero(as_tuple=True)[0]
        displaces = periodic_displace(frac[nodes], lattice[graph])
        distances = periodic_distance(displaces)
        keep = distances < cutoff
        keep.fill_diagonal_(False)
        local_edges = keep.nonzero().T
        edge_indices.append(nodes[local_edges])
        edge_vectors.append(displaces[keep])
        edge_distances.append(distances[keep])
    if not edge_indices:
        return (torch.empty((2, 0), dtype=torch.long, device=frac.device),
                frac.new_zeros((0, 3)), frac.new_zeros(0))
    return torch.cat(edge_indices, dim=1), torch.cat(edge_vectors), torch.cat(edge_distances)


@torch.no_grad()
def _max_neighbor_mask(edge_index, distances, num_nodes, max_neighbor):
    """Select each atom's first K neighbors, including ties and reverse edges.

    Input edges must form a reciprocal radius graph. Distances within 1e-6 Å
    of the K-th neighbor belong to the same shell.
    """
    center, neighbor = edge_index
    degree = torch.bincount(center, minlength=num_nodes)
    limited = degree > max_neighbor
    if not limited.any():
        return torch.ones_like(distances, dtype=torch.bool)

    # Group by center while retaining increasing distance within each group.
    order = distances.argsort()
    order = order[center[order].argsort(stable=True)]
    starts = degree.cumsum(0) - degree
    kth = starts[limited] + max_neighbor - 1
    radii = distances.new_full((num_nodes,), float("inf"))
    radii[limited] = distances[order[kth]] + 1e-6
    # Both orientations survive when either endpoint selects this neighbor.
    return distances <= torch.maximum(radii[center], radii[neighbor])


def build_periodic_graph(frac, lattice, cutoff=7.0, batch=None, max_neighbor=50):
    """Build directed periodic edges within an inclusive Cartesian cutoff (Å).

    N is the total number of input atoms, B is the number of crystals, and E
    is the total number of returned directed edges after neighbor filtering.
    Returns:
    edge_index (int64 tensor, shape [2,E])
    edge_image (int64 tensor, shape [E,3])
    edge_vector (floating tensor, shape [E,3])
    edge_distance (floating tensor, shape [E])
    num_edges (int64 tensor, shape [B])
    edge_ptr (int64 tensor, shape [B+1])
    """
    if frac.ndim != 2 or frac.shape[1] != 3:
        raise ValueError("frac must have shape [N_total, 3]")
    if lattice.shape == (3, 3):
        lattice = lattice[None]
    if lattice.ndim != 3 or lattice.shape[1:] != (3, 3):
        raise ValueError("lattice must have shape [3, 3] or [B, 3, 3]")
    if not frac.is_floating_point() or lattice.dtype != frac.dtype or lattice.device != frac.device:
        raise ValueError("frac and lattice must have the same floating dtype and device")
    if not torch.isfinite(frac).all() or not torch.isfinite(lattice).all():
        raise ValueError("frac and lattice must be finite")
    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be finite and positive")
    if max_neighbor is not None and (isinstance(max_neighbor, bool)
                                     or not isinstance(max_neighbor, Integral) or max_neighbor < 1):
        raise ValueError("max_neighbor must be a positive integer or None")
    if batch is None:
        if lattice.shape[0] > 1:
            raise ValueError("batch is required when passing multiple lattices")
        batch = torch.zeros(frac.shape[0], dtype=torch.long, device=frac.device)
    _validate_batch(batch, frac.shape[0], lattice.shape[0], frac.device)

    periods = frac.detach().floor().long()
    wrapped = frac - periods
    edge_index_list, image_list, vector_list, distance_list = [], [], [], []
    with torch.no_grad():
        # For row-vector lattices, column k of the inverse bounds |delta_frac[k]|.
        inverse, info = torch.linalg.inv_ex(lattice.double())
        if info.any() or not torch.isfinite(inverse).all():
            raise ValueError("lattice must be nonsingular")
        limits = (cutoff * torch.linalg.vector_norm(inverse, dim=-2)).ceil().long()
    for graph_id, bounds in enumerate(limits.tolist()):
        nodes = (batch == graph_id).nonzero(as_tuple=True)[0]
        if nodes.numel() == 0:
            continue
        axes = [torch.arange(-m, m + 1, device=frac.device) for m in bounds]
        images = torch.cartesian_prod(*axes)
        coords = wrapped[nodes]
        delta = coords[None, :, :] - coords[:, None, :]
        self_pair = torch.eye(nodes.numel(), dtype=torch.bool, device=frac.device)
        # Split large image searches into blocks of atom pairs and image shifts.
        chunk_size = max(1, 65_536 // nodes.numel() ** 2)
        for shifts in images.split(chunk_size):
            vectors = (delta[:, :, None, :] + shifts) @ lattice[graph_id]
            distances = periodic_distance(vectors)
            same_site = self_pair[:, :, None] & (shifts == 0).all(dim=-1)
            center, neighbor, image_id = ((distances <= cutoff) & ~same_site).nonzero(as_tuple=True)
            i, j = nodes[center], nodes[neighbor]
            edge_index_list.append(torch.stack((i, j)))
            image_list.append(shifts[image_id] + (periods[i] - periods[j]))
            vector_list.append(vectors[center, neighbor, image_id])
            distance_list.append(distances[center, neighbor, image_id])

    edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else batch.new_empty((2, 0))
    edge_image = torch.cat(image_list) if image_list else batch.new_empty((0, 3))
    edge_vector = torch.cat(vector_list) if vector_list else frac[:0]
    edge_distance = torch.cat(distance_list) if distance_list else periodic_distance(edge_vector)
    if max_neighbor is not None:
        keep = _max_neighbor_mask(edge_index, edge_distance, frac.shape[0], max_neighbor)
        edge_index, edge_image = edge_index[:, keep], edge_image[keep]
        edge_vector, edge_distance = edge_vector[keep], edge_distance[keep]

    num_edges = torch.bincount(batch[edge_index[0]], minlength=lattice.shape[0])
    edge_ptr = torch.cat((num_edges.new_zeros(1), num_edges.cumsum(0)))
    edge_length = edge_ptr[1:] - edge_ptr[:-1]
    edge_batch = batch[edge_index[0]]
    return edge_index, edge_image, edge_vector, edge_distance, num_edges, edge_ptr, edge_batch
