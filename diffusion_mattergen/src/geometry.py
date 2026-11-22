import math
from numbers import Integral

import torch


_GRAPH_CHUNK_SIZE = 65_536


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


def polar_decomposition(lattice):
    """Return the symmetric left polar factor for row-vector lattices.

    Accept [3, 3] or [B, 3, 3], preserving shape, dtype and device.
    For L = U diag(s) V.T, return P = U diag(s) U.T, so P P.T = L L.T."""
    if lattice.ndim not in (2, 3) or lattice.shape[-2:] != (3, 3):
        raise ValueError("lattice must have shape [3, 3] or [B, 3, 3]")
    if not lattice.is_floating_point() or not torch.isfinite(lattice).all():
        raise ValueError("lattice must be a finite floating-point tensor")

    axes, singular_values, _ = torch.linalg.svd(lattice, full_matrices=False)
    symmetric = (axes * singular_values.unsqueeze(-2)) @ axes.transpose(-1, -2)
    # Remove floating-point asymmetry without changing the polar factor.
    return 0.5 * (symmetric + symmetric.transpose(-1, -2))


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
def _max_neighbor_mask(center, distance_squared, max_neighbor):
    """Select the closest K directed candidates per center, in their input order."""
    keep = torch.ones_like(center, dtype=torch.bool)
    if max_neighbor is None or center.numel() == 0:
        return keep

    # Candidates for each center are contiguous, including for interleaved batches.
    _, counts = torch.unique_consecutive(center, return_counts=True)
    max_degree = int(counts.max().item())
    if max_degree <= max_neighbor:
        return keep

    starts = counts.cumsum(0) - counts
    limited = counts > max_neighbor
    columns = torch.arange(max_degree, device=center.device)
    rows_per_chunk = max(1, _GRAPH_CHUNK_SIZE // max_degree)
    # Use the same padded row width and torch.sort tie handling as MatterGen.
    for row_starts, row_counts in zip(starts[limited].split(rows_per_chunk),
                                      counts[limited].split(rows_per_chunk)):
        edge_ids = row_starts[:, None] + columns
        valid = columns < row_counts[:, None]
        distances = distance_squared.new_full(edge_ids.shape, torch.inf)
        distances[valid] = distance_squared[edge_ids[valid]]
        discarded = distances.sort(dim=1).indices[:, max_neighbor:]
        discarded_ids = row_starts[:, None] + discarded
        keep[discarded_ids[discarded < row_counts[:, None]]] = False
    return keep


@torch.no_grad()
def _periodic_candidates(cart, lattice, batch, cutoff, image_cap):
    """Enumerate candidates in MatterGen's center, neighbor, then image order."""
    inverse, info = torch.linalg.inv_ex(lattice.double())
    if info.any() or not torch.isfinite(inverse).all():
        raise ValueError("lattice must be nonsingular")
    # Bound |delta_frac[k]| by cutoff times the norm of inverse column k.
    limits = (cutoff * torch.linalg.vector_norm(inverse, dim=-2)).ceil()
    limits = limits.clamp_max(image_cap).long()

    indices, shifts, squared_distances = [], [], []
    cutoff_squared = cutoff * cutoff
    for graph_id, bounds in enumerate(limits.tolist()):
        nodes = (batch == graph_id).nonzero(as_tuple=True)[0]
        num_nodes = nodes.numel()
        if num_nodes == 0:
            continue
        axes = [torch.arange(-m, m + 1, device=batch.device) for m in bounds]
        images = torch.cartesian_prod(*axes)
        offsets = images.to(cart) @ lattice[graph_id]
        num_pairs = num_nodes * num_nodes
        pairs_per_chunk = max(1, _GRAPH_CHUNK_SIZE // images.shape[0])
        for start in range(0, num_pairs, pairs_per_chunk):
            pairs = torch.arange(start, min(start + pairs_per_chunk, num_pairs), device=batch.device)
            centers = nodes[pairs // num_nodes]
            neighbors = nodes[pairs % num_nodes]
            neighbor_positions = cart[neighbors, None, :] + offsets
            distance_squared = (cart[centers, None, :] - neighbor_positions).square().sum(dim=-1)
            valid = (distance_squared > 0.0001) & (distance_squared <= cutoff_squared)
            pair, image = valid.nonzero(as_tuple=True)
            indices.append(torch.stack((centers[pair], neighbors[pair])))
            shifts.append(images[image])
            squared_distances.append(distance_squared[pair, image])

    if not indices:
        return batch.new_empty((2, 0)), batch.new_empty((0, 3)), cart.new_empty(0)
    return torch.cat(indices, dim=1), torch.cat(shifts), torch.cat(squared_distances)


def build_periodic_graph(frac, lattice, cutoff=7.0, batch=None, max_neighbor=50,
                         max_cell_images_per_dim=5):
    """Build MatterGen's periodic radius graph for 0.01 Å < distance <= cutoff.

    Each center first selects up to max_neighbor closest candidates (all if
    None). A canonical half of those edges is mirrored to form the final graph,
    so final degrees can exceed max_neighbor. Each wrapped image shift is
    capped at max_cell_images_per_dim along each axis.

    Returns edge_index [2,E] (center, neighbor), edge_image [E,3] (shift of
    the neighbor relative to the input coordinates), edge_vector [E,3] from
    center to neighbor image, edge_distance [E], num_edges [B], edge_ptr
    [B+1], and edge_batch [E]. Each crystal's canonical edges are followed by
    their reverses in matching order.
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
    if (isinstance(max_cell_images_per_dim, bool)
            or not isinstance(max_cell_images_per_dim, Integral) or max_cell_images_per_dim < 1):
        raise ValueError("max_cell_images_per_dim must be a positive integer")
    if batch is None:
        if lattice.shape[0] > 1:
            raise ValueError("batch is required when passing multiple lattices")
        batch = torch.zeros(frac.shape[0], dtype=torch.long, device=frac.device)
    _validate_batch(batch, frac.shape[0], lattice.shape[0], frac.device)

    periods = frac.detach().floor().long()
    cart = frac_to_cart(frac - periods, lattice, batch)
    candidates, shifts, distance_squared = _periodic_candidates(
        cart, lattice, batch, cutoff, max_cell_images_per_dim)
    center, neighbor = candidates
    keep = _max_neighbor_mask(center, distance_squared, max_neighbor)

    # MatterGen stores (source, target). Locally the source is the neighbor.
    x, y, z = shifts.unbind(dim=1)
    earlier_image = (x < 0) | ((x == 0) & (y < 0)) | ((x == 0) & (y == 0) & (z < 0))
    keep &= (neighbor < center) | ((neighbor == center) & earlier_image)
    half_index = candidates[:, keep]
    center, neighbor = half_index
    shifts = shifts[keep]
    half_batch = batch[center]
    half_image = shifts + (periods[center] - periods[neighbor])
    half_vector = cart[neighbor] - cart[center] + frac_to_cart(shifts.to(frac), lattice, half_batch)

    # Keep the two halves adjacent within each crystal, in the same edge order.
    edge_batch = half_batch.repeat(2)
    order = edge_batch.argsort(stable=True)
    edge_index = torch.cat((half_index, half_index.flip([0])), dim=1)[:, order]
    edge_image = torch.cat((half_image, -half_image))[order]
    edge_vector = torch.cat((half_vector, -half_vector))[order]
    edge_distance = periodic_distance(edge_vector)
    edge_batch = edge_batch[order]
    num_edges = torch.bincount(edge_batch, minlength=lattice.shape[0])
    edge_ptr = torch.cat((num_edges.new_zeros(1), num_edges.cumsum(0)))
    return edge_index, edge_image, edge_vector, edge_distance, num_edges, edge_ptr, edge_batch


def reverse_edge_ids(num_edges, edge_ptr, edge_batch):
    """Map each edge to its paired reverse in a graph from build_periodic_graph."""
    edge_ids = torch.arange(edge_batch.numel(), device=edge_batch.device)
    half_sizes = (num_edges // 2)[edge_batch]
    # Each crystal stores its canonical half and its matched reverse half.
    in_first_half = edge_ids < edge_ptr[edge_batch] + half_sizes
    return edge_ids + torch.where(in_first_half, half_sizes, -half_sizes)


def build_triplet_graph(edge_index, edge_vectors, edge_distance, num_nodes):
    """Build all ordered pairs of distinct edges sharing a center.

    For each edge ``a -> c``, every other edge ``a -> b`` defines an angle at
    atom ``a``. Distinct periodic images have distinct edge IDs.
    """
    if (edge_index.ndim != 2 or edge_index.shape[0] != 2
            or edge_index.dtype != torch.long):
        raise ValueError("edge_index must be an int64 tensor with shape [2,E]")
    num_edges = edge_index.shape[1]
    if (edge_vectors.ndim != 2 or edge_vectors.shape != (num_edges, 3)
            or not edge_vectors.is_floating_point()):
        raise ValueError("edge_vectors must be a floating tensor with shape [E,3]")
    if (edge_distance.shape != (num_edges,)
            or edge_distance.dtype != edge_vectors.dtype
            or edge_distance.device != edge_vectors.device):
        raise ValueError("edge_distance must match edge_vectors dtype/device")
    if not isinstance(num_nodes, Integral) or num_nodes < 0:
        raise ValueError("num_nodes must be a nonnegative integer")

    if num_edges == 0:
        empty = edge_index.new_empty(0)
        return empty, empty, edge_vectors.new_empty(0)

    centers = edge_index[0]
    if (torch.any(centers < 0) or torch.any(centers >= num_nodes)
            or torch.any(edge_index[1] < 0)
            or torch.any(edge_index[1] >= num_nodes)):
        raise ValueError("edge_index contains an atom outside 0..num_nodes-1")

    edges_by_center = centers.argsort(stable=True)
    sorted_centers = centers[edges_by_center]
    degree = torch.bincount(centers, minlength=num_nodes)
    center_starts = degree.cumsum(0) - degree

    # Repeat each receiver once for every edge at its center.
    repeats = degree[sorted_centers]
    receiver = torch.arange(num_edges, device=edge_index.device).repeat_interleave(repeats)
    repeat_starts = repeats.cumsum(0) - repeats
    sender_rank = torch.arange(receiver.numel(), device=edge_index.device) - repeat_starts[receiver]
    sender = center_starts[sorted_centers[receiver]] + sender_rank

    keep = sender != receiver
    id3_b = edges_by_center[sender[keep]]
    id3_c = edges_by_center[receiver[keep]]

    valid = (edge_distance > 0) & torch.isfinite(edge_distance)
    inverse_distance = torch.where(
        valid,
        edge_distance.reciprocal(),
        torch.zeros_like(edge_distance),
    )
    unit_vectors = edge_vectors * inverse_distance[:, None]
    cos_phi = (unit_vectors[id3_b] * unit_vectors[id3_c]).sum(dim=-1)
    cos_phi = torch.where(
        torch.isfinite(cos_phi),
        cos_phi.clamp(-1.0, 1.0),
        cos_phi.new_zeros(cos_phi.shape),
    )
    return id3_b, id3_c, cos_phi
