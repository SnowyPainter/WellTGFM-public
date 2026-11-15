"""Exact-shape TGAT history materialization; no embedding deduplication or precision changes."""

import numpy as np
import torch


def history_tensors(enc, node_ids, node_interact_times, k):
    sampler = enc.neighbor_sampler
    base = getattr(sampler, "base", sampler)
    if not hasattr(base, "_recent_flat"):
        base.get_historical_neighbors(np.empty(0, dtype=np.int64), np.empty(0), k)
    offsets, neighbors, edges, times = base._recent_flat
    nodes = np.asarray(node_ids, dtype=np.int64)
    queries = np.asarray(node_interact_times)
    if hasattr(sampler, "cutoff"):
        queries = np.minimum(queries, sampler.cutoff)
    start = offsets[nodes]
    lo = start.copy()
    hi = offsets[nodes + 1].copy()
    while np.any(lo < hi):
        mid = (lo + hi) // 2
        active = lo < hi
        right = (times[mid] < queries) | np.isnan(queries)
        lo = np.where(active & right, mid + 1, lo)
        hi = np.where(active & ~right, mid, hi)
    device = enc.node_raw_features.device
    cache = getattr(enc, "_history_tensor_tables", None)
    key = (id(base), k, str(device))
    if cache is None or cache[0] != key:
        positions = np.arange(len(times))[:, None] + np.arange(-k, 0)
        positions = np.maximum(positions, 0)
        arrays = [
            neighbors[positions],
            edges[positions],
            times[positions].astype(np.float32),
        ]
        tables = [torch.from_numpy(a).to(device) for a in arrays]
        enc._history_tensor_tables = (key, tables)
    tables = enc._history_tensor_tables[1]
    idx = torch.from_numpy(lo).to(device, non_blocking=True)
    counts = torch.from_numpy(lo - start).to(device, non_blocking=True)
    valid = counts[:, None] + torch.arange(-k, 0, device=device) >= 0
    return tuple(torch.where(valid, t[idx], 0) for t in tables)


def layer_one(enc, node_ids, node_interact_times, k):
    device = enc.node_raw_features.device
    node_time = enc.time_encoder(torch.zeros((len(node_ids), 1), device=device))
    raw = enc.node_raw_features[
        torch.from_numpy(node_ids).to(device, non_blocking=True)
    ]
    ni, ei, nt = history_tensors(enc, node_ids, node_interact_times, k)
    neighbor = enc.node_raw_features[ni.flatten()].reshape(
        len(node_ids), k, enc.node_feat_dim
    )
    query = torch.from_numpy(node_interact_times).to(device, non_blocking=True)
    delta = query[:, None] - nt
    neighbor_time = enc.time_encoder(delta.float())
    from models.baselines.ctdg.tgat import TGAT

    if type(enc) is TGAT:
        edge = enc.edge_raw_features[ei]
    else:
        edge = enc.get_history_edge_features(node_ids, ni, ei)
    # Preserve original attention ops/dropout ordering, supplying a tensor mask.
    output, _ = enc.temporal_conv_layers[0](
        raw, node_time, neighbor, neighbor_time, edge, ni
    )
    return enc.merge_layers[0](output, raw)


def frozen_embedding(enc, nodes, times, neighbors):
    """Cache only deterministic, non-trainable encoders; invalidate on parameter writes."""
    from collections import OrderedDict

    if (
        enc.training
        or torch.is_grad_enabled()
        or any(p.requires_grad for p in enc.parameters())
    ):
        return enc.compute_node_temporal_embeddings(
            nodes, times, enc.num_layers, neighbors
        )
    versions = tuple(p._version for p in enc.parameters())
    cache = getattr(enc, "_frozen_embeddings", None)
    if cache is None or cache[0] != versions:
        cache = (versions, OrderedDict())
        enc._frozen_embeddings = cache
    entries = cache[1]
    key = (
        nodes.dtype.str,
        nodes.tobytes(),
        times.dtype.str,
        times.tobytes(),
        neighbors,
        getattr(enc.neighbor_sampler, "cutoff", None),
        id(enc.neighbor_sampler),
    )
    if key in entries:
        entries.move_to_end(key)
        return entries[key]
    result = enc.compute_node_temporal_embeddings(
        nodes, times, enc.num_layers, neighbors
    )
    entries[key] = result
    # Bounded per-encoder cache, including query keys and detached embedding tensors.
    while (
        entries
        and sum(
            v.numel() * v.element_size() + len(k[1]) + len(k[3])
            for k, v in entries.items()
        )
        > 64 * 1024**2
    ):
        entries.popitem(last=False)
    return result
