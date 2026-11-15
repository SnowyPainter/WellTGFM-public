"""Equivalent TGN computation with disposable history caches."""
import numpy as np
import torch
from models.baselines.ctdg.tgn import GraphAttentionEmbedding


class CachedGraphAttentionEmbedding(GraphAttentionEmbedding):
    optimized = True

    def compute_node_temporal_embeddings(self, node_memories, node_ids,
                                        node_interact_times, current_layer_num,
                                        num_neighbors=20):
        if not self.optimized:
            return super().compute_node_temporal_embeddings(
                node_memories, node_ids, node_interact_times, current_layer_num, num_neighbors)
        # Layer zero never uses the time encoder. In the original recursive
        # implementation it nevertheless allocates/encodes N * neighbors**2 times.
        if current_layer_num == 0:
            return self.get_node_features(node_memories, node_ids)
        return super().compute_node_temporal_embeddings(
            node_memories, node_ids, node_interact_times, current_layer_num, num_neighbors)

    def get_node_features(self, node_memories, node_ids):
        if not self.optimized:
            return super().get_node_features(node_memories, node_ids)
        # Dense embedding backward sorts/groups repeated indices instead of
        # contending atomic adds for millions of temporal-neighbor occurrences.
        # Node 0 is immutable padding and never receives a real event message.
        ids = torch.as_tensor(node_ids, device=node_memories.device)
        memory = torch.nn.functional.embedding(ids, node_memories, padding_idx=0)
        return memory + self.node_raw_features[ids]


@torch.no_grad()
def replay_prefix(model, data, cutoff, batch_size):
    """Reuse only complete historical batches under identical parameter versions.

    A cutoff can truncate a canonical batch (notably at split boundaries).
    Retain a snapshot BEFORE that partial batch so the next cutoff can rebuild
    it with precisely the same grouping as a fresh full replay.
    """
    versions = tuple((id(p), p._version) for p in model.parameters())
    signature = (id(data), batch_size, versions, str(model.node_raw_features.device))
    end = int(np.searchsorted(data.time, cutoff, side='left'))
    cache = getattr(model, '_prefix_cache', None)
    if cache is None or cache['signature'] != signature or end < cache['end']:
        model.memory_bank.__init_memory_bank__()
        cache = dict(signature=signature, end=0, before_partial=None)
    elif cache['before_partial'] is not None:
        model.memory_bank.reload_memory_bank(cache['before_partial'])
    start = cache['end']
    cache['before_partial'] = None
    # data.batches preserves timestamp groups, including at the cutoff.
    for part in data.batches(np.arange(start, end), batch_size):
        complete = len(part) >= batch_size
        if not complete:
            cache['before_partial'] = model.memory_bank.backup_memory_bank()
        model.replay_events(data, part)
        if complete:
            cache['end'] = int(part[-1])+1
    model.memory_bank.detach_memory_bank()
    model._prefix_cache = cache
