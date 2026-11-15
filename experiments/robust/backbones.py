"""Local model implementations with private domain inputs and causal history."""
import numpy as np
import torch
from torch import nn
from experiments.common.data import adjacency, CutoffSampler
from models.baselines.ctdg.sampling import NeighborSampler
from models.baselines.ctdg.tgn import MemoryModel
from models.baselines.graphmixer import MLPMixer
from experiments.robust.tgn_optimized import CachedGraphAttentionEmbedding, replay_prefix
from models.ours.joint import shared_key
from models.ours.model import TypedPredictor


def transferable(key):
    return shared_key(key) and 'memory_bank.' not in key


def tie_encoder(destination, source):
    for key, value in source.named_parameters():
        if transferable(key):
            parent, _, leaf = key.rpartition('.')
            setattr(destination.get_submodule(parent) if parent else destination, leaf, value)


class TypedInput:
    def initialize_typed_input(self, data, device):
        self.typed = data.metadata['family'] == 'tkg'
        if not self.typed:
            return
        rel = np.zeros(len(data.src)+1, np.int64)
        src = np.zeros_like(rel)
        rel[data.edge_id], src[data.edge_id] = data.relation, data.src
        self.register_buffer('event_relations', torch.as_tensor(rel, device=device), persistent=False)
        self.register_buffer('event_sources', torch.as_tensor(src, device=device), persistent=False)
        self.relations = nn.Embedding(data.metadata['num_relations']+1, 172, padding_idx=0, device=device)
        self.directions = nn.Embedding(4, 172, padding_idx=0, device=device)
        self.event_projection = nn.Linear(172, 172, bias=False, device=device)
        nn.init.normal_(self.relations.weight, std=.02)
        nn.init.normal_(self.directions.weight, std=.02)
        nn.init.eye_(self.event_projection.weight)
        with torch.no_grad():
            self.relations.weight[0].zero_()
            self.directions.weight[0].zero_()

    def typed_edges(self, centers, neighbors, edge_ids):
        device = self.relations.weight.device
        ids = torch.as_tensor(edge_ids, device=device)
        centers = torch.as_tensor(centers, device=device)[:, None]
        neighbors = torch.as_tensor(neighbors, device=device)
        roles = torch.where(self.event_sources[ids] == centers, 1, 2)
        roles = torch.where(neighbors == centers, 3, roles)
        roles = torch.where(ids == 0, 0, roles)
        return self.event_projection(self.relations(self.event_relations[ids]) + self.directions(roles))


class OfficialGraphMixer(nn.Module, TypedInput):
    def __init__(self, data, sampler, cfg):
        super().__init__()
        self.history = cfg['history_length']
        self.neighbor_sampler = sampler
        self.device = cfg['device']
        self.edge_raw_features = torch.as_tensor(data.edge_features, device=self.device)
        # Common 172-dimensional encoder output; author's hidden/time width is 100.
        self.network = MLPMixer(
            per_graph_size=self.history, time_channels=100, input_channels=172,
            hidden_channels=100, out_channels=172, num_layers=2, dropout=cfg['dropout'])
        self.initialize_typed_input(data, self.device)
        self.to(self.device)

    def compute_node(self, nodes, times):
        neighbors, edges, stamps = self.neighbor_sampler.get_historical_neighbors(nodes, times, self.history)
        # Author sampler puts most recent edges first and pads after real tokens.
        neighbors, edges, stamps = (np.ascontiguousarray(x[:, ::-1]) for x in (neighbors, edges, stamps))
        mask = edges != 0
        feats = self.typed_edges(nodes, neighbors, edges) if self.typed else self.edge_raw_features[torch.as_tensor(edges, device=self.device)]
        valid = torch.as_tensor(mask, device=self.device)
        delta = torch.as_tensor(times[:, None]-stamps, dtype=torch.float32, device=self.device)
        inds = torch.as_tensor(np.flatnonzero(mask.ravel()), device=self.device)
        return self.network(feats[valid], delta[valid], len(nodes), inds)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids, dst_node_ids, node_interact_times):
        return self.compute_node(src_node_ids, node_interact_times), self.compute_node(dst_node_ids, node_interact_times)


class LocalTGN(MemoryModel, TypedInput):
    def __init__(self, data, sampler, cfg):
        if np.any(data.src <= 0) or np.any(data.dst <= 0):
            raise ValueError('TGN node 0 is reserved for immutable padding')
        super().__init__(np.zeros((data.metadata['num_nodes'],172),np.float32),
            data.edge_features, sampler, 100, model_name='TGN', num_layers=2,
            num_heads=2, dropout=cfg['dropout'], device=cfg['device'])
        self.history = cfg['history_length']
        self.embedding_module.__class__ = CachedGraphAttentionEmbedding
        self.optimized = True
        self._aggregated_messages = None
        self._evaluated_memory = None
        self.initialize_typed_input(data, cfg['device'])
        # Memory is domain state, never a shared or trainable parameter.
        bank = self.memory_bank
        for key in ['node_memories', 'node_last_updated_times']:
            value = bank._parameters.pop(key).detach()
            bank.register_buffer(key, value, persistent=False)
        if self.typed:
            self.embedding_module.get_history_edge_features = self.typed_edges
        self.to(cfg['device'])

    def get_message_edge_features(self, src_node_ids, dst_node_ids, edge_ids):
        if self.typed:
            return self.typed_edges(src_node_ids, dst_node_ids[:,None], edge_ids[:,None])[:,0]
        return super().get_message_edge_features(src_node_ids, dst_node_ids, edge_ids)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids, dst_node_ids, node_interact_times):
        return super().compute_src_dst_node_temporal_embeddings(src_node_ids, dst_node_ids,
            node_interact_times, edge_ids=None, edges_are_positive=False, num_neighbors=self.history)

    def get_updated_memories(self, node_ids, node_raw_messages):
        # Query scoring always asks for all nodes. Raw messages are detached
        # history, so aggregate once per replay, but rebuild the differentiable
        # GRU graph for every gradient-bearing microbatch.
        if not self.optimized or len(node_ids) != self.num_nodes:
            return super().get_updated_memories(node_ids, node_raw_messages)
        if self._aggregated_messages is None:
            self._aggregated_messages = self.message_aggregator.aggregate_messages(
                node_ids, node_raw_messages)
        if not torch.is_grad_enabled() and self._evaluated_memory is not None:
            return self._evaluated_memory
        result = self.memory_updater.get_updated_memories(*self._aggregated_messages)
        if not torch.is_grad_enabled():
            self._evaluated_memory = result
        return result

    @torch.no_grad()
    def replay(self, data, cutoff, batch_size):
        self._aggregated_messages = None
        self._evaluated_memory = None
        if self.optimized:
            replay_prefix(self, data, cutoff, batch_size)
            return
        self._prefix_cache = None
        self.memory_bank.__init_memory_bank__()
        ids = np.flatnonzero(data.time < cutoff)
        for part in data.batches(ids, batch_size):
            self.replay_events(data, part)
        self.memory_bank.detach_memory_bank()

    @torch.no_grad()
    def replay_events(self, data, part):
        u, v, t, e = data.src[part], data.dst[part], data.time[part], data.edge_id[part]
        nodes = np.r_[u,v]
        self.update_memories(nodes, self.memory_bank.node_raw_messages)
        self.memory_bank.clear_node_raw_messages(nodes)
        messages = [self.compute_new_node_raw_messages(u,v,None,t,e),
                    self.compute_new_node_raw_messages(v,u,None,t,e)]
        for unique, raw in messages:
            self.memory_bank.store_node_raw_messages(unique, raw)


class OfficialPredictor(nn.Module):
    def __init__(self, data, target, cfg, seed, backbone):
        super().__init__()
        self.target, self.history = target, cfg['history_length']
        self.cfg = cfg
        if backbone not in {"graphmixer", "tgn"}: raise ValueError(backbone)
        self.encoders, self.heads = nn.ModuleDict(), nn.ModuleDict()
        self.samplers = {}
        self.bindings = {id(d): n for n, d in data.items()}
        for n in [target] + sorted(n for n in data if n != target):
            d = data[n]
            sampler = CutoffSampler(NeighborSampler(adjacency(d), sample_neighbor_strategy='recent', seed=seed))
            self.samplers[n] = sampler
            enc = OfficialGraphMixer(d, sampler, cfg) if backbone == 'graphmixer' else LocalTGN(d, sampler, cfg)
            self.encoders[n] = enc
            self.heads[n] = TypedPredictor() if d.metadata['family']=='tkg' else nn.Sequential(nn.Linear(344,172), nn.ReLU(), nn.Linear(172,1))
            if n != target:
                tie_encoder(enc, self.encoders[target])
        self.to(cfg['device'])
        self.bind(data[target])

    def bind(self, data): self.name, self.data = self.bindings[id(data)], data
    def reset(self): self.samplers[self.name].cutoff = float('inf')
    def begin_batch(self, ids):
        cutoff = float(self.data.time[ids[0]])
        self.samplers[self.name].cutoff = cutoff
        if isinstance(self.encoder, LocalTGN):
            size = self.cfg['tkg_batch_size'] if self.data.metadata['family']=='tkg' else self.cfg['batch_size']
            self.encoder.replay(self.data, cutoff, size)
    def observe(self, ids): pass
    @property
    def encoder(self): return self.encoders[self.name]

    def shared_state(self):
        return {k:v.detach().cpu().clone() for k,v in self.encoders[self.target].state_dict().items() if transferable(k)}

    def load_shared_state(self, state):
        missing, unexpected = self.encoders[self.target].load_state_dict(state, strict=False)
        assert not unexpected and all(not transferable(k) for k in missing)

    def score(self, ids, negative):
        d, enc, n = self.data, self.encoder, len(ids)
        if d.metadata['family']=='ctdg':
            a,b = enc.compute_src_dst_node_temporal_embeddings(np.tile(d.src[ids],2), np.r_[d.dst[ids],negative], np.tile(d.time[ids],2))
            return self.heads[self.name](torch.cat((a,b),-1)).reshape(2,n).T
        assert negative.shape == (n,2)
        u = np.r_[d.src[ids],d.src[ids],d.src[ids],negative[:,1]]
        v = np.r_[d.dst[ids],negative[:,0],d.dst[ids],d.dst[ids]]
        a,b = enc.compute_src_dst_node_temporal_embeddings(u,v,np.tile(d.time[ids],4))
        rel = torch.as_tensor(np.tile(d.relation[ids],4),device=a.device)
        z = self.heads[self.name](a,b,enc.relations(rel)).reshape(4,n)
        return torch.cat((z[:2].T,z[2:].T))
