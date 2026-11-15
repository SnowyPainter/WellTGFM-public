"""One shared encoder and LP head; bind each network's private history at runtime."""
import numpy as np
import torch
from torch import nn
from experiments.common.data import adjacency, CutoffSampler
from models.baselines.ctdg.sampling import NeighborSampler
from models.baselines.ctdg.tgat import TGAT
from models.baselines.ctdg.dygformer import DyGFormer


class SharedLinkPredictor(nn.Module):
    def __init__(self, data, backbone='tgat', history=32, device='cpu', seed=7):
        super().__init__()
        self.backbone = backbone
        self.history = history
        self.device_name = device
        self.seed = seed
        self._bindings = {}
        nodes = np.zeros((data.metadata['num_nodes'],172),np.float32)
        sampler = CutoffSampler(NeighborSampler(adjacency(data),sample_neighbor_strategy='recent',seed=seed))
        if backbone == 'tgat':
            self.encoder = TGAT(nodes,data.edge_features,sampler,100,num_layers=2,num_heads=2,dropout=.1,device=device)
        elif backbone == 'dygformer':
            self.encoder = DyGFormer(nodes,data.edge_features,sampler,100,50,patch_size=2,num_layers=2,
                                     num_heads=2,dropout=.1,max_input_sequence_length=history,device=device)
            # Fixed padding makes query geometry independent of microbatch partition.
            original = self.encoder.pad_sequences
            def pad(*args, **kwargs):
                arrays = original(*args, **kwargs)
                return tuple(np.pad(a,((0,0),(0,history-a.shape[1]))) for a in arrays)
            self.encoder.pad_sequences = pad
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')
        self.head = nn.Sequential(nn.Linear(344,172),nn.ReLU(),nn.Linear(172,1))
        self.to(device)
        self.bind(data)

    def bind(self, data):
        if data.metadata['family'] != 'ctdg':
            raise ValueError('This experiment uses CTDG link prediction')
        self.data = data
        key = id(data)
        if key not in self._bindings:
            self._bindings[key] = (
                data,
                CutoffSampler(NeighborSampler(adjacency(data),sample_neighbor_strategy='recent',seed=self.seed)),
                torch.zeros(data.metadata['num_nodes'],172,device=self.device_name),
                torch.as_tensor(data.edge_features,device=self.device_name),
            )
        _,self.sampler,nodes,edges = self._bindings[key]
        self.encoder.set_neighbor_sampler(self.sampler)
        self.encoder.node_raw_features = nodes
        self.encoder.edge_raw_features = edges

    def reset(self):
        self.sampler.cutoff = float('inf')

    def begin_batch(self, ids):
        self.sampler.cutoff = float(self.data.time[ids[0]])

    def observe(self, ids):
        pass  # Strict-past sampling reads only the bound network's history.

    def score(self, ids, negative):
        data = self.data
        source = np.tile(data.src[ids],2)
        destination = np.r_[data.dst[ids],negative]
        times = np.tile(data.time[ids],2)
        kw = {'num_neighbors':self.history} if self.backbone == 'tgat' else {}
        a,b = self.encoder.compute_src_dst_node_temporal_embeddings(source,destination,times,**kw)
        return self.head(torch.cat((a,b),-1)).reshape(2,-1).T
