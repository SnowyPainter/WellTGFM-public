"""Learning Across interfaces with truly shared history-encoder Parameters."""
import numpy as np
import torch
from torch import nn
from models.baselines.ctdg.tgat import TGAT
from models.baselines.ctdg.sampling import NeighborSampler
from models.ours.model import TypedPredictor
from experiments.common.data import adjacency, CutoffSampler

def shared_key(name):
    return not name.startswith(('relations.', 'directions.', 'event_projection.'))

class TypedTGAT(TGAT):
    """Original TGAT layers with a private relation/direction event interface."""
    def __init__(self,nodes,edges,sampler,data,device):
        super().__init__(nodes,edges,sampler,100,num_layers=2,num_heads=2,dropout=.1,device=device)
        rel=np.zeros(len(data.src)+1,np.int64);rel[data.edge_id]=data.relation
        src=np.zeros(len(data.src)+1,np.int64);src[data.edge_id]=data.src
        self.register_buffer('event_relations',torch.as_tensor(rel,device=device),persistent=False)
        self.register_buffer('event_sources',torch.as_tensor(src,device=device),persistent=False)
        self.relations=nn.Embedding(data.metadata['num_relations']+1,172,padding_idx=0,device=device)
        self.directions=nn.Embedding(4,172,padding_idx=0,device=device)
        self.event_projection=nn.Linear(172,172,bias=False,device=device)
        nn.init.normal_(self.relations.weight,std=.02);nn.init.normal_(self.directions.weight,std=.02)
        nn.init.eye_(self.event_projection.weight)
        with torch.no_grad():self.relations.weight[0].zero_();self.directions.weight[0].zero_()

    def get_history_edge_features(self,node_ids,neighbor_node_ids,neighbor_edge_ids):
        device=self.event_relations.device
        ids=torch.as_tensor(neighbor_edge_ids,device=device)
        center=torch.as_tensor(node_ids,device=device)[:,None]
        neighbor=torch.as_tensor(neighbor_node_ids,device=device)
        roles=torch.where(self.event_sources[ids]==center,1,2)
        roles=torch.where(neighbor==center,3,roles)
        roles=torch.where(ids==0,0,roles)
        return self.event_projection(self.relations(self.event_relations[ids])+self.directions(roles))


def tie_shared(destination,source):
    a=dict(source.named_parameters());b=dict(destination.named_parameters())
    assert {k for k in a if shared_key(k)}=={k for k in b if shared_key(k)}
    for key,value in a.items():
        if not shared_key(key):continue
        assert b[key].shape==value.shape
        parent,_,leaf=key.rpartition('.')
        module=destination.get_submodule(parent) if parent else destination
        setattr(module,leaf,value)

class CrossRepresentationPredictor(nn.Module):
    def __init__(self,data,target,history=32,device='cpu',seed=7):
        super().__init__()
        if data[target].metadata['family']!='ctdg':raise ValueError('Target must be CTDG')
        self.history=history;self.target=target;self.streams=data;self.encoders=nn.ModuleDict();self.heads=nn.ModuleDict()
        self.samplers={};self.bindings={id(d):n for n,d in data.items()}
        # Target first, then fixed domain order: same initial shared/private state in all conditions.
        for name in [target]+sorted(n for n in data if n!=target):
            d=data[name];nodes=np.zeros((d.metadata['num_nodes'],172),np.float32)
            sampler=CutoffSampler(NeighborSampler(adjacency(d),sample_neighbor_strategy='recent',seed=seed))
            self.samplers[name]=sampler
            if d.metadata['family']=='tkg':
                enc=TypedTGAT(nodes,d.edge_features,sampler,d,device)
                head=TypedPredictor()
            else:
                enc=TGAT(nodes,d.edge_features,sampler,100,num_layers=2,num_heads=2,dropout=.1,device=device)
                head=nn.Sequential(nn.Linear(344,172),nn.ReLU(),nn.Linear(172,1))
            self.encoders[name]=enc;self.heads[name]=head
            if name!=target:tie_shared(enc,self.encoders[target])
        self.to(device);self.bind(data[target])

    def bind(self,data):self.name=self.bindings[id(data)];self.data=data
    def reset(self):self.samplers[self.name].cutoff=float('inf')
    def begin_batch(self,ids):self.samplers[self.name].cutoff=float(self.data.time[ids[0]])
    def observe(self,ids):pass
    def shared_state(self):
        return {k:v.detach().cpu().clone() for k,v in self.encoders[self.target].state_dict().items() if shared_key(k)}
    def load_shared_state(self,state):
        missing,unexpected=self.encoders[self.target].load_state_dict(state,strict=False)
        assert not unexpected and all(not shared_key(k) for k in missing)

    def score(self,ids,negative):
        d=self.data;enc=self.encoders[self.name];n=len(ids)
        if d.metadata['family']=='ctdg':
            u=np.tile(d.src[ids],2);v=np.r_[d.dst[ids],negative];t=np.tile(d.time[ids],2)
            a,b=enc.compute_src_dst_node_temporal_embeddings(u,v,t,num_neighbors=self.history)
            return self.heads[self.name](torch.cat((a,b),-1)).reshape(2,n).T
        # Tail and head corruption both retain original fact direction/relation.
        if negative.shape!=(n,2):raise ValueError('TKG requires tail/head negative pairs')
        u=np.r_[d.src[ids],d.src[ids],d.src[ids],negative[:,1]]
        v=np.r_[d.dst[ids],negative[:,0],d.dst[ids],d.dst[ids]]
        a,b=enc.compute_src_dst_node_temporal_embeddings(u,v,np.tile(d.time[ids],4),num_neighbors=self.history)
        rel=torch.as_tensor(np.tile(d.relation[ids],4),device=a.device)
        z=self.heads[self.name](a,b,enc.relations(rel)).reshape(4,n)
        return torch.cat((z[:2].T,z[2:].T),dim=0)

    @property
    def encoder(self):return self.encoders[self.name]
