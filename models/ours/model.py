"""DyGFormer with a private TKG relation interface; backbone architecture unchanged."""
import numpy as np
import torch
from torch import nn
from models.baselines.ctdg.dygformer import DyGFormer


class TypedDyGFormer(DyGFormer):
    def __init__(self,node_features,edge_features,sampler,event_relations,nrel,device='cuda:0',event_sources=None):
        super().__init__(node_features,edge_features,sampler,time_feat_dim=100,
                         channel_embedding_dim=50,patch_size=2,num_layers=2,num_heads=2,
                         dropout=.1,max_input_sequence_length=64,device=device)
        self.register_buffer('event_relations',torch.as_tensor(event_relations,dtype=torch.long,device=device),persistent=False)
        self.relations=nn.Embedding(nrel+1,172,padding_idx=0,device=device)
        nn.init.normal_(self.relations.weight,std=.02)
        with torch.no_grad():self.relations.weight[0].zero_()
        self.register_buffer('event_sources',None if event_sources is None else
                             torch.as_tensor(event_sources,dtype=torch.long,device=device),persistent=False)
        self.direction_embedding=None
        if event_sources is not None:
            # 0=padding, 1=outgoing, 2=incoming, 3=self-loop. Without this private
            # channel the undirected sampler erases the direction of TKG history.
            self.direction_embedding=nn.Embedding(4,172,padding_idx=0,device=device)
            nn.init.normal_(self.direction_embedding.weight,std=.02)
            with torch.no_grad():self.direction_embedding.weight[0].zero_()
        # Original batch-dependent padding alters attention/pooling and therefore
        # the score of the same triple. Keep the training/evaluation geometry fixed.
        self.fixed_padding=True

    def pad_sequences(self,*args,**kwargs):
        arrays=super().pad_sequences(*args,**kwargs)
        if not self.fixed_padding:return arrays
        length=self.max_input_sequence_length
        return tuple(np.pad(a,((0,0),(0,length-a.shape[1])),mode='constant') for a in arrays)

    def get_features(self,node_interact_times,padded_nodes_neighbor_ids,padded_nodes_edge_ids,padded_nodes_neighbor_times,time_encoder):
        nodes,_,times=super().get_features(node_interact_times,padded_nodes_neighbor_ids,padded_nodes_edge_ids,padded_nodes_neighbor_times,time_encoder)
        ids=torch.as_tensor(padded_nodes_edge_ids,device=self.device)
        edges=self.relations(self.event_relations[ids])
        if self.event_sources is not None:
            neighbors=torch.as_tensor(padded_nodes_neighbor_ids,device=self.device)
            center=neighbors[:,[0]]
            roles=torch.where(self.event_sources[ids]==center,1,2)
            roles=torch.where(neighbors==center,3,roles)
            roles=torch.where(ids==0,0,roles)
            edges=edges+self.direction_embedding(roles)
        return nodes,edges,times


def vectorized_cooccurrence(encoder):
    """Equivalent counts without per-pair numpy unique calls or CPU/GPU copies."""
    def forward(src_padded_nodes_neighbor_ids,dst_padded_nodes_neighbor_ids):
        s=torch.as_tensor(src_padded_nodes_neighbor_ids,device=encoder.device)
        d=torch.as_tensor(dst_padded_nodes_neighbor_ids,device=encoder.device)
        cross=(s[:,:,None]==d[:,None,:])
        ss=(s[:,:,None]==s[:,None,:]).sum(-1)
        dd=(d[:,:,None]==d[:,None,:]).sum(-1)
        sx=torch.stack((ss,cross.sum(-1)),-1).float()
        dx=torch.stack((cross.sum(-2),dd),-1).float()
        sx=sx.masked_fill((s==0).unsqueeze(-1),0)
        dx=dx.masked_fill((d==0).unsqueeze(-1),0)
        layer=encoder.neighbor_co_occurrence_encode_layer
        return layer(sx.unsqueeze(-1)).sum(2),layer(dx.unsqueeze(-1)).sum(2)
    encoder.forward=forward


class TypedPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(172*3,172),nn.ReLU(),nn.Linear(172,1))

    def forward(self,source,target,relation):
        return self.network(torch.cat((source,target,relation),-1)).squeeze(-1)


def shared_key(name):
    # Relation vocabulary and its feature projection are target-local.
    return not (name.startswith('relations.') or name.startswith('projection_layer.edge.')
                or name.startswith('direction_embedding.'))


def load_shared(backbone,state):
    selected={k:v for k,v in state.items() if shared_key(k)}
    missing,unexpected=backbone.load_state_dict(selected,strict=False)
    assert not unexpected
    assert all(not shared_key(k) for k in missing)
    return len(selected)


def freeze_shared(backbone):
    for k,p in backbone.named_parameters():p.requires_grad_(not shared_key(k))


class CachedSampler:
    """Cache strict-past lookups by node/time, preserving original sampler semantics."""
    sample_neighbor_strategy='recent'
    seed=7

    def __init__(self,base,limit=100000):
        self.base=base;self.cache={};self.limit=limit

    def get_all_first_hop_neighbors(self,node_ids,node_interact_times):
        out=[[],[],[]]
        for node,t in zip(node_ids,node_interact_times):
            key=(int(node),float(t))
            if key not in self.cache:
                if len(self.cache)>=self.limit:self.cache.clear()
                item=self.base.find_neighbors_before(key[0],key[1])
                # All remaining consumers truncate to their last 63 neighbors.
                self.cache[key]=tuple(np.asarray(x[-63:]) for x in item[:3])
            for dest,value in zip(out,self.cache[key]):dest.append(value)
        return tuple(out)

    def reset_random_state(self):pass
