"""Model-independent events, timestamp-safe batches and shared negative queries."""
from dataclasses import dataclass
from pathlib import Path
import hashlib,json
import numpy as np


@dataclass
class Events:
    src: np.ndarray
    dst: np.ndarray
    relation: np.ndarray
    time: np.ndarray
    raw_time: np.ndarray
    split: np.ndarray
    label: np.ndarray
    edge_id: np.ndarray
    edge_features: np.ndarray
    metadata: dict

    @classmethod
    def load(cls,root,family,name):
        p=Path(root)/family/name
        meta=json.loads((p/'metadata.json').read_text())
        for f,expected in meta['artifacts'].items():
            if hashlib.sha256((p/f).read_bytes()).hexdigest()!=expected:
                raise ValueError(f'Data checksum mismatch: {p/f}')
        with np.load(p/'events.npz') as f: arrays=dict(f)
        return cls(**arrays,edge_features=np.load(p/'edge_features.npy'),metadata=meta)

    def indices(self,split):return np.flatnonzero(self.split==split)

    def batches(self,indices,batch_size):
        """Never divide a timestamp group across updates/history boundaries."""
        start=0
        while start<len(indices):
            end=min(start+batch_size,len(indices))
            while end<len(indices) and self.time[indices[end]]==self.time[indices[end-1]]:end+=1
            yield indices[start:end]
            start=end

    def negatives(self,seed):
        rng=np.random.default_rng(seed)
        vocab=np.unique(self.dst[self.split==0])
        positives={}
        for s,r,t,d in zip(self.src,self.relation,self.time,self.dst):
            positives.setdefault((int(s),int(r),float(t)),set()).add(int(d))
        out=np.empty(len(self.src),np.int64)
        for i,(s,r,t) in enumerate(zip(self.src,self.relation,self.time)):
            blocked=positives[int(s),int(r),float(t)]
            if len(blocked)>=len(vocab) and all(int(v) in blocked for v in vocab):
                raise ValueError('No valid negative destination in train vocabulary')
            for _ in range(100):
                candidate=int(rng.choice(vocab))
                if candidate not in blocked:break
            else:
                candidate=int(rng.choice([v for v in vocab if int(v) not in blocked]))
            out[i]=candidate
        return out


def adjacency(data):
    adj=[[] for _ in range(data.metadata['num_nodes'])]
    for s,d,e,t in zip(data.src,data.dst,data.edge_id,data.time):
        adj[int(s)].append((int(d),int(e),float(t)))
        adj[int(d)].append((int(s),int(e),float(t)))
    return adj


class CutoffSampler:
    """All models see only events strictly before the current batch start."""
    sample_neighbor_strategy='recent'
    seed=7
    def __init__(self,base):self.base=base;self.cutoff=float('inf')
    def get_all_first_hop_neighbors(self,node_ids,node_interact_times):
        return self.base.get_all_first_hop_neighbors(node_ids,np.minimum(node_interact_times,self.cutoff))
    def get_historical_neighbors(self,node_ids,node_interact_times,num_neighbors=20):
        return self.base.get_historical_neighbors(node_ids,np.minimum(node_interact_times,self.cutoff),num_neighbors)
    def reset_random_state(self):pass
