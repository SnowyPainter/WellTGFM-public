"""Small synthetic events for CPU checks; no downloaded data or GPU required."""
import numpy as np
from etl.prepare_benchmark import temporal_split
from experiments.common.data import Events


def events(name='fixture', offset=0., typed=False):
    n=60
    times=np.arange(1,n+1,dtype=float)+offset
    src=np.tile([1,2,3],20)
    dst=np.tile([2,3,1],20)
    features=np.random.default_rng(7).normal(size=(n+1,172)).astype(np.float32)
    features[0]=0
    return Events(src,dst,np.ones(n,np.int64) if typed else np.zeros(n,np.int64),times,times.copy(),
                  temporal_split(times),np.ones(n,np.float32),np.arange(1,n+1),features,
                  dict(dataset=name,family='tkg' if typed else 'ctdg',num_nodes=8,num_relations=2 if typed else 1))
