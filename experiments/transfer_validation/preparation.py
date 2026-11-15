"""Data and source-only checkpoints for the prospective suite."""
import json
import time

import numpy as np
import pandas as pd
import torch

from etl.prepare_benchmark import sha, temporal_split
from experiments.common.engine import train_batch
from experiments.common.runtime import atomic_json, atomic_torch, capture_rng, restore_rng, evaluate
from experiments.transfer_validation.core import ROOT, TARGETS, DEFAULT, read, data_for, build, digest
from experiments.transfer_validation.design import source_checkpoint
from experiments.transfer.cross_representation import typed_negatives


def prepare_large():
    """A disjoint later window; never train on the small benchmark's held-out events."""
    out=ROOT/'data/benchmarks/tfra_validation_late131072'
    for n in TARGETS:
        dest=out/'ctdg'/n
        if (dest/'metadata.json').exists():
            m=read(dest/'metadata.json')
            assert m['version']=='tfra_late_window_v1' and m['raw_row_offset']==32768
            for f,h in m['artifacts'].items(): assert sha(dest/f)==h
            continue
        raw=ROOT/'data/raw/ctdg'/f'{n}.csv'
        # Explicit raw-row window: remove the entire old prefix before taking data.
        v=pd.read_csv(raw,skiprows=1,header=None,nrows=32768+131072+2).to_numpy(dtype=np.float64)
        assert np.all(np.diff(v[:,2])>=0), 'Raw JODIE timestamps must already be ordered'
        start=32768
        while start<len(v) and v[start,2]==v[start-1,2]: start+=1
        end=min(start+131072,len(v))
        if end<len(v):
            while end>start and v[end-1,2]==v[end,2]:end-=1
        v=v[start:end]
        assert len(v)>32768, (n,len(v))
        _,s=np.unique(v[:,0],return_inverse=True);_,d=np.unique(v[:,1],return_inverse=True)
        src=s+1;dst=d+s.max()+2;raw_time=v[:,2];split=temporal_split(raw_time)
        scale=float(np.median(np.diff(np.unique(raw_time[split==0]))))
        features=np.zeros((len(v)+1,172),np.float32);features[1:,:v.shape[1]-4]=v[:,4:]
        dest.mkdir(parents=True,exist_ok=True)
        np.savez(dest/'events.npz',src=src,dst=dst,relation=np.zeros(len(v),np.int64),
            time=(raw_time-raw_time[0])/max(scale,1e-12)+1,raw_time=raw_time,split=split,
            label=v[:,3].astype(np.float32),edge_id=np.arange(1,len(v)+1))
        np.save(dest/'edge_features.npy',features)
        m=dict(version='tfra_late_window_v1',dataset=n,family='ctdg',max_events=131072,
            num_events=len(v),num_nodes=int(max(src.max(),dst.max()))+1,num_relations=1,
            split_counts=np.bincount(split,minlength=3).tolist(),time_origin=float(raw_time[0]),time_scale=scale,
            raw_row_offset=32768,actual_raw_start=start,actual_raw_end=end,raw_sha256=sha(raw),
            split_policy='later disjoint window; timestamp-safe 70/15/15; no old benchmark Test events',
            artifacts={f:sha(dest/f) for f in ['events.npz','edge_features.npy']})
        atomic_json(dest/'metadata.json',m)
        print(json.dumps(dict(prepared=n,events=len(v),raw_window=[start,end])),flush=True)


def typed_eval_negatives(d,seed):
    rng=np.random.default_rng(seed);ids=d.indices(0)
    vocab=np.unique(np.r_[d.src[ids],d.dst[ids]])
    tails={};heads={}
    for u,v,r,t in zip(d.src,d.dst,d.relation,d.time):
        tails.setdefault((int(u),int(r),float(t)),set()).add(int(v))
        heads.setdefault((int(v),int(r),float(t)),set()).add(int(u))
    out=np.empty((len(d.src),2),np.int64)
    def draw(blocked):
        for _ in range(100):
            a=int(rng.choice(vocab))
            if a not in blocked:return a
        allowed=np.array([a for a in vocab if a not in blocked])
        if not len(allowed):raise ValueError('No valid typed evaluation negative')
        return int(rng.choice(allowed))
    for i,(u,v,r,t) in enumerate(zip(d.src,d.dst,d.relation,d.time)):
        out[i]=draw(tails[int(u),int(r),float(t)]),draw(heads[int(v),int(r),float(t)])
    return out


def ensure_pretraining(c,out):
    p=source_checkpoint(c,out)
    if p is None:return
    if p.exists():return
    if (out/'test_freeze.json').exists():raise RuntimeError('Source training sealed')
    dest=p.parent;dest.mkdir(parents=True,exist_ok=True)
    cfg={**DEFAULT,'epochs':5}
    data=data_for(c);seed=c['pretrain_seed'];sources=c['sources']
    manifest=dict(case=c,config=cfg,data={n:d.metadata['artifacts'] for n,d in data.items()},
                  implementation=digest(__file__),updates=c['source_budget'],test_used=False)
    if (dest/'manifest.json').exists():assert read(dest/'manifest.json')==manifest
    atomic_json(dest/'manifest.json',manifest)
    model=build(c,cfg,data,seed)
    opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    batches={n:list(data[n].batches(data[n].indices(0),cfg['tkg_batch_size'] if data[n].metadata['family']=='tkg' else cfg['batch_size'])) for n in sources}
    history=[];first=0;negatives={};start=time.monotonic()
    if (dest/'resume.pt').exists():
        s=torch.load(dest/'resume.pt',map_location=cfg['device'],weights_only=False)
        model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);restore_rng(s['rng'],cfg['device'])
        first=s['step'];history=s['history']
    for step in range(first,c['source_budget']):
        n=sources[step%len(sources)];local=step//len(sources);cycle=local//len(batches[n]);d=data[n]
        if (n,cycle) not in negatives:
            negatives[n,cycle]=typed_negatives(d,seed+cycle) if d.metadata['family']=='tkg' else d.negatives(seed+cycle)
        model.bind(d);ids=batches[n][local%len(batches[n])]
        loss=train_batch(model,d,ids,negatives[n,cycle],cfg,opt,1.)
        history.append(dict(step=step+1,source=n,events=len(ids),loss=loss))
        if (step+1)%12==0 or step+1==c['source_budget']:
            atomic_torch(dest/'resume.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step+1,
                         history=history,rng=capture_rng(cfg['device'])))
            print(json.dumps(dict(stage='source_training',target=c['target'],sources=sources,
                backbone=c['backbone'],preseed=seed,step=step+1,total=c['source_budget'])),flush=True)
    atomic_torch(p,dict(shared=model.shared_state(),case=c))
    atomic_json(dest/'result.json',dict(status='complete',case=c,history=history,seconds=time.monotonic()-start,test_used=False))


def source_fit(c,out):
    """Evaluate each old Source using its OWN saved pretraining head on Validation."""
    if not c['sources']:return
    p=source_checkpoint(c,out)
    from experiments.target_adaptation.runtime import canonical_hash
    dest=out/'source_fit'/canonical_hash(c)[:20]
    if (dest/'result.json').exists():return
    saved=p.parent/'resume.pt'
    s=torch.load(saved,map_location='cpu',weights_only=False)
    cfg={**DEFAULT};data=data_for(c);model=build(c,cfg,data,c['pretrain_seed'])
    model.load_state_dict(s['model'])
    rows=[]
    for n in c['sources']:
        d=data[n];model.bind(d)
        neg=typed_eval_negatives(d,100007) if d.metadata['family']=='tkg' else d.negatives(100007)
        eval_cfg={**cfg,'batch_size':cfg['tkg_batch_size']} if d.metadata['family']=='tkg' else cfg
        metrics,_=evaluate(model,d,1,neg,eval_cfg)
        rows.append(dict(source=n,**metrics))
    atomic_json(dest/'result.json',dict(case=c,checkpoint_sha256=digest(p),rows=rows,split='validation'))
