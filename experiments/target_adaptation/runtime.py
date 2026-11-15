"""Fixed development inputs and shared model construction."""
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.common.data import Events
from experiments.common.runtime import seed_all
from models.ours.joint import CrossRepresentationPredictor
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def array_hash(arrays):
    h=hashlib.sha256()
    for key in ('event_ids','src','dst','time','negatives','labels'):
        value=np.ascontiguousarray(arrays[key]);h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(value.shape).encode());h.update(value.tobytes())
    return h.hexdigest()

def build(cfg,data,seed):
    seed_all(seed)
    return CrossRepresentationPredictor(data,cfg['target'],cfg['history_length'],cfg['device'],seed)

def load_inputs(load_data=True):
    plan=json.loads((HERE/'inputs/plan.json').read_text());jobs=plan['jobs']
    assert len(jobs)==54 and plan['pretrain_seed']==7 and plan['test_used'] is False
    assert {j['adaptation_seed'] for j in jobs}=={7,17,37}
    assert all(j['stage']=='development' and j['pretrain_seed']==(7 if j['sources'] else None) for j in jobs)
    cfgs=json.loads((HERE/'inputs/configs.json').read_text())
    data={}
    if load_data:
        cfg=cfgs['reddit']
        data={n:Events.load(ROOT/cfg['data_root'],cfg['families'][n],n) for n in cfg['datasets']}
    return jobs,cfgs,data
