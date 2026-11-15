"""Shared small-data supervised LP benchmark; dry-run by default."""
from __future__ import annotations
import argparse,contextlib,fcntl,hashlib,json,os,random,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score,roc_auc_score
from experiments.common.data import Events


def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)


def atomic_torch(path,obj):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def capture_rng(device):
    state=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state())
    if str(device).startswith('cuda'):state['cuda']=torch.cuda.get_rng_state(device)
    return state


def restore_rng(state,device):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if 'cuda' in state:torch.cuda.set_rng_state(state['cuda'].cpu(),device)


@contextlib.contextmanager
def gpu_guard(device,minimum_free_gib):
    if device=='cpu':yield;return
    if device!='cuda:0':raise ValueError('This profile supports cpu or cuda:0')
    with open('/tmp/wellgfm-baselines-gpu0.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        if torch.cuda.get_device_capability(0)>=(12,0) and 'sm_120' not in torch.cuda.get_arch_list():
            raise RuntimeError('PyTorch must support Blackwell sm_120')
        torch.set_float32_matmul_precision('high')
        yield


def pass_batches(model,data,indices,negatives,cfg,optimizer=None):
    scores=[];labels=[];loss_sum=0.;count=0;updates=0
    model.train(optimizer is not None)
    for ids in data.batches(indices,cfg['batch_size']):
        model.begin_batch(ids)
        if optimizer is not None:optimizer.zero_grad(set_to_none=True)
        for start in range(0,len(ids),cfg['microbatch']):
            part=ids[start:start+cfg['microbatch']]
            with torch.set_grad_enabled(optimizer is not None):
                logits=model.score(part,negatives[part]);truth=torch.zeros_like(logits);truth[:,0]=1.
                if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite scores')
                loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,truth)
                if optimizer is not None:(loss*len(part)/len(ids)).backward()
            loss_sum+=float(loss.detach())*len(part);count+=len(part)
            scores.extend(logits.detach().float().cpu().numpy().reshape(-1).tolist())
            labels.extend(np.tile([1,0],len(part)).tolist())
        if optimizer is not None:
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'])
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradients')
            optimizer.step();updates+=1
        # Observed positives only; equal-time events are revealed together after scoring.
        model.observe(ids)
    if not count:raise ValueError('Empty split')
    return dict(loss=loss_sum/count,ap=float(average_precision_score(labels,scores)),
        auc=float(roc_auc_score(labels,scores)),events=count,updates=updates),np.asarray(scores)


@torch.no_grad()
def evaluate(model,data,split,negatives,cfg):
    model.eval();model.reset()
    for prior in range(split):
        for ids in data.batches(data.indices(prior),cfg['batch_size']):model.observe(ids)
    return pass_batches(model,data,data.indices(split),negatives,cfg)


