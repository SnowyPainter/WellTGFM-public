"""Legacy Batch-wise trainer, preserved equations and operation order for replay."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from experiments.target_adaptation.runtime import ROOT, digest, array_hash, build
from experiments.target_adaptation.readout import hidden, labels, ridge, refit
from experiments.common.engine import save_predictions
from experiments.common.runtime import atomic_json, atomic_torch, capture_rng, restore_rng, evaluate

def execute(job,method,cfg,data,out,phash,smoke=False):
    dest=out/method/job['job_id'];dest.mkdir(parents=True,exist_ok=True)
    ck=ROOT/job['checkpoint'] if job['checkpoint'] else None
    assert job['pretrain_seed'] in (7,None) and job['adaptation_seed'] in (7,17,37)
    if ck: assert digest(ck)==job['checkpoint_sha256']
    manifest=dict(job=job,method=method,config=cfg,protocol_sha256=phash,
        runner_sha256=digest(Path(__file__)),
        solver_sha256=digest(Path(__file__).with_name('readout.py')),
        ridge_lambda=.001,support_queries=256,query_queries=0,
        encoder_updates_per_epoch=12 if method=='ridge_profile' else 0,
        head='fixed original hidden layer; last linear refit on full Train each epoch',
        dropout='disabled during support,query,and readout fitting',
        direction_objective='profiled support MSE; envelope gradient with fitted readout detached',test_used=False)
    if (dest/'manifest.json').exists(): assert json.loads((dest/'manifest.json').read_text())==manifest
    atomic_json(dest/'manifest.json',manifest)
    if (dest/'result.json').exists() and not smoke: return
    started=time.monotonic();seed=job['adaptation_seed'];d=data[job['target']]
    model=build(cfg,data,seed)
    if ck: model.load_shared_state(torch.load(ck,map_location='cpu',weights_only=False)['shared'])
    model.bind(d)
    for p in model.parameters():p.requires_grad_(False)
    ep=list(model.encoders[model.target].parameters())
    for p in ep:p.requires_grad_(method=='ridge_profile')
    opt=torch.optim.Adam(ep,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    fixed=d.negatives(100007);batches=list(d.batches(d.indices(0),cfg['batch_size']))
    history=[];decisions=[];best=-float('inf');best_epoch=None;start=0;prior=0.
    if (dest/'resume.pt').exists() and not smoke:
        s=torch.load(dest/'resume.pt',map_location=cfg['device'],weights_only=False)
        model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);restore_rng(s['rng'],cfg['device'])
        history=s['history'];decisions=s['decisions'];best=s['best'];best_epoch=s['best_epoch'];start=s['epoch'];prior=s['seconds']
    for epoch in range(start,cfg['epochs']):
        negatives=d.negatives(seed+epoch);losses=[]
        if method=='ridge_profile':
            for bi,native in enumerate(batches):
                support=native[np.linspace(0,len(native)-1,min(256,len(native)),dtype=int)]
                assert np.isin(support,d.indices(0)).all()
                model.bind(d);model.eval();opt.zero_grad(set_to_none=True)
                model.begin_batch(native);xs=hidden(model,d,support,negatives);ys=labels(len(support),xs.device,xs.dtype)
                w,b=ridge(xs,ys)
                loss=((xs@w.detach()+b.detach()-ys)**2).mean()
                if not torch.isfinite(loss):raise FloatingPointError('Ridge query loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(ep,cfg['grad_clip'])
                if not torch.isfinite(norm):raise FloatingPointError('Ridge gradient')
                opt.step();losses.append(float(loss.detach()))
                decisions.append(dict(epoch=epoch+1,batch=bi,audit=dict(query_forwards=1,query_backwards=1,
                    support_size=len(support),query_size=0,support_max_time=float(d.time[support].max()),
                    profiled_risk=float(loss.detach()+.001*w.detach().square().sum()),loss=float(loss.detach()),gradient_norm=float(norm))))
                if smoke and len(decisions)==3:
                    atomic_json(dest/'smoke.json',dict(status='passed',decisions=decisions));return
        fit=refit(model,d,negatives,cfg)
        metrics,scores=evaluate(model,d,1,fixed,cfg)
        history.append(dict(epoch=epoch+1,validation=metrics,train_loss=float(np.mean(losses)) if losses else None,readout=fit))
        if metrics['ap']>best:
            best=metrics['ap'];best_epoch=epoch+1
            atomic_torch(dest/'best.pt',dict(model=model.state_dict(),epoch=best_epoch))
            save_predictions(dest/'validation_predictions.npz',d,1,fixed,scores)
        atomic_torch(dest/'resume.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=capture_rng(cfg['device']),
            history=history,decisions=decisions,best=best,best_epoch=best_epoch,epoch=epoch+1,seconds=prior+time.monotonic()-started))
        print(json.dumps(dict(method=method,job_id=job['job_id'],epoch=epoch+1,ap=metrics['ap'])),flush=True)
        if smoke:
            atomic_json(dest/'smoke.json',dict(status='passed',readout=fit));return
    with np.load(dest/'validation_predictions.npz') as arrays:ehash=array_hash(arrays)
    atomic_json(dest/'result.json',dict(status='complete',method=method,job=job,ap=best,best_epoch=best_epoch,
        last_epoch_ap=history[-1]['validation']['ap'],history=history,decisions=decisions,seconds=prior+time.monotonic()-started,
        protocol_sha256=phash,evaluation_sha256=ehash,checkpoint_sha256=job['checkpoint_sha256'],
        evaluation_negative_seed=100007,query_forwards=sum(x['audit']['query_forwards'] for x in decisions),
        query_backwards=sum(x['audit']['query_backwards'] for x in decisions),
        full_train_readout_passes=5,encoder_updates=len(decisions),initialization=None))
