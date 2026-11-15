"""Sealed Test evaluation and read-only reuse of completed powerset models."""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.common.runtime import atomic_json, evaluate
from experiments.common.engine import save_predictions
from experiments.transfer_validation.core import ROOT, BASE, DEFAULT, data_for, model_for, read, digest, canonical_hash

CANDIDATES = {}


def resolve_reuse(dest):
    seen=set()
    while (dest/'reuse.json').exists():
        if str(dest) in seen:raise ValueError('Cyclic reuse pointer')
        seen.add(str(dest));p=read(dest/'reuse.json')
        dest=ROOT/p['reused_from']
        if p.get('result_sha256'):assert digest(dest/'result.json')==p['result_sha256']
        if p.get('predictions_sha256'):assert digest(dest/'validation_predictions.npz')==p['predictions_sha256']
    assert (dest/'best.pt').exists(), dest
    return dest


def legacy_models():
    """1,536 saved models, including method-matched scratch; no retraining."""
    from experiments.target_adaptation.powerset import plan
    p=plan();rows=[]
    for job in p['jobs']:
        for method,folder,oldname in [
            ('finetune','powerset_v1','finetune'),('tfra','powerset_v1','ridge_epoch'),
            ('lp_ft','lp_ft_powerset_v1','lp_ft'),('refine','refine_powerset_v1','refine')]:
            dest=resolve_reuse(BASE/'outputs'/folder/oldname/job['job_id'])
            j={**job,'id':'legacy__'+canonical_hash([method,job['job_id']])[:20],
                'method':method,'cohort':'legacy_powerset','profile':'original','backbone':'tgat','source_budget':None,
                'config':{**DEFAULT,**p['configs'][job['target']]}}
            rows.append(dict(job=j,best=str((dest/'best.pt').relative_to(ROOT)),
                             validation_ap=read(dest/'result.json')['ap'],config=j['config']))
    assert len(rows)==1536
    return rows


def verify_development(jobs,out):
    rows=[]
    for j in jobs:
        dest=out/'jobs'/j['id'];r=read(dest/'result.json');cfg=read(dest/'manifest.json')['config']
        assert r['status']=='complete' and r['test_used'] is False
        assert r['job']==j
        assert len(r['history'])==cfg['epochs']
        initial=[r['initial_validation']['ap']] if r.get('initial_validation') else []
        assert r['ap']==max(initial+[e['validation']['ap'] for e in r['history']])
        with np.load(dest/'validation_predictions.npz') as p:
            ap=float(average_precision_score(p['labels'].ravel(),p['scores'].ravel()))
        assert abs(ap-r['ap'])<1e-10
        assert np.isfinite(r['ap'])
        rows.append(dict(job=j,best=str((dest/'best.pt').relative_to(ROOT)),validation_ap=r['ap'],config=cfg))
        if j['cohort']=='long_budget':
            for budget in [5,20]:
                assert (dest/f'budget{budget}.pt').exists()
                jb={**j,'id':j['id']+f'__budget{budget}','cohort':f'long_budget_{budget}'}
                rows.append(dict(job=jb,best=str((dest/f'budget{budget}.pt').relative_to(ROOT)),
                    validation_ap=max(initial+[e['validation']['ap'] for e in r['history'][:budget]]),config=cfg))
    return rows


def seal(rows,out,protocol_hash):
    groups={}
    for r in rows:
        j=r['job']
        if j['sources']:continue
        key=(j['cohort'],j['profile'],j['backbone'],j['target'],j.get('config',{}).get('query_fraction',1.))
        groups.setdefault(key,{}).setdefault(j['method'],[]).append(r['validation_ap'])
    references=[dict(cohort=k[0],profile=k[1],backbone=k[2],target=k[3],query_fraction=k[4],
                     method=max(sorted(v),key=lambda m:np.mean(v[m])),
                     selection='largest mean scratch Validation AP over adaptation seeds') for k,v in groups.items()]
    frozen=dict(protocol_sha256=protocol_hash,models=[dict(**r,best_sha256=digest(ROOT/r['best'])) for r in rows],
                common_scratch_references=references,
                test_negative_seed=200007,selection='Validation only; no post-Test training in this suite')
    p=out/'test_freeze.json'
    if p.exists():assert read(p)==frozen
    else:atomic_json(p,frozen)
    return frozen


def negative_matrix(d,split,k,seed,historical=False,batch_size=2048):
    """Causal historical candidates, with timestamp filtering and explicit fallback."""
    ids=d.indices(split);rng=np.random.default_rng(seed)
    vocab=np.unique(d.dst[d.indices(0)]);vocab_set=set(vocab.tolist())
    blocked={}
    for u,v,t in zip(d.src,d.dst,d.time):blocked.setdefault((int(u),float(t)),set()).add(int(v))
    matrix=np.zeros((len(d.src),k),np.int64);fallback=0;prior={};cursor=0
    for native in d.batches(ids,batch_size):
        cutoff=d.time[native[0]]
        while cursor<len(d.src) and d.time[cursor]<cutoff:
            prior.setdefault(int(d.src[cursor]),set()).add(int(d.dst[cursor]));cursor+=1
        for i in native:
            exclusions=blocked[int(d.src[i]),float(d.time[i])]
            candidates=sorted((prior.get(int(d.src[i]),set()) & vocab_set)-exclusions) if historical else []
            if not candidates:
                if historical:fallback+=1
                candidates=vocab
            allowed=np.asarray([int(v) for v in candidates if int(v) not in exclusions],dtype=np.int64)
            if not len(allowed):raise ValueError('No evaluation negative')
            chosen=rng.choice(allowed,min(k,len(allowed)),replace=False).tolist()
            if len(chosen)<k:
                rest=np.asarray([int(v) for v in vocab if int(v) not in exclusions and int(v) not in chosen])
                if len(rest):chosen.extend(rng.choice(rest,min(k-len(chosen),len(rest)),replace=False).tolist())
            if len(chosen)<k:chosen.extend(rng.choice(allowed,k-len(chosen),replace=True).tolist())
            matrix[i]=chosen
    return matrix,dict(queries=len(ids),historical_empty_queries=fallback,negatives_per_query=k)


@torch.no_grad()
def rank_evaluate(model,d,cfg,negatives):
    model.eval();model.reset();all_scores=[]
    for native in d.batches(d.indices(2),cfg['batch_size']):
        model.begin_batch(native)
        for off in range(0,len(native),cfg['microbatch']):
            ids=native[off:off+cfg['microbatch']];cols=[];positive=None
            for col in range(negatives.shape[1]):
                z=model.score(ids,negatives[ids,col]).detach().cpu().numpy()
                if positive is None:positive=z[:,0]
                else:np.testing.assert_allclose(z[:,0],positive,rtol=1e-5,atol=1e-6)
                cols.append(z[:,1])
            all_scores.append(np.column_stack([positive]+cols))
    scores=np.concatenate(all_scores);truth=np.zeros_like(scores);truth[:,0]=1
    rank=1+(scores[:,1:]>scores[:,:1]).sum(1)+.5*(scores[:,1:]==scores[:,:1]).sum(1)
    return dict(ap=float(average_precision_score(truth.ravel(),scores.ravel())),
                auc=float(roc_auc_score(truth.ravel(),scores.ravel())),mrr=float(np.mean(1/rank)),
                hits1=float(np.mean(rank==1)),hits10=float(np.mean(rank<=10))),scores


def evaluate_frozen(row,out,rich=False):
    assert (out/'test_freeze.json').exists()
    job=row['job'];dest=out/'test'/job['id'];dest.mkdir(parents=True,exist_ok=True)
    if (dest/'result.json').exists():
        assert read(dest/'result.json')['best_sha256']==row['best_sha256'];return
    assert digest(ROOT/row['best'])==row['best_sha256']
    cfg=row['config'];data=data_for(job);d=data[job['target']]
    model=model_for(job,cfg,data)
    saved=torch.load(ROOT/row['best'],map_location=cfg['device'],weights_only=False)
    model.load_state_dict(saved['model']);model.bind(d)
    fixed=d.negatives(200007)
    metrics,scores=evaluate(model,d,2,fixed,cfg)
    save_predictions(dest/'test_predictions.npz',d,2,fixed,scores)
    results={'random_1':metrics}
    if rich:
        for historical in [False,True]:
            name='historical_10' if historical else 'random_10'
            key=(canonical_hash(d.metadata['artifacts']),historical,cfg['batch_size'])
            if key not in CANDIDATES:
                if len(CANDIDATES)>=24:CANDIDATES.pop(next(iter(CANDIDATES)))
                CANDIDATES[key]=negative_matrix(d,2,10,200017,historical,cfg['batch_size'])
            neg,audit=CANDIDATES[key]
            r,s=rank_evaluate(model,d,cfg,neg);results[name]=dict(**r,**audit)
            ids=d.indices(2)
            np.savez_compressed(dest/(name+'.npz'),event_ids=d.edge_id[ids],negatives=neg[ids],scores=s)
    atomic_json(dest/'result.json',dict(status='complete',job=job,selected_epoch=saved['epoch'],
        validation_ap=row['validation_ap'],metrics=results,best_sha256=row['best_sha256'],split='test'))
    print(json.dumps(dict(stage='test',job=job['id'],ap=metrics['ap'])),flush=True)
