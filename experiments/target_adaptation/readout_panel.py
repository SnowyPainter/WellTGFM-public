"""Complete the fixed 54-run readout-strategy panel, reusing verified results."""
import argparse
import csv
import fcntl
import json
import time
from pathlib import Path
from statistics import mean,pstdev
import numpy as np
import torch
from experiments.target_adaptation.runtime import ROOT,HERE,digest,canonical_hash,array_hash,build,load_inputs
from experiments.target_adaptation.readout import hidden,labels,refit
from experiments.target_adaptation.powerset import read_result
from experiments.common.engine import train_batch,save_predictions
from experiments.common.runtime import atomic_json,atomic_torch,capture_rng,restore_rng,evaluate,gpu_guard
OUT=HERE/'outputs/readout_panel_v2'
BASE=ROOT/'archive/2026-11-05-adaptation-search/experiments/adaptation_screening/outputs/control_comparison_v1'
METHODS=['finetune','ridge_frozen','ridge_warmstart','ridge_profile','ridge_joint','ridge_epoch']
NEW=['ridge_frozen','ridge_warmstart','ridge_joint']

def execute_readout(job,method,cfg,data,out,phash,smoke=False):
    dest=out/method/job['job_id'];dest.mkdir(parents=True,exist_ok=True)
    ck=ROOT/job['checkpoint'] if job['checkpoint'] else None
    assert job['pretrain_seed'] in (7,None) and job['adaptation_seed'] in (7,17,37)
    if ck: assert digest(ck)==job['checkpoint_sha256']
    manifest=dict(job=job,method=method,config=cfg,protocol_sha256=phash,
        runner_sha256=digest(Path(__file__)),
        solver_sha256=digest(HERE/'readout.py'),
        ridge_lambda=.001,support_queries=256,query_queries=0,
        encoder_updates_per_epoch=12 if method == 'ridge_joint' else 0,
        head='fixed original hidden layer; last linear refit on full Train each epoch',
        dropout='disabled during support,query,and readout fitting',
        direction_objective=('joint support MSE plus .001 readout penalty' if method=='ridge_joint' else 'support MSE under epoch-fitted fixed readout'),
        training_readout=('ordinary Adam, restored after validation refit' if method=='ridge_joint' else 'full Train refit at epoch start'),test_used=False)
    if (dest/'manifest.json').exists(): assert json.loads((dest/'manifest.json').read_text())==manifest
    atomic_json(dest/'manifest.json',manifest)
    if (dest/'result.json').exists() and not smoke: return
    started=time.monotonic();seed=job['adaptation_seed'];d=data[job['target']]
    model=build(cfg,data,seed)
    if ck: model.load_shared_state(torch.load(ck,map_location='cpu',weights_only=False)['shared'])
    model.bind(d)
    for p in model.parameters():p.requires_grad_(False)
    ep=list(model.encoders[model.target].parameters())
    for p in ep:p.requires_grad_(method == 'ridge_joint')
    if method=='ridge_joint':
        readout_params=list(model.heads[model.target][2].parameters())
        for p in readout_params:p.requires_grad_(True)
        ep += readout_params
    opt=torch.optim.Adam(ep,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    fixed=d.negatives(100007);batches=list(d.batches(d.indices(0),cfg['batch_size']))
    history=[];decisions=[];best=-float('inf');best_epoch=None;start=0;prior=0.
    if (dest/'resume.pt').exists() and not smoke:
        s=torch.load(dest/'resume.pt',map_location=cfg['device'],weights_only=False)
        model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);restore_rng(s['rng'],cfg['device'])
        history=s['history'];decisions=s['decisions'];best=s['best'];best_epoch=s['best_epoch'];start=s['epoch'];prior=s['seconds']
    for epoch in range(start,cfg['epochs']):
        negatives=d.negatives(seed+epoch);losses=[]
        if method=='ridge_epoch':refit(model,d,negatives,cfg)
        if method == 'ridge_joint':
            for bi,native in enumerate(batches):
                support=native[np.linspace(0,len(native)-1,min(256,len(native)),dtype=int)]
                assert np.isin(support,d.indices(0)).all()
                model.bind(d);model.eval();opt.zero_grad(set_to_none=True)
                model.begin_batch(native);xs=hidden(model,d,support,negatives);ys=labels(len(support),xs.device,xs.dtype)
                w=model.heads[model.target][2].weight[0];b=model.heads[model.target][2].bias[0]
                loss=((xs@w+b-ys)**2).mean()+.001*w.square().sum()
                if not torch.isfinite(loss):raise FloatingPointError('Ridge query loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(ep,cfg['grad_clip'])
                if not torch.isfinite(norm):raise FloatingPointError('Ridge gradient')
                opt.step();losses.append(float(loss.detach()))
                decisions.append(dict(epoch=epoch+1,batch=bi,audit=dict(query_forwards=1,query_backwards=1,
                    support_size=len(support),query_size=0,support_max_time=float(d.time[support].max()),
                    training_risk=float(loss.detach()),loss=float(loss.detach()),gradient_norm=float(norm))))
                if smoke and len(decisions)==3:
                    atomic_json(dest/'smoke.json',dict(status='passed',decisions=decisions));return
        training_readout={k:v.detach().clone() for k,v in model.heads[model.target][2].state_dict().items()}
        fit=refit(model,d,negatives,cfg)
        metrics,scores=evaluate(model,d,1,fixed,cfg)
        history.append(dict(epoch=epoch+1,validation=metrics,train_loss=float(np.mean(losses)) if losses else None,readout=fit))
        if metrics['ap']>best:
            best=metrics['ap'];best_epoch=epoch+1
            atomic_torch(dest/'best.pt',dict(model=model.state_dict(),epoch=best_epoch))
            save_predictions(dest/'validation_predictions.npz',d,1,fixed,scores)
        if method=='ridge_joint':model.heads[model.target][2].load_state_dict(training_readout)
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
        full_train_readout_passes=10 if method=='ridge_epoch' else 5,encoder_updates=len(decisions),initialization=None))

def execute_warmstart(job,cfg,data,out,phash):
    dest=out/'ridge_warmstart'/job['job_id'];dest.mkdir(parents=True,exist_ok=True)
    manifest=dict(job=job,config=cfg,protocol_sha256=phash,runner_sha256=digest(Path(__file__)),test_used=False)
    if (dest/'manifest.json').exists(): assert json.loads((dest/'manifest.json').read_text())==manifest
    atomic_json(dest/'manifest.json',manifest)
    if (dest/'result.json').exists():return
    start=time.monotonic();model=build(cfg,data,job['adaptation_seed']);d=data[job['target']]
    if job['checkpoint']:
        ck=ROOT/job['checkpoint'];assert digest(ck)==job['checkpoint_sha256']
        model.load_shared_state(torch.load(ck,map_location='cpu',weights_only=False)['shared'])
    model.bind(d)
    if not (dest/'resume.pt').exists():
        rng=capture_rng(cfg['device'])
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        fit=refit(model,d,d.negatives(job['adaptation_seed']),cfg)
        assert all(torch.equal(p,before[n]) for n,p in model.named_parameters() if not n.startswith(f'heads.{model.target}.2.'))
        restore_rng(rng,cfg['device'])
        atomic_json(dest/'readout_initialization.json',fit)
    opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    history=[];best=-float('inf');best_epoch=None;first=0;prior=0
    if (dest/'resume.pt').exists():
        r=torch.load(dest/'resume.pt',map_location=cfg['device'],weights_only=False)
        model.load_state_dict(r['model']);opt.load_state_dict(r['optimizer']);restore_rng(r['rng'],cfg['device'])
        history=r['history'];best=r['best'];best_epoch=r['best_epoch'];first=r['epoch'];prior=r['seconds']
    fixed=d.negatives(100007)
    for epoch in range(first,5):
        neg=d.negatives(job['adaptation_seed']+epoch)
        losses=[train_batch(model,d,ids,neg,cfg,opt,1.) for ids in d.batches(d.indices(0),cfg['batch_size'])]
        metrics,scores=evaluate(model,d,1,fixed,cfg)
        history.append(dict(epoch=epoch+1,validation=metrics,train_loss=float(np.mean(losses))))
        if metrics['ap']>best:
            best=metrics['ap'];best_epoch=epoch+1
            atomic_torch(dest/'best.pt',dict(model=model.state_dict(),epoch=best_epoch))
            save_predictions(dest/'validation_predictions.npz',d,1,fixed,scores)
        atomic_torch(dest/'resume.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=capture_rng(cfg['device']),history=history,best=best,best_epoch=best_epoch,epoch=epoch+1,seconds=prior+time.monotonic()-start))
        print(json.dumps(dict(method='ridge_warmstart',job_id=job['job_id'],epoch=epoch+1,ap=metrics['ap'])),flush=True)
    with np.load(dest/'validation_predictions.npz') as a:ehash=array_hash(a)
    atomic_json(dest/'result.json',dict(status='complete',method='ridge_warmstart',job=job,ap=best,best_epoch=best_epoch,last_epoch_ap=history[-1]['validation']['ap'],history=history,seconds=prior+time.monotonic()-start,evaluation_sha256=ehash,evaluation_negative_seed=100007,test_used=False))

def prepare():
    jobs,cfgs,_=load_inputs(False);OUT.mkdir(exist_ok=True,parents=True)
    p=dict(jobs=jobs,configs=cfgs,methods=METHODS,runner_sha256=digest(Path(__file__)),readout_sha256=digest(HERE/'readout.py'),test_used=False)
    if (OUT/'plan.json').exists():assert json.loads((OUT/'plan.json').read_text())==p
    atomic_json(OUT/'plan.json',p);count=0
    for m in METHODS:
        for j in jobs:
            source=(HERE/'outputs'/m/j['job_id']) if m in ['finetune','ridge_epoch'] else BASE/m/j['job_id']
            if not (source/'result.json').exists():continue
            r=json.loads((source/'result.json').read_text());manifest=json.loads((source/'manifest.json').read_text())
            assert manifest['config']==cfgs[j['target']]
            for k in ['target','adaptation_seed','checkpoint_sha256']:assert r['job'][k]==j[k]
            assert set(r['job']['sources'])==set(j['sources'])
            dest=OUT/m/j['job_id'];dest.mkdir(exist_ok=True,parents=True)
            atomic_json(dest/'reuse.json',dict(reused_from=str(source.relative_to(ROOT)),result_sha256=digest(source/'result.json'),predictions_sha256=digest(source/'validation_predictions.npz')))
            read_result(OUT,m,j);count+=1
    assert count==243,count
    return p

def report(p):
    jobs=p['jobs'];rs={(m,j['job_id']):read_result(OUT,m,j) for m in METHODS for j in jobs}
    for j in jobs:assert len({rs[m,j['job_id']]['evaluation_sha256'] for m in METHODS})==1
    rows=[];summaries=[]
    for m in METHODS:
        to={(j['target'],j['adaptation_seed']):rs[m,j['job_id']]['ap'] for j in jobs if not j['sources']}
        stats=[]
        for t in p['configs']:
            group=[j for j in jobs if j['target']==t and j['sources']]
            aps=[rs[m,j['job_id']]['ap'] for j in group];controls=[to[t,j['adaptation_seed']] for j in group];gs=[a-b for a,b in zip(aps,controls)]
            sd={}
            for j,a,g in zip(group,aps,gs):
                sd.setdefault(j['case_id'],[]).append(a);rows.append(dict(method=m,job_id=j['job_id'],target=t,AP=a,G=g))
            stats.append(dict(AP=mean(aps),target_only_AP=mean(controls),G=mean(gs),D_G=mean(max(-g,0) for g in gs),B_G=mean(max(g,0) for g in gs),seed_SD=mean(pstdev(v) for v in sd.values())))
        summaries.append(dict(method=m,**{k:mean(v[k] for v in stats) for k in stats[0]}))
    def dominates(a,b):
        av=(a['AP'],-a['D_G'],a['B_G']);bv=(b['AP'],-b['D_G'],b['B_G'])
        return all(x>=y for x,y in zip(av,bv)) and any(x>y for x,y in zip(av,bv))
    for r in summaries:r['dominated_by']=[a['method'] for a in summaries if dominates(a,r)];r['pareto']=not r['dominated_by']
    atomic_json(OUT/'summary.json',dict(methods=summaries,aggregation='equal Target weight',jobs_per_method=54,test_used=False))
    with (OUT/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Readout strategy comparison: full fixed panel','', '14 Source configurations × 3 adaptation seeds + 12 matched Target-only = 54 runs/method; six methods, 324 runs. Pretraining seed 7. Validation only. Equal Target weighting.','', '| Method | Target-only AP | Transfer AP | G | D_G | B_G | Seed SD | Pareto |','|---|---:|---:|---:|---:|---:|---:|---|']
    for r in summaries:lines.append('| '+r['method']+' | '+' | '.join(f"{r[k]:.6f}" for k in ['target_only_AP','AP','G','D_G','B_G','seed_SD'])+f" | {r['pareto']} |")
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    atomic_json(OUT/'verification.json',dict(passed=True,runs=324,checks=['paired evaluation candidates','prediction AP','best epoch','reuse hashes','three replay checks'],test_used=False))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args();p=prepare();phash=canonical_hash(p)
    if not args.execute:print('324 total: 243 verified reuse, 81 new, plus 3 implementation replay checks');return
    atomic_json(OUT/'progress.json',dict(status='waiting_for_gpu',reused=243,new_total=81,total=324))
    # Wait without running GPU work alongside the ongoing LP-FT powerset.
    with open('/tmp/wellgfm-baselines-gpu0.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
    _,_,data=load_inputs();torch.set_num_threads(2)
    try:
        with gpu_guard('cuda:0',0):
            checks=[]
            for m in NEW:
                j=next(j for j in p['jobs'] if j['case_id']=='lastfm_mooc' and j['adaptation_seed']==7)
                replay=OUT/'replay'
                if m=='ridge_warmstart':execute_warmstart(j,p['configs'][j['target']],data,replay,phash)
                else:execute_readout(j,m,p['configs'][j['target']],data,replay,phash)
                fresh=read_result(replay,m,j);old=read_result(OUT,m,j)
                assert abs(fresh['ap']-old['ap'])<1e-6,(m,fresh['ap'],old['ap'])
                source=ROOT/json.loads((OUT/m/j['job_id']/'reuse.json').read_text())['reused_from']
                with np.load(source/'validation_predictions.npz') as a,np.load(replay/m/j['job_id']/'validation_predictions.npz') as b:np.testing.assert_allclose(a['scores'],b['scores'],rtol=1e-4,atol=1e-5)
                checks.append(dict(method=m,passed=True,ap=fresh['ap']))
            atomic_json(OUT/'replay_verification.json',checks)
            for m in NEW:
                for j in p['jobs']:
                    dest=OUT/m/j['job_id']
                    if (dest/'reuse.json').exists() or (dest/'result.json').exists():continue
                    n=sum((OUT/mm/jj['job_id']/'result.json').exists() for mm in NEW for jj in p['jobs'])
                    atomic_json(OUT/'progress.json',dict(status='running',completed=243+n,total=324,current_method=m,current=j['job_id']))
                    if m=='ridge_warmstart':execute_warmstart(j,p['configs'][j['target']],data,OUT,phash)
                    else:execute_readout(j,m,p['configs'][j['target']],data,OUT,phash)
                    read_result(OUT,m,j);torch.cuda.empty_cache()
        report(p);atomic_json(OUT/'progress.json',dict(status='complete',completed=324,total=324))
    except Exception as e:
        atomic_json(OUT/'progress.json',dict(status='failed',error=repr(e)));raise

if __name__=='__main__':main()
