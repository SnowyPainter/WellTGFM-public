"""LP-FT adapted from Kumar et al. (ICLR 2022), Sections 4.1–4.2."""
import argparse
import csv
import json
import time
import warnings
from pathlib import Path
from statistics import mean, pstdev
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score
from experiments.target_adaptation.runtime import ROOT, HERE, load_inputs, digest, canonical_hash, array_hash, build
from experiments.target_adaptation.readout import hidden
from experiments.common.engine import train_batch, save_predictions
from experiments.common.runtime import atomic_json, atomic_torch, capture_rng, restore_rng, evaluate, gpu_guard

@torch.no_grad()
def features(model,d,cfg,split,neg):
    model.bind(d);model.eval();xs=[];ys=[]
    for native in d.batches(d.indices(split),cfg['batch_size']):
        model.begin_batch(native)
        for offset in range(0,len(native),cfg['microbatch']):
            ids=native[offset:offset+cfg['microbatch']]
            xs.append(hidden(model,d,ids,neg).cpu().double().numpy())
            ys.append(np.r_[np.ones(len(ids)),np.zeros(len(ids))])
    return np.concatenate(xs),np.concatenate(ys)

@torch.no_grad()
def fit_probe(model,d,cfg,seed,dest):
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    x,y=features(model,d,cfg,0,d.negatives(seed))
    xv,yv=features(model,d,cfg,1,d.negatives(100007))
    trials=[];best=None
    # Fixed adaptation-specific grid, not claimed to be the paper's unpublished grid.
    for c in [0.001,0.01,0.1,1.,10.,100.,1000.]:
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            clf=LogisticRegression(C=c,solver='newton-cholesky',max_iter=200,tol=1e-8,random_state=seed).fit(x,y)
        ap=float(average_precision_score(yv,clf.decision_function(xv)))
        trials.append(dict(C=c,validation_ap=ap,iterations=int(clf.n_iter_[0])))
        if best is None or ap>best[0]:best=(ap,c,clf)
    head=model.heads[model.target][2];clf=best[2]
    head.weight.copy_(torch.as_tensor(clf.coef_,device=head.weight.device,dtype=head.weight.dtype))
    head.bias.copy_(torch.as_tensor(clf.intercept_,device=head.bias.device,dtype=head.bias.dtype))
    prefix=f'heads.{model.target}.2.'
    assert all(torch.equal(p,before[n]) for n,p in model.named_parameters() if not n.startswith(prefix))
    atomic_json(dest/'linear_probe.json',dict(trials=trials,selected_C=best[1],validation_ap=best[0],train_examples=len(y),encoder_and_hidden_head_unchanged=True,test_used=False))

def execute(job,cfg,data,out,phash):
    dest=out/'lp_ft'/job['job_id'];dest.mkdir(parents=True,exist_ok=True)
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
        rng = capture_rng(cfg['device'])
        fit_probe(model,d,cfg,job['adaptation_seed'],dest)
        restore_rng(rng,cfg['device'])
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
        print(json.dumps(dict(method='lp_ft',job_id=job['job_id'],epoch=epoch+1,ap=metrics['ap'])),flush=True)
    with np.load(dest/'validation_predictions.npz') as a:ehash=array_hash(a)
    atomic_json(dest/'result.json',dict(status='complete',method='lp_ft',job=job,ap=best,best_epoch=best_epoch,last_epoch_ap=history[-1]['validation']['ap'],history=history,seconds=prior+time.monotonic()-start,evaluation_sha256=ehash,evaluation_negative_seed=100007,test_used=False))

def summarize(jobs,out):
    results={}
    for method in ['finetune','ridge_epoch','lp_ft']:
        base=out if method=='lp_ft' else HERE/'outputs'
        results[method]={}
        for j in jobs:
            dest=base/method/j['job_id'];r=json.loads((dest/'result.json').read_text())
            with np.load(dest/'validation_predictions.npz') as a:
                assert abs(average_precision_score(a['labels'].ravel(),a['scores'].ravel())-r['ap'])<1e-9
                assert array_hash(a)==r['evaluation_sha256']
            assert abs(r['ap']-max(h['validation']['ap'] for h in r['history']))<1e-9
            results[method][j['job_id']]=r
    for j in jobs:
        assert len({results[m][j['job_id']]['evaluation_sha256'] for m in results})==1
    rows=[];lines=['# LP-FT development-panel comparison','', 'Pretraining seed 7; adaptation seeds 7, 17, 37. Validation only. Each method uses its own Target-only. Transfer runs are equally weighted.','', '| Method | AP | D_G | B_G | Mean seed SD | Mean seconds (all runs) |','|---|---:|---:|---:|---:|---:|']
    for m,rs in results.items():
        to={(j['target'],j['adaptation_seed']):rs[j['job_id']]['ap'] for j in jobs if not j['sources']}
        aps=[];gs=[];groups={}
        for j in jobs:
            if not j['sources']:continue
            r=rs[j['job_id']];g=r['ap']-to[j['target'],j['adaptation_seed']]
            aps.append(r['ap']);gs.append(g);groups.setdefault(j['case_id'],[]).append(r['ap'])
            rows.append(dict(method=m,job_id=j['job_id'],target=j['target'],ap=r['ap'],gain=g))
        lines.append(f"| {m} | {mean(aps):.6f} | {mean(max(-g,0) for g in gs):.6f} | {mean(max(g,0) for g in gs):.6f} | {mean(pstdev(v) for v in groups.values()):.6f} | {mean(r['seconds'] for r in rs.values()):.1f} |")
    with (out/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    atomic_json(out/'verification.json',dict(passed=True,evaluation_candidates_identical=True,stored_prediction_ap_verified=True,test_used=False))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    jobs,cfgs,data=load_inputs(args.execute);out=HERE/'outputs/lp_ft_v1';out.mkdir(parents=True,exist_ok=True)
    p=dict(jobs=jobs,configs=cfgs,runner_sha256=digest(Path(__file__)),C_grid=[.001,.01,.1,1.,10.,100.,1000.],lp='L2 logistic regression on frozen penultimate features; validation AP selects C',ft='all parameters BCE Adam five epochs, existing fixed LR',test_used=False)
    if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==p
    atomic_json(out/'plan.json',p)
    if not args.execute:print(f'{len(jobs)} LP-FT runs planned');return
    torch.set_num_threads(2)
    with gpu_guard('cuda:0',0):
        for i,j in enumerate(jobs):
            atomic_json(out/'progress.json',dict(status='running',completed=i,total=len(jobs),current=j['job_id']))
            execute(j,cfgs[j['target']],data,out,canonical_hash(p));torch.cuda.empty_cache()
    summarize(jobs,out)
    atomic_json(out/'progress.json',dict(status='complete',completed=len(jobs),total=len(jobs)))

if __name__=='__main__':main()
