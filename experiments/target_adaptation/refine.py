"""REFINE's frozen-feature / trainable-residual concatenation adapted to TGAT."""
import argparse
import csv
import json
import time
from pathlib import Path
from statistics import mean, pstdev
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from experiments.target_adaptation.runtime import ROOT, HERE, build, load_inputs, digest, canonical_hash, array_hash
from experiments.target_adaptation.readout import hidden
from experiments.common.engine import train_batch, save_predictions
from experiments.common.runtime import atomic_json, atomic_torch, capture_rng, restore_rng, evaluate, gpu_guard

class RefineTGAT(torch.nn.Module):
    def __init__(self, cfg, data, job):
        super().__init__()
        seed=job['adaptation_seed']
        self.source=build(cfg,data,seed)
        if job['checkpoint']:
            ck=ROOT/job['checkpoint'];assert digest(ck)==job['checkpoint_sha256']
            self.source.load_shared_state(torch.load(ck,map_location='cpu',weights_only=False)['shared'])
        self.source.requires_grad_(False)
        self.residual=build(cfg,data,seed)
        self.residual.requires_grad_(False)
        self.residual.encoders[self.residual.target].requires_grad_(True)
        self.residual.heads[self.residual.target][0].requires_grad_(True)
        self.classifier=torch.nn.Linear(344,1,device=cfg['device'])
        self.bind(data[job['target']])
    @property
    def encoder(self): return self.residual.encoder
    def bind(self,data):
        self.data=data;self.source.bind(data);self.residual.bind(data)
    def reset(self): self.source.reset();self.residual.reset()
    def begin_batch(self,ids): self.source.begin_batch(ids);self.residual.begin_batch(ids)
    def observe(self,ids): self.source.observe(ids);self.residual.observe(ids)
    def train(self, mode=True):
        super().train(mode);self.source.eval();return self
    def score(self,ids,negative):
        # hidden() accepts a full event-indexed negative array. Only ids are accessed.
        neg=np.empty(len(self.data.src),dtype=np.int64);neg[ids]=negative
        with torch.no_grad(): frozen=hidden(self.source,self.data,ids,neg)
        residual=hidden(self.residual,self.data,ids,neg)
        return self.classifier(torch.cat((residual,frozen),dim=-1)).reshape(2,len(ids)).T

def state_hash(model):
    import hashlib
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def execute(job,cfg,data,out,phash,smoke=False):
    dest=out/'refine'/job['job_id'];dest.mkdir(parents=True,exist_ok=True)
    manifest=dict(job=job,config=cfg,protocol_sha256=phash,runner_sha256=digest(Path(__file__)),test_used=False)
    if (dest/'manifest.json').exists():assert json.loads((dest/'manifest.json').read_text())==manifest
    atomic_json(dest/'manifest.json',manifest)
    if (dest/'result.json').exists():return
    start=time.monotonic();d=data[job['target']];model=RefineTGAT(cfg,data,job)
    frozen_hash=state_hash(model.source)
    # Actual inference graph counts (exclude unused domains and discarded final layers).
    source_modules=[model.source.encoder,model.source.heads[job['target']][0]]
    trainable=[p for p in model.parameters() if p.requires_grad]
    counts=dict(trainable=sum(p.numel() for p in trainable),frozen=sum(p.numel() for m in source_modules for p in m.parameters()))
    initial_residual=state_hash(model.residual)
    opt=torch.optim.Adam(trainable,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    fixed=d.negatives(100007);history=[];best=-float('inf');best_epoch=None;first=0;prior=0
    if (dest/'resume.pt').exists():
        r=torch.load(dest/'resume.pt',map_location=cfg['device'],weights_only=False)
        model.load_state_dict(r['model']);opt.load_state_dict(r['optimizer']);restore_rng(r['rng'],cfg['device'])
        history=r['history'];best=r['best'];best_epoch=r['best_epoch'];first=r['epoch'];prior=r['seconds']
    for epoch in range(first,5):
        neg=d.negatives(job['adaptation_seed']+epoch);losses=[]
        for bi,ids in enumerate(d.batches(d.indices(0),cfg['batch_size'])):
            losses.append(train_batch(model,d,ids,neg,cfg,opt,1.))
            assert all(p.grad is None for p in model.source.parameters())
            if smoke and bi==1:break
        assert state_hash(model.source)==frozen_hash
        assert state_hash(model.residual)!=initial_residual
        metrics,scores=evaluate(model,d,1,fixed,cfg)
        history.append(dict(epoch=epoch+1,validation=metrics,train_loss=mean(losses)))
        if metrics['ap']>best:
            best=metrics['ap'];best_epoch=epoch+1
            atomic_torch(dest/'best.pt',dict(model=model.state_dict(),epoch=best_epoch))
            save_predictions(dest/'validation_predictions.npz',d,1,fixed,scores)
        print(json.dumps(dict(method='refine',job=job['job_id'],epoch=epoch+1,ap=metrics['ap'])),flush=True)
        if smoke:
            # Saved-model inference must reproduce the same scores and AP.
            model.load_state_dict(torch.load(dest/'best.pt',map_location=cfg['device'],weights_only=False)['model'])
            again,ss=evaluate(model,d,1,fixed,cfg)
            assert np.allclose(ss,scores,rtol=1e-6,atol=1e-7)
            with np.load(HERE/'outputs/finetune'/job['job_id']/'validation_predictions.npz') as a:
                with np.load(dest/'validation_predictions.npz') as b:assert array_hash(a)==array_hash(b)
            atomic_json(dest/'smoke.json',dict(passed=True,frozen_unchanged=True,residual_updated=True,saved_inference_equal=True,candidates_equal=True,parameters=counts))
            return
        atomic_torch(dest/'resume.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=capture_rng(cfg['device']),history=history,best=best,best_epoch=best_epoch,epoch=epoch+1,seconds=prior+time.monotonic()-start))
    with np.load(dest/'validation_predictions.npz') as a:
        assert abs(average_precision_score(a['labels'].ravel(),a['scores'].ravel())-best)<1e-9
        ehash=array_hash(a)
    atomic_json(dest/'result.json',dict(status='complete',method='refine',job=job,ap=best,best_epoch=best_epoch,last_epoch_ap=history[-1]['validation']['ap'],history=history,seconds=prior+time.monotonic()-start,evaluation_sha256=ehash,parameters=counts,frozen_unchanged=True,test_used=False))

def summarize(jobs,out):
    results={};rows=[];summaries=[]
    for method,base in [('finetune',HERE/'outputs'),('lp_ft',HERE/'outputs/lp_ft_v1'),('ridge_epoch',HERE/'outputs'),('refine',out)]:
        rs={}
        for j in jobs:
            dest=base/method/j['job_id'];r=json.loads((dest/'result.json').read_text())
            with np.load(dest/'validation_predictions.npz') as a:
                assert abs(average_precision_score(a['labels'].ravel(),a['scores'].ravel())-r['ap'])<1e-9
                assert array_hash(a)==r['evaluation_sha256']
            assert abs(max(h['validation']['ap'] for h in r['history'])-r['ap'])<1e-9
            rs[j['job_id']]=r
        results[method]=rs
        controls={(j['target'],j['adaptation_seed']):rs[j['job_id']]['ap'] for j in jobs if not j['sources']}
        stats=[]
        for target in sorted(controls.keys()):pass
        for target in sorted({j['target'] for j in jobs}):
            aps=[];gs=[];groups={}
            for j in jobs:
                if j['target']!=target or not j['sources']:continue
                ap=rs[j['job_id']]['ap'];g=ap-controls[target,j['adaptation_seed']]
                rows.append(dict(method=method,job_id=j['job_id'],target=target,AP=ap,G=g))
                aps.append(ap);gs.append(g);groups.setdefault(j['case_id'],[]).append(ap)
            stats.append(dict(target=target,AP=mean(aps),D_G=mean(max(-g,0) for g in gs),B_G=mean(max(g,0) for g in gs),seed_SD=mean(pstdev(v) for v in groups.values())))
        summaries.append(dict(method=method,**{k:mean(x[k] for x in stats) for k in ['AP','D_G','B_G','seed_SD']},seconds=mean(r['seconds'] for r in rs.values()),by_target=stats))
    for j in jobs:assert len({rs[j['job_id']]['evaluation_sha256'] for rs in results.values()})==1
    atomic_json(out/'summary.json',dict(methods=summaries))
    with (out/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# REFINE TGAT comparison','', '14 fixed Source configurations, adaptation seeds 7/17/37. Equal Target weighting. Same-architecture Target-only for each method. Validation only; no Test.','', '| Method | AP | D_G | B_G | Seed SD | Seconds |','|---|---:|---:|---:|---:|---:|']
    for r in summaries:lines.append('| '+r['method']+' | '+' | '.join(f'{r[k]:.6f}' for k in ['AP','D_G','B_G','seed_SD','seconds'])+' |')
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    atomic_json(out/'verification.json',dict(passed=True,runs=54,common_candidates=True,prediction_ap=True,best_epoch=True,test_used=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--smoke',action='store_true');args=p.parse_args()
    jobs,cfgs,data=load_inputs(args.execute)
    if args.smoke:jobs=[next(j for j in jobs if j['case_id']=='lastfm_mooc'),next(j for j in jobs if j['target']=='lastfm' and not j['sources'])]
    out=HERE/'outputs'/('refine_smoke_v1' if args.smoke else 'refine_v1');out.mkdir(parents=True,exist_ok=True)
    plan=dict(jobs=jobs,configs=cfgs,runner_sha256=digest(Path(__file__)),upstream_commit='3f3afe223c096fcb12294856d4960f44e085a177',test_used=False)
    if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==plan
    atomic_json(out/'plan.json',plan)
    if not args.execute:print(f'{len(jobs)} runs planned');return
    torch.set_num_threads(2)
    try:
        with gpu_guard('cuda:0',0):
            for i,j in enumerate(jobs):
                atomic_json(out/'progress.json',dict(status='running',completed=i,total=len(jobs),current=j['job_id']))
                execute(j,cfgs[j['target']],data,out,canonical_hash(plan),args.smoke);torch.cuda.empty_cache()
        if not args.smoke:summarize(jobs,out)
        atomic_json(out/'progress.json',dict(status='complete',completed=len(jobs),total=len(jobs),verification='passed'))
    except Exception as e:
        atomic_json(out/'progress.json',dict(status='failed',error=repr(e)));raise
if __name__=='__main__':main()
