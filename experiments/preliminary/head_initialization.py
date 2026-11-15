"""Cross Head initialization with all other adaptation randomness on fixed checkpoints."""
import argparse
import hashlib
import json
import time
import shutil
from pathlib import Path
from statistics import mean
import numpy as np
import torch
from experiments.target_adaptation.runtime import ROOT,load_inputs,build,digest,array_hash
from experiments.common.engine import train_batch,save_predictions
from experiments.common.runtime import atomic_json,atomic_torch,capture_rng,restore_rng,evaluate,gpu_guard

HERE=Path(__file__).resolve().parent
DEFAULT_OUT=HERE/'head_initialization_v1'
SEEDS=[7,17,37]


def same(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def make_plan():
    source,cfgs,_=load_inputs(False)
    targets=sorted({j['target'] for j in source})
    cases={t:min((j for j in source if j['target']==t and j['sources']),key=lambda j:(len(j['sources']),j['case_id']))['case_id'] for t in targets}
    selected=[j for j in source if not j['sources'] or j['case_id']==cases[j['target']]]
    jobs=[dict(base=j,head_seed=h,id=f"{j['job_id']}__h{h}") for j in selected for h in SEEDS]
    return dict(selection_rule='Within each Target: minimum Source count, then lexical case_id; no AP used',cases=cases,jobs=jobs,configs=cfgs,pretrain_seed=7,adaptation_seeds=SEEDS,head_seeds=SEEDS,test_used=False)


def initialize(job,cfg,data):
    j=job['base'];model=build(cfg,data,j['adaptation_seed'])
    if j['checkpoint']:
        ck=ROOT/j['checkpoint'];assert digest(ck)==j['checkpoint_sha256']
        model.load_shared_state(torch.load(ck,map_location='cpu',weights_only=False)['shared'])
    before={n:v.detach().clone() for n,v in model.state_dict().items()}
    rng=capture_rng(cfg['device'])
    donor=build(cfg,data,job['head_seed'])
    model.heads[j['target']].load_state_dict(donor.heads[j['target']].state_dict())
    del donor
    restore_rng(rng,cfg['device']);assert same(rng,capture_rng(cfg['device']))
    prefix=f"heads.{j['target']}."
    state=model.state_dict()
    assert all(torch.equal(v,state[n]) for n,v in before.items() if not n.startswith(prefix))
    if job['head_seed']==j['adaptation_seed']:assert all(torch.equal(v,state[n]) for n,v in before.items())
    h=hashlib.sha256()
    for n,v in sorted(model.heads[j['target']].state_dict().items()):h.update(n.encode());h.update(v.detach().cpu().numpy().tobytes())
    return model,dict(non_head_state_identical=True,postbuild_rng_identical=True,head_sha256=h.hexdigest(),head_seed=job['head_seed'])


def execute(job,plan,data,out,smoke=False):
    j=job['base'];cfg=plan['configs'][j['target']];dest=out/job['id'];dest.mkdir(parents=True,exist_ok=True)
    manifest=dict(job=job,config=cfg,runner_sha256=digest(Path(__file__)),evaluation_negative_seed=100007,test_used=False)
    if (dest/'manifest.json').exists():assert json.loads((dest/'manifest.json').read_text())==manifest
    atomic_json(dest/'manifest.json',manifest)
    if (dest/'result.json').exists() and not smoke:return
    start=time.monotonic();model,audit=initialize(job,cfg,data);d=data[j['target']];model.bind(d)
    if job['head_seed']==j['adaptation_seed'] and not smoke:
        ref=ROOT/'experiments/target_adaptation/outputs/finetune'/j['job_id']
        original=json.loads((ref/'result.json').read_text())
        original_manifest=json.loads((ref/'manifest.json').read_text())
        assert original_manifest['config']==cfg
        assert original['job']['checkpoint_sha256']==j['checkpoint_sha256']
        assert original['job']['adaptation_seed']==j['adaptation_seed']
        assert original['status']=='complete' and len(original['history'])==5
        for name in ['best.pt','validation_predictions.npz']:shutil.copy2(ref/name,dest/name)
        atomic_json(dest/'result.json',dict(status='complete',job=job,intervention=audit,ap=original['ap'],last_epoch_ap=original['last_epoch_ap'],best_epoch=original['best_epoch'],history=original['history'],evaluation_sha256=original['evaluation_sha256'],seconds=0,test_used=False,reused_from=str(ref.relative_to(ROOT)),reused_result_sha256=digest(ref/'result.json'),diagonal_identity_checked=True))
        print(json.dumps(dict(job=job['id'],reused_diagonal=True,ap=original['ap'])),flush=True)
        return
    opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    assert not opt.state
    fixed=d.negatives(100007);history=[];best=-float('inf');step=0
    # Off-diagonal interventions use the same ordinary BCE adaptation as the reference.
    for epoch in range(5):
        neg=d.negatives(j['adaptation_seed']+epoch);losses=[]
        for ids in d.batches(d.indices(0),cfg['batch_size']):
            losses.append(train_batch(model,d,ids,neg,cfg,opt,1.));step+=1
            if smoke and step==3:
                atomic_json(dest/'smoke.json',dict(status='passed',intervention=audit,steps=step,losses=losses));return
        metrics,scores=evaluate(model,d,1,fixed,cfg)
        history.append(dict(epoch=epoch+1,validation=metrics,train_loss=float(np.mean(losses))))
        if metrics['ap']>best:
            best=metrics['ap'];best_epoch=epoch+1
            atomic_torch(dest/'best.pt',dict(model=model.state_dict(),epoch=best_epoch))
            save_predictions(dest/'validation_predictions.npz',d,1,fixed,scores)
        print(json.dumps(dict(job=job['id'],epoch=epoch+1,ap=metrics['ap'])),flush=True)
    with np.load(dest/'validation_predictions.npz') as a:
        ehash=array_hash(a)
        ref=ROOT/'experiments/target_adaptation/outputs/finetune'/j['job_id']
        with np.load(ref/'validation_predictions.npz') as b:
            assert array_hash(b)==ehash
            if job['head_seed']==j['adaptation_seed']:
                np.testing.assert_allclose(a['scores'],b['scores'],rtol=1e-5,atol=1e-6)
                reference=json.loads((ref/'result.json').read_text())
                assert abs(best-reference['ap'])<1e-6
    atomic_json(dest/'result.json',dict(status='complete',job=job,intervention=audit,ap=best,last_epoch_ap=history[-1]['validation']['ap'],best_epoch=best_epoch,history=history,evaluation_sha256=ehash,seconds=time.monotonic()-start,test_used=False,diagonal_replay=job['head_seed']==j['adaptation_seed']))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=DEFAULT_OUT);p.add_argument('--execute',action='store_true');p.add_argument('--smoke',action='store_true');args=p.parse_args()
    plan=make_plan();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'plan.json').exists():assert json.loads((args.out/'plan.json').read_text())==plan
    atomic_json(args.out/'plan.json',plan)
    print(json.dumps(dict(jobs=len(plan['jobs']),cases=plan['cases'],execute=args.execute)),flush=True)
    if not args.execute:return
    _,_,data=load_inputs();torch.set_num_threads(2)
    jobs=plan['jobs']
    if args.smoke:
        b=next(j for j in jobs if j['base']['sources'] and j['head_seed']==17 and j['base']['adaptation_seed']==7)
        jobs=[b]
    with gpu_guard('cuda:0',0):
        for job in jobs:execute(job,plan,data,args.out,args.smoke);torch.cuda.empty_cache()
    atomic_json(args.out/'progress.json',dict(status='smoke_complete' if args.smoke else 'complete',jobs=len(jobs)))

if __name__=='__main__':main()
