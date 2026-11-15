"""Fixed TFRA vs Fine-tuning evaluation on all existing seed-7 source subsets."""
import argparse
import csv
import itertools
import json
import time
from pathlib import Path
from statistics import mean, pstdev
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from experiments.target_adaptation.runtime import ROOT, HERE, load_inputs, digest, canonical_hash, array_hash, build
from experiments.target_adaptation.run import execute as execute_tfra
from experiments.common.engine import train_batch, save_predictions
from experiments.common.runtime import atomic_json, atomic_torch, capture_rng, restore_rng, evaluate, gpu_guard

OUT = HERE / 'outputs/powerset_v1'
METHODS = ['finetune', 'ridge_epoch']

def plan():
    old, cfgs, _ = load_inputs(False)
    lookup = {(j['target'], tuple(sorted(j['sources'])), j['adaptation_seed']): j for j in old}
    jobs = []
    base = ROOT / 'experiments/transfer/outputs/cross_representation_two_tkg_v1'
    for target, cfg in cfgs.items():
        pool = [n for n in cfg['datasets'] if n != target]
        for count in range(6):
            for sources in itertools.combinations(pool, count):
                ck = base / target / '+'.join(sources) / 'seed7/pretrain/final.pt' if sources else None
                if ck: assert ck.is_file(), ck
                sha = digest(ck) if ck else 'random_initialization'
                for seed in [7,17,37]:
                    key = (target, tuple(sorted(sources)), seed)
                    case = target + '__' + ('+'.join(sources) if sources else 'target_only')
                    job = dict(job_id=f'{case}__p{7 if sources else "none"}__a{seed}', case_id=case,
                        target=target, sources=list(sources), pretrain_seed=7 if sources else None,
                        adaptation_seed=seed, stage='powerset', evaluation_negative_seed=100007,
                        checkpoint=str(ck.relative_to(ROOT)) if ck else None, checkpoint_sha256=sha)
                    if key in lookup:
                        assert lookup[key]['checkpoint_sha256'] == sha
                        job['reuse_id'] = lookup[key]['job_id']
                    jobs.append(job)
    assert len(jobs)==384 and len({j['job_id'] for j in jobs})==384
    return dict(jobs=jobs, configs=cfgs, pretrain_seed=7, adaptation_seeds=[7,17,37],
        evaluation_negative_seed=100007, epochs=5, model_selection='best_validation_ap', test_used=False,
        runner_sha256=digest(Path(__file__)), tfra_runner_sha256=digest(HERE/'run.py'),
        readout_sha256=digest(HERE/'readout.py'))

def reuse(job, cfg, method, out):
    if 'reuse_id' not in job: return False
    source = HERE/'outputs'/method/job['reuse_id']
    result = json.loads((source/'result.json').read_text())
    manifest = json.loads((source/'manifest.json').read_text())
    assert manifest['config']==cfg
    assert result['status']=='complete' and len(result['history'])==5
    for k in ['target','adaptation_seed','pretrain_seed','checkpoint_sha256']:
        assert result['job'][k]==job[k], k
    assert set(result['job']['sources'])==set(job['sources'])
    assert result['evaluation_negative_seed']==100007
    with np.load(source/'validation_predictions.npz') as a:
        assert array_hash(a)==result['evaluation_sha256']
        assert abs(average_precision_score(a['labels'].ravel(),a['scores'].ravel())-result['ap'])<1e-10
    dest=out/method/job['job_id'];dest.mkdir(parents=True,exist_ok=True)
    pointer=dict(status='complete', reused_from=str(source.relative_to(ROOT)),
                 result_sha256=digest(source/'result.json'), predictions_sha256=digest(source/'validation_predictions.npz'))
    atomic_json(dest/'reuse.json',pointer)
    return True

def finetune(job,cfg,data,out,phash):
    dest=out/'finetune'/job['job_id'];dest.mkdir(parents=True,exist_ok=True)
    manifest=dict(job=job,config=cfg,protocol_sha256=phash,runner_sha256=digest(Path(__file__)),test_used=False)
    if (dest/'manifest.json').exists(): assert json.loads((dest/'manifest.json').read_text())==manifest
    atomic_json(dest/'manifest.json',manifest)
    if (dest/'result.json').exists():return
    start=time.monotonic();model=build(cfg,data,job['adaptation_seed']);d=data[job['target']]
    if job['checkpoint']:
        ck=ROOT/job['checkpoint'];assert digest(ck)==job['checkpoint_sha256']
        model.load_shared_state(torch.load(ck,map_location='cpu',weights_only=False)['shared'])
    model.bind(d);opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
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
        print(json.dumps(dict(method='finetune',job_id=job['job_id'],epoch=epoch+1,ap=metrics['ap'])),flush=True)
    with np.load(dest/'validation_predictions.npz') as a:ehash=array_hash(a)
    atomic_json(dest/'result.json',dict(status='complete',method='finetune',job=job,ap=best,best_epoch=best_epoch,last_epoch_ap=history[-1]['validation']['ap'],history=history,seconds=prior+time.monotonic()-start,evaluation_sha256=ehash,evaluation_negative_seed=100007,test_used=False))

def read_result(out,method,job):
    dest=out/method/job['job_id']
    if (dest/'reuse.json').exists():
        pointer=json.loads((dest/'reuse.json').read_text());dest=ROOT/pointer['reused_from']
        assert digest(dest/'result.json')==pointer['result_sha256']
        assert digest(dest/'validation_predictions.npz')==pointer['predictions_sha256']
    r=json.loads((dest/'result.json').read_text())
    assert r['status']=='complete' and len(r['history'])==5
    assert abs(r['ap']-max(h['validation']['ap'] for h in r['history']))<1e-12
    with np.load(dest/'validation_predictions.npz') as a:
        assert array_hash(a)==r['evaluation_sha256']
        assert abs(average_precision_score(a['labels'].ravel(),a['scores'].ravel())-r['ap'])<1e-10
    return r

def summarize(p,out):
    results={(m,j['job_id']):read_result(out,m,j) for j in p['jobs'] for m in METHODS}
    controls={(j['target'],j['adaptation_seed']):j['job_id'] for j in p['jobs'] if not j['sources']}
    hashes={}
    rows=[]
    for j in p['jobs']:
        for m in METHODS:
            h=results[m,j['job_id']]['evaluation_sha256'];t=j['target']
            if t in hashes: assert hashes[t]==h
            hashes[t]=h
        if not j['sources']:continue
        row={k:j[k] for k in ['job_id','case_id','target','adaptation_seed']};row['source_count']=len(j['sources'])
        for m in METHODS:
            r=results[m,j['job_id']];row[m+'_ap']=r['ap'];row[m+'_gain']=r['ap']-results[m,controls[j['target'],j['adaptation_seed']]]['ap']
        rows.append(row)
    def stats(rs,m):
        groups={}
        for r in rs:groups.setdefault(r['case_id'],[]).append(r[m+'_ap'])
        return dict(AP=mean(r[m+'_ap'] for r in rs),D_G=mean(max(-r[m+'_gain'],0) for r in rs),B_G=mean(max(r[m+'_gain'],0) for r in rs),seed_SD=mean(pstdev(v) for v in groups.values()),negative=sum(r[m+'_gain']<0 for r in rs),negative_over_005=sum(r[m+'_gain']<-.005 for r in rs))
    overall={m:stats(rows,m) for m in METHODS}
    by_target={t:{m:stats([r for r in rows if r['target']==t],m) for m in METHODS} for t in p['configs']}
    by_count={k:{m:stats([r for r in rows if r['source_count']==k],m) for m in METHODS} for k in range(1,6)}
    positive=[r for r in rows if r['finetune_gain']>0]
    retention=dict(baseline_positive=len(positive),retained=sum(r['ridge_epoch_gain']>0 for r in positive))
    edges=[]
    index={(j['target'],frozenset(j['sources']),j['adaptation_seed']):j for j in p['jobs']}
    for j in p['jobs']:
        if not j['sources']:continue
        for source in p['configs'][j['target']]['datasets']:
            if source==j['target'] or source in j['sources']:continue
            k=index[j['target'],frozenset(j['sources']+[source]),j['adaptation_seed']]
            edges.append(dict(target=j['target'],before=j['job_id'],after=k['job_id'],added=source,**{m:results[m,k['job_id']]['ap']-results[m,j['job_id']]['ap'] for m in METHODS}))
    assert len(rows)==372 and len(edges)==900
    composition={m:dict(mean=mean(e[m] for e in edges),D_C=mean(max(-e[m],0) for e in edges),negative=sum(e[m]<0 for e in edges)) for m in METHODS}
    costs={m:dict(mean_seconds=mean(results[m,j['job_id']]['seconds'] for j in p['jobs']),total_seconds=sum(results[m,j['job_id']]['seconds'] for j in p['jobs'])) for m in METHODS}
    atomic_json(out/'summary.json',dict(overall=overall,by_target=by_target,by_source_count=by_count,retention=retention,composition=composition,costs=costs))
    for name,records in [('comparison.csv',rows),('source_addition.csv',edges)]:
        with (out/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    labels={'finetune':'Fine-tuning','ridge_epoch':'TFRA'}
    lines=['# Target-Fitted Readout Adaptation (TFRA): 전체 Source 조합 비교','', '사전학습 seed 7, 적응 seed 7·17·37. 네 Target × 32구성 × 3 seed × 2방법 = 768회. Validation 최고 epoch 선택, Test 미사용. 개발 패널을 포함하는 전체 조합 확장 평가이며 독립 Test 평가는 아니다.','']
    for title,group in [('전체',overall)]+[(t,by_target[t]) for t in by_target]+[(f'Source {k}개',by_count[k]) for k in by_count]:
        lines += [f'## {title}','','| 방법 | AP | D_G | B_G | Seed SD | G < −0.005 |','|---|---:|---:|---:|---:|---:|']
        for m,v in group.items():lines.append('| '+labels[m]+' | '+' | '.join(f'{v[k]:.6f}' for k in ['AP','D_G','B_G','seed_SD'])+f" | {v['negative_over_005']} |")
        lines.append('')
    lines += ['## 양의 전이 보존', '', f"Fine-tuning의 양의 전이 {retention['baseline_positive']}건 중 TFRA에서 {retention['retained']}건 유지.", '', '## Source 추가 효과', '', '| 방법 | 평균 C | D_C | C < 0 / 900 |','|---|---:|---:|---:|']
    for m,v in composition.items():lines.append(f"| {labels[m]} | {v['mean']:.6f} | {v['D_C']:.6f} | {v['negative']} |")
    lines+=['','## 계산 비용','','실행당 평균 시간은 재사용된 기존 실행의 기록을 포함하므로 하드웨어 부하에 따른 차이가 있다. TFRA는 전체 Train 예측층 적합 10회와 부분 배치 Encoder 업데이트 60회, Fine-tuning은 전체 배치 업데이트 60회를 수행한다.']
    for m,v in costs.items():lines.append(f"- {labels[m]}: 평균 {v['mean_seconds']:.2f}초")
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    atomic_json(out/'verification.json',dict(passed=True,results=768,transfer_comparisons=372,source_additions=900,evaluation_hashes=hashes,checks=['prediction AP','best epoch AP','paired evaluation candidates','reused file hashes'],test_used=False))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--replay-check',action='store_true');args=parser.parse_args()
    p=plan();out=args.out;out.mkdir(parents=True,exist_ok=True)
    if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==p
    atomic_json(out/'plan.json',p)
    phash=canonical_hash(p)
    if not args.execute:
        for j in p['jobs']:
            for m in METHODS:reuse(j,p['configs'][j['target']],m,out)
        print(json.dumps(dict(total=768,reused=sum('reuse_id' in j for j in p['jobs'])*2,new=sum('reuse_id' not in j for j in p['jobs'])*2)));return
    _,_,data=load_inputs();torch.set_num_threads(2)
    with gpu_guard('cuda:0',0):
        if args.replay_check:
            j=next(j for j in p['jobs'] if j.get('reuse_id')=='wiki_reddit__p7__a7')
            finetune(j,p['configs'][j['target']],data,out,phash)
            fresh=read_result(out,'finetune',j);old=json.loads((HERE/'outputs/finetune'/j['reuse_id']/'result.json').read_text())
            assert abs(fresh['ap']-old['ap'])<1e-7
            with np.load(out/'finetune'/j['job_id']/'validation_predictions.npz') as a,np.load(HERE/'outputs/finetune'/j['reuse_id']/'validation_predictions.npz') as b:
                np.testing.assert_allclose(a['scores'],b['scores'],rtol=1e-5,atol=1e-6)
            atomic_json(out/'replay_check.json',dict(passed=True,ap=fresh['ap']));return
        completed=0
        for j in p['jobs']:
            for m in METHODS:
                atomic_json(out/'progress.json',dict(status='running',completed=completed,total=768,current_method=m,current_job=j['job_id']))
                if not reuse(j,p['configs'][j['target']],m,out):
                    (finetune if m=='finetune' else execute_tfra)(j,p['configs'][j['target']],data,out,phash)
                completed+=1;torch.cuda.empty_cache()
    summarize(p,out)
    atomic_json(out/'progress.json',dict(status='complete',completed=768,total=768,verification='passed'))

if __name__=='__main__':main()
