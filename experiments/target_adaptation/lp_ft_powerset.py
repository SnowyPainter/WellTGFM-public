"""Expand unchanged LP-FT to all existing pretraining-seed-7 checkpoints."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean,pstdev
import torch
from experiments.target_adaptation.runtime import ROOT,HERE,load_inputs,digest,canonical_hash
from experiments.target_adaptation.powerset import plan as base_plan,read_result
from experiments.target_adaptation.lp_ft import execute
from experiments.common.runtime import atomic_json,gpu_guard

OUT=HERE/'outputs/lp_ft_powerset_v1'
METHODS=['finetune','ridge_epoch','lp_ft']

def prepare(out):
    p=base_plan()
    p.update(lp_ft_runner_sha256=digest(HERE/'lp_ft.py'),expansion_runner_sha256=digest(Path(__file__)),methods=METHODS,
        C_grid=[.001,.01,.1,1.,10.,100.,1000.],lp_ft_protocol='unchanged lp_ft_v1; L2 logistic probe, Validation C selection, existing five-epoch BCE FT')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==p
    atomic_json(out/'plan.json',p)
    reused=0
    for j in p['jobs']:
        for m in METHODS:
            if m=='lp_ft':
                if 'reuse_id' not in j:continue
                source=HERE/'outputs/lp_ft_v1/lp_ft'/j['reuse_id']
            else:
                source=HERE/'outputs/powerset_v1'/m/j['job_id']
                if (source/'reuse.json').exists():source=ROOT/json.loads((source/'reuse.json').read_text())['reused_from']
            r=json.loads((source/'result.json').read_text());manifest=json.loads((source/'manifest.json').read_text())
            assert manifest['config']==p['configs'][j['target']]
            for key in ['target','adaptation_seed','pretrain_seed','checkpoint_sha256']:assert r['job'][key]==j[key]
            assert set(r['job']['sources'])==set(j['sources'])
            if m=='lp_ft':assert manifest['runner_sha256']==p['lp_ft_runner_sha256']
            dest=out/m/j['job_id'];dest.mkdir(parents=True,exist_ok=True)
            atomic_json(dest/'reuse.json',dict(status='complete',reused_from=str(source.relative_to(ROOT)),result_sha256=digest(source/'result.json'),predictions_sha256=digest(source/'validation_predictions.npz')))
            read_result(out,m,j)
            reused+=m=='lp_ft'
    assert reused==54
    return p

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
    labels={'finetune':'Fine-tuning','ridge_epoch':'TFRA','lp_ft':'LP-FT'}
    lines=['# Fine-tuning, TFRA, LP-FT: 전체 Source 조합 비교','', '사전학습 seed 7, 적응 seed 7·17·37. 네 Target × 32구성 × 3 seed × 3방법 = 1,152회. Validation 최고 epoch 선택, Test 미사용. 개발 패널을 포함하는 전체 조합 확장 평가이며 독립 Test 평가는 아니다.','']
    for title,group in [('전체',overall)]+[(t,by_target[t]) for t in by_target]+[(f'Source {k}개',by_count[k]) for k in by_count]:
        lines += [f'## {title}','','| 방법 | AP | D_G | B_G | Seed SD | G < −0.005 |','|---|---:|---:|---:|---:|---:|']
        for m,v in group.items():lines.append('| '+labels[m]+' | '+' | '.join(f'{v[k]:.6f}' for k in ['AP','D_G','B_G','seed_SD'])+f" | {v['negative_over_005']} |")
        lines.append('')
    lines += ['## 양의 전이 보존', '', f"Fine-tuning의 양의 전이 {retention['baseline_positive']}건 중 TFRA에서 {retention['retained']}건 유지.", '', '## Source 추가 효과', '', '| 방법 | 평균 C | D_C | C < 0 / 900 |','|---|---:|---:|---:|']
    for m,v in composition.items():lines.append(f"| {labels[m]} | {v['mean']:.6f} | {v['D_C']:.6f} | {v['negative']} |")
    lines+=['','## 계산 비용','','실행당 평균 시간은 재사용된 기존 실행의 기록을 포함하므로 하드웨어 부하에 따른 차이가 있다. TFRA는 전체 Train 예측층 적합 10회와 부분 배치 Encoder 업데이트 60회, Fine-tuning은 전체 배치 업데이트 60회를 수행한다.']
    for m,v in costs.items():lines.append(f"- {labels[m]}: 평균 {v['mean_seconds']:.2f}초")
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    atomic_json(out/'verification.json',dict(passed=True,results=1152,transfer_comparisons=372,source_additions=900,evaluation_hashes=hashes,checks=['prediction AP','best epoch AP','paired evaluation candidates','reused file hashes'],test_used=False))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    p=prepare(OUT);jobs=p['jobs'];phash=canonical_hash(p)
    if not args.execute:print('LP-FT: 384 total, 54 verified reused, 330 new; FT/TFRA: 768 verified reused');return
    _,_,data=load_inputs();torch.set_num_threads(2)
    completed=sum((OUT/'lp_ft'/j['job_id']/'reuse.json').exists() or (OUT/'lp_ft'/j['job_id']/'result.json').exists() for j in jobs)
    try:
        with gpu_guard('cuda:0',0):
            for j in jobs:
                dest=OUT/'lp_ft'/j['job_id']
                if (dest/'reuse.json').exists() or (dest/'result.json').exists():
                    read_result(OUT,'lp_ft',j);continue
                atomic_json(OUT/'progress.json',dict(status='running',completed=completed,total=384,reused=54,new_total=330,current=j['job_id']))
                execute(j,p['configs'][j['target']],data,OUT,phash)
                read_result(OUT,'lp_ft',j);completed+=1;torch.cuda.empty_cache()
        summarize(p,OUT)
        atomic_json(OUT/'progress.json',dict(status='complete',completed=384,total=384,reused=54,verification='passed'))
    except Exception as e:
        atomic_json(OUT/'progress.json',dict(status='failed',completed=completed,total=384,error=repr(e)))
        raise

if __name__=='__main__':main()
