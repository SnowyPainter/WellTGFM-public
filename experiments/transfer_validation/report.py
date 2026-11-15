"""Paired summaries; experimental conditions remain separate, with no Test selection."""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.common.runtime import atomic_json
from experiments.transfer_validation.core import ROOT, read


def csv_write(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def records(out,split):
    rows=[]
    files=(out/('test' if split=='test' else 'jobs')).glob('*/result.json')
    for p in files:
        r=read(p)
        if r['status']!='complete':continue
        j=r['job'];cfg=j.get('config',{})
        common=dict(job_id=j['id'],cohort=j['cohort'],profile=j['profile'],backbone=j['backbone'],
            target=j['target'],sources='+'.join(sorted(j['sources'])),pretrain_seed=j['pretrain_seed'],
            adaptation_seed=j['adaptation_seed'],method=j['method'],query_fraction=cfg.get('query_fraction',1.),
            source_budget=j.get('source_budget'),hidden_seed=j.get('hidden_seed'),final_seed=j.get('final_seed'),
            trial=j.get('trial'),split=split)
        if split=='test':
            for evaluation,m in r['metrics'].items():rows.append(dict(**common,evaluation=evaluation,ap=m['ap'],mrr=m.get('mrr')))
        else:
            rows.append(dict(**common,evaluation='random_1',ap=r['ap'],seconds=r['seconds'],best_epoch=r['best_epoch']))
    return rows


def control_key(r):
    return tuple(r.get(k) for k in ['cohort','profile','backbone','target','method','adaptation_seed',
                                   'query_fraction','hidden_seed','final_seed','trial','evaluation'])


def summarize(rows):
    controls={control_key(r):r['ap'] for r in rows if not r['sources']}
    transfer=[]
    for r in rows:
        if not r['sources']:continue
        key=control_key(r)
        if key not in controls:continue
        transfer.append(dict(**r,target_only_ap=controls[key],gain=r['ap']-controls[key]))
    groups=defaultdict(list)
    keys=['cohort','profile','backbone','query_fraction','source_budget','method','evaluation']
    # Tuning has different trials; never average distinct trial arms as one method.
    keys+=['trial']
    for r in transfer:groups[tuple(r.get(k) for k in keys)].append(r)
    summary=[]
    for k,rr in sorted(groups.items(),key=lambda x:str(x[0])):
        per_target=[]
        for t in sorted({r['target'] for r in rr}):
            x=[r for r in rr if r['target']==t];g=np.array([r['gain'] for r in x])
            # Seed SD is within the same checkpoint; do not mix pretraining effects.
            by=defaultdict(list)
            for r in x:by[(r['sources'],r['pretrain_seed'],r['hidden_seed'],r['final_seed'])].append(r)
            sd=[np.std([r['ap'] for r in z]) for z in by.values() if len(z)>=2]
            stable=[all(r['gain']>0 for r in z) for z in by.values() if len(z)>=3]
            row=dict(**dict(zip(keys,k)),target=t,runs=len(x),ap=float(np.mean([r['ap'] for r in x])),
                target_only_ap=float(np.mean([r['target_only_ap'] for r in x])),
                D_G=float(np.maximum(-g,0).mean()),B_G=float(np.maximum(g,0).mean()),
                mean_gain=float(g.mean()),negative=int((g<0).sum()),large_negative=int((g<-.005).sum()),
                seed_sd=float(np.mean(sd)) if sd else None,
                stable_positive=sum(stable),stability_groups=len(stable))
            per_target.append(row);summary.append(row)
        overall=dict(**dict(zip(keys,k)),target='target_equal_mean',runs=len(rr))
        for metric in ['ap','target_only_ap','D_G','B_G','mean_gain']:
            overall[metric]=float(np.mean([r[metric] for r in per_target]))
        for metric in ['negative','large_negative','stable_positive','stability_groups']:
            overall[metric]=sum(r[metric] for r in per_target)
        s=[r['seed_sd'] for r in per_target if r['seed_sd'] is not None]
        overall['seed_sd']=float(np.mean(s)) if s else None
        summary.append(overall)
    return transfer,summary


def paired_seed_intervals(transfer,draws=2000):
    """Conditional on observed graphs/compositions; crossed seeds are shared draws.

    We never resample 900 source-addition edges as IID observations. A bootstrap
    replicate draws each pretraining/adaptation seed axis once and uses it across
    all compositions, methods and scratch controls within the same fixed Target.
    These intervals describe seed sensitivity, not uncertainty over unseen graphs.
    """
    rr=[r for r in transfer if r['cohort']=='seed_factorial' and r['evaluation']=='random_1']
    lookup={(r['target'],r['sources'],r['pretrain_seed'],r['adaptation_seed'],r['method']):r for r in rr}
    rows=[];rng=np.random.default_rng(910007)
    for target in sorted({r['target'] for r in rr}):
        compositions=sorted({r['sources'] for r in rr if r['target']==target})
        ps=sorted({r['pretrain_seed'] for r in rr if r['target']==target})
        ads=sorted({r['adaptation_seed'] for r in rr if r['target']==target})
        for baseline in ['finetune','lp_ft','refine']:
            keys=[(target,s,p,a) for s in compositions for p in ps for a in ads]
            if not all(k+(m,) in lookup for k in keys for m in ['tfra',baseline]):continue
            arrays={}
            for metric in ['ap','gain','damage','benefit']:
                def value(r):
                    if metric=='damage':return max(-r['gain'],0.)
                    if metric=='benefit':return max(r['gain'],0.)
                    return r[metric]
                ar=np.array([[[value(lookup[(target,s,p,a,'tfra')])-value(lookup[(target,s,p,a,baseline)])
                              for a in ads] for p in ps] for s in compositions])
                samples=[]
                for _ in range(draws):
                    ip=rng.integers(len(ps),size=len(ps));ia=rng.integers(len(ads),size=len(ads))
                    samples.append(ar[:,ip][:,:,ia].mean())
                rows.append(dict(target=target,baseline=baseline,metric=metric,difference=float(ar.mean()),
                    lower=float(np.quantile(samples,.025)),upper=float(np.quantile(samples,.975)),
                    interpretation='seed-conditional percentile interval; fixed observed graphs and source sets'))
    return rows


def source_contributions(transfer):
    rr=[r for r in transfer if r['cohort']=='source_budget'];rows=[]
    by=defaultdict(list)
    for r in rr:
        # None = total-60; other values are 60 or 300 updates per Source.
        regime='total60' if r['source_budget'] is None else 'per_source'+str(r['source_budget']//len(r['sources'].split('+')))
        by[(r['target'],r['method'],r['pretrain_seed'],r['adaptation_seed'],regime,r['evaluation'])].append(r)
    for key,items in by.items():
        for a in items:
            sa=set(a['sources'].split('+'))
            for b in items:
                sb=set(b['sources'].split('+'))
                if sa<sb and len(sb)==len(sa)+1:
                    rows.append(dict(target=key[0],method=key[1],pretrain_seed=key[2],adaptation_seed=key[3],
                        regime=key[4],evaluation=key[5],before=a['sources'],after=b['sources'],
                        contribution=b['ap']-a['ap'],before_gain=a['gain'],after_gain=b['gain']))
    return rows


def head_decomposition(rows):
    groups=defaultdict(list);out=[]
    for r in rows:
        if r['cohort']=='head_components' and r['evaluation']=='random_1':
            groups[(r['target'],r['sources'],r['method'],r['adaptation_seed'])].append(r)
    for key,rr in groups.items():
        if len(rr)!=9:continue
        h=sorted({r['hidden_seed'] for r in rr});f=sorted({r['final_seed'] for r in rr})
        a=np.array([[next(r['ap'] for r in rr if r['hidden_seed']==i and r['final_seed']==j) for j in f] for i in h])
        hidden=a.mean(1)-a.mean();final=a.mean(0)-a.mean();interaction=a-a.mean()-hidden[:,None]-final[None,:]
        out.append(dict(target=key[0],sources=key[1],method=key[2],adaptation_seed=key[3],
            hidden_rms=float(np.sqrt(np.mean(hidden**2))),final_rms=float(np.sqrt(np.mean(final**2))),
            interaction_rms=float(np.sqrt(np.mean(interaction**2))),range=float(np.ptp(a))))
    return out


def common_scratch(rows,out):
    """A common practical reference selected exclusively on Validation."""
    freeze=out/'test_freeze.json'
    if not freeze.exists():return []
    refs=read(freeze)['common_scratch_references']
    references={(r['cohort'],r['profile'],r['backbone'],r['target'],r['query_fraction']):r['method'] for r in refs}
    controls={(r['cohort'],r['profile'],r['backbone'],r['target'],r['query_fraction'],r['method'],r['adaptation_seed'],r['evaluation']):r['ap']
              for r in rows if not r['sources'] and r['hidden_seed'] is None and r['trial'] is None}
    result=[]
    for r in rows:
        if not r['sources'] or r['hidden_seed'] is not None or r['trial'] is not None:continue
        key=(r['cohort'],r['profile'],r['backbone'],r['target'],r['query_fraction'])
        m=references.get(key);lookup=key+(m,r['adaptation_seed'],r['evaluation'])
        if lookup not in controls:continue
        result.append(dict(**r,common_scratch_method=m,common_scratch_ap=controls[lookup],
                           gain_over_common_scratch=r['ap']-controls[lookup]))
    return result


def learning_curves(out):
    rows=[]
    for p in (out/'jobs').glob('*/result.json'):
        r=read(p);j=r['job']
        if j['cohort'] not in {'long_budget','query_budget','tuning'}:continue
        best=r['initial_validation']['ap'] if r.get('initial_validation') else -np.inf
        for e in r['history']:
            best=max(best,e['validation']['ap'])
            rows.append(dict(job_id=j['id'],cohort=j['cohort'],target=j['target'],sources='+'.join(j['sources']),
                method=j['method'],seed=j['adaptation_seed'],query_fraction=j['config']['query_fraction'],
                epoch=e['epoch'],seconds=e['seconds'],ap=e['validation']['ap'],best_ap=best,
                updates=e['encoder_updates'],fits=e['readout_fits']))
    return rows


def source_fit_pairs(out,contributions):
    source={}
    for p in (out/'source_fit').glob('*/result.json'):
        x=read(p);c=x['case'];s='+'.join(sorted(c['sources']))
        regime='total60' if c['source_budget'] is None else 'per_source'+str(c['source_budget']//len(c['sources']))
        source[c['target'],s,c['pretrain_seed'],regime]={r['source']:r['ap'] for r in x['rows']}
    rows=[]
    for r in contributions:
        k=r['target'],r['before'],r['pretrain_seed'],r['regime']
        kk=r['target'],r['after'],r['pretrain_seed'],r['regime']
        if k not in source or kk not in source:continue
        deltas={n:source[kk][n]-v for n,v in source[k].items()}
        rows.append(dict(**r,mean_source_change=float(np.mean(list(deltas.values()))),
            all_old_sources_improve=all(v>0 for v in deltas.values()),
            all_improve_but_target_declines=all(v>0 for v in deltas.values()) and r['contribution']<0,
            per_source=json_text(deltas)))
    return rows


def json_text(value):
    import json
    return json.dumps(value,sort_keys=True)


def report(out,final=False):
    root=out/'analysis';root.mkdir(parents=True,exist_ok=True)
    csv_write(root/'learning_curves.csv',learning_curves(out))
    for split in ['validation','test']:
        rows=records(out,split)
        if not rows:continue
        tr,summary=summarize(rows)
        csv_write(root/f'{split}_runs.csv',rows);csv_write(root/f'{split}_transfer.csv',tr)
        csv_write(root/f'{split}_summary.csv',summary)
        csv_write(root/f'{split}_source_additions.csv',source_contributions(tr))
        csv_write(root/f'{split}_source_fit_pairs.csv',source_fit_pairs(out,source_contributions(tr)))
        csv_write(root/f'{split}_common_scratch.csv',common_scratch(rows,out))
        csv_write(root/f'{split}_head_components.csv',head_decomposition(rows))
        if final:csv_write(root/f'{split}_paired_seed_intervals.csv',paired_seed_intervals(tr))
        atomic_json(root/f'{split}_summary.json',summary)
    diagnostics=[]
    for p in (out/'jobs').glob('*/result.json'):
        r=read(p);j=r['job']
        diagnostics.extend(dict(job_id=j['id'],target=j['target'],sources='+'.join(j['sources']),
                                method=j['method'],seed=j['adaptation_seed'],**d) for d in r['diagnostics'])
    csv_write(root/'gradient_diagnostics.csv',diagnostics)
    return root


def append_claim(out):
    """Append measured results only after the complete sealed evaluation succeeds."""
    rows=read(out/'analysis/test_summary.json')
    chosen=[r for r in rows if r['target']=='target_equal_mean' and r['evaluation']=='random_1'
            and r['cohort'] in {'seed_factorial','breadth','scale','long_budget'}]
    lines=['<!-- transfer-validation-v1:start -->','## Appendix D. Additional Independent Evaluation','',
        '개발 실험의 학습·선택을 완료한 뒤 모델 목록과 해시를 고정하고, 각 모델을 Test에서 한 번 평가했다. '
        '모든 epoch 및 Hyperparameter 선택은 Validation으로 수행했다. 아래 값은 관측한 Target별 평균에 동일 비중을 부여한 AP와 전이 지표다. '
        '방법별 Target-only를 기준으로 계산하며, 서로 다른 평가 범위는 별도 행으로 구분한다.','',
        '| 평가 범위 | 데이터 | Backbone | 방법 | Test AP | D_G | B_G |',
        '|---|---|---|---|---:|---:|---:|']
    for r in chosen:
        lines.append(f"| {r['cohort']} | {r['profile']} | {r['backbone']} | {r['method']} | {r['ap']:.6f} | {r['D_G']:.6f} | {r['B_G']:.6f} |")
    lines+=['', '상세 실행 계획과 재현 절차는 [추가 검증 실험](../experiments/transfer_validation/README.md)에, '
            '전체 조건별 결과와 seed 조건부 구간 추정은 해당 실험의 `outputs/v1/analysis`에 보존한다. '
            '새 수치는 기존 Validation 표와 평가 분할이 다르므로 별도로 해석한다.',
            '<!-- transfer-validation-v1:end -->']
    p=ROOT/'docs/claim.md';text=p.read_text();a='<!-- transfer-validation-v1:start -->';b='<!-- transfer-validation-v1:end -->'
    if a in text:
        start=text.index(a);end=text.index(b,start)+len(b);text=text[:start]+'\n'.join(lines)+text[end:]
    else:text=text.rstrip()+'\n\n'+'\n'.join(lines)+'\n'
    p.write_text(text)
