"""Complete-grid summaries with method/settings-matched scratch and equal Target weight."""
import csv
import json
from collections import defaultdict
import numpy as np
from experiments.robust.training import read
from experiments.robust.design import LABELS


def write_csv(path,rows):
    if not rows: return
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def summarize(plan,out):
    jobs={j['id']:j for j in plan['jobs']}
    analysis=out/'analysis';analysis.mkdir(exist_ok=True)
    all_summary=[];target_summary=[]
    for stage,views in plan['stages'].items():
        runs=[];expected=defaultdict(int)
        for v in views:
            j=jobs[v['job_id']];budget=v.get('budget',j['config']['epochs'])
            settings=json.dumps({k:v[k] for k in ['axis','value','budget'] if k in v},sort_keys=True)
            group=(settings,j['backbone'],j['method'])
            expected[group]+=1
            p=out/'jobs'/j['id']/f'test_budget{budget}.json'
            if not p.exists(): continue
            r=read(p)
            runs.append(dict(job_id=j['id'],settings=settings,backbone=j['backbone'],method=j['method'],
                target=j['target'],sources='+'.join(sorted(j['sources'])),pretrain_seed=j['pretrain_seed'],
                adaptation_seed=j['adaptation_seed'],budget=budget,selected_epoch=r['selected_epoch'],
                validation_ap=r['validation_ap'],test_ap=r['metrics']['ap']))
        write_csv(analysis/f'{stage}_runs.csv',runs)
        groups=defaultdict(list)
        for r in runs: groups[r['settings'],r['backbone'],r['method']].append(r)
        transfer=[];summary=[]
        for group,rs in groups.items():
            # A partially finished grid can be inspected as raw runs, never as a full result.
            if len(rs)!=expected[group]: continue
            scratch={(r['target'],r['adaptation_seed']):r['test_ap'] for r in rs if not r['sources']}
            tr=[]
            for r in rs:
                if not r['sources']: continue
                gain=r['test_ap']-scratch[r['target'],r['adaptation_seed']]
                tr.append(dict(**r,G=gain,D_G=max(-gain,0.),B_G=max(gain,0.)))
            transfer+=tr
            pvalues=sorted({r['pretrain_seed'] for r in tr})
            for pre in [None]+pvalues:
                selected=[r for r in tr if pre is None or r['pretrain_seed']==pre]
                targets=sorted({r['target'] for r in selected})
                def macro(key):
                    return float(np.mean([np.mean([r[key] for r in selected if r['target']==t]) for t in targets]))
                sd=[]
                for t in targets:
                    compositions=defaultdict(list)
                    for r in selected:
                        if r['target']==t: compositions[r['sources'],r['pretrain_seed']].append(r['test_ap'])
                    sd.append(np.mean([np.std(v,ddof=0) for v in compositions.values()]))
                seed_aggregates=[]
                for a in sorted({r['adaptation_seed'] for r in selected}):
                    seed_aggregates.append(np.mean([np.mean([r['test_ap'] for r in selected if r['target']==t and r['adaptation_seed']==a]) for t in targets]))
                summary.append(dict(stage=stage,settings=group[0],backbone=group[1],method=group[2],
                    pretrain_seed='all' if pre is None else pre,transfer_runs=len(selected),
                    AP=macro('test_ap'),target_only_AP=float(np.mean(list(scratch.values()))),
                    G=macro('G'),D_G=macro('D_G'),B_G=macro('B_G'),seed_SD=float(np.mean(sd)),
                    aggregate_seed_SD=float(np.std(seed_aggregates,ddof=1)),
                    negative=sum(r['G']<0 for r in selected),large_negative=sum(r['G']<-.005 for r in selected)))
                for target in targets:
                    rr=[r for r in selected if r['target']==target]
                    target_summary.append(dict(stage=stage,settings=group[0],backbone=group[1],method=group[2],
                        pretrain_seed='all' if pre is None else pre,target=target,transfer_runs=len(rr),
                        AP=float(np.mean([r['test_ap'] for r in rr])),
                        target_only_AP=float(np.mean([v for (t,a),v in scratch.items() if t==target])),
                        G=float(np.mean([r['G'] for r in rr])),D_G=float(np.mean([r['D_G'] for r in rr])),
                        B_G=float(np.mean([r['B_G'] for r in rr])),negative=sum(r['G']<0 for r in rr)))
        write_csv(analysis/f'{stage}_transfer.csv',transfer)
        write_csv(analysis/f'{stage}_summary.csv',summary)
        all_summary+=summary
    write_csv(analysis/'summary.csv',all_summary)
    write_csv(analysis/'by_target.csv',target_summary)
    lines=['# Robustness results','', 'Only complete method/settings groups are summarized. All metrics below use Test; models are selected on Validation.','',
           '| Stage | Settings | Backbone | Method | Pretraining seed | AP | D_G | B_G | Seed SD |',
           '|---|---|---|---|---|---:|---:|---:|---:|']
    for r in all_summary:
        if r['pretrain_seed']!='all' and r['stage']!='pretrain_seed': continue
        lines.append(f"| {r['stage']} | {r['settings']} | {r['backbone']} | {LABELS[r['method']]} | {r['pretrain_seed']} | {r['AP']:.6f} | {r['D_G']:.6f} | {r['B_G']:.6f} | {r['seed_SD']:.6f} |")
    (analysis/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return all_summary
