"""Pareto comparison of completed paired readout controls, without scalar weights."""
import csv
import json
from pathlib import Path
from statistics import mean

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'experiments/target_adaptation/outputs/readout_panel_v2'
OUT=Path(__file__).resolve().parent/'outputs/ablation_pareto'
METHODS=['finetune','ridge_frozen','ridge_warmstart','ridge_profile','ridge_joint','ridge_epoch']

def main():
    assert json.loads((BASE/'verification.json').read_text())['passed']
    data=list(csv.DictReader((BASE/'comparison.csv').open()))
    rows=[]
    for method in METHODS:
        results=[r for r in data if r['method']==method]
        assert len(results)==42
        target_stats=[]
        for target in sorted({r['target'] for r in results}):
            rs=[r for r in results if r['target']==target]
            aps=[float(r['AP']) for r in rs];g=[float(r['G']) for r in rs]
            target_stats.append(dict(AP=mean(aps),target_only_AP=mean(a-b for a,b in zip(aps,g)),G=mean(g),D_G=mean(max(-x,0) for x in g),B_G=mean(max(x,0) for x in g)))
        rows.append(dict(method=method,**{k:mean(t[k] for t in target_stats) for k in target_stats[0]}))
    def dominates(a,b):
        av=(a['AP'],-a['D_G'],a['B_G']);bv=(b['AP'],-b['D_G'],b['B_G'])
        return all(x>=y for x,y in zip(av,bv)) and any(x>y for x,y in zip(av,bv))
    for r in rows:
        r['dominated_by']=[a['method'] for a in rows if dominates(a,r)]
        r['pareto']=not r['dominated_by']
    assert [r['method'] for r in rows if not r['pareto']]==['finetune','ridge_warmstart']
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'analysis.json').write_text(json.dumps(dict(objectives=['AP maximize','D_G minimize','B_G maximize'],aggregation='equal Target weight',methods=rows),indent=2)+'\n')
    with (OUT/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Readout-control Pareto analysis','', 'Fourteen fixed Source configurations, three adaptation seeds, matched Target-only. Metrics are averaged within Target then equally across four Targets. Dominance uses full-precision AP (maximize), D_G (minimize), B_G (maximize). No composite score or fitted threshold is used.','', '| Method | AP | D_G | B_G | Nondominated |','|---|---:|---:|---:|---|']
    for r in rows:lines.append(f"| {r['method']} | {r['AP']:.6f} | {r['D_G']:.6f} | {r['B_G']:.6f} | {r['pareto']} |")
    lines+=['','Frozen, Batch, Joint and TFRA are nondominated. TFRA dominates Fine-tuning and Ridge Warm-start and has the lowest transfer damage among the six methods. TFRA is a Pareto trade-off, not a unique optimum. This is a descriptive comparison of observed panel means, not a statistical population-frontier claim. G is derived from B_G-D_G and is not counted as a fourth objective.','']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))
    print('\n'.join(lines))

if __name__=='__main__':main()
