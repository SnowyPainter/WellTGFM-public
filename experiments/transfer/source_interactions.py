"""Configuration-dependent transfer and paired source-retention diagnostics."""
import csv,json,itertools
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from experiments.transfer.analyze_source_pool import BASE,OUT,NAMES,SEEDS
from experiments.common.runtime import atomic_json


def sign_table(rows,threshold=0.):
    def sign(x):return 'down' if x < -threshold else 'up' if x > threshold else 'flat'
    counts={a+'_'+b:0 for a in ['down','up','flat'] for b in ['down','up','flat']}
    for r in rows:counts[sign(r['target_delta'])+'_'+sign(r['retained_source_delta'])]+=1
    rho=float(spearmanr([r['target_delta'] for r in rows],[r['retained_source_delta'] for r in rows]).statistic)
    return dict(n=len(rows),threshold=threshold,target_then_retained_source=counts,spearman=rho)


def main():
    records=json.loads((OUT/'all_source_evaluations.json').read_text())
    assert len(records)==372
    lookup={(r['target_context'],r['seed'],frozenset(r['sources'])):r for r in records}
    edges=[]
    for target in NAMES[:4]:
        summary=json.loads((BASE/target/'summary.json').read_text())['rows']
        pool=[n for n in NAMES if n!=target]
        for row in summary:
            seed=row['seed']
            for k in range(1,5):
                for old in itertools.combinations(pool,k):
                    s=frozenset(old)
                    for a in pool:
                        if a in s:continue
                        new=s|{a};newkey='+'.join(n for n in pool if n in new);oldkey='+'.join(old)
                        before=lookup[target,seed,s]['metrics'];after=lookup[target,seed,new]['metrics']
                        changes={n:after[n]['ap']-before[n]['ap'] for n in old}
                        edges.append(dict(target=target,seed=seed,sources=oldkey,added_source=a,source_count=k,
                            target_before=row[oldkey],target_after=row[newkey],target_delta=row[newkey]-row[oldkey],
                            target_only=row['target_only'],source_deltas=changes,
                            retained_source_delta=float(np.mean(list(changes.values()))),
                            any_source_decline=any(d<0 for d in changes.values()),all_sources_decline=all(d<0 for d in changes.values())))
    assert len(edges)==900
    grouped=defaultdict(list)
    for r in edges:grouped[r['target'],r['added_source'],r['sources']].append(r)
    averaged=[]
    for (t,a,s),rs in grouped.items():
        assert len(rs)==3
        averaged.append(dict(target=t,added_source=a,sources=s,source_count=rs[0]['source_count'],
            target_delta=float(np.mean([r['target_delta'] for r in rs])),
            retained_source_delta=float(np.mean([r['retained_source_delta'] for r in rs])),
            all_seeds_target_positive=all(r['target_delta']>0 for r in rs),
            all_seeds_target_negative=all(r['target_delta']<0 for r in rs)))
    assert len(averaged)==300
    context_groups=[]
    for t in NAMES[:4]:
        for a in NAMES:
            if a==t:continue
            rs=[r for r in averaged if r['target']==t and r['added_source']==a]
            positive=[r for r in rs if r['target_delta']>0];negative=[r for r in rs if r['target_delta']<0]
            robust_positive=[r for r in rs if r['all_seeds_target_positive']];robust_negative=[r for r in rs if r['all_seeds_target_negative']]
            pairs=[(x,y) for x in robust_positive for y in robust_negative if x['source_count']==y['source_count']]
            context_groups.append(dict(target=t,added_source=a,contexts=len(rs),mean_sign_flip=bool(positive and negative),
                same_count_mean_sign_flip=any(x['source_count']==y['source_count'] for x in positive for y in negative),
                all_seed_sign_flip=bool(robust_positive and robust_negative),same_count_all_seed_sign_flip=bool(pairs),
                strongest_same_count_pair=max(pairs,key=lambda z:z[0]['target_delta']-z[1]['target_delta']) if pairs else None))
    output=BASE/'analysis/source_interactions';output.mkdir(exist_ok=True)
    atomic_json(output/'edges.json',edges);atomic_json(output/'mean_edges.json',averaged);atomic_json(output/'context_groups.json',context_groups)
    summary=dict(comparisons=900,mean_comparisons=300,groups=20,
        reversals={k:sum(r[k] for r in context_groups) for k in ['mean_sign_flip','same_count_mean_sign_flip','all_seed_sign_flip','same_count_all_seed_sign_flip']},
        association_by_seed=sign_table(edges),association_seed_means=sign_table(averaged),
        per_target={t:sign_table([r for r in edges if r['target']==t]) for t in NAMES[:4]},
        per_source_count={str(k):sign_table([r for r in edges if r['source_count']==k]) for k in range(1,5)},
        sensitivity={str(e):dict(association=sign_table(edges,e),mean_sign_flip_groups=sum(any(r['target_delta']>e for r in averaged if r['target']==g['target'] and r['added_source']==g['added_source']) and any(r['target_delta']< -e for r in averaged if r['target']==g['target'] and r['added_source']==g['added_source']) for g in context_groups)) for e in [.001,.005]})
    for epsilon in [.001,.005]:
        same=robust=0
        for group in context_groups:
            contexts=[r for r in averaged if r['target']==group['target'] and r['added_source']==group['added_source']]
            pos=[r for r in contexts if r['target_delta']>epsilon]
            neg=[r for r in contexts if r['target_delta']< -epsilon]
            same+=any(x['source_count']==y['source_count'] for x in pos for y in neg)
            pp=[r for r in contexts if all(e['target_delta']>epsilon for e in grouped[r['target'],r['added_source'],r['sources']])]
            nn=[r for r in contexts if all(e['target_delta']< -epsilon for e in grouped[r['target'],r['added_source'],r['sources']])]
            robust+=any(x['source_count']==y['source_count'] for x in pp for y in nn)
        summary['sensitivity'][str(epsilon)].update(same_count_mean_sign_flip_groups=same,same_count_all_seed_sign_flip_groups=robust)
    summary['individual_source_signs']=dict(
        target_down_all_existing_up=sum(r['target_delta']<0 and all(v>0 for v in r['source_deltas'].values()) for r in edges),
        target_up_all_existing_down=sum(r['target_delta']>0 and all(v<0 for v in r['source_deltas'].values()) for r in edges),
        target_down_any_existing_down=sum(r['target_delta']<0 and r['any_source_decline'] for r in edges))
    atomic_json(output/'summary.json',summary)
    with (output/'edges.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=edges[0].keys());w.writeheader()
        for r in edges:w.writerow(dict(r,source_deltas=json.dumps(r['source_deltas'])))
    print(json.dumps(summary,indent=2))
    for group in context_groups:
        if group['strongest_same_count_pair']:
            print(group['target'],group['added_source'],group['strongest_same_count_pair'])

if __name__=='__main__':main()
