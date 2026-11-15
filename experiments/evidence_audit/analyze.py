"""CPU-only audit of existing diagnostics; does not train or select models."""
from pathlib import Path
import hashlib, json, itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'outputs'
OUT.mkdir(exist_ok=True)
paths={
 'factorial':ROOT/'experiments/preliminary/seed_factorial/v1/comparison.csv',
 'edges':ROOT/'experiments/transfer/outputs/cross_representation_two_tkg_v1/analysis/source_interactions/edges.json',
 'gradients':ROOT/'experiments/transfer_validation/outputs/readout_mechanism_v2/analysis/gradient_diagnostics.csv'}
result={'provenance':{k:{'path':str(v.relative_to(ROOT)),'sha256':hashlib.sha256(v.read_bytes()).hexdigest()} for k,v in paths.items()}}
f=pd.read_csv(paths['factorial']); b=f[f.method=='bce'].copy(); base=f[f.method=='target_only'].set_index(['target','adaptseed']).ap
assert len(b)==288
assert all(np.isclose(r.gain,r.ap-base.loc[(r.target,r.adaptseed)]) for r in b.itertuples())
pairs=[]; checkpoints=[]
for (target,sources),g in b.groupby(['target','sources']):
 a=g.pivot(index='preseed',columns='adaptseed',values='ap')
 assert a.shape==(3,3) and a.notna().all().all()
 for p,q in itertools.combinations(a.index,2):
  delta=a.loc[p]-a.loc[q]
  pairs.append(dict(target=target,sources=sources,preseed_a=p,preseed_b=q,min_delta=delta.min(),max_delta=delta.max(),reversal=bool(delta.min()<0<delta.max()),reversal_005=bool(delta.min()<-.005 and delta.max()>.005)))
 for p,h in g.groupby('preseed'):
  vals=h.gain
  checkpoints.append(dict(target=target,sources=sources,preseed=p,min_gain=vals.min(),max_gain=vals.max(),reversal=bool(vals.min()<0<vals.max()),reversal_005=bool(vals.min()<-.005 and vals.max()>.005)))
p=pd.DataFrame(pairs); c=pd.DataFrame(checkpoints)
p.to_csv(OUT/'checkpoint_rank_pairs.csv',index=False); c.to_csv(OUT/'transfer_sign_checkpoints.csv',index=False)
rows=[]
for target in ['all',*sorted(b.target.unique())]:
 pp=p if target=='all' else p[p.target==target]; cc=c if target=='all' else c[c.target==target]
 rows.append(dict(target=target,configurations=len(pp.groupby(['target','sources'])),rank_reversal_configs=int(pp.groupby(['target','sources']).reversal.any().sum()),rank_pairs=len(pp),rank_reversals=int(pp.reversal.sum()),rank_reversals_005=int(pp.reversal_005.sum()),checkpoints=len(cc),sign_reversals=int(cc.reversal.sum()),sign_reversals_005=int(cc.reversal_005.sum())))
result['seed_reversals']=rows
pd.DataFrame(rows).to_csv(OUT/'seed_reversal_summary.csv',index=False)
e=pd.DataFrame(json.loads(paths['edges'].read_text())); assert len(e)==900
e['all_up']=e.source_deltas.map(lambda d:all(v>0 for v in d.values()))
e['all_up_005']=e.source_deltas.map(lambda d:all(v>.005 for v in d.values()))
rows=[]
for target in ['all',*sorted(e.target.unique())]:
 for k in ['all',1,2,3,4]:
  g=e[(pd.Series(True,index=e.index) if target=='all' else e.target==target)&(True if k=='all' else e.source_count==k)]
  rows.append(dict(target=target,source_count=k,n=len(g),rho=float(spearmanr(g.retained_source_delta,g.target_delta).statistic),source_mean_up=int((g.retained_source_delta>0).sum()),mean_up_target_down=int(((g.retained_source_delta>0)&(g.target_delta<0)).sum()),all_up=int(g.all_up.sum()),all_up_target_down=int((g.all_up&(g.target_delta<0)).sum()),all_up_target_drop_005=int((g.all_up&(g.target_delta<-.005)).sum()),all_up_margin_005_target_drop_005=int((g.all_up_005&(g.target_delta<-.005)).sum())))
s=pd.DataFrame(rows); s.to_csv(OUT/'source_disagreement_strata.csv',index=False)
result['source_disagreement']=rows
g=pd.read_csv(paths['gradients']); assert g.notna().all().all()
assert not g.duplicated(['job_id','epoch','batch']).any()
assert (g.epsilon>=-1e-6).all()
assert np.allclose(g.objective_current-g.objective_fitted,g.epsilon)
g['relative_gap']=g.gradient_gap/g.fitted_norm
runs=g.groupby('job_id').size(); result['gradient_snapshot_counts']=runs.value_counts().to_dict()
rows=[]
for method,h in g.groupby('method'):
 rho=[float(spearmanr(j.epsilon,j.gradient_gap).statistic) for _,j in h.groupby('job_id')]
 rows.append(dict(method=method,runs=h.job_id.nunique(),snapshots=len(h),targets=sorted(h.target.unique()),epsilon_median=float(h.epsilon.median()),gradient_gap_median=float(h.gradient_gap.median()),relative_gap_median=float(h.relative_gap.median()),cosine_median=float(h.cosine.median()),nonpositive_cosine=int((h.cosine<=0).sum()),within_run_rho_median=float(np.median(rho)),within_run_positive_rho=int((np.array(rho)>0).sum())))
result['gradients']=rows
g.to_csv(OUT/'gradient_snapshots.csv',index=False); pd.DataFrame(rows).to_csv(OUT/'gradient_summary.csv',index=False)
(OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['provenance','source_disagreement']},indent=2))
print(s[(s.target=='all')|(s.source_count=='all')].to_string(index=False))
