"""Link source-fit diagnostics to adaptation outcomes and initial-fit trajectories."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[2]; O=Path(__file__).resolve().parent/'outputs'; provenance={}
def read(p):
 p=R/p; provenance[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
def csv(p):
 p=R/p;provenance[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest();return pd.read_csv(p)
def key(t,s,a):return (t,tuple(sorted(s)),int(a))
e=pd.DataFrame(read('experiments/transfer/outputs/cross_representation_two_tkg_v1/analysis/source_interactions/edges.json'))
c=csv('experiments/target_adaptation/outputs/powerset_v1/comparison.csv')
lookup={key(r.target,r.case_id.split('__')[1].split('+'),r.adaptation_seed):r for r in c.itertuples()}
rows=[]
# Only pretrain seed 7 aligns with the powerset adaptation benchmark.
for r in e[e.seed==7].itertuples():
 src=r.sources.split('+');before=lookup[key(r.target,src,7)];after=lookup[key(r.target,src+[r.added_source],7)]
 assert np.isclose(before.finetune_ap,r.target_before,atol=1e-12,rtol=0)
 assert np.isclose(after.finetune_ap,r.target_after,atol=1e-12,rtol=0)
 allup=all(v>0 for v in r.source_deltas.values())
 rows.append(dict(target=r.target,sources=r.sources,added=r.added_source,source_count=r.source_count,all_source_up=allup,ft_C=r.target_delta,tfra_C=after.ridge_epoch_ap-before.ridge_epoch_ap,ft_G_after=after.finetune_gain,tfra_G_after=after.ridge_epoch_gain,AP_improvement_after=after.ridge_epoch_ap-after.finetune_ap))
p=pd.DataFrame(rows);assert len(p)==300;p.to_csv(O/'matched_source_paths.csv',index=False)
summ=[]
for name,g in [('all_matched',p),('all_sources_up',p[p.all_source_up]),('all_sources_up_ft_drop',p[p.all_source_up&(p.ft_C<0)])]:
 for t in ['all',*sorted(g.target.unique())]:
  h=g if t=='all' else g[g.target==t]
  summ.append(dict(cohort=name,target=t,n=len(h),ft_C=float(h.ft_C.mean()),tfra_C=float(h.tfra_C.mean()),tfra_C_positive=int((h.tfra_C>0).sum()),tfra_C_better=int((h.tfra_C>h.ft_C).sum()),AP_improvement=float(h.AP_improvement_after.mean()),AP_wins=int((h.AP_improvement_after>0).sum()),ft_D_G=float((-h.ft_G_after).clip(lower=0).mean()),tfra_D_G=float((-h.tfra_G_after).clip(lower=0).mean()),ft_B_G=float(h.ft_G_after.clip(lower=0).mean()),tfra_B_G=float(h.tfra_G_after.clip(lower=0).mean())))
pd.DataFrame(summ).to_csv(O/'matched_source_summary.csv',index=False)
# Existing paired trajectories; no selection of good epochs or configurations.
old={}
for directory in sorted((R/'experiments/target_adaptation/outputs/readout_panel_v2/ridge_joint').iterdir()):
 while (directory/'reuse.json').exists():
  reuse=read((directory/'reuse.json').relative_to(R)); directory=R/reuse['reused_from']
  assert hashlib.sha256((directory/'result.json').read_bytes()).hexdigest()==reuse['result_sha256']
 file=directory/'result.json'
 r=read(file.relative_to(R));j=r['job'];old[key(j['target'],j['sources'],j['adaptation_seed'])]=r
rows=[]
for file in sorted((R/'experiments/transfer_validation/outputs/readout_mechanism_v2/jobs').glob('*/result.json')):
 r=read(file.relative_to(R));j=r['job']
 if j['method']!='joint_init':continue
 other=old[key(j['target'],j['sources'],j['adaptation_seed'])]
 assert not j['sources'] or r['checkpoint_sha256']==other['checkpoint_sha256']
 assert len(r['history'])==len(other['history'])==5
 for a,b in zip(other['history'],r['history']):
  assert a['epoch']==b['epoch']
  rows.append(dict(target=j['target'],sources='+'.join(sorted(j['sources'])),adaptation_seed=j['adaptation_seed'],epoch=a['epoch'],joint_ap=a['validation']['ap'],initial_fit_ap=b['validation']['ap'],delta=b['validation']['ap']-a['validation']['ap']))
h=pd.DataFrame(rows);assert len(h)==270
h.to_csv(O/'initial_fit_validation_trajectories.csv',index=False)
base=h[h.sources==''].set_index(['target','adaptation_seed','epoch'])
trans=h[h.sources!=''].copy()
for m in ['joint','initial_fit']:
 trans[m+'_gain']=[getattr(r,m+'_ap')-base.loc[(r.target,r.adaptation_seed,r.epoch),m+'_ap'] for r in trans.itertuples()]
rows=[]
for epoch,g in trans.groupby('epoch'):
 row={'epoch':int(epoch),'runs':len(g),'wins':int((g.delta>0).sum())}
 for m in ['joint','initial_fit']:
  row[m+'_AP']=float(g.groupby('target')[m+'_ap'].mean().mean())
  for name,vals in [('D_G',(-g[m+'_gain']).clip(lower=0)),('B_G',g[m+'_gain'].clip(lower=0))]:
   row[m+'_'+name]=float(vals.groupby(g.target).mean().mean())
 row['delta_AP']=row['initial_fit_AP']-row['joint_AP']; rows.append(row)
pd.DataFrame(rows).to_csv(O/'initial_fit_epoch_summary.csv',index=False)
trans.groupby(['target','epoch']).delta.agg(['mean','min','max']).to_csv(O/'initial_fit_target_epochs.csv')
(O/'connection_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(pd.DataFrame(summ).query("target=='all'").to_string(index=False));print(pd.DataFrame(rows).to_string(index=False))
