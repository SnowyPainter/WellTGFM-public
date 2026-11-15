"""Fixed practical comparison: paired timing/replay, then complete Batch powerset."""
import json
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiments.target_adaptation.runtime import ROOT, HERE, digest, canonical_hash, load_inputs
from experiments.target_adaptation.powerset import plan
from experiments.target_adaptation.run import execute as tfra
from experiments.target_adaptation.batch_practical_training import execute as batch
from experiments.target_adaptation.test_evaluation import Evaluator
from experiments.common.runtime import atomic_json, gpu_guard

OUT=HERE/'outputs/batch_practical_v1'
OLD=HERE/'outputs/test_v1'

def read(p):return json.loads(p.read_text())
def key(j):return j['target'],tuple(sorted(j['sources'])),j['adaptation_seed']

def report(rows):
    records=[]
    for r in rows:
        j=r['job'];p=OUT/'test'/j['id']/'result.json'
        if not p.exists():p=OLD/'test'/j['id']/'result.json'
        if not p.exists():continue
        q=read(p)
        records.append(dict(target=j['target'],sources='+'.join(sorted(j['sources'])),seed=j['adaptation_seed'],method=j['method'],ap=q['metrics']['random_1']['ap']))
    frame=pd.DataFrame(records)
    if frame.empty:return
    frame.to_csv(OUT/'runs.csv',index=False)
    controls={(r.target,r.seed,r.method):r.ap for r in frame.itertuples() if not r.sources}
    matched=frame[frame.sources!=''].pivot(index=['target','sources','seed'],columns='method',values='ap').dropna(subset=['batch','tfra','finetune'])
    summaries=[]
    for m in ['batch','tfra','finetune']:
        rs=[]
        for (t,s,a),row in matched.iterrows():
            if (t,a,m) not in controls:continue
            rs.append(dict(target=t,sources=s,seed=a,ap=row[m],gain=row[m]-controls[t,a,m],common_gain=row[m]-controls[t,a,'finetune']))
        if not rs:continue
        x=pd.DataFrame(rs)
        for t,g in x.groupby('target'):
            summaries.append(dict(method=m,target=t,runs=len(g),ap=g.ap.mean(),D_G=(-g.gain).clip(lower=0).mean(),B_G=g.gain.clip(lower=0).mean(),common_D=(-g.common_gain).clip(lower=0).mean(),common_B=g.common_gain.clip(lower=0).mean(),seed_sd=g.groupby('sources').ap.std(ddof=0).mean(),mean_worst_seed=g.groupby('sources').ap.min().mean()))
    sm=pd.DataFrame(summaries)
    if sm.empty:return
    means=sm.groupby('method').mean(numeric_only=True).reset_index();means['target']='target_equal_mean'
    sm=pd.concat([sm,means],ignore_index=True);sm.to_csv(OUT/'summary.csv',index=False)
    timed=[]
    for p in (OUT/'timing').glob('*/*/measurement.json'):
        q=read(p)
        if q['uninterrupted']:timed.append(q)
    costs=pd.DataFrame(timed)
    if not costs.empty:costs.to_csv(OUT/'timing.csv',index=False)
    lines=['# TFRA vs Batch-wise: practical comparison','',
        'Exploratory extension specified after inspecting the original panel Test results. No Test-based tuning. Best Validation epoch; Test negative seed 200007. Original checkpoints remain unchanged.','',
        f'Matched transfer executions so far: {len(matched)} / 372. Partial results include the development panel and are not representative of the full powerset.','',
        '```',sm.to_string(index=False),'```','',
        'common_D/common_B use the SAME Fine-tuning Target-only control. Own-method D_G/B_G describe the incremental contribution of pretraining and do not establish superior deployment performance.','']
    if not costs.empty:
        lines+=['## Fresh interleaved timing','', 'Synchronized wall time includes model construction, training, Validation selection and checkpoint I/O. Peak allocated memory is process-local. Shared external GPU load remains a limitation.','', '```',costs.groupby('method')[['seconds','peak_allocated_mib']].mean().to_string(),'```']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    p=plan();jobs=p['jobs'];cfgs=p['configs']
    frozen=read(OLD/'test_freeze.json')['models']
    old={(r['job']['method'],key(r['job'])):r for r in frozen if r['job']['method'] in ['batch','tfra','finetune']}
    # Eight predetermined conditions: scratch and smallest/name-first panel source for each target.
    timing=[]
    for t in sorted(cfgs):
        candidates=[r['job'] for r in frozen if r['job']['method']=='batch' and r['job']['target']==t and r['job']['sources']]
        source=min(candidates,key=lambda j:(len(j['sources']),tuple(sorted(j['sources']))))['sources']
        timing.extend(j for j in jobs if j['target']==t and (not j['sources'] or set(j['sources'])==set(source)))
    timing.sort(key=lambda j:(j['adaptation_seed'],j['target'],len(j['sources']),j['job_id']))
    assert len(timing)==24
    spec=dict(jobs=jobs,timing_jobs=timing,configs=cfgs,epochs=5,selection='best_validation_ap',test_negative_seed=200007,
        endpoints=['absolute Test AP','common FT target-only damage/benefit','worst of three seeds','seed SD','synchronized wall time','peak allocated memory'],
        interpretation='Exploratory after original panel Test inspection; no hyperparameter tuning or automatic manuscript edits',
        hashes={n:digest(HERE/n) for n in ['batch_practical.py','batch_practical_training.py','run.py','readout.py']})
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==spec
    else:atomic_json(OUT/'protocol.json',spec)
    phash=canonical_hash(spec)
    rows=[r for r in frozen if r['job']['cohort']=='legacy_powerset' and r['job']['method'] in ['tfra','finetune']]
    rows += [r for r in frozen if r['job']['method']=='batch']
    report(rows)
    torch.set_num_threads(2)
    with gpu_guard('cuda:0',0):
        _,_,data=load_inputs()
        # Warm up both forward/backward paths; excluded from all measurements.
        if not (OUT/'warmup_complete.json').exists():
            for m in ['tfra','batch']:
                j=timing[0];cfg=cfgs[j['target']]
                if m=='tfra':tfra(j,cfg,data,OUT/'warmup',phash,smoke=True)
                else:batch(j,'ridge_profile',cfg,data,OUT/'warmup',phash,smoke=True)
            atomic_json(OUT/'warmup_complete.json',dict(done=True))
        for i,j in enumerate(timing):
            for m in (['tfra','batch'] if i%2==0 else ['batch','tfra']):
                name='ridge_epoch' if m=='tfra' else 'ridge_profile';dest=OUT/'timing'/name/j['job_id']
                if (dest/'measurement.json').exists():continue
                atomic_json(OUT/'progress.json',dict(status='running',stage='paired_timing_and_training_replay',completed=len(list((OUT/'timing').glob('*/*/measurement.json'))),total=48,method=m,job=j['job_id']))
                uninterrupted=not (dest/'resume.pt').exists()
                torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
                if m=='tfra':tfra(j,cfgs[j['target']],data,OUT/'timing',phash)
                else:batch(j,name,cfgs[j['target']],data,OUT/'timing',phash)
                torch.cuda.synchronize();seconds=time.monotonic()-start;peak=torch.cuda.max_memory_allocated()/2**20
                original=ROOT/old[m,key(j)]['best']
                with np.load(dest/'validation_predictions.npz') as a,np.load(original.parent/'validation_predictions.npz') as b:
                    np.testing.assert_allclose(a['scores'],b['scores'],rtol=1e-5,atol=1e-6)
                atomic_json(dest/'measurement.json',dict(method=m,target=j['target'],sources=j['sources'],seed=j['adaptation_seed'],seconds=seconds,peak_allocated_mib=peak,uninterrupted=uninterrupted,training_replay_passed=True))
                report(rows)
        missing=[j for j in jobs if ('batch',key(j)) not in old]
        assert len(missing)==330
        for i,j in enumerate(missing):
            atomic_json(OUT/'progress.json',dict(status='running',stage='batch_powerset_training',completed=i,total=len(missing),job=j['job_id']))
            batch(j,'ridge_profile',cfgs[j['target']],data,OUT/'powerset',phash)
            dest=OUT/'powerset/ridge_profile'/j['job_id'];r=read(dest/'result.json')
            cfg={**old['tfra',key(j)]['config']}
            job={**j,'id':'practical_batch__'+canonical_hash(j['job_id'])[:20],'method':'batch','cohort':'batch_practical','profile':'original','backbone':'tgat','source_budget':None,'config':cfg}
            rows.append(dict(job=job,best=str((dest/'best.pt').relative_to(ROOT)),best_sha256=digest(dest/'best.pt'),validation_ap=r['ap'],config=cfg))
            torch.cuda.empty_cache()
        seal=dict(protocol_sha256=phash,models=rows)
        if (OUT/'test_freeze.json').exists():assert read(OUT/'test_freeze.json')==seal
        else:atomic_json(OUT/'test_freeze.json',seal)
        evaluator=Evaluator()
        pending=[r for r in rows if r['job']['cohort']=='batch_practical']
        for i,r in enumerate(pending):
            atomic_json(OUT/'progress.json',dict(status='running',stage='frozen_batch_test',completed=i,total=len(pending)))
            evaluator.test(r,OUT);report(rows)
        report(rows)
        atomic_json(OUT/'progress.json',dict(status='complete',batch_models=384,transfer_comparisons=372,timing_runs=48,claim_updated=False))

if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        atomic_json(OUT/'error.json',dict(error=traceback.format_exc()))
        raise
