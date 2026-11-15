"""TFRA-B full powerset: verified panel reuse, 330 new fits, frozen Test evaluation."""
import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.target_adaptation.runtime import ROOT, HERE, digest, canonical_hash, load_inputs, array_hash
from experiments.target_adaptation.powerset import plan, read_result
from experiments.target_adaptation.batch_practical_training import execute
from experiments.target_adaptation.test_evaluation import Evaluator
from experiments.common.runtime import atomic_json, gpu_guard

OUT = HERE / 'outputs/batch_powerset_v1'
OLD = HERE / 'outputs/test_v1'
METHODS = ['finetune', 'lp_ft', 'refine', 'tfra', 'batch']
LABELS = dict(finetune='Fine-tuning', lp_ft='LP-FT', refine='REFINE', tfra='TFRA-E', batch='TFRA-B')


def read(path):
    return json.loads(path.read_text())


def key(job):
    return job['target'], tuple(sorted(job['sources'])), job['adaptation_seed']


def prepare():
    base = plan()
    frozen = read(OLD / 'test_freeze.json')['models']
    baseline = [r for r in frozen if r['job']['cohort'] == 'legacy_powerset' and r['job']['method'] in METHODS[:-1]]
    panel = [r for r in frozen if r['job']['method'] == 'batch']
    assert len(baseline) == 1536 and len(panel) == 54
    lookup = {key(r['job']): r for r in panel}
    by_key = {key(j): j for j in base['jobs']}
    for r in panel:
        j = by_key[key(r['job'])]
        source = ROOT / r['best']
        assert digest(source) == r['best_sha256']
        result = read(source.parent / 'result.json')
        assert read(source.parent / 'manifest.json')['config'] == base['configs'][j['target']]
        for field in ['checkpoint_sha256', 'pretrain_seed', 'adaptation_seed', 'target']:
            assert result['job'][field] == j[field]
        assert result['status'] == 'complete' and len(result['history']) == 5
        assert result['encoder_updates'] == 60
        assert result['evaluation_negative_seed'] == 100007
        assert abs(result['ap'] - max(h['validation']['ap'] for h in result['history'])) < 1e-12
        test = read(OLD / 'test' / r['job']['id'] / 'result.json')
        assert test['best_sha256'] == r['best_sha256'] and test['split'] == 'test'
    missing = [j for j in base['jobs'] if key(j) not in lookup]
    assert len(missing) == 330
    spec = dict(jobs=base['jobs'], configs=base['configs'], reused_models=panel,
                epochs=5, pretrain_seed=7, adaptation_seeds=[7,17,37],
                validation_negative_seed=100007, test_negative_seed=200007,
                selection='best Validation AP; Test never used for selection',
                interpretation='Powerset extension after panel Test inspection; unchanged hyperparameters',
                hashes={name:digest(HERE/name) for name in ['batch_powerset.py','batch_practical_training.py','readout.py','test_evaluation.py']})
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'protocol.json').exists():
        assert read(OUT/'protocol.json') == spec
    else:
        atomic_json(OUT/'protocol.json', spec)
    atomic_json(OUT/'reuse_verification.json', dict(passed=True, reused=54, new=330, total=384))
    return spec, baseline, panel, missing


def report(rows):
    records = []
    candidates = {}
    for r in rows:
        j = r['job']
        root = OUT if j['cohort'] == 'batch_powerset' else OLD
        dest = root/'test'/j['id']
        q = read(dest/'result.json')
        assert q['best_sha256'] == r['best_sha256'] and q['split'] == 'test'
        with np.load(dest/'test_predictions.npz') as arr:
            h = array_hash(arr)
        if j['target'] in candidates:
            assert candidates[j['target']] == h
        candidates[j['target']] = h
        records.append(dict(method=j['method'], target=j['target'], sources='+'.join(sorted(j['sources'])),
                            seed=j['adaptation_seed'], ap=q['metrics']['random_1']['ap']))
    frame = pd.DataFrame(records)
    assert len(frame) == 1920 and not frame.duplicated(['method','target','sources','seed']).any()
    controls = {(r.method,r.target,r.seed):r.ap for r in frame.itertuples() if not r.sources}
    frame['gain'] = [r.ap-controls[r.method,r.target,r.seed] for r in frame.itertuples()]
    summary = []
    for (method,target), group in frame[frame.sources!=''].groupby(['method','target']):
        assert len(group) == 93
        summary.append(dict(method=method,target=target,AP=group.ap.mean(),
            target_only_AP=frame[(frame.method==method)&(frame.target==target)&(frame.sources=='')].ap.mean(),
            D_G=(-group.gain).clip(lower=0).mean(), B_G=group.gain.clip(lower=0).mean(),
            seed_SD=group.groupby('sources').ap.std(ddof=0).mean(),
            negative_over_005=int((group.gain < -.005).sum()),
            stable_positive=int(group.groupby('sources').gain.apply(lambda g:(g>0).all()).sum())))
    sm = pd.DataFrame(summary)
    for method in METHODS:
        g = sm[sm.method==method]
        row = dict(method=method,target='overall')
        for field in ['AP','target_only_AP','D_G','B_G','seed_SD']:
            row[field] = float(g[field].mean())
        for field in ['negative_over_005','stable_positive']:
            row[field] = int(g[field].sum())
        summary.append(row)
    frame.to_csv(OUT/'runs.csv',index=False)
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
    atomic_json(OUT/'summary.json', dict(rows=summary, aggregation='equal Target weight', test_candidates=candidates))
    lines=['# Full powerset Test benchmark with TFRA-B','',
           'Pretraining seed 7; adaptation seeds 7/17/37; 5 epochs. 372 transfers + 12 same-method Target-only per method.',
           'TFRA-B: 54 verified panel models reused, 330 newly trained. Extension after panel Test inspection; no Test tuning.', '',
           '| Target | Method | AP | Target-only AP | D_G | B_G | Seed SD | G < -0.005 | Stable positive compositions |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for target in list(spec_target for spec_target in sorted(candidates))+['overall']:
        for method in METHODS:
            r=next(x for x in summary if x['target']==target and x['method']==method)
            lines.append('| '+target+' | '+LABELS[method]+' | '+' | '.join(f'{r[f]:.6f}' for f in ['AP','target_only_AP','D_G','B_G','seed_SD'])+f" | {r['negative_over_005']} | {r['stable_positive']} |")
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    atomic_json(OUT/'verification.json',dict(passed=True, total_models=1920, batch_models=384, paired_test_candidates=True))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    spec,baseline,panel,missing=prepare()
    phash=canonical_hash(spec)
    if not args.execute:
        print('TFRA-B: 384 total = 54 verified reuse + 330 new. Existing four methods: 1536 models.');return
    atomic_json(OUT/'progress.json',dict(status='starting',completed=54,total=384))
    torch.set_num_threads(2)
    with gpu_guard('cuda:0',0):
        _,_,data=load_inputs()
        if not (OUT/'replay_verification.json').exists():
            reference=panel[0]
            j=next(j for j in spec['jobs'] if key(j)==key(reference['job']))
            atomic_json(OUT/'progress.json',dict(status='running',stage='training_replay',completed=54,total=384,current=j['job_id']))
            execute(j,'ridge_profile',spec['configs'][j['target']],data,OUT/'replay',phash)
            fresh=read_result(OUT/'replay','ridge_profile',j)
            with np.load(OUT/'replay/ridge_profile'/j['job_id']/'validation_predictions.npz') as a, np.load((ROOT/reference['best']).parent/'validation_predictions.npz') as b:
                np.testing.assert_allclose(a['scores'],b['scores'],rtol=1e-5,atol=1e-6)
            assert abs(fresh['ap']-reference['validation_ap'])<1e-8
            atomic_json(OUT/'replay_verification.json',dict(passed=True,job=j['job_id'],ap=fresh['ap']))
        rows=baseline+panel
        template={key(r['job']):r['config'] for r in baseline if r['job']['method']=='tfra'}
        for i,j in enumerate(missing):
            atomic_json(OUT/'progress.json',dict(status='running',stage='training',completed=54+i,total=384,current=j['job_id']))
            execute(j,'ridge_profile',spec['configs'][j['target']],data,OUT/'training',phash)
            result=read_result(OUT/'training','ridge_profile',j)
            assert result['encoder_updates']==60
            dest=OUT/'training/ridge_profile'/j['job_id']
            cfg=template[key(j)]
            job={**j,'id':'batch_powerset__'+canonical_hash(j['job_id'])[:20], 'method':'batch',
                 'cohort':'batch_powerset','profile':'original','backbone':'tgat','source_budget':None,'config':cfg}
            rows.append(dict(job=job,best=str((dest/'best.pt').relative_to(ROOT)),best_sha256=digest(dest/'best.pt'),validation_ap=result['ap'],config=cfg))
            torch.cuda.empty_cache()
        seal=dict(protocol_sha256=phash,models=rows)
        if (OUT/'test_freeze.json').exists():assert read(OUT/'test_freeze.json')==seal
        else:atomic_json(OUT/'test_freeze.json',seal)
        del data
        evaluator=Evaluator()
        for i,r in enumerate(rows[-330:]):
            atomic_json(OUT/'progress.json',dict(status='running',stage='test',completed=i,total=330,training_completed=384,current=r['job']['id']))
            evaluator.test(r,OUT)
        report(rows)
        atomic_json(OUT/'progress.json',dict(status='complete',completed=384,total=384,new_training=330,test_complete=330,verification='passed'))


if __name__=='__main__':
    try:
        main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        atomic_json(OUT/'progress.json',dict(status='failed',error=traceback.format_exc()))
        raise
