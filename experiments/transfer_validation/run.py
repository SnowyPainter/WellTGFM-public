"""Run all six authorized validation packages, then seal and evaluate Test."""
import argparse
import collections
import fcntl
import json
import shutil
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from experiments.common.runtime import atomic_json, gpu_guard
from experiments.transfer_validation.core import ROOT, HERE, read, digest, canonical_hash, train_job
from experiments.transfer_validation.design import (OUT, METHODS, MAIN, protocol, tuning_jobs, mechanism_jobs,
    head_jobs, confirmation_jobs, factorial_jobs, breadth_jobs, query_budget_jobs, source_budget_jobs,
    source_budget_cases)
from experiments.transfer_validation.preparation import prepare_large, ensure_pretraining, source_fit
from experiments.transfer_validation.evaluation import legacy_models, verify_development, seal, evaluate_frozen
from experiments.transfer_validation.report import report, append_claim


def prospective_counts(out):
    dummy={m:{'config':{}} for m in METHODS}
    return {name:len(jobs) for name,jobs in [
        ('mechanism',mechanism_jobs(out)),('head_components',head_jobs(out)),('tuning',tuning_jobs(out)),
        ('long_budget',confirmation_jobs(dummy,out)),('seed_factorial',factorial_jobs(dummy,out)),
        ('breadth_and_scale',breadth_jobs(dummy,out)),('query_budget',query_budget_jobs(dummy,out)),
        ('source_budget',source_budget_jobs(dummy,out))]}


def implementation_hashes():
    paths=list(HERE.glob('*.py'))
    paths+=[ROOT/p for p in ['experiments/common/data.py','experiments/common/runtime.py',
        'experiments/common/engine.py','experiments/common/backbones.py','experiments/target_adaptation/readout.py',
        'models/ours/joint.py','models/ours/model.py','models/baselines/ctdg/tgat.py',
        'models/baselines/ctdg/dygformer.py','models/baselines/ctdg/modules.py','models/baselines/ctdg/sampling.py']]
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)}


def initialize(out):
    out.mkdir(parents=True,exist_ok=True)
    spec=dict(protocol=protocol(),counts=prospective_counts(out),implementation=implementation_hashes())
    if (out/'protocol.json').exists():
        assert read(out/'protocol.json')==spec, 'Protocol or code changed: use a new suite version'
    else:
        atomic_json(out/'protocol.json',spec)
        import sklearn
        atomic_json(out/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,
            numpy=np.__version__,sklearn=sklearn.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            timing_note='Other GPU workloads are not stopped; elapsed time includes their possible contention.'))
        for name in spec['implementation']:
            dest=out/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    return spec


def choose_winners(jobs,out):
    scores=collections.defaultdict(list);configs={}
    for j in jobs:
        r=read(out/'jobs'/j['id']/'result.json')
        assert r['status']=='complete'
        key=j['method'],j['trial'];scores[key].append(r['ap']);configs[key]=j['config']
    winners={}
    for m in METHODS:
        assert all(len(scores[m,i])==8 for i in range(9))
        trial=max(range(9),key=lambda i:np.mean(scores[m,i]))
        winners[m]=dict(trial=trial,config=configs[m,trial],mean_validation_ap=float(np.mean(scores[m,trial])),
            trials=[dict(trial=i,mean_validation_ap=float(np.mean(scores[m,i]))) for i in range(9)])
    p=out/'selected_configurations.json'
    if p.exists():assert read(p)==winners
    else:atomic_json(p,winners)
    return winners


def execute_stage(name,jobs,out,all_jobs):
    stage=out/'stages'/f'{name}.json'
    listing=dict(stage=name,jobs=jobs)
    if stage.exists():assert read(stage)==listing
    else:atomic_json(stage,listing)
    started=time.monotonic()
    for index,j in enumerate(jobs):
        atomic_json(out/'progress.json',dict(status='running',stage=name,stage_index=index+1,
            stage_jobs=len(jobs),completed_previous_stages=len(all_jobs),job=j['id'],test_used=False))
        c={k:j[k] for k in ['target','sources','pretrain_seed','profile','backbone','source_budget']}
        ensure_pretraining(c,out)
        train_job(j,out)
        if torch.cuda.is_available():torch.cuda.empty_cache()
    all_jobs.extend(jobs)
    atomic_json(out/'stages'/f'{name}_complete.json',dict(status='complete',jobs=len(jobs),seconds=time.monotonic()-started))
    report(out)


def run(out):
    spec=initialize(out)
    # Prepare data without reading any performance from the new Target domains.
    prepare_large()
    for t in ['ADX','BAG']:
        assert (ROOT/'data/benchmarks/mint32768/ctdg'/t/'metadata.json').exists()
    torch.set_num_threads(2)
    all_jobs=[]
    execute_stage('mechanism',mechanism_jobs(out),out,all_jobs)
    execute_stage('head_components',head_jobs(out),out,all_jobs)
    tune=tuning_jobs(out);execute_stage('tuning',tune,out,all_jobs)
    winners=choose_winners(tune,out)
    for name,jobs in [
        ('long_budget',confirmation_jobs(winners,out)),
        ('seed_factorial',factorial_jobs(winners,out)),
        ('breadth_and_scale',breadth_jobs(winners,out)),
        ('query_budget',query_budget_jobs(winners,out)),
        ('source_budget',source_budget_jobs(winners,out))]:
        execute_stage(name,jobs,out,all_jobs)
    for c in source_budget_cases():source_fit(c,out)
    checked=verify_development(all_jobs,out)
    atomic_json(out/'development_verification.json',dict(status='passed',jobs=len(all_jobs),test_used=False))
    # Search trials stay on Validation. All confirmatory models are sealed together.
    rows=[r for r in checked if r['job']['cohort']!='tuning']+legacy_models()
    frozen=seal(rows,out,canonical_hash(spec))
    for index,row in enumerate(frozen['models']):
        j=row['job']
        atomic_json(out/'progress.json',dict(status='running',stage='sealed_test',index=index+1,
            total=len(rows),job=j['id'],test_used=True))
        rich=(j['cohort'] in {'seed_factorial','breadth','scale'} and j['adaptation_seed']==7
              and j['pretrain_seed'] in {None,7})
        evaluate_frozen(row,out,rich)
        torch.cuda.empty_cache()
    report(out,final=True)
    from experiments.transfer_validation.plots import render
    render(out)
    append_claim(out)
    atomic_json(out/'progress.json',dict(status='complete',development_jobs=len(all_jobs),test_models=len(rows),
        test_used=True,claim_updated=True,analysis=str((out/'analysis').relative_to(ROOT))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--report',action='store_true')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args();out=args.out.resolve()
    if args.report:
        print(report(out));return
    if not args.execute:
        print(json.dumps(dict(protocol=protocol(),counts=prospective_counts(out)),indent=2));return
    out.mkdir(parents=True,exist_ok=True)
    with (out/'suite.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            while True:
                try:
                    with gpu_guard('cuda:0',0):
                        if args.smoke:
                            torch.set_num_threads(2)
                            # Real GPU/data smoke stays entirely on Validation.
                            jobs=mechanism_jobs(out)
                            probes=[next(j for j in jobs if j['method']==m and j['sources']) for m in ['tfra','joint_init','full_step','batch']]
                            for j in probes:train_job(j,out,smoke=True)
                            atomic_json(out/'progress.json',dict(status='smoke_complete',jobs=len(probes),test_used=False))
                        else:run(out)
                    break
                except BlockingIOError:
                    atomic_json(out/'progress.json',dict(status='waiting_for_gpu_lock',test_used=False))
                    time.sleep(30)
        except Exception as e:
            atomic_json(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()))
            prior=read(out/'progress.json') if (out/'progress.json').exists() else {}
            atomic_json(out/'progress.json',{**prior,'status':'failed','error':repr(e)})
            print(traceback.format_exc(),flush=True)
            raise


if __name__=='__main__':main()
