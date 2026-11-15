"""Serial resumable queue. Default: print plan. --execute: train and evaluate."""
import argparse
import collections
import fcntl
import json
import platform
import shutil
import time
import traceback
from pathlib import Path
import torch
import numpy as np
from experiments.common.runtime import atomic_json,gpu_guard
from experiments.robust.design import ROOT,HERE,OUT,plan
from experiments.robust.training import read,digest,canonical_hash,train_job
from experiments.robust.provenance import local_implementations
from experiments.robust.preparation import ensure_pretraining
from experiments.robust.reuse import import_selected,continue_from_five
from experiments.robust.evaluation import evaluate_job
from experiments.robust.report import summarize


def initialize(out):
    p=plan(out)
    paths=[]
    for root in ['experiments/robust','experiments/common','experiments/target_adaptation',
                 'experiments/transfer_validation','experiments/transfer','models']:
        paths += [f for f in (ROOT/root).rglob('*.py') if 'outputs' not in f.parts and 'reference' not in f.parts]
    p['implementation']={str(f.relative_to(ROOT)):digest(f) for f in sorted(set(paths))}
    p['local_implementations']=local_implementations()
    p['source_checkpoints']={j['checkpoint']:digest(ROOT/j['checkpoint']) for j in p['jobs'] if j['checkpoint'] and j['backbone']=='tgat'}
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists(): assert read(out/'protocol.json')==p,'Protocol or implementation changed; use a new output path'
    else:
        atomic_json(out/'protocol.json',p)
        for name in p['implementation']:
            dest=out/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        atomic_json(out/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,
            numpy=np.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)))
    return p


def execute(out):
    p=initialize(out);jobs={j['id']:j for j in p['jobs']}
    started=time.time();failures={};done=set()
    with (out/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Shared lock prevents competing with other wellgfm experiment runners.
        while True:
            guard=gpu_guard('cuda:0',0)
            try: guard.__enter__();break
            except BlockingIOError:
                atomic_json(out/'progress.json',dict(status='waiting_for_gpu_lock',updated_at=time.time()))
                time.sleep(15)
        try:
            torch.set_num_threads(2)
            for stage,views in p['stages'].items():
                ids=list(dict.fromkeys(v['job_id'] for v in views))
                for i,ident in enumerate(ids):
                    if ident in done or ident in failures: continue
                    j=jobs[ident];dest=out/'jobs'/ident
                    atomic_json(out/'progress.json',dict(status='running',stage=stage,stage_index=i+1,
                        stage_jobs=len(ids),completed=len(done),failed=len(failures),total_jobs=len(jobs),
                        job=j,started_at=started,updated_at=time.time()))
                    print(dict(stage=stage,index=i+1,total=len(ids),target=j['target'],sources=j['sources'],
                        backbone=j['backbone'],method=j['method'],pretrain_seed=j['pretrain_seed'],seed=j['adaptation_seed']),flush=True)
                    try:
                        ensure_pretraining(j,out)
                        if not (dest/'result.json').exists() and not import_selected(j,out):
                            continue_from_five(j,out)
                            train_job(j,out)
                        evaluate_job(j,out)
                        done.add(ident)
                        if (dest/'failure.json').exists(): (dest/'failure.json').unlink()
                    except Exception as error:
                        failure=dict(job=j,error=repr(error),traceback=traceback.format_exc(),updated_at=time.time())
                        atomic_json(dest/'failure.json',failure);failures[ident]=failure
                        print(failure['traceback'],flush=True)
                    finally:
                        import gc
                        gc.collect();torch.cuda.empty_cache()
                    # Persist useful partial reports, without touching manuscript files.
                    if (i+1)%18==0: summarize(p,out)
                summarize(p,out)
            atomic_json(out/'progress.json',dict(status='completed_with_failures' if failures else 'complete',
                completed=len(done),failed=len(failures),total_jobs=len(jobs),started_at=started,updated_at=time.time()))
            atomic_json(out/'failures.json',failures)
        finally: guard.__exit__(None,None,None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--report',action='store_true')
    args=parser.parse_args();out=args.out.resolve()
    if args.execute: execute(out)
    elif args.report: summarize(read(out/'protocol.json'),out)
    else:
        p=plan(out)
        print(json.dumps(dict(unique_training_jobs=len(p['jobs']),
            stages={k:len(set(v['job_id'] for v in vs)) for k,vs in p['stages'].items()},
            per_method=collections.Counter(j['method'] for j in p['jobs'])),indent=2))


if __name__=='__main__': main()
