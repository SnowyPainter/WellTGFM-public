"""Evaluate existing Validation-selected checkpoints on Test without training."""
import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from experiments.common.runtime import atomic_json, evaluate, gpu_guard
from experiments.common.engine import save_predictions
from experiments.transfer_validation.core import ROOT, BASE, DEFAULT, data_for, model_for, read, digest, canonical_hash
from experiments.transfer_validation.evaluation import legacy_models, resolve_reuse, seal

OUT = BASE / 'outputs/test_v1'
METHOD_MAP = {'finetune': 'finetune', 'ridge_epoch': 'tfra', 'ridge_frozen': 'frozen',
              'ridge_warmstart': 'warmstart', 'ridge_profile': 'batch', 'ridge_joint': 'joint'}


def saved_models():
    rows = legacy_models()
    panel = read(BASE / 'outputs/readout_panel_v2/plan.json')
    # The two overlapping methods are evaluated once and reused in panel aggregation.
    for old_method, method in METHOD_MAP.items():
        if method in {'finetune', 'tfra'}:
            continue
        for job in panel['jobs']:
            dest = resolve_reuse(BASE / 'outputs/readout_panel_v2' / old_method / job['job_id'])
            cfg = {**DEFAULT, **panel['configs'][job['target']]}
            j = {**job, 'id': 'panel__' + canonical_hash([method, job['job_id']])[:20],
                 'method': method, 'cohort': 'legacy_panel', 'profile': 'original',
                 'backbone': 'tgat', 'source_budget': None, 'config': cfg}
            rows.append(dict(job=j, best=str((dest/'best.pt').relative_to(ROOT)),
                             validation_ap=read(dest/'result.json')['ap'], config=cfg))
    assert len(rows) == 1752
    for row in rows:
        dest = (ROOT/row['best']).parent
        result = read(dest/'result.json')
        assert result['status'] == 'complete' and len(result['history']) == 5
        assert abs(result['ap'] - max(e['validation']['ap'] for e in result['history'])) < 1e-12
        with np.load(dest/'validation_predictions.npz') as p:
            ap = average_precision_score(p['labels'].ravel(), p['scores'].ravel())
        assert abs(ap-row['validation_ap']) < 1e-10
    # Fixed order includes all scratch models first; no ranking by observed performance.
    return sorted(rows, key=lambda r: (r['job']['cohort'] != 'legacy_powerset',
        len(r['job']['sources']), r['job']['adaptation_seed'], r['job']['target'],
        tuple(sorted(r['job']['sources'])), r['job']['method']))


class Evaluator:
    def __init__(self):
        self.models = OrderedDict()
        self.negatives = {}

    def load(self, row):
        job, cfg = row['job'], row['config']
        data = data_for(job); d = data[job['target']]
        key = (job['target'], job['method'] == 'refine', job['adaptation_seed'])
        if key not in self.models:
            # Every persistent weight is restored from the full saved model below.
            # Recent-neighbor sampling is deterministic; cache preserves seed as well.
            self.models[key] = model_for({**job, 'checkpoint': None}, cfg, data)
            if len(self.models) > 4:
                self.models.popitem(last=False)
        self.models.move_to_end(key)
        model = self.models[key]
        saved = torch.load(ROOT/row['best'], map_location=cfg['device'], weights_only=False)
        model.load_state_dict(saved['model'], strict=True); model.bind(d)
        return model, d, saved

    def negative(self, d, seed):
        key = (canonical_hash(d.metadata['artifacts']), seed)
        if key not in self.negatives:
            self.negatives[key] = d.negatives(seed)
        return self.negatives[key]

    def replay(self, row):
        model, d, saved = self.load(row)
        metrics, scores = evaluate(model, d, 1, self.negative(d, 100007), row['config'])
        with np.load((ROOT/row['best']).parent/'validation_predictions.npz') as old:
            np.testing.assert_allclose(scores, old['scores'].ravel(), rtol=1e-5, atol=1e-6)
        assert abs(metrics['ap']-row['validation_ap']) < 1e-8
        return dict(job=row['job']['id'], ap=metrics['ap'], expected=row['validation_ap'])

    def test(self, row, out):
        dest = out/'test'/row['job']['id']; dest.mkdir(parents=True, exist_ok=True)
        if (dest/'result.json').exists():
            result = read(dest/'result.json')
            assert result['best_sha256'] == row['best_sha256'] and result['split'] == 'test'
            return result
        assert digest(ROOT/row['best']) == row['best_sha256']
        started = time.monotonic()
        model, d, saved = self.load(row)
        fixed = self.negative(d, 200007)
        metrics, scores = evaluate(model, d, 2, fixed, row['config'])
        assert np.isfinite(scores).all()
        save_predictions(dest/'test_predictions.npz', d, 2, fixed, scores)
        with np.load(dest/'test_predictions.npz') as p:
            assert abs(average_precision_score(p['labels'].ravel(), p['scores'].ravel())-metrics['ap']) < 1e-12
        selected_epoch = saved.get('epoch', read((ROOT/row['best']).parent/'result.json')['best_epoch'])
        result = dict(status='complete', job=row['job'], selected_epoch=selected_epoch,
            validation_ap=row['validation_ap'], metrics={'random_1': metrics},
            best_sha256=row['best_sha256'], split='test', seconds=time.monotonic()-started)
        atomic_json(dest/'result.json', result)
        print(json.dumps(dict(job=row['job']['id'], method=row['job']['method'],
            target=row['job']['target'], seconds=result['seconds'])), flush=True)
        return result


WORKER_EVALUATOR = None


def worker_evaluate(row, out):
    global WORKER_EVALUATOR
    if WORKER_EVALUATOR is None:
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('high')
        WORKER_EVALUATOR = Evaluator()
    WORKER_EVALUATOR.test(row, Path(out))
    return row['job']['id']


def run(out):
    torch.set_num_threads(2)
    rows = saved_models()
    spec = dict(models=len(rows), selection='Existing best Validation epoch; no training, refitting, or Test selection',
        test_negative_seed=200007, panel=read(BASE/'outputs/readout_panel_v2/plan.json'),
        prior_development_protocol_sha256=digest(ROOT/'experiments/transfer_validation/outputs/v1/protocol.json'),
        code_sha256=digest(Path(__file__)),
        interpretation='Existing-model Test precedes additional experiments by explicit user request. '
        'The additional protocol was fixed before this Test; its continuation must not be tuned to these results.')
    p = out/'protocol.json'
    if p.exists(): assert read(p) == spec
    else: atomic_json(p, spec)
    frozen = seal(rows, out, canonical_hash(spec))
    for path in [Path(__file__), ROOT/'experiments/transfer_validation/evaluation.py',
                 ROOT/'experiments/transfer_validation/core.py', ROOT/'experiments/common/runtime.py']:
        dest=out/'code_snapshot'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists(): shutil.copy2(path,dest)
    evaluator = Evaluator()
    if not (out/'validation_replay.json').exists():
        checks=[];seen=set()
        for row in frozen['models']:
            j=row['job'];key=(j['method'],j['target'])
            if key in seen:continue
            seen.add(key);checks.append(evaluator.replay(row))
        atomic_json(out/'validation_replay.json',dict(passed=True,checks=checks))
    del evaluator
    torch.cuda.empty_cache()
    started=time.monotonic()
    completed=0;pending=[]
    for row in frozen['models']:
        p=out/'test'/row['job']['id']/'result.json'
        if p.exists():
            assert read(p)['best_sha256']==row['best_sha256'];completed+=1
        else:pending.append(row)
    atomic_json(out/'progress.json',dict(status='running',stage='test_inference',completed=completed,
        total=len(rows),test_used=True,workers=4))
    # Independent read-only inference; the parent retains the shared GPU lock.
    # Spawn avoids inheriting a CUDA context. No optimization or sample order changes.
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(worker_evaluate,row,str(out)):row for row in pending}
        for future in as_completed(futures):
            job_id=future.result();completed+=1;row=futures[future]
            atomic_json(out/'progress.json',dict(status='running',stage='test_inference',completed=completed,
                total=len(rows),job=job_id,method=row['job']['method'],target=row['job']['target'],test_used=True,
                workers=4,elapsed_seconds=time.monotonic()-started))
    from experiments.target_adaptation.test_report import report
    report(out, update_claim=True)
    atomic_json(out/'progress.json',dict(status='complete',completed=len(rows),total=len(rows),test_used=True,
        claim_updated=False,manuscript_draft_ready=True,seconds=time.monotonic()-started))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if not args.execute:
        rows=saved_models();print(json.dumps(dict(saved_models=len(rows),training=False)));return
    with (out/'suite.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            with gpu_guard('cuda:0',0):run(out)
        except Exception as e:
            atomic_json(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()))
            atomic_json(out/'progress.json',dict(status='failed',error=repr(e)))
            raise


if __name__=='__main__':main()
