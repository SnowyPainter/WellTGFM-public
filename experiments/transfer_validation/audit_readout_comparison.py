"""Read-only audit of LP-FT and the three completed readout mechanism arms.

Recompute metrics from saved predictions, check paired inputs/provenance, and
optionally replay saved models. No training, model selection, or refitting of
evaluated models is performed. Initial readout fits only check saved invariants.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from experiments.common.runtime import evaluate, gpu_guard
from experiments.target_adaptation.runtime import build as legacy_build
from experiments.transfer_validation.core import (
    ROOT, build, data_for, digest, fit, model_for, read,
)

NEW = ROOT / 'experiments/transfer_validation/outputs/readout_mechanism_v2'
OLD = ROOT / 'experiments/target_adaptation/outputs/test_v1'
METHODS = ('lp_ft', 'initial_fixed', 'joint_init', 'tfra')


def condition(row):
    j = row['job']
    return j['target'], tuple(sorted(j['sources'])), j['adaptation_seed']


def array_digest(values):
    h = hashlib.sha256()
    for name, value in values:
        a = np.ascontiguousarray(value)
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


@lru_cache(maxsize=None)
def file_digest(path):
    return digest(path)


def independent_summary(records):
    controls = {}
    for r in records:
        if r['sources']:
            continue
        k = r['method'], r['target'], r['seed']
        assert k not in controls, ('duplicate scratch', k)
        controls[k] = r['ap']
    assert len(controls) == 48
    transfer = []
    for r in records:
        if not r['sources']:
            continue
        baseline = controls[r['method'], r['target'], r['seed']]
        transfer.append(dict(**r, target_only_ap=baseline, gain=r['ap']-baseline))
    summary = []
    for m in METHODS:
        groups = defaultdict(list)
        for r in transfer:
            if r['method'] == m:
                groups[r['target']].append(r)
        assert sum(map(len, groups.values())) == 42 and len(groups) == 4
        per_target = []
        for target, rr in sorted(groups.items()):
            row = dict(method=m, target=target, runs=len(rr),
                       ap=mean(r['ap'] for r in rr),
                       target_only_ap=mean(r['target_only_ap'] for r in rr),
                       D_G=mean(max(-r['gain'], 0) for r in rr),
                       B_G=mean(max(r['gain'], 0) for r in rr),
                       negative=sum(r['gain'] < 0 for r in rr))
            per_target.append(row)
            summary.append(row)
        summary.append(dict(method=m, target='target_equal_mean', runs=42,
                            **{k: mean(r[k] for r in per_target)
                               for k in ['ap', 'target_only_ap', 'D_G', 'B_G']}))
    return transfer, summary


def audit(replay=False):
    new = read(NEW/'test_freeze.json')['models']
    keys = {condition(r) for r in new}
    assert len(new) == 162 and len(keys) == 54
    old = [r for r in read(OLD/'test_freeze.json')['models']
           if r['job']['method'] == 'lp_ft' and condition(r) in keys]
    assert len(old) == 54
    all_rows = [(NEW, r) for r in new] + [(OLD, r) for r in old]
    paired, source_hashes, records = {}, {}, []
    for root, row in all_rows:
        j = row['job']
        assert j['target'] not in j['sources']
        assert file_digest(str(ROOT/row['best'])) == row['best_sha256']
        src = file_digest(str(ROOT/j['checkpoint'])) if j['sources'] else None
        assert source_hashes.setdefault(condition(row), src) == src
        test = read(root/'test'/j['id']/'result.json')
        train_dir = (ROOT/row['best']).parent
        tr = read(train_dir/'result.json')
        man = read(train_dir/'manifest.json')
        history = tr['history']
        assert len(history) == 5 and tr['ap'] == max(x['validation']['ap'] for x in history)
        assert test['best_sha256'] == row['best_sha256']
        assert test['selected_epoch'] == tr['best_epoch']
        assert test['validation_ap'] == row['validation_ap'] == tr['ap']
        if j['method'] != 'lp_ft':
            assert tr['updates'] == 60
            assert tr['fits'] == (10 if j['method'] == 'tfra' else 6)
        else:
            assert file_digest(str(ROOT/'experiments/target_adaptation/lp_ft.py')) == man['runner_sha256']
            probe = read(train_dir/'linear_probe.json')
            assert len(probe['trials']) == 7
        for split, path, expected in [
            ('validation', train_dir/'validation_predictions.npz', row['validation_ap']),
            ('test', root/'test'/j['id']/'test_predictions.npz', test['metrics']['random_1']['ap']),
        ]:
            with np.load(path) as p:
                ap = float(average_precision_score(p['labels'].ravel(), p['scores'].ravel()))
                assert abs(ap-expected) < 1e-12
                candidates = array_digest((k, p[k]) for k in
                                          ['event_ids', 'src', 'dst', 'time', 'negatives', 'labels'])
                assert paired.setdefault((j['target'], split), candidates) == candidates
        records.append(dict(method=j['method'], target=j['target'],
                            sources=sorted(j['sources']), seed=j['adaptation_seed'],
                            ap=test['metrics']['random_1']['ap']))
    transfer, summary = independent_summary(records)
    recorded = read(NEW/'analysis/test_summary.json')
    for row in summary:
        if row['method'] == 'lp_ft':
            continue
        oldrow = next(r for r in recorded if r['method'] == row['method'] and r['target'] == row['target'])
        for metric in ['ap', 'target_only_ap', 'D_G', 'B_G']:
            assert abs(row[metric]-oldrow[metric]) < 1e-12
    result = dict(status='passed', saved_models=216, prediction_files_checked=432,
                  model_hashes=True, paired_validation_and_test_candidates=True,
                  same_source_checkpoint_per_condition=True,
                  validation_epoch_selection=True, independent_metric_recomputation=True,
                  summary=summary, transfer=transfer, runtime_checks=[])
    print('Saved predictions, provenance and independent summaries passed: 216 models.', flush=True)
    if not replay:
        return result

    cfgs = read(ROOT/'experiments/target_adaptation/inputs/configs.json')
    # Constructors should give identical initial target features for all 12 seeds/targets.
    for target in ['wikipedia', 'reddit', 'mooc', 'lastfm']:
        for seed in [7, 17, 37]:
            row = next(r for r in new if r['job']['target'] == target
                       and r['job']['adaptation_seed'] == seed and not r['job']['sources'])
            j, cfg = row['job'], {**row['config'], 'device': 'cpu'}
            data = data_for(j)
            a = build(j, cfg, data, seed)
            b = legacy_build({**cfgs[target], 'device': 'cpu'}, data, seed)
            assert list(a.state_dict()) == list(b.state_dict())
            assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
            result['runtime_checks'].append(dict(check='same_initial_model', target=target, seed=seed))
            del a, b
    print('Legacy/new constructors identical for 12 Target/seed combinations.', flush=True)

    with gpu_guard('cuda:0', 0):
        # One transfer and its scratch control, all four methods; no model changes.
        for root, row in all_rows:
            j = row['job']
            if j['target'] != 'reddit' or j['adaptation_seed'] != 7:
                continue
            if j['sources'] and set(j['sources']) != {'wikipedia', 'GDELT'}:
                continue
            cfg, data = row['config'], data_for(j)
            d = data[j['target']]
            model = model_for(j, cfg, data)
            saved = torch.load(ROOT/row['best'], map_location=cfg['device'], weights_only=False)
            model.load_state_dict(saved['model'], strict=True)
            model.bind(d)
            for split, split_id, seed in [('validation', 1, 100007), ('test', 2, 200007)]:
                metrics, scores = evaluate(model, d, split_id, d.negatives(seed), cfg)
                path = ((ROOT/row['best']).parent/'validation_predictions.npz' if split_id == 1
                        else root/'test'/j['id']/'test_predictions.npz')
                with np.load(path) as p:
                    error = float(np.max(np.abs(scores-p['scores'].ravel())))
                    np.testing.assert_allclose(scores, p['scores'].ravel(), rtol=1e-5, atol=1e-6)
                    assert abs(metrics['ap']-average_precision_score(p['labels'].ravel(),p['scores'].ravel())) < 1e-10
                result['runtime_checks'].append(dict(check='saved_model_replay', method=j['method'],
                                                    sources=j['sources'], split=split, max_score_error=error))
            print('Replayed', j['method'], j['sources'] or 'scratch', flush=True)
            del model

        # Actual initial ridge vs the training readout saved after five epochs.
        for row in new:
            j = row['job']
            if j['method'] != 'initial_fixed' or j['adaptation_seed'] != 7:
                continue
            choices = {'wikipedia': {'reddit'}, 'reddit': {'wikipedia', 'GDELT'},
                       'mooc': {'wikipedia', 'lastfm'}, 'lastfm': {'mooc'}}
            if j['sources'] and set(j['sources']) != choices[j['target']]:
                continue
            cfg, data = row['config'], data_for(j)
            model = model_for(j, cfg, data)
            d = data[j['target']]
            fit(model, d, d.negatives(j['adaptation_seed']), cfg)
            resume = torch.load((ROOT/row['best']).parent/'resume.pt', map_location=cfg['device'], weights_only=False)
            prefix = f"heads.{j['target']}.2."
            assert all(torch.equal(v, resume['model'][prefix+k])
                       for k, v in model.heads[j['target']][2].state_dict().items())
            hidden_prefix = f"heads.{j['target']}.0."
            assert all(torch.equal(v, resume['model'][hidden_prefix+k])
                       for k, v in model.heads[j['target']][0].state_dict().items())
            result['runtime_checks'].append(dict(check='initial_fixed_head_unchanged_after_60_updates',
                                                target=j['target'], sources=j['sources']))
            print('Verified fixed training Head', j['target'], j['sources'] or 'scratch', flush=True)
            del model, resume
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = audit(args.replay)
    dest = NEW/'audit'/'lpft_readout_comparison.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    result['audit_code_sha256'] = digest(Path(__file__))
    dest.write_text(json.dumps(result, indent=2)+'\n')
    print('Audit passed:', dest, flush=True)
