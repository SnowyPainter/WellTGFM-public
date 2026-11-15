"""Run only the three agreed readout arms on the existing 14-composition panel.

Training definitions are an unchanged subset of the earlier frozen mechanism
plan. This entry point never advances to other suites or edits the manuscript.
"""
import argparse
import collections
import fcntl
import json
import platform
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from experiments.common.runtime import atomic_json, evaluate, gpu_guard
from experiments.transfer_validation.core import (
    ROOT, HERE, data_for, model_for, read, digest, canonical_hash, train_job,
)
from experiments.transfer_validation.design import OUT as PARENT, mechanism_jobs
from experiments.transfer_validation.evaluation import (
    verify_development, seal, evaluate_frozen, resolve_reuse,
)
from experiments.transfer_validation.report import records, summarize, csv_write

PREVIOUS = HERE / 'outputs/readout_mechanism_v1'
OUT = HERE / 'outputs/readout_mechanism_v2'
METHODS = ('initial_fixed', 'joint_init', 'tfra')
LABELS = {
    'initial_fixed': '최초 적합 후 유지',
    'joint_init': '최초 적합 후 공동 갱신',
    'tfra': 'TFRA',
}


def planned_jobs():
    jobs = [j for j in mechanism_jobs(PARENT) if j['method'] in METHODS]
    assert len(jobs) == 162
    assert collections.Counter(j['method'] for j in jobs) == {m: 54 for m in METHODS}
    assert len({canonical_hash(j['config']) for j in jobs}) == 1
    return jobs


def initialize(out):
    parent = read(PARENT / 'protocol.json')
    # Refuse silent implementation changes relative to the pre-existing plan.
    for path, sha in parent['implementation'].items():
        assert digest(ROOT / path) == sha, f'Parent implementation changed: {path}'
    jobs = planned_jobs()
    parent_jobs = read(PARENT / 'stages/mechanism.json')['jobs']
    assert jobs == [j for j in parent_jobs if j['method'] in METHODS]
    implementation = {**parent['implementation'],
                      str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))}
    spec = dict(
        version=2, parent_protocol_sha256=digest(PARENT / 'protocol.json'),
        methods=list(METHODS), jobs=jobs, implementation=implementation,
        source_checkpoints={j['checkpoint']: digest(ROOT / j['checkpoint'])
                            for j in jobs if j['checkpoint']},
        selection='best Validation AP among epochs 1 through 5',
        transfer_reference='same method, Target, and adaptation seed Target-only',
        aggregation='equal Target weight after averaging each Target',
        test_context='Existing benchmark Test results were already seen. These arms, '
                     'settings, and cases were fixed in the earlier parent plan.',
        evaluation_readout='All arms refit on Train for epoch evaluation; the '
                           'initial_fixed/joint_init training head is then restored.',
        dropout='MSE arms run in eval mode throughout: dropout is inactive.',
        manuscript_autowrite=False, continue_to_other_stages=False,
        reuse_policy='Preserve completed and resumable v1 jobs. Reuse verified legacy '
                     'TFRA models for every remaining non-diagnostic TFRA condition.',
    )
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'protocol.json'
    if path.exists():
        assert read(path) == spec, 'Protocol changed; use a new output directory'
    else:
        atomic_json(path, spec)
        for name in implementation:
            dest = out / 'code_snapshot' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        atomic_json(out / 'environment.json', dict(
            python=platform.python_version(), torch=torch.__version__,
            numpy=np.__version__, cuda=torch.version.cuda,
        ))
    return spec


def import_completed(jobs, out):
    """Preserve exact prior jobs, including interrupted epoch checkpoints."""
    reused = []
    for j in jobs:
        dest = out / 'jobs' / j['id']
        if (dest / 'result.json').exists():
            if (dest / 'reuse.json').exists():
                reused.append(j['id'])
            continue
        if dest.exists():
            continue
        source = next((root / 'jobs' / j['id'] for root in [PREVIOUS, PARENT]
                       if (root / 'jobs' / j['id'] / 'manifest.json').exists()), None)
        if source is None:
            continue
        if not (source / 'result.json').exists():
            manifest = read(source / 'manifest.json')
            assert manifest['job'] == j and manifest['code_sha256'] == digest(HERE / 'core.py')
            if (source / 'resume.pt').exists():
                dest.mkdir(parents=True)
                for name in ['manifest.json', 'resume.pt', 'best.pt', 'validation_predictions.npz']:
                    shutil.copy2(source / name, dest / name)
                atomic_json(dest / 'resume_from.json', dict(source=str(source.relative_to(ROOT))))
            continue
        result, manifest = read(source / 'result.json'), read(source / 'manifest.json')
        assert result['status'] == 'complete' and result['job'] == j
        assert manifest['job'] == j and manifest['config'] == j['config']
        assert manifest['code_sha256'] == digest(HERE / 'core.py')
        checkpoint_hash = digest(ROOT / j['checkpoint']) if j['checkpoint'] else None
        assert manifest['checkpoint_sha256'] == checkpoint_hash
        required = ['manifest.json', 'result.json', 'best.pt', 'resume.pt',
                    'validation_predictions.npz']
        hashes = {name: digest(source / name) for name in required}
        dest.mkdir(parents=True)
        for name in required:
            shutil.copy2(source / name, dest / name)
            assert digest(dest / name) == hashes[name]
        atomic_json(dest / 'reuse.json', dict(source=str(source.relative_to(ROOT)),
                                             artifacts=hashes))
        reused.append(j['id'])
    return reused


def import_legacy_tfra(jobs, out):
    """Reuse original TFRA artifacts with explicit original-code provenance."""
    from experiments.target_adaptation.runtime import load_inputs
    old_jobs, _, _ = load_inputs(False)
    imported = []
    for j in jobs:
        if j['method'] != 'tfra' or j.get('diagnostics'):
            continue
        dest = out / 'jobs' / j['id']
        if (dest / 'result.json').exists() or (dest / 'resume.pt').exists():
            continue
        old = next(x for x in old_jobs if x['target'] == j['target']
                   and set(x['sources']) == set(j['sources'])
                   and x['adaptation_seed'] == j['adaptation_seed'])
        source = resolve_reuse(ROOT / 'experiments/target_adaptation/outputs/readout_panel_v2/ridge_epoch' / old['job_id'])
        r, m = read(source / 'result.json'), read(source / 'manifest.json')
        assert r['status'] == 'complete' and m['method'] == 'ridge_epoch'
        for recorded in [r['job'], m['job']]:
            assert {k: v for k, v in recorded.items() if k != 'checkpoint'} == {
                k: v for k, v in old.items() if k != 'checkpoint'}
            if old['checkpoint']:
                assert digest(ROOT / recorded['checkpoint']) == digest(ROOT / old['checkpoint'])
            else:
                assert recorded['checkpoint'] == old['checkpoint']
        for key in ['epochs', 'learning_rate', 'weight_decay', 'grad_clip', 'batch_size',
                    'microbatch', 'history_length', 'device']:
            assert m['config'][key] == j['config'][key], key
        assert m['config']['data_root'] == 'data/benchmarks/events32768'
        assert m['ridge_lambda'] == j['config']['ridge_lambda']
        assert m['support_queries'] == j['config']['support_queries']
        assert m['dropout'] == 'disabled during support,query,and readout fitting'
        assert j['config']['query_fraction'] == 1
        checkpoint_hash = digest(ROOT / j['checkpoint']) if j['checkpoint'] else None
        assert r['checkpoint_sha256'] == old['checkpoint_sha256'] == (
            checkpoint_hash if j['checkpoint'] else 'random_initialization')
        assert len(r['history']) == 5 and r['encoder_updates'] == 60
        assert r['full_train_readout_passes'] == 10
        assert r['evaluation_negative_seed'] == 100007
        assert r['ap'] == max(e['validation']['ap'] for e in r['history'])
        with np.load(source / 'validation_predictions.npz') as p:
            assert abs(average_precision_score(p['labels'].ravel(), p['scores'].ravel()) - r['ap']) < 1e-10
        provenance = dict(source=str(source.relative_to(ROOT)),
                          original_method='ridge_epoch',
                          original_runner_sha256=m['runner_sha256'],
                          original_solver_sha256=m['solver_sha256'],
                          artifacts={name: digest(source / name) for name in
                                     ['manifest.json', 'result.json', 'best.pt', 'validation_predictions.npz']})
        dest.mkdir(parents=True, exist_ok=True)
        for name in ['best.pt', 'validation_predictions.npz']:
            shutil.copy2(source / name, dest / name)
        # This manifest identifies the code that actually trained the reused model.
        atomic_json(dest / 'manifest.json', dict(job=j, config=j['config'],
            code_sha256=m['runner_sha256'], checkpoint_sha256=checkpoint_hash,
            split='validation', test_used=False, legacy_reuse=provenance))
        atomic_json(dest / 'result.json', dict(status='complete', job=j,
            ap=r['ap'], best_epoch=r['best_epoch'], history=r['history'], diagnostics=[],
            seconds=r['seconds'], updates=r['encoder_updates'], fits=r['full_train_readout_passes'],
            initial_validation=None, checkpoint_sha256=checkpoint_hash,
            test_used=False, legacy_reuse=provenance))
        atomic_json(dest / 'reuse.json', provenance)
        imported.append(j['id'])
    return imported


def verify_legacy_reuse(dest):
    r = read(dest / 'result.json')
    provenance = r['legacy_reuse']
    source = ROOT / provenance['source']
    for name, sha in provenance['artifacts'].items():
        assert digest(source / name) == sha, f'Reused source changed: {source / name}'
    for name in ['best.pt', 'validation_predictions.npz']:
        assert digest(dest / name) == provenance['artifacts'][name]
    assert read(dest / 'manifest.json')['legacy_reuse'] == provenance
    return r


def verify_training(rows, out):
    """Check actual updates/fits, paired candidates, and saved-model replays."""
    candidates = {}
    replayed = []
    for row in rows:
        j = row['job']
        dest = out / 'jobs' / j['id']
        result = read(dest / 'result.json')
        assert result['updates'] == 60 and result['best_epoch'] in range(1, 6)
        assert result['fits'] == (10 if j['method'] == 'tfra' else 6)
        with np.load(dest / 'validation_predictions.npz') as p:
            # Candidate identity must match across methods and adaptation seeds.
            candidate = canonical_hash({k: p[k].tolist() for k in
                ['event_ids', 'src', 'dst', 'time', 'negatives', 'labels']})
            previous = candidates.setdefault(j['target'], candidate)
            assert previous == candidate, 'Unpaired Validation candidates'
        if j['target'] == 'lastfm' and j['sources'] == ['mooc'] and j['adaptation_seed'] == 7:
            cfg = row['config']
            data = data_for(j)
            d = data[j['target']]
            model = model_for(j, cfg, data)
            saved = torch.load(ROOT / row['best'], map_location=cfg['device'], weights_only=False)
            model.load_state_dict(saved['model'])
            model.bind(d)
            metrics, scores = evaluate(model, d, 1, d.negatives(100007), cfg)
            assert abs(metrics['ap'] - row['validation_ap']) < 1e-10
            with np.load(dest / 'validation_predictions.npz') as p:
                np.testing.assert_allclose(scores, p['scores'].ravel(), rtol=1e-6, atol=1e-7)
            replayed.append(dict(method=j['method'], job=j['id'], ap=metrics['ap']))
            del model
    assert {r['method'] for r in replayed} == set(METHODS)
    atomic_json(out / 'development_verification.json', dict(
        status='passed', jobs=len(rows), paired_candidates=True,
        updates_per_run=60, replayed=replayed, test_used=False,
    ))


def write_report(out, jobs):
    """Keep method-matched transfer metrics and gradient diagnostics together."""
    root = out / 'analysis'
    root.mkdir(exist_ok=True)
    for split in ['validation', 'test']:
        rows = records(out, split)
        assert len(rows) == len(jobs)
        transfer, summary = summarize(rows)
        assert len(transfer) == 126
        csv_write(root / f'{split}_runs.csv', rows)
        csv_write(root / f'{split}_transfer.csv', transfer)
        csv_write(root / f'{split}_summary.csv', summary)
        atomic_json(root / f'{split}_summary.json', summary)
    diagnostics = []
    for j in jobs:
        result = read(out / 'jobs' / j['id'] / 'result.json')
        diagnostics.extend(dict(job_id=j['id'], target=j['target'],
                                sources='+'.join(j['sources']), method=j['method'],
                                seed=j['adaptation_seed'], **d)
                           for d in result['diagnostics'])
    csv_write(root / 'gradient_diagnostics.csv', diagnostics)
    overall = {r['method']: r for r in summary if r['target'] == 'target_equal_mean'}
    lines = [
        '# 예측층 반복 적합의 구성 요소 분석', '',
        '14개 Source 구성 × 3 적응 seed + 12회 Target-only = 방법당 54조건, 총 162조건.',
        '사전학습 seed 7, 5 epoch, 60회 Encoder 업데이트. 기존 계획의 설정을 그대로 사용했다.',
        '모든 방법은 평가 시 전체 Train에 예측층을 적합한다. 최초 적합 후 유지·공동 갱신에서는 '
        '평가 뒤 학습용 예측층을 복원한다.',
        '전이 이득은 같은 방법·Target·적응 seed의 Target-only를 차감하며, 네 Target에 동일 비중을 부여한다.',
        '모델은 Validation에서 선택한 뒤 고정하여 Test에서 평가했다. 기존 벤치마크 Test를 확인한 이후의 '
        '실행이며, 실험 조건은 그 전에 고정된 계획에서 가져왔다.', '',
        '| 방법 | Target-only AP | 전이 Test AP | 평균 전이 이득 | 음수 피해 D_G | 양수 이득 B_G | Seed SD |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for method in METHODS:
        r = overall[method]
        values = [r[k] for k in ['target_only_ap', 'ap', 'mean_gain', 'D_G', 'B_G', 'seed_sd']]
        lines.append('| ' + LABELS[method] + ' | ' + ' | '.join(f'{v:.6f}' for v in values) + ' |')
    lines += ['', '조건별 결과와 Target별 집계는 `analysis/`에 저장한다. '
              'Gradient 진단은 기존 계획에 지정된 네 대표 전이 구성의 TFRA와 최초 적합 후 공동 갱신에 해당한다.',
              '이 보고서 생성은 `docs/claim.md`를 변경하지 않는다.', '']
    (out / 'RESULTS.md').write_text('\n'.join(lines))


def verify_test(rows, out):
    candidates = {}
    for row in rows:
        j = row['job']
        dest = out / 'test' / j['id']
        result = read(dest / 'result.json')
        assert result['status'] == 'complete' and result['split'] == 'test'
        assert result['job'] == j and result['best_sha256'] == row['best_sha256']
        assert digest(ROOT / row['best']) == row['best_sha256']
        with np.load(dest / 'test_predictions.npz') as p:
            ap = average_precision_score(p['labels'].ravel(), p['scores'].ravel())
            assert abs(ap - result['metrics']['random_1']['ap']) < 1e-10
            candidate = canonical_hash({k: p[k].tolist() for k in
                ['event_ids', 'src', 'dst', 'time', 'negatives', 'labels']})
            assert candidates.setdefault(j['target'], candidate) == candidate
    atomic_json(out / 'verification.json', dict(
        status='passed', test_models=len(rows), prediction_ap=True,
        frozen_model_hashes=True, paired_candidates=True,
    ))


def run(out, spec):
    jobs = spec['jobs']
    import_completed(jobs, out)
    import_legacy_tfra(jobs, out)
    reused = [j['id'] for j in jobs if (out / 'jobs' / j['id'] / 'reuse.json').exists()]
    if not (out / 'test_freeze.json').exists():
        for index, j in enumerate(jobs):
            count = sum((out / 'jobs' / x['id'] / 'result.json').exists() for x in jobs)
            atomic_json(out / 'progress.json', dict(
                status='running', stage='training', completed=count, total=len(jobs),
                reused=len(reused), index=index+1, method=j['method'], target=j['target'],
                sources=j['sources'], job=j['id'], test_used=False,
            ))
            dest = out / 'jobs' / j['id']
            if (dest / 'result.json').exists() and 'legacy_reuse' in read(dest / 'result.json'):
                assert verify_legacy_reuse(dest)['job'] == j
            else:
                train_job(j, out)
            torch.cuda.empty_cache()
        atomic_json(out / 'progress.json', dict(status='running', stage='verification',
                                               completed=len(jobs), total=len(jobs), test_used=False))
        rows = verify_development(jobs, out)
        verify_training(rows, out)
        frozen = seal(rows, out, canonical_hash(spec))
    else:
        frozen = read(out / 'test_freeze.json')
        assert frozen['protocol_sha256'] == canonical_hash(spec)
    for index, row in enumerate(frozen['models']):
        atomic_json(out / 'progress.json', dict(
            status='running', stage='test', completed=index, total=len(jobs),
            training_completed=len(jobs), job=row['job']['id'], test_used=True,
        ))
        evaluate_frozen(row, out)
        torch.cuda.empty_cache()
    verify_test(frozen['models'], out)
    write_report(out, jobs)
    atomic_json(out / 'progress.json', dict(
        status='complete', training_completed=len(jobs), completed=len(jobs), total=len(jobs),
        reused=len(reused), test_used=True, verification='passed', claim_updated=False,
        report=str((out / 'RESULTS.md').relative_to(ROOT)),
    ))
    print(json.dumps(read(out / 'progress.json')), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    if not args.execute:
        jobs = planned_jobs()
        print(json.dumps(dict(total=len(jobs), methods=list(METHODS),
                              per_method=54, out=str(args.out)), indent=2))
        return
    out = args.out.resolve()
    if out == PARENT.resolve() or out == (HERE / 'outputs').resolve():
        raise ValueError('Use a separate subset output directory')
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'suite.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            spec = initialize(out)
            with (out / 'gpu_environment.json').open('w') as f:
                json.dump(dict(gpu=torch.cuda.get_device_name(0),
                               timing='Other GPU workloads may affect runtime.'), f, indent=2)
            while True:
                try:
                    with gpu_guard('cuda:0', 0):
                        torch.set_num_threads(2)
                        run(out, spec)
                    break
                except BlockingIOError:
                    atomic_json(out / 'progress.json', dict(status='waiting_for_gpu_lock',
                                                           total=len(spec['jobs']), test_used=False))
                    time.sleep(30)
        except Exception as exc:
            atomic_json(out / 'failure.json', dict(error=repr(exc), traceback=traceback.format_exc()))
            prior = read(out / 'progress.json') if (out / 'progress.json').exists() else {}
            atomic_json(out / 'progress.json', {**prior, 'status': 'failed', 'error': repr(exc)})
            raise


if __name__ == '__main__':
    main()
