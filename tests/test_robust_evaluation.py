"""Legacy checkpoint selection must preserve Validation selection and provenance."""
import json

import pytest

from experiments.robust.evaluation import selected_epoch
from experiments.robust.training import digest


def history(values):
    return [dict(epoch=i, validation=dict(ap=ap)) for i, ap in enumerate(values, 1)]


@pytest.fixture
def legacy(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    dest = tmp_path/'job'
    dest.mkdir()
    h = history([.6, .8, .7, .7, .7])
    (source/'best.pt').write_bytes(b'frozen model-only checkpoint')
    (source/'resume.pt').write_bytes(b'frozen optimizer and RNG')
    (source/'result.json').write_text(json.dumps(dict(best_epoch=2, history=h)))
    sha = digest(source/'best.pt')
    return source, dest, dict(history=h), dict(budget=5, validation_ap=.8, sha256=sha)


@pytest.mark.parametrize('mode', ['reuse', 'continuation'])
def test_legacy_selection_preserves_artifacts_and_uses_original_epoch(legacy, mode):
    source, dest, result, row = legacy
    before = {p.name: digest(p) for p in source.iterdir()}
    if mode == 'reuse':
        provenance = dict(source=str(source), artifacts=before)
        result['provenance'] = provenance
    else:
        provenance = dict(source=str(source), original_best_sha256=before['best.pt'],
                          original_resume_sha256=before['resume.pt'])
        # A longer run may still select the original epoch-2 checkpoint.
        result['history'] += history([.7] * 15)
        for i, h in enumerate(result['history'], 1):
            h['epoch'] = i
        row['budget'] = 20
    (dest/f'{mode}.json').write_text(json.dumps(provenance))
    assert selected_epoch({}, result, row, dest) == 2
    assert {p.name: digest(p) for p in source.iterdir()} == before
    row['sha256'] = 'wrong checkpoint hash'
    with pytest.raises(AssertionError):
        selected_epoch({}, result, row, dest)


def test_native_selection_uses_budget_and_first_maximum(tmp_path):
    result = dict(history=history([.6, .8, .8, .9]))
    row = dict(budget=3, validation_ap=.8)
    assert selected_epoch(dict(epoch=2), result, row, tmp_path) == 2
    for wrong_epoch in [0, 3, 4]:
        with pytest.raises(AssertionError):
            selected_epoch(dict(epoch=wrong_epoch), result, row, tmp_path)


def test_model_only_checkpoint_without_provenance_is_rejected(legacy):
    _, dest, result, row = legacy
    with pytest.raises(FileNotFoundError):
        selected_epoch({}, result, row, dest)


def test_changed_source_result_is_rejected(legacy):
    source, dest, result, row = legacy
    provenance = dict(source=str(source), artifacts={
        p.name: digest(p) for p in source.iterdir()})
    result['provenance'] = provenance
    (dest/'reuse.json').write_text(json.dumps(provenance))
    (source/'result.json').write_text(json.dumps(dict(best_epoch=3, history=result['history'])))
    with pytest.raises(AssertionError):
        selected_epoch({}, result, row, dest)
