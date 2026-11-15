"""Verify the matched initialization and isolation of the bounded experiment."""
from unittest.mock import patch

import numpy as np
import pytest
import torch

from tests.fixtures import events
from experiments.transfer_validation import core
from experiments.transfer_validation.readout_mechanism import METHODS, planned_jobs


def fixture_job(method, diagnostics=False):
    return dict(
        id=method, cohort='fixture', target='fixture', sources=[], pretrain_seed=None,
        adaptation_seed=7, method=method, backbone='tgat', profile='fixture',
        checkpoint=None, diagnostics=diagnostics,
        config={**core.DEFAULT, 'device': 'cpu', 'history_length': 4,
                'batch_size': 12, 'microbatch': 12, 'epochs': 2},
    )


def test_exact_three_arm_subset_with_paired_controls():
    jobs = planned_jobs()
    assert len(jobs) == 162 and {j['method'] for j in jobs} == set(METHODS)
    for method in METHODS:
        arm = [j for j in jobs if j['method'] == method]
        assert sum(bool(j['sources']) for j in arm) == 42
        assert sum(not j['sources'] for j in arm) == 12
        controls = {(j['target'], j['adaptation_seed']) for j in arm if not j['sources']}
        assert all((j['target'], j['adaptation_seed']) in controls for j in arm)


def test_initial_readout_is_matched_and_fixed_arm_restores_training_head(tmp_path):
    torch.set_num_threads(1)
    d = events()
    data = {'fixture': d}
    initial_heads, results = {}, {}
    actual_fit = core.fit
    for method in METHODS:
        job = fixture_job(method)
        calls = []

        def tracked_fit(model, *args, **kwargs):
            result = actual_fit(model, *args, **kwargs)
            calls.append({k: v.detach().clone() for k, v in model.heads['fixture'][2].state_dict().items()})
            return result

        with patch.object(core, 'data_for', return_value=data), patch.object(core, 'fit', side_effect=tracked_fit):
            results[method] = core.train_job(job, tmp_path)
        initial_heads[method] = calls[0]
        assert len(calls) == (4 if method == 'tfra' else 3)
        saved = torch.load(tmp_path / 'jobs' / method / 'resume.pt', map_location='cpu', weights_only=False)
        initial_model = core.model_for(job, job['config'], data)
        for k, value in initial_model.heads['fixture'][0].state_dict().items():
            torch.testing.assert_close(saved['model']['heads.fixture.0.' + k], value, atol=0, rtol=0)
        assert any(not torch.equal(saved['model']['encoders.fixture.' + k], v)
                   for k, v in initial_model.encoders['fixture'].state_dict().items())
        if method == 'initial_fixed':
            for k, v in calls[0].items():
                torch.testing.assert_close(saved['model']['heads.fixture.2.' + k], v, atol=0, rtol=0)
        if method == 'joint_init':
            assert any(not torch.equal(saved['model']['heads.fixture.2.' + k], v)
                       for k, v in calls[0].items())
    for method in METHODS:
        for key in initial_heads['tfra']:
            torch.testing.assert_close(initial_heads[method][key], initial_heads['tfra'][key], atol=0, rtol=0)
    # Before the next epoch's refit, the two fixed-readout arms are identical.
    assert results['initial_fixed']['history'][0]['validation'] == results['tfra']['history'][0]['validation']


@pytest.mark.parametrize('method', ['tfra', 'joint_init'])
def test_gradient_diagnostics_preserve_training_and_predictions(tmp_path, method):
    torch.set_num_threads(1)
    d = events()
    with patch.object(core, 'data_for', return_value={'fixture': d}):
        plain = core.train_job(fixture_job(method), tmp_path / 'plain')
        measured = core.train_job(fixture_job(method, diagnostics=True), tmp_path / 'measured')
    assert plain['ap'] == measured['ap'] and plain['best_epoch'] == measured['best_epoch']
    assert measured['diagnostics'] and not plain['diagnostics']
    with np.load(tmp_path / 'plain/jobs' / method / 'validation_predictions.npz') as a, \
         np.load(tmp_path / 'measured/jobs' / method / 'validation_predictions.npz') as b:
        np.testing.assert_array_equal(a['scores'], b['scores'])
