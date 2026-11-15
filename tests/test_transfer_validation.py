"""Protocol, intervention, and optimization checks for the prospective suite."""
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from tests.fixtures import events
from experiments.transfer_validation.core import (DEFAULT, build, model_for, features, fit,
    train_queries, selected_batches, gradient_snapshot, train_job)
from experiments.transfer_validation.evaluation import negative_matrix


def job(method='tfra', backbone='tgat'):
    return dict(id='fixture',cohort='fixture',target='fixture',sources=[],pretrain_seed=None,adaptation_seed=7,
                method=method,backbone=backbone,profile='fixture',checkpoint=None,
                config={**DEFAULT,'device':'cpu','history_length':4,'batch_size':12,'microbatch':12,'epochs':1})


def test_nested_query_budget_and_native_cutoff():
    d=events();a=train_queries(d,.1);b=train_queries(d,.25)
    assert set(a)<=set(b)<=set(d.indices(0))
    cfg={**DEFAULT,'query_fraction':.25,'batch_size':12}
    for native,chosen in selected_batches(d,cfg):
        assert np.isin(chosen,native).all()
        assert d.time[native[0]]<=d.time[chosen[0]]


@pytest.mark.parametrize('backbone',['tgat','dygformer'])
def test_readout_fit_and_gradient_snapshot_restore(backbone):
    torch.set_num_threads(1);d=events();j=job(backbone=backbone);cfg=j['config']
    m=build(j,cfg,{'fixture':d},7);m.requires_grad_(False);m.encoders['fixture'].requires_grad_(True)
    neg=d.negatives(7);fit(m,d,neg,cfg)
    before={k:v.clone() for k,v in m.state_dict().items()}
    result=gradient_snapshot(m,d,neg,cfg)
    assert abs(result['epsilon'])<1e-6
    assert result['gradient_gap']<1e-5
    for k,v in m.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)


def test_head_replacement_does_not_change_encoder_or_other_rng():
    from experiments.common.runtime import capture_rng
    d=events();data={'fixture':d};base=job('finetune');cfg=base['config']
    a=model_for(base,cfg,data);ra=capture_rng('cpu')
    b=model_for({**base,'hidden_seed':17,'final_seed':37},cfg,data);rb=capture_rng('cpu')
    for x,y in zip(a.encoder.parameters(),b.encoder.parameters()):torch.testing.assert_close(x,y,atol=0,rtol=0)
    assert not torch.equal(a.heads['fixture'][0].weight,b.heads['fixture'][0].weight)
    assert not torch.equal(a.heads['fixture'][2].weight,b.heads['fixture'][2].weight)
    assert torch.equal(ra['torch'],rb['torch'])
    assert ra['python']==rb['python']
    np.testing.assert_array_equal(ra['numpy'][1],rb['numpy'][1])


@pytest.mark.parametrize('method',['finetune','lp_ft','refine','tfra','joint','joint_init','full_step','batch','frozen','initial_fixed'])
def test_training_never_uses_test_and_saved_model_replays(method):
    from experiments.common.runtime import evaluate as original_evaluate
    torch.set_num_threads(1);d=events();j=job(method);data={'fixture':d};seen=[]
    def checked(*args,**kwargs):
        seen.append(args[2]);assert args[2]==1
        return original_evaluate(*args,**kwargs)
    with tempfile.TemporaryDirectory() as tmp, patch('experiments.transfer_validation.core.data_for',return_value=data), patch('experiments.transfer_validation.core.evaluate',side_effect=checked):
        out=Path(tmp);r=train_job(j,out)
        assert r['status']=='complete' and r['test_used'] is False and seen==[1]
        saved=torch.load(out/'jobs/fixture/best.pt',map_location='cpu',weights_only=False)
        m=model_for(j,j['config'],data);m.load_state_dict(saved['model']);m.bind(d)
        metrics,scores=original_evaluate(m,d,1,d.negatives(100007),j['config'])
        assert abs(metrics['ap']-r['ap'])<1e-10
        with np.load(out/'jobs/fixture/validation_predictions.npz') as p:
            np.testing.assert_allclose(scores,p['scores'].ravel(),rtol=1e-6,atol=1e-7)
        assert train_job(j,out)==r
        (out/'test_freeze.json').write_text('{}')
        with pytest.raises(RuntimeError):train_job({**j,'id':'new'},out)


def test_historical_negatives_strictly_past_and_filtered():
    d=events();matrix,audit=negative_matrix(d,2,2,1,True,12)
    vocab=set(d.dst[d.indices(0)])
    for native in d.batches(d.indices(2),12):
        cutoff=d.time[native[0]]
        for i in native:
            assert d.dst[i] not in matrix[i]
            assert set(matrix[i])<=vocab
    assert audit['queries']==len(d.indices(2))


@pytest.mark.parametrize('method',['lp_ft','tfra','joint'])
def test_epoch_zero_can_be_selected_without_changing_training_length(method):
    torch.set_num_threads(1);d=events();j=job(method)
    j['config']['include_epoch0']=True
    with tempfile.TemporaryDirectory() as tmp, patch('experiments.transfer_validation.core.data_for',return_value={'fixture':d}):
        r=train_job(j,Path(tmp))
        assert len(r['history'])==1 and r['initial_validation'] is not None
        assert r['ap']==max(r['initial_validation']['ap'],r['history'][0]['validation']['ap'])
        assert r['best_epoch'] in {0,1}


def test_dygformer_shared_source_updates_and_checkpoint_loading():
    from experiments.common.engine import train_batch
    torch.set_num_threads(1);d=events();s=events('source',offset=100.)
    j=job(backbone='dygformer');cfg=j['config'];data={'fixture':d,'source':s}
    m=build(j,cfg,data,7)
    assert all(a is b for a,b in zip(m.encoders['fixture'].parameters(),m.encoders['source'].parameters()))
    before={k:v.clone() for k,v in m.shared_state().items()};m.bind(s)
    opt=torch.optim.Adam(m.parameters(),lr=1e-3)
    ids=s.indices(0)[:12];train_batch(m,s,ids,s.negatives(7),cfg,opt,1.)
    assert any(not torch.equal(before[k],v) for k,v in m.shared_state().items())
    target=build(j,cfg,data,17);target.load_shared_state(m.shared_state())
    for k,v in target.shared_state().items():torch.testing.assert_close(v,m.shared_state()[k],atol=0,rtol=0)


def test_sealed_evaluation_and_analysis_end_to_end():
    from experiments.transfer_validation import evaluation as ev
    from experiments.transfer_validation.report import report
    torch.set_num_threads(1);d=events();d.metadata['artifacts']={'fixture':'deterministic'}
    data={'fixture':d}
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);out=root/'suite';jobs=[]
        for method in ['finetune','tfra']:
            j=job(method);j['cohort']='seed_factorial';j['profile']='original';j['source_budget']=None
            j['id']=method;j['config']['include_epoch0']=True;jobs.append(j)
        with patch('experiments.transfer_validation.core.data_for',return_value=data):
            for j in jobs:train_job(j,out)
        with patch.object(ev,'ROOT',root),patch.object(ev,'data_for',return_value=data):
            rows=ev.verify_development(jobs,out)
            frozen=ev.seal(rows,out,'test_protocol')
            for row in frozen['models']:ev.evaluate_frozen(row,out,rich=True)
            assert len(frozen['common_scratch_references'])==1
            assert all((out/'test'/j['id']/'result.json').exists() for j in jobs)
        report(out,final=True)
        assert (out/'analysis/test_runs.csv').exists()
