"""Regression checks for causal replay caches and deterministic query sharing."""
import numpy as np
import pytest
import torch
from tests.fixtures import events
from experiments.robust.backbones import OfficialPredictor
from experiments.robust.training import DEFAULT


def make(typed):
    torch.set_num_threads(2)
    d=events(typed=typed)
    # Include timestamp groups that cross nominal batch boundaries.
    d.time=np.floor(d.time/3)
    cfg={**DEFAULT,'device':'cpu','history_length':4,'batch_size':7,'tkg_batch_size':7}
    return d,OfficialPredictor({'fixture':d},'fixture',cfg,7,'tgn')


def toggle(m,on):
    m.encoder.optimized=on
    m.encoder.embedding_module.optimized=on
    m.encoder._prefix_cache=None


@pytest.mark.parametrize('typed',[False,True])
def test_replay_cache(typed):
    d,m=make(typed)
    cuts=[0.,4.,7.,8.,10.,7.,15.,20.]
    toggle(m,False)
    expected=[]
    for cutoff in cuts:
        m.encoder.replay(d,cutoff,7)
        expected.append(m.encoder.memory_bank.backup_memory_bank())
    toggle(m,True)
    for cutoff,ref in zip(cuts,expected):
        m.encoder.replay(d,cutoff,7)
        actual=m.encoder.memory_bank.backup_memory_bank()
        for x,y in zip(ref[:2],actual[:2]):torch.testing.assert_close(x,y,rtol=0,atol=0)
        for k in set(ref[2])|set(actual[2]):
            left,right=ref[2].get(k,[]),actual[2].get(k,[])
            assert len(left)==len(right)
            for (x,t),(y,u) in zip(left,right):
                assert t==u
                torch.testing.assert_close(x,y,rtol=0,atol=0)
    # Invalidate after parameter loading, even at an unchanged history cutoff.
    state={k:v.clone() for k,v in m.state_dict().items()}
    for k in state:
        if 'weight' in k:state[k].add_(.01)
    m.load_state_dict(state)
    m.encoder.replay(d,20.,7)
    actual=m.encoder.memory_bank.node_memories.clone()
    toggle(m,False);m.encoder.replay(d,20.,7)
    torch.testing.assert_close(actual,m.encoder.memory_bank.node_memories,rtol=0,atol=0)


@pytest.mark.parametrize('typed',[False,True])
@pytest.mark.parametrize('training',[False,True])
def test_scores_gradients_and_dropout_rng(typed,training):
    d,m=make(typed);ids=np.arange(24,32)
    neg=np.full((len(ids),2) if typed else len(ids),4,dtype=np.int64)
    results=[]
    for on in [False,True]:
        toggle(m,on);m.train(training);m.begin_batch(ids);m.zero_grad(set_to_none=True)
        torch.manual_seed(123)
        z=m.score(ids,neg);z.square().mean().backward()
        results.append((z.detach(),{k:p.grad.clone() for k,p in m.named_parameters() if p.grad is not None},torch.get_rng_state()))
    a,b=results
    torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)
    assert a[1].keys()==b[1].keys()
    for k in a[1]:torch.testing.assert_close(a[1][k],b[1][k],rtol=1e-5,atol=1e-7,msg=k)
    assert torch.equal(a[2],b[2])


def test_manifest_compatibility_is_narrow(tmp_path):
    import json
    from experiments.robust.preparation import compatible_manifest
    old={'adapter_sha256':'old','trainer_sha256':'train','config':{'epochs':5}}
    new={**old,'adapter_sha256':'new'}
    assert not compatible_manifest(old,new,tmp_path)
    (tmp_path/'tgn_optimization.json').write_text(json.dumps({'compatible_pretraining_hashes':{'adapter_sha256':{'before':'old','after':'new'}}}))
    assert compatible_manifest(old,new,tmp_path)
    assert not compatible_manifest(old,{**new,'config':{'epochs':6}},tmp_path)
    assert not compatible_manifest(old,{**new,'trainer_sha256':'other'},tmp_path)


def test_manifest_compatibility_chain(tmp_path):
    import json
    from experiments.robust.preparation import compatible_manifest
    for name,before,after in [('tgn_optimization.json','a','b'),('tgn_optimization_v2.json','b','c')]:
        (tmp_path/name).write_text(json.dumps({'compatible_pretraining_hashes':{'trainer_sha256':{'before':before,'after':after}}}))
    assert compatible_manifest({'trainer_sha256':'a','setting':1},{'trainer_sha256':'c','setting':1},tmp_path)
    assert not compatible_manifest({'trainer_sha256':'a','setting':1},{'trainer_sha256':'c','setting':2},tmp_path)


def test_padding_memory_is_constant_and_gradient_matches_double_reference():
    from experiments.robust.tgn_optimized import CachedGraphAttentionEmbedding
    d,m=make(False)
    enc=m.encoder.embedding_module
    enc.double()
    enc.node_raw_features=enc.node_raw_features.double()
    ids=np.array([0,1,1,2,0,1,3,2,0],dtype=np.int64)
    x=torch.randn(8,172,dtype=torch.float64,requires_grad=True)
    weights=torch.randn(len(ids),172,dtype=torch.float64)
    # Padding is constant in the TGN computation graph. Build an independent
    # float64 reference with that same boundary condition.
    ref=torch.cat([x[:1].detach(),x[1:]])[torch.from_numpy(ids)]+enc.node_raw_features[torch.from_numpy(ids)]
    fast=enc.get_node_features(x,ids)
    torch.testing.assert_close(ref,fast,rtol=0,atol=0)
    a=torch.autograd.grad((ref*weights).sum(),x)[0]
    b=torch.autograd.grad((fast*weights).sum(),x)[0]
    torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
