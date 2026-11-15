import numpy as np
import torch
from tests.fixtures import events
from experiments.transfer.analyze_source_pool import negatives,evaluate_source
from experiments.transfer.cross_representation import build


def test_typed_evaluation_negatives_and_no_parameter_updates():
    torch.set_num_threads(2)
    data={'target':events('target'),'kg':events('kg',typed=True)}
    d=data['kg'];neg=negatives(d,7)
    np.testing.assert_array_equal(neg,negatives(d,7))
    vocab=set(np.r_[d.src[d.indices(0)],d.dst[d.indices(0)]])
    for i in np.r_[d.indices(0),d.indices(1)]:
        same=(d.time==d.time[i])&(d.relation==d.relation[i])
        assert neg[i,0] in vocab and neg[i,1] in vocab
        assert neg[i,0] not in d.dst[same&(d.src==d.src[i])]
        assert neg[i,1] not in d.src[same&(d.dst==d.dst[i])]
    cfg=dict(target='target',device='cpu',history_length=2,batch_size=8,tkg_batch_size=4,microbatch=2)
    model=build(cfg,data,7)
    before={k:p.detach().clone() for k,p in model.named_parameters()}
    metrics,scores=evaluate_source(model,d,neg,cfg)
    assert scores.shape==(2*len(d.indices(1)),2)
    assert 0<=metrics['ap']<=1
    for k,p in model.named_parameters():
        assert p.grad is None
        torch.testing.assert_close(before[k],p,rtol=0,atol=0)
