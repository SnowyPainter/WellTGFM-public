import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from tests.fixtures import events
from experiments.common.backbones import SharedLinkPredictor
from experiments.common.runtime import seed_all
from experiments.transfer.cross_representation import typed_negatives,build,pretrain,adapt
from models.ours.joint import shared_key

class CrossRepresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def setUp(self):
        self.data={n:events(n,typed=(n=='kg')) for n in ('lastfm','mooc','kg')}
        self.cfg=dict(target='lastfm',history_length=2,device='cpu',batch_size=16,tkg_batch_size=8,microbatch=8,learning_rate=1e-4,weight_decay=0.,grad_clip=1.,pretrain_updates=2,epochs=1)
    def test_ctdg_is_original_tgat(self):
        d=self.data['lastfm'];seed_all(7);old=SharedLinkPredictor(d,'tgat',2,'cpu',7)
        new=build(self.cfg,self.data,7);old.eval();new.eval()
        for k,v in old.encoder.state_dict().items():self.assertTrue(torch.equal(v,new.encoders['lastfm'].state_dict()[k]))
        for k,v in old.head.state_dict().items():self.assertTrue(torch.equal(v,new.heads['lastfm'].state_dict()[k]))
        ids=d.indices(1)[:4];neg=d.negatives(7)[ids]
        old.begin_batch(ids);new.begin_batch(ids)
        torch.testing.assert_close(old.score(ids,neg),new.score(ids,neg),rtol=0,atol=0)
    def test_tkg_updates_shared_encoder_and_private_interface(self):
        m=build(self.cfg,self.data,7);a=dict(m.encoders['lastfm'].named_parameters());b=dict(m.encoders['kg'].named_parameters())
        for k,v in a.items():self.assertIs(v,b[k])
        self.assertNotIn('relations.weight',a)
        d=self.data['kg'];ids=d.indices(0)[10:14];neg=typed_negatives(d,7)
        m.bind(d);m.begin_batch(ids);z=m.score(ids,neg[ids]);self.assertEqual(tuple(z.shape),(8,2))
        y=torch.zeros_like(z);y[:,0]=1
        torch.nn.functional.binary_cross_entropy_with_logits(z,y).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in a.values()))
        self.assertGreater(float(m.encoders['kg'].relations.weight.grad.abs().sum()),0)
        self.assertGreater(float(m.encoders['kg'].directions.weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in m.heads['lastfm'].parameters()))
        for i in d.indices(0):
            same=(d.relation==d.relation[i])&(d.time==d.time[i])
            self.assertNotIn(neg[i,0],d.dst[same&(d.src==d.src[i])])
            self.assertNotIn(neg[i,1],d.src[same&(d.dst==d.dst[i])])
    def test_training_transfer_and_no_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'pretrain';p.mkdir();t=root/'target';t.mkdir()
            state=pretrain(self.cfg,self.data,['mooc','kg'],7,p)
            self.assertTrue(all(shared_key(k) for k in state))
            adapt(self.cfg,self.data,state,7,t)
            r=json.loads((t/'result.json').read_text());self.assertEqual(r['split'],'validation');self.assertTrue(0<=r['metrics']['ap']<=1)
            self.assertFalse(list(root.rglob('*test*')))
            with np.load(t/'validation_predictions.npz') as f:
                np.testing.assert_array_equal(f['event_ids'],self.data['lastfm'].edge_id[self.data['lastfm'].indices(1)])

if __name__=='__main__':unittest.main()
