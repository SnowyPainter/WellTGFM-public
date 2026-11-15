"""Integration checks for adapters, paired baselines, reuse, and continuation."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import numpy as np
import torch

from experiments.robust.design import OUT, plan, job, case
from experiments.robust.training import DEFAULT, data_for, build, model_for, train_job, read
from experiments.robust.backbones import LocalTGN, MLPMixer
from experiments.robust.reuse import candidate, import_selected, continue_from_five
from experiments.robust.evaluation import evaluate_job
from experiments.robust.preparation import ensure_pretraining
from experiments.common.runtime import gpu_guard,atomic_json,evaluate,restore_rng
from experiments.transfer.cross_representation import typed_negatives
from experiments.transfer_validation.core import build as old_build


def checks(out):
    torch.set_num_threads(2)
    p=plan(out)
    assert len(p['jobs'])==1848
    assert len(p['panel'])==18 and sum(bool(c['sources']) for c in p['panel'])==14
    assert len({j['id'] for j in p['jobs']})==len(p['jobs'])
    done=[]
    # TGAT is unchanged at initialization and transfer loading.
    j=job(case('lastfm',['mooc']), 'tfra', 7, out=out)
    data=data_for(j)
    a=build(j,DEFAULT,data,7);b=old_build(j,DEFAULT,data,7)
    assert a.state_dict().keys()==b.state_dict().keys()
    for k,v in a.state_dict().items(): torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
    del a,b
    done.append('TGAT state equals original implementation')
    for backbone in ['tgn','graphmixer']:
        torch.set_float32_matmul_precision('highest')
        c=case('lastfm',backbone=backbone,source_budget=60)
        model=build(c,DEFAULT,data,7);model.eval()
        for name in ['lastfm','ICEWS14','GDELT']:
            d=data[name];ids=d.indices(0)[1000:1008]
            model.bind(d);model.begin_batch(ids)
            neg=typed_negatives(d,7) if d.metadata['family']=='tkg' else d.negatives(7)
            scores=model.score(ids,neg[ids]);assert torch.isfinite(scores).all()
            scores.square().mean().backward()
            if d.metadata['family']=='tkg': assert model.encoder.relations.weight.grad.abs().sum()>0
            sample=model.samplers[name]
            ns,es,ts=sample.get_historical_neighbors(d.src[ids],d.time[ids],32)
            assert np.all(ts[es>0]<d.time[ids[0]])
            if backbone=='tgn':
                assert model.encoder.memory_updater.memory_updater.weight_ih.grad.abs().sum()>0
                memory, _ = model.encoder.get_updated_memories(np.arange(model.encoder.num_nodes), model.encoder.memory_bank.node_raw_messages)
                assert memory.abs().sum()>0
                model.begin_batch(ids)
                torch.testing.assert_close(scores, model.score(ids,neg[ids]),rtol=0,atol=0)
            if d.metadata['family']=='ctdg':
                with torch.no_grad():
                    parts=torch.cat([model.score(ids[i:i+1],neg[ids[i:i+1]]) for i in range(len(ids))])
                torch.testing.assert_close(scores,parts,rtol=1e-4,atol=1e-5)
        # Adapters use local model implementations.
        if backbone=='graphmixer': assert isinstance(model.encoders['lastfm'].network, MLPMixer)
        else: assert isinstance(model.encoders['lastfm'],LocalTGN)
        del model
        torch.set_float32_matmul_precision('high')
        done.append(backbone+': CTDG/TKG gradients, strict history cutoff, chunk-invariant predictions')
        # All requested backbone arms do a real epoch, including LP-FT and REFINE.
        for method in ['finetune','lp_ft','refine','tfra','batch']:
            j=job(c,method,7,dict(epochs=1),out=out)
            r=train_job(j,out)
            assert r['updates']==12 and len(r['history'])==1
            evaluate_job(j,out)
            done.append(backbone+': '+method+' training/selection/Test replay')
            atomic_json(out/'checks_progress.json',dict(passed=done))
            torch.cuda.empty_cache()
    # Reuse each family member and extend its actual epoch-5 training state by one epoch.
    for method in ['finetune','lp_ft','refine','tfra','joint','batch','initial_fixed','joint_init']:
        c=case('lastfm',['mooc'])
        j=job(c,method,7,out=out)
        assert import_selected(j,out)
        evaluate_job(j,out)
        j=job(c,method,7,dict(epochs=6,save_budgets=[5,6]),out=out)
        continued=continue_from_five(j,out)
        r=train_job(j,out)
        assert len(r['history'])==6 and r['updates']==72
        evaluate_job(j,out)
        done.append(method+': reuse/replay and continuation' if continued else method+': replay and full retraining fallback')
        atomic_json(out/'checks_progress.json',dict(passed=done))
        torch.cuda.empty_cache()
    atomic_json(out/'checks.json',dict(status='passed',checks=done))


def pretraining_checks(out):
    torch.set_num_threads(2)
    completed=[]
    for backbone in ['tgn','graphmixer']:
        # A real mixed CTDG/TKG composition from the existing LastFM panel.
        c=case('lastfm',['reddit','mooc','ICEWS14'],backbone=backbone,source_budget=60)
        j=job(c,'tfra',7,dict(epochs=1),out=out)
        ensure_pretraining(j,out)
        p=Path(j['checkpoint'])
        assert p.exists()
        source=read(p.parent/'result.json')
        assert len(source['history'])==60
        assert {x['source'] for x in source['history']}==set(c['sources'])
        train_job(j,out)
        evaluate_job(j,out)
        completed.append(backbone)
        torch.cuda.empty_cache()
    atomic_json(out/'pretraining_checks.json',dict(status='passed',backbones=completed,
        updates_per_backbone=60,ctdg_tkg_mixed=True,shared_checkpoint_transfer_replayed=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT.parent/'checks_v2_tgn')
    parser.add_argument('--pretraining-only',action='store_true')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    with gpu_guard('cuda:0',0):
        (pretraining_checks if args.pretraining_only else checks)(args.out.resolve())
