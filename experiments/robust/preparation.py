"""Source-only pretraining for the two local backbone adapters."""
from experiments.robust.provenance import local_implementations
import time
import shutil
from pathlib import Path
import torch
from experiments.robust.training import DEFAULT, build, data_for, read, digest
from experiments.transfer_validation.design import source_checkpoint
from experiments.transfer.cross_representation import typed_negatives
from experiments.common.engine import train_batch
from experiments.common.runtime import atomic_json, atomic_torch, capture_rng, restore_rng


def compatible_manifest(existing, expected, out):
    if existing == expected:
        return True
    normalized = dict(existing)
    for filename in ['tgn_optimization.json', 'tgn_optimization_v2.json']:
        revision = out / filename
        if not revision.exists():
            continue
        for field, pair in read(revision).get('compatible_pretraining_hashes', {}).items():
            if normalized.get(field) == pair['before']:
                normalized[field] = pair['after']
    return normalized == expected


def ensure_pretraining(c,out):
    c={k:c[k] for k in ['target','sources','pretrain_seed','profile','backbone','source_budget']}
    p=source_checkpoint(c,out)
    if p is None: return
    if c['backbone']=='tgat':
        assert p.exists()
        return
    assert c['backbone'] in {'graphmixer','tgn'} and c['source_budget']==60
    dest=p.parent;dest.mkdir(parents=True,exist_ok=True)
    cfg=dict(DEFAULT);data=data_for(c);seed=c['pretrain_seed'];sources=c['sources']
    manifest=dict(case={k:c[k] for k in ['target','sources','pretrain_seed','profile','backbone','source_budget']},
                  config=cfg,data={n:d.metadata['artifacts'] for n,d in data.items()},test_used=False,
                  adapter_sha256=digest(Path(__file__).with_name('backbones.py')),
                  trainer_sha256=digest(__file__),implementation=local_implementations())
    # The integration check trains two real 60-update source models. Reuse those
    # unchanged source-only artifacts when their complete manifests match.
    checked=Path(__file__).parent/'outputs/checks_v2_tgn/pretraining'/p.parent.name
    if checked.resolve()!=dest.resolve() and not p.exists() and (checked/'result.json').exists():
        if compatible_manifest(read(checked/'manifest.json'),manifest,out):
            for name in ['manifest.json','result.json','final.pt','resume.pt']:
                shutil.copy2(checked/name,dest/name)
            atomic_json(dest/'reuse.json',dict(source=str(checked),final_sha256=digest(checked/'final.pt')))
    if (dest/'manifest.json').exists(): assert compatible_manifest(read(dest/'manifest.json'),manifest,out)
    if p.exists():
        assert read(dest/'result.json')['final_sha256']==digest(p)
        return
    atomic_json(dest/'manifest.json',manifest)
    model=build(c,cfg,data,seed)
    opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    batches={n:list(data[n].batches(data[n].indices(0),cfg['tkg_batch_size'] if data[n].metadata['family']=='tkg' else cfg['batch_size'])) for n in sources}
    history=[];first=0;negatives={};start=time.monotonic()
    if (dest/'resume.pt').exists():
        s=torch.load(dest/'resume.pt',map_location=cfg['device'],weights_only=False)
        model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);restore_rng(s['rng'],cfg['device'])
        first=s['step'];history=s['history']
    for step in range(first,60):
        n=sources[step%len(sources)];local=step//len(sources);cycle=local//len(batches[n]);d=data[n]
        if (n,cycle) not in negatives:
            negatives[n,cycle]=typed_negatives(d,seed+cycle) if d.metadata['family']=='tkg' else d.negatives(seed+cycle)
        model.bind(d);ids=batches[n][local%len(batches[n])]
        loss=train_batch(model,d,ids,negatives[n,cycle],cfg,opt,1.)
        history.append(dict(step=step+1,source=n,events=len(ids),loss=loss))
        if (step+1)%12==0:
            atomic_torch(dest/'resume.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step+1,
                history=history,rng=capture_rng(cfg['device'])))
            event=dict(stage='pretraining',target=c['target'],sources=sources,backbone=c['backbone'],step=step+1,total=60,updated_at=time.time())
            atomic_json(out/'heartbeat.json',event)
            print(event,flush=True)
    atomic_torch(p,dict(shared=model.shared_state(),case=manifest['case']))
    atomic_json(dest/'result.json',dict(status='complete',history=history,seconds=time.monotonic()-start,
        final_sha256=digest(p),test_used=False))
