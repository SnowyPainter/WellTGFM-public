"""Verified, read-only reuse of selected models and compatible epoch states."""
from functools import lru_cache
import shutil
import torch
from sklearn.metrics import average_precision_score
import numpy as np

from experiments.robust.training import ROOT, DEFAULT, read, digest
from experiments.common.runtime import atomic_json, atomic_torch

FREEZES = ['experiments/target_adaptation/outputs/test_v1/test_freeze.json',
           'experiments/transfer_validation/outputs/readout_mechanism_v2/test_freeze.json']


def key(j):
    return (j['target'],tuple(sorted(j['sources'])),j['pretrain_seed'],j['adaptation_seed'],j['method'])


@lru_cache(None)
def index():
    rows = {}
    for freeze in FREEZES:
        for r in read(ROOT/freeze)['models']:
            r = dict(r, freeze=freeze)
            rows.setdefault(key(r['job']), []).append(r)
    return rows


def candidate(j, continuation=False):
    if j['backbone'] != 'tgat': return None
    for k,v in DEFAULT.items():
        if k=='epochs': continue
        if j['config'][k] != v: return None
    if not continuation and j['config']['epochs'] != 5: return None
    candidates = index().get(key(j), [])
    for r in reversed(candidates):
        cfg = r['config']
        for k,v in DEFAULT.items():
            if cfg.get(k,v) != v: raise ValueError(f'Unexpected legacy setting {k}: {cfg}')
        best = ROOT/r['best']
        assert digest(best) == r['best_sha256']
        if continuation and not (best.parent/'resume.pt').exists(): continue
        manifest=read(best.parent/'manifest.json')
        if j['method'] in {'tfra','joint','batch','initial_fixed','joint_init'}:
            if 'ridge_lambda' in manifest:
                assert manifest['ridge_lambda']==j['config']['ridge_lambda']
                assert manifest['support_queries']==j['config']['support_queries']
                assert manifest['dropout']=='disabled during support,query,and readout fitting'
            else:
                assert manifest['config']['ridge_lambda']==j['config']['ridge_lambda']
                assert manifest['config']['support_queries']==j['config']['support_queries']
        old_ck = r['job'].get('checkpoint')
        assert bool(old_ck) == bool(j['checkpoint'])
        if old_ck: assert digest(ROOT/old_ck) == digest(ROOT/j['checkpoint'])
        with np.load(best.parent/'validation_predictions.npz') as a:
            ap = float(average_precision_score(a['labels'].ravel(),a['scores'].ravel()))
        assert abs(ap-r['validation_ap']) < 1e-10
        return r
    return None


def import_selected(j, out):
    dest = out/'jobs'/j['id']
    if (dest/'result.json').exists(): return True
    r = candidate(j)
    if r is None: return False
    src = (ROOT/r['best']).parent
    old = read(src/'result.json')
    assert len(old['history']) == 5
    assert abs(max(h['validation']['ap'] for h in old['history'])-r['validation_ap']) < 1e-10
    dest.mkdir(parents=True,exist_ok=True)
    for name in ['best.pt','validation_predictions.npz']:
        shutil.copy2(src/name,dest/name)
    provenance = dict(source=str(src.relative_to(ROOT)),freeze=r['freeze'],
        artifacts={name:digest(src/name) for name in ['manifest.json','best.pt','result.json','validation_predictions.npz']})
    atomic_json(dest/'reuse.json',provenance)
    atomic_json(dest/'result.json',dict(status='complete',job=j,ap=r['validation_ap'],
        best_epoch=old['best_epoch'],history=old['history'],seconds=old['seconds'],
        test_used=False,reused=True,provenance=provenance))
    return True


def continue_from_five(j, out):
    dest = out/'jobs'/j['id']
    if (dest/'resume.pt').exists() or j['config']['epochs'] <= 5: return False
    r = candidate(j,continuation=True)
    if r is None: return False
    src = (ROOT/r['best']).parent
    state = torch.load(src/'resume.pt',map_location='cpu',weights_only=False)
    assert state['epoch'] == 5 and len(state['history']) == 5
    assert state['best'] == r['validation_ap']
    # This bridge changes only metadata. Model/Adam/RNG and training head are kept.
    state.setdefault('diagnostics',[])
    state.setdefault('initial_validation',None)
    state.setdefault('updates',60)
    state.setdefault('fits',{'tfra':10,'batch':65,'joint':5,'initial_fixed':6,'joint_init':6}.get(j['method'],0))
    dest.mkdir(parents=True,exist_ok=True)
    for source, target in [('best.pt','best.pt'),('best.pt','budget5.pt'),
                           ('validation_predictions.npz','validation_predictions.npz'),
                           ('validation_predictions.npz','budget5_validation.npz')]:
        shutil.copy2(src/source,dest/target)
    atomic_torch(dest/'resume.pt',state)
    atomic_json(dest/'continuation.json',dict(source=str(src.relative_to(ROOT)),
        original_resume_sha256=digest(src/'resume.pt'),original_best_sha256=r['best_sha256'],
        epoch=5,optimizer_and_rng_preserved=True))
    return True
