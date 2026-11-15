"""Freeze each completed Validation selection before evaluating Test."""
import torch
import numpy as np
from sklearn.metrics import average_precision_score
from experiments.robust.training import ROOT, read, digest, data_for, model_for
from experiments.common.runtime import atomic_json, evaluate
from experiments.common.engine import save_predictions


def selected_epoch(saved, result, row, dest):
    """Resolve legacy model-only checkpoints without modifying frozen artifacts."""
    history = result['history'][:row['budget']]
    assert len(history) == row['budget']
    assert [h['epoch'] for h in history] == list(range(1, row['budget'] + 1))
    # Training saves on strict improvement, so the first maximum wins ties.
    selected = max(history, key=lambda h: h['validation']['ap'])
    epoch = selected['epoch']
    assert abs(selected['validation']['ap'] - row['validation_ap']) < 1e-10
    if 'epoch' in saved:
        assert saved['epoch'] == epoch, (saved['epoch'], epoch)
        return epoch

    # Only verified reuse/continuation may omit checkpoint epoch metadata.
    if (dest/'reuse.json').exists():
        provenance = read(dest/'reuse.json')
        source = ROOT/provenance['source']
        assert provenance == result['provenance']
        for name, sha in provenance['artifacts'].items():
            assert digest(source/name) == sha, name
        assert row['sha256'] == provenance['artifacts']['best.pt']
    else:
        provenance = read(dest/'continuation.json')
        source = ROOT/provenance['source']
        assert row['sha256'] == provenance['original_best_sha256']
        assert digest(source/'best.pt') == row['sha256']
        assert digest(source/'resume.pt') == provenance['original_resume_sha256']
    original = read(source/'result.json')
    assert original['best_epoch'] == epoch
    assert original['history'] == result['history'][:len(original['history'])]
    return epoch


def evaluate_job(j,out):
    dest=out/'jobs'/j['id'];r=read(dest/'result.json')
    assert r['status']=='complete' and r['job']==j and r['test_used'] is False
    rows=[]
    for budget in j['config'].get('save_budgets',[j['config']['epochs']]):
        best=dest/(f'budget{budget}.pt' if 'save_budgets' in j['config'] else 'best.pt')
        preds=dest/(f'budget{budget}_validation.npz' if 'save_budgets' in j['config'] else 'validation_predictions.npz')
        val=max(h['validation']['ap'] for h in r['history'][:budget])
        with np.load(preds) as a:
            assert abs(average_precision_score(a['labels'].ravel(),a['scores'].ravel())-val)<1e-10
        rows.append(dict(budget=budget,path=str(best),sha256=digest(best),validation_ap=val))
    freeze=dict(job=j,selections=rows,test_negative_seed=200007,validation_negative_seed=100007)
    if (dest/'selection.json').exists(): assert read(dest/'selection.json')==freeze
    else: atomic_json(dest/'selection.json',freeze)
    pending=[row for row in rows if not (dest/f"test_budget{row['budget']}.json").exists()]
    if not pending: return
    data=data_for(j);d=data[j['target']];model=model_for(j,j['config'],data)
    neg=d.negatives(200007)
    for row in pending:
        saved=torch.load(row['path'],map_location=j['config']['device'],weights_only=False)
        epoch=selected_epoch(saved,r,row,dest)
        assert 1<=epoch<=row['budget']
        model.load_state_dict(saved['model']);model.bind(d)
        # Exact Validation replay tests adapter/state compatibility for reused models too.
        val,_=evaluate(model,d,1,d.negatives(100007),j['config'])
        assert abs(val['ap']-row['validation_ap'])<1e-8,(j['id'],val['ap'],row['validation_ap'])
        metrics,scores=evaluate(model,d,2,neg,j['config'])
        save_predictions(dest/f"test_budget{row['budget']}.npz",d,2,neg,scores)
        atomic_json(dest/f"test_budget{row['budget']}.json",dict(status='complete',job_id=j['id'],
            budget=row['budget'],selected_epoch=epoch,validation_ap=row['validation_ap'],
            best_sha256=row['sha256'],metrics=metrics,split='test'))
        print(dict(stage='test',job=j['id'],budget=row['budget'],ap=metrics['ap']),flush=True)
