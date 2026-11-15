import ast
from pathlib import Path
import importlib
import json
import numpy as np


def test_legacy_batch_execution_preserved():
    root=Path(__file__).resolve().parents[1]
    old=(root/'archive/2026-11-05-adaptation-search/experiments/adaptation_screening/run_profiled_support.py').read_text()
    expected=old[old.index('def execute('):old.index('\ndef main():')]
    expected=expected.replace('panel.ROOT','ROOT').replace('panel.digest','digest').replace("Path(__file__).with_name('profiled_readout.py')","Path(__file__).with_name('readout.py')")
    new=(root/'experiments/target_adaptation/batch_practical_training.py').read_text()
    actual=next(n for n in ast.parse(new).body if isinstance(n,ast.FunctionDef) and n.name=='execute')
    assert ast.dump(actual)==ast.dump(ast.parse(expected).body[0])


def test_common_control_and_own_control_are_distinct(tmp_path,monkeypatch):
    mod=importlib.import_module('experiments.target_adaptation.batch_practical')
    monkeypatch.setattr(mod,'OUT',tmp_path)
    monkeypatch.setattr(mod,'OLD',tmp_path/'old')
    rows=[]
    for m,base,transfer in [('finetune',.7,.71),('tfra',.6,.75),('batch',.8,.82)]:
        for sources,ap in [([],base),(['s'],transfer)]:
            for seed in [7,17,37]:
                ident=f'{m}_{len(sources)}_{seed}'
                j=dict(id=ident,method=m,target='t',sources=sources,adaptation_seed=seed)
                rows.append(dict(job=j))
                dest=tmp_path/'test'/ident;dest.mkdir(parents=True)
                (dest/'result.json').write_text(json.dumps(dict(metrics={'random_1':{'ap':ap}})))
    mod.report(rows)
    frame=mod.pd.read_csv(tmp_path/'summary.csv')
    values=frame[frame.target=='t'].set_index('method')
    assert values.loc['tfra','B_G']>values.loc['batch','B_G']
    assert values.loc['batch','common_B']>values.loc['tfra','common_B']
    assert np.isclose(values.loc['batch','common_B'],.12)
    assert np.isclose(values.loc['tfra','common_B'],.05)
    assert 'Matched transfer executions so far: 3 / 372' in (tmp_path/'RESULTS.md').read_text()
