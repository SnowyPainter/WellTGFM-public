"""Frozen-checkpoint evaluation must preserve weights and use the held-out split."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import json

from tests.fixtures import events
from tests.test_transfer_validation import job
from experiments.common.runtime import evaluate
from experiments.common.engine import save_predictions
from experiments.transfer_validation.core import model_for, digest
from experiments.target_adaptation.test_evaluation import Evaluator


@pytest.mark.parametrize('method', ['tfra', 'refine'])
def test_cached_models_restore_each_checkpoint_and_evaluate_test(tmp_path, method):
    torch.set_num_threads(1);d=events();d.metadata['artifacts']={'fixture':'fixed'}
    data={'fixture':d};j=job(method);cfg=j['config'];rows=[]
    for i in range(2):
        m=model_for(j,cfg,data)
        with torch.no_grad():
            for p in m.parameters():p.add_(i*.005)
        dest=tmp_path/str(i);dest.mkdir()
        torch.save(dict(model=m.state_dict(),**({'epoch':3} if i==0 else {})),dest/'best.pt')
        (dest/'result.json').write_text(json.dumps(dict(best_epoch=3)))
        metrics,scores=evaluate(m,d,1,d.negatives(100007),cfg)
        save_predictions(dest/'validation_predictions.npz',d,1,d.negatives(100007),scores)
        rows.append(dict(job={**j,'id':str(i)},config=cfg,best=str(dest/'best.pt'),
            best_sha256=digest(dest/'best.pt'),validation_ap=metrics['ap']))
    evaluator=Evaluator();out=tmp_path/'out'
    with patch('experiments.target_adaptation.test_evaluation.data_for',return_value=data):
        for row in rows+rows[:1]:
            evaluator.replay(row)
            result=evaluator.test(row,out)
            fresh=model_for(j,cfg,data)
            fresh.load_state_dict(torch.load(row['best'],weights_only=False)['model'])
            expected,scores=evaluate(fresh,d,2,d.negatives(200007),cfg)
            assert result['split']=='test' and result['selected_epoch']==3
            assert result['metrics']['random_1']['ap']==expected['ap']
            with np.load(out/'test'/row['job']['id']/'test_predictions.npz') as p:
                np.testing.assert_allclose(p['scores'].ravel(),scores,atol=1e-7,rtol=1e-6)
                np.testing.assert_array_equal(p['event_ids'],d.edge_id[d.indices(2)])
        assert len(evaluator.models)==1
        changed={**rows[0],'job':{**j,'id':'changed'},'best_sha256':'invalid'}
        with pytest.raises(AssertionError):evaluator.test(changed,out)


def test_complete_powerset_aggregation_and_manuscript_replacement(tmp_path):
    import itertools
    from experiments.target_adaptation import test_report as report
    from experiments.transfer_validation.report import summarize
    from plots import tfra_pareto_test
    targets=report.TARGETS;pool=targets+['ICEWS14','GDELT'];rows=[]
    for t in targets:
        available=[n for n in pool if n!=t]
        for n in range(6):
            for sources in itertools.combinations(available,n):
                for a in [7,17,37]:
                    for m in report.MAIN:
                        base=.6+.01*targets.index(t)+.0001*a
                        gain=(.01 if m=='tfra' else -.003 if n%2 else .003) if n else 0.
                        rows.append(dict(cohort='fixture',profile='original',backbone='tgat',target=t,
                            sources='+'.join(sorted(sources)),pretrain_seed=7 if n else None,adaptation_seed=a,
                            method=m,query_fraction=1.,source_budget=None,hidden_seed=None,final_seed=None,
                            trial=None,evaluation='random_1',ap=base+gain))
    tr,ss=summarize(rows);summary={(r['method'],r['target']):r for r in ss}
    pairs={m:report.paired(tr,m) for m in report.MAIN if m!='tfra'}
    json.dumps(pairs)
    assert pairs['finetune']['wins']==372 and pairs['finetune']['all_seed_ap_improved']==124
    edges,composition=report.additions(tr)
    assert len(edges)==3600 and composition['tfra']['zero']==900
    panel={}
    for i,m in enumerate(report.PANEL):
        r=dict(summary['tfra','target_equal_mean']);r.update(method=m,ap=.60+i*.01,D_G=.01/(i+1),B_G=.02+i*.001)
        r['mean_gain']=r['B_G']-r['D_G'];r['target_only_ap']=r['ap']-r['mean_gain'];panel[m]=r
    for m,r in panel.items():
        r['dominated_by']=[n for n in report.PANEL if report.dominates(panel[n],r)];r['pareto']=not r['dominated_by']
    assert panel['tfra']['pareto'] and panel['finetune']['dominated_by']
    (tmp_path/'docs').mkdir()
    (tmp_path/'docs/claim.md').write_bytes((report.ROOT/'docs/claim.md').read_bytes())
    counts={m:{k:.001 for k in range(1,6)} for m in report.MAIN}
    with patch.object(report,'ROOT',tmp_path),patch.object(tfra_pareto_test,'ROOT',tmp_path):
        report.manuscript(tmp_path,summary,panel,pairs,composition,counts)
        tfra_pareto_test.render(panel)
    assert (tmp_path/'docs/claim.md').read_bytes()==(report.ROOT/'docs/claim.md').read_bytes()
    text=(tmp_path/'claim_test_draft.md').read_text()
    assert (tmp_path/'claim_test_draft.diff').exists()
    assert '### 5.3. Transfer Performance on the Held-out Test Split' in text
    assert '**Table 8. 전체 Source 조합에서의 Test 성능' in text
    assert '### 5.2. Transfer Damage and Adaptation Dependence (Validation)' in text
    assert '91.9%' not in text
    assert (tmp_path/'plots/outputs/tfra_pareto_test.pdf').exists()
