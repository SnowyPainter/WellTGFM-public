"""Check the aggregation contract independently using known synthetic gains."""
from pathlib import Path
import tempfile
import unittest
from experiments.common.runtime import atomic_json
from experiments.robust.report import summarize


class ReportingTest(unittest.TestCase):
    def test_equal_target_weight_and_missing_scratch(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);jobs=[];views=[]
            # Unequal numbers of source configurations must not change Target weights.
            for t,base,gains in [('a',.9,[-.2]),('b',.4,[.1,.1,.1])]:
                for seed in [7,17,37]:
                    for i,gain in enumerate([None]+gains):
                        ident=f'{t}_{seed}_{i}'
                        jobs.append(dict(id=ident,config={'epochs':5},backbone='tgat',method='tfra',
                            target=t,sources=[] if i==0 else [str(i)],pretrain_seed=None if i==0 else 7,adaptation_seed=seed))
                        views.append(dict(job_id=ident))
                        atomic_json(out/'jobs'/ident/'test_budget5.json',dict(selected_epoch=1,
                            validation_ap=.5,metrics={'ap':base+(gain or 0)}))
            p=dict(jobs=jobs,stages={'test':views})
            r=next(r for r in summarize(p,out) if r['pretrain_seed']=='all')
            self.assertAlmostEqual(r['AP'],.6)
            self.assertAlmostEqual(r['D_G'],.1)
            self.assertAlmostEqual(r['B_G'],.05)
            (out/'jobs/a_7_0/test_budget5.json').unlink()
            self.assertEqual(summarize(p,out),[])


if __name__=='__main__': unittest.main()
