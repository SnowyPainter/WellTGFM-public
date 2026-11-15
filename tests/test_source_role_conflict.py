import unittest
from copy import deepcopy
import numpy as np
from scipy.optimize import minimize_scalar
from experiments.common.source_patterns import aggregate,features,compare,DATASETS,VIEWS,nll
from tests.fixtures import events


class ConflictTests(unittest.TestCase):
    def test_mle_is_weighted_by_cell_frequency(self):
        tables={}
        for i,name in enumerate(DATASETS):
            cell=[100,90] if i==0 else [900,90]
            rest=[900,410] if i==0 else [100,10]
            tables[name]={s:{v:{'cell':cell.copy(),'other':rest.copy()} for v in VIEWS} for s in (0,1)}
        row=next(r for r in compare(tables,7) if r['source_a']=='wikipedia' and r['source_b']=='reddit' and r['view']=='recurrence' and r['cell']=='cell')
        self.assertAlmostEqual(row['pooled_train_p'],.18)
        optimum=minimize_scalar(lambda p:.1*nll(p,90,100)+.9*nll(p,90,900),bounds=(.001,.999),method='bounded')
        self.assertAlmostEqual(optimum.x,row['pooled_train_p'],places=5)
        self.assertTrue(row['val_strong_opposition'])
        tables['reddit'][1]['recurrence']['cell']=[900,810]
        row=next(r for r in compare(tables,7) if r['source_a']=='wikipedia' and r['source_b']=='reddit' and r['view']=='recurrence' and r['cell']=='cell')
        self.assertTrue(row['train_discovered']);self.assertFalse(row['val_strong_opposition'])

    def test_history_and_candidate_balance(self):
        d=events();negative=np.full(len(d.src),7);a=features(d,negative,16)
        self.assertEqual(sum(r[1] for r in a),len(a)//2)
        self.assertTrue(all(r[2][0]=='0' for r in a[:32]))
        changed=events();changed.dst[35:]=6;b=features(changed,negative,16)
        self.assertEqual(a[:64],b[:64])
        table=aggregate(a)
        self.assertEqual(sum(v[0] for v in table[0]['recurrence'].values()),2*len(d.indices(0)))

if __name__=='__main__':unittest.main()
