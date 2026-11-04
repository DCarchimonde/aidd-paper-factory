import unittest
import numpy as np
import pandas as pd
from paper2_admet_benchmark.revision_2026.summarize import paired_cells,bootstrap_comparison

class SummaryTests(unittest.TestCase):
    def fixture(self):
        rows=[]
        for e in ['a','b']:
            for seed in [1,2]:
                for method in ['x','ref']:
                    for i in range(20):
                        rows.append(dict(phase='retrospective_epa',endpoint=e,seed=seed,
                          predictor='stack',method=method,alpha=.1,sample_id=str(i),structure_id=str(i),
                          y=i%2,include_0=int(i%2==0),include_1=int(i%2==1),eligible=1))
        return pd.DataFrame(rows)
    def test_identical_methods_zero_paired_uncertainty(self):
        cells,structures,seeds=paired_cells(self.fixture(),'x','ref','stack','stack',.1,['a','b'],'retrospective_epa')
        r=bootstrap_comparison(cells,len(structures),100,3)
        for x in r:
            self.assertEqual(x['delta'],0)
            self.assertEqual(x['ci_95_low'],0)
            self.assertEqual(x['ci_95_high'],0)
    def test_seed_repetition_is_not_extra_evidence(self):
        f=self.fixture()
        f.loc[(f.method=='x') & f.sample_id.isin(['0','2','4']), 'include_1']=1
        c,s,_=paired_cells(f,'x','ref','stack','stack',.1,['a','b'],'retrospective_epa')
        g=f[f.seed==1]
        d,t,_=paired_cells(g,'x','ref','stack','stack',.1,['a','b'],'retrospective_epa')
        self.assertEqual(bootstrap_comparison(c,len(s),100,3),bootstrap_comparison(d,len(t),100,3))
    def test_repeated_endpoint_shared_molecules_preserves_interval(self):
        f=self.fixture()
        f.loc[(f.method=='x') & f.sample_id.isin(['0','2','4']), 'include_1']=1
        c,s,_=paired_cells(f,'x','ref','stack','stack',.1,['a','b'],'retrospective_epa')
        d,t,_=paired_cells(f[f.endpoint=='a'],'x','ref','stack','stack',.1,['a'],'retrospective_epa')
        self.assertEqual(bootstrap_comparison(c,len(s),100,3),bootstrap_comparison(d,len(t),100,3))
    def test_identity_mismatch_fails(self):
        f=self.fixture(); f.loc[0,'sample_id']='wrong'
        with self.assertRaises(ValueError): paired_cells(f,'x','ref','stack','stack',.1,['a','b'],'retrospective_epa')
    def test_label_mismatch_fails(self):
        f=self.fixture(); f.loc[0,'y']=1
        with self.assertRaises(ValueError): paired_cells(f,'x','ref','stack','stack',.1,['a','b'],'retrospective_epa')
    def test_missing_class_fails(self):
        f=self.fixture(); f['y']=0
        with self.assertRaises(ValueError): paired_cells(f,'x','ref','stack','stack',.1,['a','b'],'retrospective_epa')
if __name__=='__main__': unittest.main()
