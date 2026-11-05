"""Numerical and semantic checks for the full KMM-CP procedures."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from paper2_admet_benchmark.revision_2026.full_baselines.kmm_cp import (
    KMMError, METHOD_NAMES, _mmd_for_assignments, calibrate_sets, evaluate_selection,
    fit_joint_selection, fit_predict, fit_standard_kmm, select_bandwidth,
)


class KMMCPTests(unittest.TestCase):
    def test_mmd_statistic_matches_direct_pair_sums(self):
        rng=np.random.default_rng(82)
        x=rng.normal(size=(7,3))
        K=np.exp(-np.sum((x[:,None]-x[None,:])**2,axis=2)/2)
        assignments=np.array([[1,1,1,0,0,0,0],[0,1,0,1,0,1,0]])
        got=_mmd_for_assignments(K,assignments,3)
        expected=[]
        for a in assignments:
            I,J=np.flatnonzero(a),np.flatnonzero(1-a)
            ss=sum(K[i,j] for i in I for j in I if i!=j)/(len(I)*(len(I)-1))
            tt=sum(K[i,j] for i in J for j in J if i!=j)/(len(J)*(len(J)-1))
            st=sum(K[i,j] for i in I for j in J)/(len(I)*len(J))
            expected.append(ss+tt-2*st)
        np.testing.assert_allclose(got,expected,atol=1e-12)

    def test_permutation_bandwidth_selection_is_deterministic_and_auditable(self):
        rng=np.random.default_rng(42)
        x,z=rng.normal(size=(15,3)),rng.normal(2,1,(12,3))
        a,da=select_bandwidth(x,z,seed=12,config={"permutations":12})
        b,db=select_bandwidth(x,z,seed=12,config={"permutations":12})
        self.assertEqual(a,b)
        self.assertEqual(da,db)
        scores=[r["standardized_mmd"] for r in da["records"] if r["usable_null_variance"]]
        self.assertAlmostEqual(da["records"][da["selected_index"]]["standardized_mmd"],max(scores))
        self.assertEqual(len(da["records"]),5)

    def test_constant_feature_bandwidth_is_explicit_not_nan(self):
        bw,d=select_bandwidth(np.zeros((3,2)),np.zeros((4,2)),config={"permutations":4})
        self.assertEqual(bw,1.)
        self.assertEqual(d["median_policy"],"all_identical_fixed_bandwidth")
        self.assertIn("degenerate",d["reason"])

    def test_standard_qp_identity_domain_recovers_uniform_solution(self):
        x=np.array([[-2.],[-1.],[0.],[1.],[2.]])
        w,d=fit_standard_kmm(x,x,bandwidth=.7)
        np.testing.assert_allclose(w,1.,atol=2e-5)
        self.assertAlmostEqual(w.mean(),1.,places=5)
        self.assertEqual(d["objective_ridge"],0.)
        self.assertLess(d["constraint_residual"],5e-6)

    def test_joint_qp_matches_mass_and_filters_unsupported_targets(self):
        rng=np.random.default_rng(5)
        x=rng.normal(0,.15,(15,2))
        z=np.r_[rng.normal(0,.15,(10,2)),rng.normal(8,.15,(10,2))]
        w,a,d=fit_joint_selection(x,z,bandwidth=.7)
        self.assertGreaterEqual(a.mean(),.5-5e-6)
        self.assertAlmostEqual(w.mean(),a.mean(),places=5)
        self.assertTrue(np.all((a>=0)&(a<=1)))
        self.assertGreater(a[:10].mean(),a[10:].mean()+.5)
        self.assertEqual(d["status_value"],1)

    def test_literal_paper_threshold_and_candidate_threshold_remain_distinct(self):
        cp=np.array([.1,.1,.2,.2]);y=np.array([0,0,1,1]);tp=np.array([.6]);w=np.ones(4)
        candidate,_=calibrate_sets(cp,y,tp,w,.1,"mondrian_candidate")
        paper,_=calibrate_sets(cp,y,tp,w,.1,"mondrian_paper_predicted")
        np.testing.assert_array_equal(candidate,[[False,True]])
        np.testing.assert_array_equal(paper,[[True,True]])

    def test_empirical_quantile_does_not_silently_add_query_atom(self):
        result,_=calibrate_sets(np.array([.1,.9]),np.array([0,1]),np.array([.5]),
                                np.ones(2),.1,"global")
        np.testing.assert_array_equal(result,[[False,False]])

    def test_missing_calibration_class_has_explicit_infinite_threshold(self):
        member,q=calibrate_sets(np.array([.1,.2]),np.array([0,0]),np.array([.9]),
                                np.ones(2),.1,"mondrian_candidate")
        self.assertTrue(np.isinf(q["1"]))
        self.assertTrue(member[0,1])

    def test_full_fit_refits_on_selected_cohort_and_keeps_population_denominators(self):
        rng=np.random.default_rng(2)
        x=rng.normal(0,1,(20,2));z=np.r_[rng.normal(0,1,(10,2)),rng.normal(6,1,(10,2))]
        cp=1/(1+np.exp(-x[:,0]));tp=1/(1+np.exp(-z[:,0]));y=(x[:,0]>0).astype(int)
        result=fit_predict(x,y,cp,z,tp,alphas=(.1,.2),config={"permutations":10})
        self.assertEqual(set(result["predictions"][.1]),set(METHOD_NAMES))
        self.assertEqual(result["diagnostics"]["selected_refit"]["target_n"],result["selected"].sum())
        for method in METHOD_NAMES:
            if method.startswith("skmm_"):
                self.assertTrue(result["predictions"][.1][method][~result["selected"]].all())
            else:
                self.assertTrue(result["method_selected"][method].all())
        target_y=(z[:,0]>0).astype(int)
        metrics=evaluate_selection(target_y,result["raw_predictions"][.1]["skmm_cp_global"],result["selected"])
        self.assertEqual(metrics["all_population"]["n"],len(z))
        self.assertEqual(metrics["selected_only"]["n"],result["selected"].sum())

    def test_outcome_changes_do_not_change_density_or_selection_fit(self):
        rng=np.random.default_rng(8)
        x=rng.normal(size=(12,2));z=rng.normal(.4,1,(10,2));p=np.linspace(.1,.9,12);q=np.linspace(.1,.9,10)
        y=np.arange(12)%2
        a=fit_predict(x,y,p,z,q,config={"permutations":4})
        b=fit_predict(x,1-y,p,z,q,config={"permutations":4})
        np.testing.assert_array_equal(a["selection_weights"],b["selection_weights"])
        np.testing.assert_array_equal(a["source_weights"]["kmm"],b["source_weights"]["kmm"])

    def test_solver_failure_and_caps_are_not_silent_fallbacks(self):
        x=np.arange(3.)[:,None]
        fake=SimpleNamespace(x=np.ones(3),info=SimpleNamespace(status_val=7,status="maximum iterations reached",iter=1))
        with patch("osqp.OSQP.solve",return_value=fake):
            with self.assertRaisesRegex(KMMError,"failed"):
                fit_standard_kmm(x,x,1.)
        with self.assertRaisesRegex(KMMError,"no subsampling"):
            select_bandwidth(x,x,config={"max_total_rows":4,"permutations":4})


if __name__ == "__main__":
    unittest.main()
