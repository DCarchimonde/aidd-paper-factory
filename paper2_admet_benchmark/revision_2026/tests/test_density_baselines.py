import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from paper2_admet_benchmark.revision_2026.density_baselines import (
    DensityEstimationError, kde_density_ratio, kde_mondrian_component,
    kernel_mean_matching, kmm_mondrian_component, weighted_empirical_mondrian,
)


class DensityTests(unittest.TestCase):
    def test_kde_equal_domains_ratio_one_and_sample_size_normalization(self):
        source = np.array([[-1.], [0.], [1.]])
        target = np.repeat(source, 2, axis=0)
        sw, tw, diag = kde_density_ratio(source, target)
        np.testing.assert_allclose(sw, 1, atol=1e-12)
        np.testing.assert_allclose(tw, 1, atol=1e-12)
        self.assertEqual(diag["query_mass"], "estimated_query_weight_at_infinity")

    def test_kde_extreme_shift_is_clipped_without_overflow(self):
        source = np.array([[0.], [.1], [.2]])
        target = np.array([[100.], [101.], [102.]])
        sw, tw, diag = kde_density_ratio(source, target, bandwidth=.1)
        np.testing.assert_allclose(sw, .05)
        np.testing.assert_allclose(tw, 20.)
        self.assertEqual(diag["source_clip_fraction"], 1.)
        self.assertEqual(diag["target_clip_fraction"], 1.)

    def test_kmm_identity_domain_feasible_weights_and_improved_objective(self):
        x = np.array([[-2.], [-1.], [0.], [1.], [2.]])
        weights, diag = kernel_mean_matching(x, x)
        np.testing.assert_allclose(weights, 1., atol=1e-6)
        self.assertTrue(diag["solver_success"])
        self.assertLessEqual(diag["sum_residual"], 1e-6)
        z = x + .6
        shifted, diag2 = kernel_mean_matching(x, z, epsilon=.1)
        self.assertTrue(np.all(shifted >= 0))
        self.assertLessEqual(abs(shifted.mean()-1), .100001)
        self.assertLessEqual(diag2["objective_final"], diag2["objective_initial"]+1e-8)

    def test_kmm_size_cap_is_not_silent_subsampling(self):
        with self.assertRaisesRegex(DensityEstimationError, "no subsampling"):
            kernel_mean_matching(np.ones((4,1)), np.ones((4,1)), max_source=3)

    def test_kmm_rejects_failed_or_infeasible_solver(self):
        module = "paper2_admet_benchmark.revision_2026.density_baselines.minimize"
        for success, weight in ((False, np.ones(3)), (True, np.zeros(3))):
            fake = SimpleNamespace(success=success, x=weight, status=9,
                                   message="test failure", nit=1)
            with patch(module, return_value=fake):
                with self.assertRaises(DensityEstimationError):
                    kernel_mean_matching(np.arange(3.)[:,None], np.arange(3.)[:,None])

    def test_empirical_quantile_uses_candidate_class_and_no_query_mass(self):
        cal_p = np.array([.1,.5,.8,.9])
        y = np.array([0,0,1,1])
        w = np.array([1.,3.,1.,1.])
        actual = weighted_empirical_mondrian(cal_p, y, np.array([.4,.85]), w, alpha=.25)
        np.testing.assert_array_equal(actual, [[True,False], [False,True]])
        # With one example per class, empirical 0.9 quantile stays finite;
        # finite-sample split CP would return infinity instead.
        result = weighted_empirical_mondrian(np.array([.1,.9]), np.array([0,1]),
                                            np.array([.5]), np.ones(2), alpha=.1)
        np.testing.assert_array_equal(result, [[False,False]])

    def test_zero_weight_class_conservative_infinite_threshold(self):
        result = weighted_empirical_mondrian(np.array([.1,.9]), np.array([0,1]),
                                            np.array([.99]), np.array([0.,1.]))
        self.assertTrue(result[0,0])

    def test_components_execute_and_label_alignment_is_checked(self):
        x = np.array([[-1.],[-.5],[.5],[1.]])
        y = np.array([0,0,1,1])
        p = np.array([.1,.2,.8,.9])
        for method in (kde_mondrian_component, kmm_mondrian_component):
            sets, diag = method(x,y,x,p,p)
            self.assertEqual(sets.shape, (4,2))
            self.assertFalse(diag["finite_sample_shift_guarantee"])
            with self.assertRaises(ValueError):
                method(x,y[:-1],x,p,p)
        with self.assertRaises(ValueError):
            kernel_mean_matching(x,x,epsilon=1.)
        with self.assertRaises(ValueError):
            kde_density_ratio(x,x,bandwidth=0.)


if __name__ == "__main__":
    unittest.main()
