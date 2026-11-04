import unittest

import numpy as np

from paper2_admet_benchmark.revision_2026.core import (
    candidate_scores, conformal_sets, decision_metrics, deferral_priority,
    fixed_budget_deferral, matched_budget_deferral, point_abstention,
    repair_empty, tame_variants,
)
from paper2_admet_benchmark.scripts.racer_c4.racer_c4_core import TransportAudit


class CoreTests(unittest.TestCase):
    def test_candidate_specific_mondrian_quantiles(self):
        # The predicted class of every query is 1. Class-0's threshold must
        # nevertheless govern whether candidate 0 enters its prediction set.
        cal_p = np.array([0.1] * 9 + [0.8] * 9)
        y = np.array([0] * 9 + [1] * 9)
        actual = conformal_sets(cal_p, y, np.array([0.85, 0.05]), alpha=0.2)
        np.testing.assert_array_equal(actual, [[False, True], [True, False]])

    def test_finite_sample_and_missing_calibration_class(self):
        result = conformal_sets(np.array([0.1]), np.array([0]), np.array([0.99]), alpha=0.1)
        np.testing.assert_array_equal(result, [[True, True]])

    def test_aps_candidates_and_tie_rule(self):
        actual = candidate_scores(np.array([0.2, 0.8, 0.5]), "aps")
        np.testing.assert_allclose(actual, [[0.8, 1.0], [1.0, 0.8], [0.5, 1.0]])
        # All variants execute with actual finite-sample quantiles.
        for method in ("mondrian_aps", "marginal_aps", "mondrian_lac", "marginal_lac"):
            self.assertEqual(conformal_sets(np.array([0.2, 0.8]), np.array([0, 1]),
                                            np.array([0.4]), method=method).shape, (1, 2))

    def test_invalid_probabilities_and_fractional_labels_fail(self):
        for bad in ([-0.1], [1.1], [np.nan], [np.inf]):
            with self.assertRaises(ValueError):
                candidate_scores(np.array(bad))
        for bad_label in (0.2, np.nan, -1, 2):
            with self.assertRaises(ValueError):
                conformal_sets(np.array([0.2]), np.array([bad_label]), np.array([0.5]))
        for alpha in (0, 1, np.nan, {0: .1}, {0: .1, 1: 1.1}):
            with self.assertRaises(ValueError):
                conformal_sets(np.array([0.2]), np.array([0]), np.array([0.5]), alpha=alpha)
        with self.assertRaises(ValueError):
            conformal_sets(np.array([0.2]), np.array([0]), np.array([0.5]),
                           alpha={0: .1, 1: .2}, method="marginal_lac")

    def test_class_matched_budget_and_row_order_invariance(self):
        base = np.array([[1,0], [1,0], [0,1], [0,1], [1,1], [0,0]], dtype=bool)
        ref = repair_empty(base)
        ref[[0, 3]] = True
        ids = np.array(["b", "a", "d", "c", "e", "f"])
        priority = np.ones(6)
        got = matched_budget_deferral(base, ref, priority, ids)
        # Ties break by ID, with one deferral from each predicted class.
        np.testing.assert_array_equal(got.sum(axis=1), [1,2,1,2,2,2])
        perm = np.array([5, 3, 1, 4, 0, 2])
        shuffled = matched_budget_deferral(base[perm], ref[perm], priority[perm], ids[perm])
        np.testing.assert_array_equal(shuffled, got[perm])
        self.assertFalse(np.any(repair_empty(base) & ~got))

    def test_bad_reference_cannot_remove_baseline_label(self):
        with self.assertRaises(ValueError):
            matched_budget_deferral(np.array([[1, 0]]), np.array([[0, 1]]),
                                    np.array([1.]), ["a"])
        for budget in (-1, 2, 0.5):
            with self.assertRaises(ValueError):
                fixed_budget_deferral(np.array([[1,0]]), budget, np.array([1.]), ["a"])

    def test_random_ad_and_point_rules(self):
        p = np.array([.1, .5, .9])
        ids = np.array(["z", "y", "x"])
        a = deferral_priority(p, ids, "random", seed=99)
        b = deferral_priority(p[::-1], ids[::-1], "random", seed=99)
        np.testing.assert_array_equal(a, b[::-1])
        np.testing.assert_allclose(deferral_priority(p, ids, "ad", similarity=p), 1-p)
        member = point_abstention(p, 1, deferral_priority(p, ids), ids)
        np.testing.assert_array_equal(member, [[True, False], [True, True], [False, True]])

    def test_metrics_denominators_and_empty_deferral(self):
        y = np.array([0,0,0,1,1,1])
        member = np.array([[1,0],[0,1],[1,1],[1,0],[0,1],[0,0]])
        result = decision_metrics(y, member)
        self.assertAlmostEqual(result["false_clearance"], 1/3)
        self.assertAlmostEqual(result["clearance_contamination"], 1/2)
        self.assertAlmostEqual(result["false_discard"], 1/3)
        self.assertAlmostEqual(result["clear_yield"], 1/3)
        self.assertAlmostEqual(result["minimum_class_coverage"], 1/3)
        self.assertAlmostEqual(result["macro_csy"], 1/3)
        self.assertAlmostEqual(result["deferred_rate"], 1/3)
        self.assertEqual(result["singleton_fn"], 1)

    def test_missing_class_does_not_look_like_good_minimum_coverage(self):
        result = decision_metrics(np.array([0,0]), np.array([[1,0],[1,1]]))
        self.assertTrue(np.isnan(result["minimum_class_coverage"]))
        self.assertTrue(np.isnan(result["macro_csy"]))
        self.assertTrue(np.isnan(result["false_clearance"]))
        empty = decision_metrics(np.array([]), np.empty((0,2), bool))
        self.assertTrue(np.isnan(empty["overall_coverage"]))

    def test_protected_label_variants_and_baseline_containment(self):
        base = np.array([[1,0], [0,1], [0,0]], dtype=bool)
        candidate = np.ones((3,2), dtype=bool)
        audit = lambda view: TransportAudit(view, True, "active", 100, 100, 50, 50,
                                            100, 50, 50, .6, 0, 0, 1, .5)
        variants = tame_variants(base, [candidate, candidate], [audit("a"), audit("b")])
        np.testing.assert_array_equal(variants["tame_label0_historical"],
                                      [[True,False],[True,True],[True,True]])
        np.testing.assert_array_equal(variants["tame_label1"],
                                      [[True,True],[False,True],[True,True]])
        self.assertTrue(variants["tame_both"].all())


if __name__ == "__main__":
    unittest.main()
