"""Numerical and exact small-population tests, not molecular experiments."""
import itertools
import math
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from scipy.stats import betabinom, hypergeom

from paper2_admet_benchmark.revision_2026.batch_triage.core import (
    CountCS, InconsistentConfidenceSet, OptimizationFailure,
    certify_triage, finite_population_cs, log_e_value, optimize_triage,
    update_count_interval,
)


class CountConfidenceSequenceTests(unittest.TestCase):
    def test_uniform_total_prior_predictive_identity(self):
        # Independent enumeration of a uniform prior over original totals.
        for N in (1, 7, 12):
            for n in range(N + 1):
                for s in range(n + 1):
                    mass = sum(hypergeom.pmf(s, N, K, n) for K in range(N + 1)) / (N + 1)
                    self.assertAlmostEqual(mass, 1 / (n + 1), places=13)

    def test_exact_enumeration_anytime_coverage(self):
        # Every possible randomly ordered fixed-population label sequence has
        # probability 1/C(N,K); checking all stopping times checks optional stop.
        N, delta = 8, .2
        for prior in ('uniform', 'edge_mixture'):
            for K in range(N + 1):
                failures = 0
                for positives in itertools.combinations(range(N), K):
                    positions = set(positives)
                    total, failed = 0, False
                    for n in range(1, N + 1):
                        total += int(n - 1 in positions)
                        lo, hi = finite_population_cs(N, n, total, delta, prior=prior)
                        failed |= not lo <= K <= hi
                    failures += failed
                self.assertLessEqual(failures / math.comb(N, K), delta + 1e-14)

    def test_edge_predictive_mass_matches_original_total_prior(self):
        N = 9
        for n in range(N + 1):
            for s in range(n + 1):
                mixture = sum((.4 * (K == 0) + .4 * (K == N) + .2 * betabinom.pmf(K, N, .5, .5))
                              * hypergeom.pmf(s, N, K, n) for K in range(N + 1))
                feasible_K = s
                from_e = math.exp(log_e_value(N, n, s, feasible_K, prior='edge_mixture')) * hypergeom.pmf(s, N, feasible_K, n)
                self.assertAlmostEqual(mixture, from_e, places=13)

    def test_prior_choice_propagates_to_state_and_cannot_be_unknown(self):
        state = CountCS(20, .05, prior='edge_mixture')
        state.observe_many([0, 0, 0])
        self.assertEqual(state.snapshot()['prior'], 'edge_mixture')
        self.assertEqual(state.upper_total, finite_population_cs(20, 3, 0, .05, prior='edge_mixture')[1])
        state.prior = 'uniform'
        with self.assertRaises(ValueError):
            state.observe(0)
        with self.assertRaises(ValueError):
            CountCS(20, .05, prior='posthoc_best')

    def test_boundary_null_is_supermartingale_not_martingale(self):
        # With K=0 every draw is zero, yet e_t=1/(t+1), not expectation one.
        self.assertAlmostEqual(math.exp(log_e_value(5, 0, 0, 0)), 1)
        self.assertAlmostEqual(math.exp(log_e_value(5, 1, 0, 0)), .5)
        self.assertTrue(math.isinf(log_e_value(5, 1, 1, 0)))

    def test_history_intersection_remaining_counts_and_census(self):
        state = CountCS(7, .05)
        lo, hi = 0, 7
        for y in (0, 1, 0, 0, 1, 1, 0):
            state.observe(y)
            lo, hi = update_count_interval(7, state.n, state.s, lo, hi, .05)
            self.assertEqual((state.lower_total, state.upper_total), (lo, hi))
            self.assertEqual(state.remaining_bounds(), (max(0, lo - state.s), min(7 - state.n, hi - state.s)))
        self.assertEqual((lo, hi), (3, 3))
        self.assertEqual(state.remaining_bounds(), (0, 0))
        with self.assertRaises(ValueError):
            state.observe(0)

    def test_invalid_batch_is_atomic(self):
        state = CountCS(5, .05)
        before = state.snapshot()
        with self.assertRaises(ValueError):
            state.observe_many([0, 1, 2])
        self.assertEqual(state.snapshot(), before)
        for values in ((5, 6, 0, .05), (5, 2, 3, .05), (5, 2, 1, 0), (5.5, 2, 1, .05)):
            with self.assertRaises(ValueError):
                finite_population_cs(*values)

    def test_empty_intersection_is_fail_closed(self):
        with self.assertRaises(InconsistentConfidenceSet):
            update_count_interval(5, 5, 3, 0, 2, .05)


class TriageCertificateTests(unittest.TestCase):
    def test_worst_case_bounds_match_all_integer_configurations(self):
        M, L, U, a, b = np.array([4, 3, 2]), np.array([0, 1, 0]), np.array([2, 3, 1]), 2, 3
        for actions in itertools.product(range(3), repeat=3):
            clear, discard = np.array(actions) == 0, np.array(actions) == 1
            got = certify_triage(M, L, U, a, b, clear, discard, .2, .2)
            fn, fp = [], []
            for positive in itertools.product(*[range(l, u + 1) for l, u in zip(L, U)]):
                positive = np.array(positive)
                fn.append(positive[clear].sum() / (a + positive.sum()))
                fp.append((M - positive)[discard].sum() / (b + (M - positive).sum()))
            self.assertEqual(got['fn_upper'], max(fn))
            self.assertEqual(got['fp_upper'], max(fp))

    def test_denominator_coupling_beats_separate_fraction_bound(self):
        # Independent numerator/denominator worst cases are incompatible.
        got = certify_triage([10, 90], [0, 40], [2, 50], 0, 0, [1, 0], [0, 0], .048, .1)
        self.assertAlmostEqual(got['fn_upper'], 2 / 42)
        self.assertTrue(got['certified'])
        self.assertGreater(2 / 40, .048)

    def test_exact_boundary_not_float_tolerance(self):
        got = certify_triage([1, 9], [1, 9], [1, 9], 0, 0, [1, 0], [0, 0], .1, .1)
        self.assertTrue(got['certified'])
        got = certify_triage([1, 9], [1, 9], [1, 9], 0, 0, [1, 0], [0, 0], .09999999999999999, .1)
        self.assertFalse(got['certified'])

    def test_absent_class_flags_and_zero_error_convention(self):
        got = certify_triage([4], [0], [0], 0, 2, [1], [0], .1, .1)
        self.assertTrue(got['class_1_absence_possible'])
        self.assertEqual(got['fn_upper'], 0)
        self.assertTrue(got['certified'])
        got = certify_triage([4], [0], [4], 0, 0, [1], [0], .1, .1)
        self.assertEqual(got['fn_upper'], 1)
        self.assertFalse(got['certified'])

    def test_invalid_overlap_counts_or_probabilities_rejected(self):
        with self.assertRaises(ValueError):
            certify_triage([4], [0], [4], 0, 0, [1], [1], .1, .1)
        with self.assertRaises(ValueError):
            certify_triage([4], [2], [1], 0, 0, [1], [0], .1, .1)
        with self.assertRaises(ValueError):
            certify_triage([4], [0], [4], 0, 0, [1], [0], np.nan, .1)


class TriageOptimizationTests(unittest.TestCase):
    def test_milp_matches_exhaustive_three_action_search(self):
        M, L, U, a, b = [5, 4, 3], [0, 3, 1], [1, 4, 2], 10, 10
        costs = np.array([1., 1.7, .8])
        best = 0.
        for actions in itertools.product(range(3), repeat=3):
            clear, discard = np.array(actions) == 0, np.array(actions) == 1
            certificate = certify_triage(M, L, U, a, b, clear, discard, .15, .15)
            if certificate['certified']:
                best = max(best, float((np.array(M) * costs)[clear | discard].sum()))
        got = optimize_triage(M, L, U, a, b, .15, .15, costs=costs)
        self.assertTrue(got['certificate']['certified'])
        self.assertAlmostEqual(got['saved_cost'], best)
        self.assertTrue(np.array_equal(got['actions'] == 0, got['clear_mask']))

    def test_no_audit_no_informative_interval_cannot_release(self):
        got = optimize_triage([30, 20], [0, 0], [30, 20], 0, 0, .1, .1)
        self.assertEqual(got['saved_items'], 0)
        self.assertTrue(np.all(got['actions'] == 2))

    def test_allowed_actions_and_empty_strata(self):
        got = optimize_triage([0, 5, 8], [0, 0, 8], [0, 0, 8], 2, 3, .1, .1,
                              eligible_clear=[1, 0, 0], eligible_discard=[1, 0, 1])
        self.assertEqual(got['saved_items'], 8)
        self.assertEqual(got['actions'].tolist(), [2, 2, 1])
        empty = optimize_triage([], [], [], 1, 2, .1, .1)
        self.assertEqual(empty['saved_items'], 0)

    def test_solver_failure_is_not_reported_as_optimum(self):
        with patch('paper2_admet_benchmark.revision_2026.batch_triage.core.milp',
                   return_value=SimpleNamespace(status=1, x=None, message='test time limit')):
            with self.assertRaises(OptimizationFailure):
                optimize_triage([5], [0], [0], 1, 1, .1, .1)


if __name__ == '__main__':
    unittest.main()
