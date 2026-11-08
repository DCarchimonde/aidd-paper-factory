"""Tests of audit allocation isolation and equally certified policy comparisons."""
import itertools
import copy
import unittest
from unittest.mock import patch

import numpy as np

from paper2_admet_benchmark.revision_2026.batch_triage.core import CountCS
from paper2_admet_benchmark.revision_2026.batch_triage.policies import (
    _lookahead_values, choose_stratum, make_strata, optimize_policy,
)


def box_state(remaining, lower, upper, audited_positive=0, audited_negative=0):
    """Synthetic admissible interval for optimizer tests, not a simulated CS."""
    state = CountCS(remaining + audited_positive + audited_negative, .01)
    state.n = audited_positive + audited_negative
    state.s = audited_positive
    state.lower_total = audited_positive + lower
    state.upper_total = audited_positive + upper
    return state


class StrataTests(unittest.TestCase):
    def test_label_free_row_invariance_and_tied_scores(self):
        ids = np.asarray([f"id{i:02}" for i in range(32)])
        score = np.repeat([.1, .4, .6, .9], 8)
        support = np.tile([.1, .9], 16)
        permutation = np.random.default_rng(23).permutation(32)
        for mode in ("score_support", "score_only"):
            groups = make_strata(score, support, ids, mode=mode)
            permuted = make_strata(score[permutation], support[permutation], ids[permutation], mode=mode)
            np.testing.assert_array_equal(permuted, groups[permutation])
            self.assertEqual(set(groups), set(range(8)))
        tied = make_strata([.5] * 8, None, list("hgfedcba"), mode="score_only")
        np.testing.assert_array_equal(tied, np.arange(7, -1, -1))

    def test_empty_cross_cells_are_removed_and_bad_features_rejected(self):
        groups = make_strata([.1, .2, .8, .9], [.1, .2, .8, .9], list("abcd"))
        np.testing.assert_array_equal(groups, np.arange(4))
        self.assertEqual(len(make_strata([], [], [])), 0)
        with self.assertRaises(ValueError):
            make_strata([.1, .2], [.1, .2], ["same", "same"])
        for score, support in (([np.nan], [.2]), ([1.1], [.2]), ([.2], [np.inf])):
            with self.assertRaises(ValueError):
                make_strata(score, support, ["a"])


class OptimizerPoliciesTests(unittest.TestCase):
    def test_nonmonotone_safe_strata_separate_joint_from_threshold_family(self):
        states = [box_state(5, k, k, 50, 50) for k in (0, 5, 0)]
        joint = optimize_policy(states, .01, .01, mode="joint")
        nested = optimize_policy(states, .01, .01, mode="thresholds", order=[0, 1, 2])
        self.assertEqual(joint["saved_items"], 15)
        self.assertEqual(nested["saved_items"], 5)
        np.testing.assert_array_equal(joint["actions"], [0, 1, 0])
        self.assertTrue(nested["certificate"]["certified"])
        with self.assertRaises(ValueError):
            optimize_policy(states, .1, .1, mode="thresholds", order=[0, 0, 1])

    def test_joint_denominator_beats_loose_without_relaxing_actual_risk(self):
        # True remaining positives K lie in [0,2]. Clearing all 10 has
        # worst FN 2/(18+2)=.1, but a loose ratio 2/18>.1 must reject it.
        states = [box_state(10, 0, 2, audited_positive=18)]
        exact = optimize_policy(states, .1, .1, mode="joint")
        loose = optimize_policy(states, .1, .1, mode="loose")
        self.assertEqual(exact["saved_items"], 10)
        self.assertEqual(loose["saved_items"], 0)
        self.assertAlmostEqual(exact["certificate"]["fn_upper"], .1)

    def test_all_optimizers_respect_every_admissible_integer_count_assignment(self):
        states = [box_state(3, 0, 1, 4, 5), box_state(4, 2, 4, 6, 7),
                  box_state(2, 0, 2, 4, 3)]
        for mode in ("joint", "thresholds", "loose"):
            result = optimize_policy(states, .2, .2, mode=mode, order=[0, 2, 1])
            for positive in itertools.product(range(2), range(2, 5), range(3)):
                positive = np.asarray(positive)
                remaining = np.asarray([3, 4, 2])
                fn = positive[result["clear_mask"]].sum() / (14 + positive.sum())
                fp = (remaining - positive)[result["discard_mask"]].sum() / (15 + (remaining - positive).sum())
                self.assertLessEqual(fn, .2)
                self.assertLessEqual(fp, .2)
            self.assertEqual(result["saved_items"] + result["certificate"]["deferred_items"], 9)
            self.assertLessEqual(result["saved_items"], 9)  # Audited 29 are not savings.


class AllocationTests(unittest.TestCase):
    def test_uniform_uses_remaining_size_entropy_avoids_certain_priors(self):
        states = [CountCS(3, .02), CountCS(9, .02)]
        rng = np.random.default_rng(42)
        choices = [choose_stratum(states, [.0, .5], rng, policy="uniform") for _ in range(4000)]
        self.assertLess(abs(np.mean(np.asarray(choices) == 0) - .25), .03)
        for _ in range(15):
            self.assertEqual(choose_stratum(states, [.0, .5], rng, policy="entropy"), 1)

    def test_lookahead_does_not_mutate_observed_states_and_respects_depletion(self):
        states = [CountCS(8, .02), CountCS(7, .02), CountCS(0, .01)]
        states[0].observe_many([0, 1, 0])
        states[1].observe_many([1, 1])
        before = [s.snapshot() for s in states]
        for optimizer in ("joint", "thresholds", "loose"):
            for policy in ("uniform", "entropy", "neyman", "lookahead"):
                a = choose_stratum(states, [.2, .8, .5], np.random.default_rng(9), policy=policy,
                                   optimizer_mode=optimizer, order=[0, 2, 1], block_size=4)
                b = choose_stratum(states, [.2, .8, .5], np.random.default_rng(9), policy=policy,
                                   optimizer_mode=optimizer, order=[0, 2, 1], block_size=4)
                self.assertIn(a, (0, 1))
                self.assertEqual(a, b)
        self.assertEqual([s.snapshot() for s in states], before)

    def test_forecast_savings_pay_for_new_audits_once(self):
        # With known all-negative remaining items, auditing two of them cannot
        # create a saving: they already could be released and now cost assays.
        state = box_state(3, 0, 0, audited_negative=9)
        current, forecast = _lookahead_values([state], np.array([0.]), "joint", None, 2, .1, .1)
        self.assertEqual(current, 3)
        self.assertEqual(forecast[0], 1)

    def test_one_step_forecast_uses_the_same_frozen_prior_as_actual_observation(self):
        # At block size one there is no unobserved intermediate order: forecast
        # bounds must exactly match an actual CountCS.observe under EACH prior.
        for prior in ("uniform", "edge_mixture"):
            state = CountCS(100, .025, prior=prior)
            state.observe_many([0] * 12)
            seen = {}

            def capture(candidate_states, *args, **kwargs):
                candidate = candidate_states[0]
                seen[(candidate.n, candidate.s)] = candidate.remaining_bounds()
                self.assertEqual(candidate.prior, prior)
                return {"saved_cost": 0.0}

            with patch("paper2_admet_benchmark.revision_2026.batch_triage.policies.optimize_policy",
                       side_effect=capture):
                _lookahead_values([state], np.array([.5]), "joint", None, 1, .1, .1)
            for label in (0, 1):
                actual = copy.copy(state)
                actual.observe(label)
                self.assertEqual(seen[(actual.n, actual.s)], actual.remaining_bounds())

    def test_no_positive_gain_falls_back_to_widest_count_interval(self):
        states = [CountCS(2, .01), CountCS(15, .01)]
        # With essentially no data neither virtual one-step audit can certify
        # a whole stratum at these caps. The 15-item uncertainty wins fallback.
        self.assertEqual(choose_stratum(states, [.5, .5], np.random.default_rng(1),
                                        block_size=1, eps_fn=.001, eps_fp=.001), 1)

    def test_exhausted_states_invalid_priors_and_large_forecasts_are_rejected(self):
        with self.assertRaises(ValueError):
            choose_stratum([CountCS(0, .05)], [.5], np.random.default_rng(1))
        for prior in ([np.nan], [1.1], [.5, .6]):
            with self.assertRaises(ValueError):
                choose_stratum([CountCS(4, .05)], prior, np.random.default_rng(1))
        with self.assertRaises(ValueError):
            choose_stratum([CountCS(4, .05)], [.5], np.random.default_rng(1), block_size=10)


if __name__ == "__main__":
    unittest.main()
