from __future__ import annotations

import math
import inspect
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.special import expit, logit


ROOT = Path(__file__).resolve().parents[3]
CORE_DIR = ROOT / "paper2_admet_benchmark" / "scripts" / "racer_c3"
sys.path.insert(0, str(CORE_DIR))

import racer_c3_core as core


class RacerC3CoreTests(unittest.TestCase):
    def synthetic_inputs(self, n: int = 120):
        rng = np.random.default_rng(23)
        target = np.asarray([0, 1] * (n // 2), dtype=np.int8)
        signal = 0.15 + 0.70 * target
        probability = np.column_stack(
            [np.clip(signal + rng.normal(0.0, 0.07, n), 0.01, 0.99) for _ in range(5)]
        )
        risk = np.column_stack(
            [
                rng.uniform(0.0, 1.0, n),
                rng.uniform(0.1, 0.8, n),
                rng.uniform(0.0, 0.2, n),
                rng.uniform(0.0, 0.4, n),
                rng.uniform(0.0, 1.0, n),
            ]
        )
        folds = np.arange(n) % 3
        return probability, risk, target, folds

    def test_fallback_exactly_matches_v1_attenuation(self) -> None:
        probability = np.asarray([0.1, 0.3, 0.7, 0.9])
        risk = np.asarray([0.0, 0.25, 0.75, 1.0])
        temperature = 1.0 + 0.5 * risk
        expected_p1 = expit(logit(probability) / temperature)
        observed = core.fallback_scores(probability, risk, t_max=1.5)
        np.testing.assert_allclose(observed[:, 0], expected_p1)
        np.testing.assert_allclose(observed[:, 1], 1.0 - expected_p1)

    def test_risk_tempering_moves_support_toward_half(self) -> None:
        support = np.asarray([0.1, 0.9])
        risk = np.ones(2)
        tempered = core.temper_support(support, risk, 2.0)
        self.assertGreater(tempered[0], support[0])
        self.assertLess(tempered[1], support[1])

    def test_candidate_crossfit_is_complete_and_external_prediction_is_finite(self) -> None:
        probability, risk, target, folds = self.synthetic_inputs()
        fitted = core.crossfit_candidate_correctness(
            probability, risk, target, folds, seed=71
        )
        self.assertEqual(fitted.oof_correctness.shape, (len(target), 2))
        self.assertTrue(np.isfinite(fitted.oof_correctness).all())
        external = fitted.ensemble.predict_correctness(probability[:11], risk[:11])
        self.assertEqual(external.shape, (11, 2))
        self.assertTrue(np.all((external > 0.0) & (external < 1.0)))

    def test_candidate_scores_are_not_required_to_sum_to_one(self) -> None:
        probability, risk, target, folds = self.synthetic_inputs()
        fitted = core.crossfit_candidate_correctness(
            probability, risk, target, folds, seed=73
        )
        scores = core.frontier_scores(
            probability,
            risk,
            fitted.oof_correctness,
            risk[:, 4],
        )
        self.assertEqual(scores.shape, (len(target), 2))
        self.assertFalse(np.allclose(scores.sum(axis=1), 1.0))

    def test_frontier_gate_is_permutation_invariant(self) -> None:
        development = ["A", "B", "C"]
        union = np.asarray([f"N{i}" for i in range(120)], dtype=object)
        distance = np.linspace(0.58, 0.72, len(union))
        first = core.symmetric_frontier_gate(development, union, distance)
        order = np.random.default_rng(17).permutation(len(union))
        second = core.symmetric_frontier_gate(
            development, union[order], distance[order]
        )
        self.assertTrue(first.active)
        self.assertEqual(first, second)

    def test_frontier_gate_has_no_target_argument(self) -> None:
        parameters = inspect.signature(core.symmetric_frontier_gate).parameters
        self.assertNotIn("target", parameters)
        self.assertNotIn("targets", parameters)

    def test_frontier_gate_falls_back_for_overlap_or_small_union(self) -> None:
        development = ["A", "B", "C"]
        overlap = np.asarray(["A"] * 60 + [f"N{i}" for i in range(60)])
        distance = np.full(120, 0.70)
        observed = core.symmetric_frontier_gate(development, overlap, distance)
        self.assertFalse(observed.active)
        small = core.symmetric_frontier_gate(
            development,
            np.asarray([f"N{i}" for i in range(20)]),
            np.full(20, 0.70),
        )
        self.assertFalse(small.active)
        self.assertEqual(small.reason, "insufficient_union")

    def test_route_has_exact_fallback_identity(self) -> None:
        fallback = np.arange(12, dtype=float).reshape(6, 2)
        frontier = fallback + 100.0
        off = core.FrontierGateDecision(False, 0.4, 0.5, 200, "fallback")
        on = core.FrontierGateDecision(True, 0.0, 0.7, 200, "frontier")
        np.testing.assert_array_equal(core.routed_scores(fallback, frontier, off), fallback)
        np.testing.assert_array_equal(core.routed_scores(fallback, frontier, on), frontier)

    def test_mondrian_quantile_and_set_states_match_hand_calculation(self) -> None:
        scores = np.column_stack([np.arange(18) / 20.0, np.arange(18)[::-1] / 20.0])
        target = np.asarray([0] * 9 + [1] * 9)
        thresholds = core.mondrian_thresholds(scores, target, alpha=0.10)
        self.assertEqual(thresholds[0], 0.4)
        self.assertEqual(thresholds[1], 0.4)
        membership = core.prediction_membership(
            np.asarray([[0.2, 0.8], [0.8, 0.2], [0.3, 0.3]]), thresholds
        )
        np.testing.assert_array_equal(
            membership, [[True, False], [False, True], [True, True]]
        )

    def test_small_class_quantile_is_infinite(self) -> None:
        self.assertEqual(
            core.finite_sample_quantile(np.arange(8) / 10.0, 0.10), math.inf
        )

    def test_mondrian_accepts_more_conservative_critical_alpha(self) -> None:
        scores = np.column_stack([np.arange(40) / 50.0, np.arange(40) / 50.0])
        target = np.asarray([0] * 20 + [1] * 20)
        ordinary = core.mondrian_thresholds(scores, target, 0.10)
        protected = core.mondrian_thresholds(scores, target, {0: 0.10, 1: 0.05})
        self.assertEqual(ordinary[0], protected[0])
        self.assertGreaterEqual(protected[1], ordinary[1])


if __name__ == "__main__":
    unittest.main()
