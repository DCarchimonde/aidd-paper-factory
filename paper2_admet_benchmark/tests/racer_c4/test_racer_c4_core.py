from __future__ import annotations

import inspect
import math
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
CORE_DIR = ROOT / "paper2_admet_benchmark" / "scripts" / "racer_c4"
sys.path.insert(0, str(CORE_DIR))

import racer_c4_core as core


def audit(view: str, active: bool) -> core.TransportAudit:
    return core.TransportAudit(
        view=view,
        active=active,
        reason="active" if active else "failed",
        source_n=100,
        target_n=100,
        class_0_n=50,
        class_1_n=50,
        source_ess=90.0,
        class_0_ess=45.0,
        class_1_ess=45.0,
        domain_auc=0.6,
        lower_clip_fraction=0.0,
        upper_clip_fraction=0.0,
        mean_gap_before=0.2,
        mean_gap_after=0.1,
    )


class RacerC4CoreTests(unittest.TestCase):
    def test_binary_nonconformity_columns_are_candidate_specific(self) -> None:
        observed = core.binary_nonconformity(np.asarray([0.2, 0.8]))
        np.testing.assert_allclose(observed, [[0.2, 0.8], [0.8, 0.2]])

    def test_finite_sample_quantile_matches_hand_calculation(self) -> None:
        values = np.arange(8) / 10.0
        self.assertEqual(core.finite_sample_quantile(values, 0.10), math.inf)
        values = np.arange(10) / 10.0
        self.assertEqual(core.finite_sample_quantile(values, 0.10), 0.9)

    def test_weighted_query_mass_at_infinity_matches_hand_calculation(self) -> None:
        scores = np.column_stack(
            [np.asarray([0.1, 0.2, 0.3, 0.4]), np.asarray([0.4, 0.3, 0.2, 0.1])]
        )
        target = np.asarray([0, 0, 1, 1], dtype=np.int8)
        thresholds = core.weighted_test_thresholds(
            scores,
            target,
            np.ones(4),
            np.asarray([0.1, 3.0]),
            alpha=0.25,
        )
        self.assertEqual(thresholds[0, 0], 0.2)
        self.assertEqual(thresholds[0, 1], 0.2)
        self.assertTrue(np.isinf(thresholds[1]).all())

    def test_envelope_contains_baseline_and_creates_no_singleton(self) -> None:
        baseline = np.asarray(
            [[True, False], [False, True], [True, True], [False, False]], dtype=bool
        )
        first = np.asarray(
            [[True, True], [True, True], [True, True], [True, False]], dtype=bool
        )
        second = np.asarray(
            [[True, False], [True, True], [False, True], [True, False]], dtype=bool
        )
        decision = core.transport_envelope(
            baseline,
            [first, second],
            [audit("a", True), audit("b", True)],
            protected_labels=[0],
            quorum="all",
            minimum_active_views=2,
        )
        self.assertFalse(decision.failed_closed)
        self.assertFalse(np.any(baseline & ~decision.membership))
        self.assertFalse(np.any(decision.membership.sum(axis=1) == 0))
        new_singletons = (decision.membership.sum(axis=1) == 1) & (
            baseline.sum(axis=1) != 1
        )
        self.assertFalse(new_singletons.any())
        np.testing.assert_array_equal(decision.membership[3], [True, True])

    def test_insufficient_active_views_fail_closed_to_baseline(self) -> None:
        baseline = np.asarray([[True, False], [False, False]], dtype=bool)
        candidate = np.asarray([[True, True], [True, False]], dtype=bool)
        decision = core.transport_envelope(
            baseline,
            [candidate, candidate],
            [audit("a", True), audit("b", False)],
            minimum_active_views=2,
        )
        self.assertTrue(decision.failed_closed)
        np.testing.assert_array_equal(
            decision.membership,
            [[True, False], [True, True]],
        )

    def test_envelope_cannot_increase_wrong_singleton_exposure(self) -> None:
        rng = np.random.default_rng(43)
        baseline = rng.integers(0, 2, size=(500, 2)).astype(bool)
        candidates = [rng.integers(0, 2, size=(500, 2)).astype(bool) for _ in range(2)]
        targets = rng.integers(0, 2, size=500).astype(np.int8)
        decision = core.transport_envelope(
            baseline,
            candidates,
            [audit("a", True), audit("b", True)],
            protected_labels=[0],
            quorum="all",
            minimum_active_views=2,
        )
        ordinary = core.set_metrics(targets, baseline)
        tame = core.set_metrics(targets, decision.membership)
        for label in (0, 1):
            self.assertLessEqual(
                tame[f"class_{label}_wrong_singleton_exposure"],
                ordinary[f"class_{label}_wrong_singleton_exposure"],
            )

    def test_density_ratio_is_deterministic_and_uses_no_target_label_argument(self) -> None:
        rng = np.random.default_rng(5)
        source = rng.normal(0.0, 1.0, size=(120, 6))
        target = rng.normal(0.2, 1.0, size=(100, 6))
        source_y = np.asarray([0, 1] * 60, dtype=np.int8)
        first = core.cross_fitted_density_ratio(source, target, source_y, seed=17)
        second = core.cross_fitted_density_ratio(source, target, source_y, seed=17)
        np.testing.assert_allclose(first.source_weights, second.source_weights)
        np.testing.assert_allclose(first.target_weights, second.target_weights)
        parameters = inspect.signature(core.cross_fitted_density_ratio).parameters
        self.assertNotIn("target_labels", parameters)
        self.assertNotIn("target_targets", parameters)

    def test_transport_audit_rejects_low_class_ess(self) -> None:
        result = core.DensityRatioResult(
            source_weights=np.ones(100),
            target_weights=np.ones(100),
            domain_auc=0.6,
            source_ess=80.0,
            class_0_ess=2.0,
            class_1_ess=40.0,
            lower_clip_fraction=0.0,
            upper_clip_fraction=0.0,
            mean_gap_before=0.2,
            mean_gap_after=0.1,
        )
        observed = core.audit_density_ratio(
            "ecfp", result, np.asarray([0] * 50 + [1] * 50), 100
        )
        self.assertFalse(observed.active)
        self.assertIn("class_ess", observed.reason)


if __name__ == "__main__":
    unittest.main()
