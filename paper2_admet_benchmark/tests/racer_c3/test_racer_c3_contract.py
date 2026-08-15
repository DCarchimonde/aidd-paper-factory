from __future__ import annotations

import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
P2 = ROOT / "paper2_admet_benchmark"


class RacerC3ContractTests(unittest.TestCase):
    def test_development_lock_is_explicitly_retrospective_and_unfrozen(self) -> None:
        lock = yaml.safe_load(
            (P2 / "configs" / "racer_c3" / "development_lock_v0.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            lock["lock_status"],
            "retrospective_architecture_candidate_not_freeze_ready",
        )
        self.assertTrue(
            lock["retrospective_signal"][
                "architecture_was_chosen_after_outer_outcomes_were_known"
            ]
        )
        self.assertFalse(
            lock["retrospective_signal"]["scientific_superiority_claim_authorized"]
        )
        self.assertFalse(lock["prospective_validation"]["test_predictions_allowed"])

    def test_route_and_fallback_are_numerically_locked(self) -> None:
        lock = yaml.safe_load(
            (P2 / "configs" / "racer_c3" / "development_lock_v0.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(lock["frontier_route"]["overlap_fraction_max"], 0.05)
        self.assertEqual(lock["frontier_route"]["median_ecfp_distance_min"], 0.57)
        self.assertTrue(lock["frontier_route"]["permutation_invariant_union"])
        self.assertEqual(lock["fallback"]["method"], "RACER-C-v1-no-gate")
        self.assertTrue(lock["fallback"]["exact_identity_required"])
        self.assertEqual(lock["conformal"]["fallback_alpha_by_class"], {0: 0.1, 1: 0.1})
        self.assertEqual(
            lock["conformal"]["frontier_alpha_by_class"], {0: 0.1, 1: 0.095}
        )

    def test_retrospective_runner_cannot_masquerade_as_confirmatory(self) -> None:
        source = (
            P2
            / "scripts"
            / "racer_c3"
            / "run_retrospective_architecture_audit.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"known_test_labels_used": True', source)
        self.assertIn('"scientific_superiority_claim_authorized": False', source)
        self.assertIn('"prospective_test_predictions_authorized": False', source)
        self.assertNotIn("scientific_superiority_claim_authorized\": True", source)


if __name__ == "__main__":
    unittest.main()
