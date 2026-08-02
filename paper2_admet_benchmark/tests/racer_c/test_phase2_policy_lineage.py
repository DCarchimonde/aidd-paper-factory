from __future__ import annotations

import importlib.util
import hashlib
import json
import random
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
P2 = ROOT / "paper2_admet_benchmark"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


POLICY = load_module(
    "racer_policy_selection",
    P2 / "scripts" / "racer_c" / "policy_selection.py",
)
LINEAGE = load_module(
    "racer_lineage_contract",
    P2 / "scripts" / "racer_c" / "lineage_contract.py",
)
ELIGIBILITY = load_module(
    "racer_endpoint_eligibility",
    P2 / "scripts" / "racer_c" / "finalize_endpoint_eligibility.py",
)


def policy_rows(n_per_class: int = 180, error_every: int = 40):
    rows = []
    for label in (0, 1):
        for index in range(n_per_class):
            predicted = label if index % error_every else 1 - label
            rows.append(
                {
                    "structure_id": f"s{label}_{index}",
                    "true_class": label,
                    "predicted_class": predicted,
                    "risk_percentile": min(1.0, (index + 1) / n_per_class),
                }
            )
    return rows


class PolicySelectionTests(unittest.TestCase):
    def test_perfect_large_policy_sample_is_feasible(self) -> None:
        rows = policy_rows(n_per_class=180, error_every=1000)
        # Remove the index-zero errors introduced by the helper.
        for row in rows:
            row["predicted_class"] = row["true_class"]
        status, chosen, evaluations = POLICY.select_policy(rows)
        self.assertEqual(status, "selected")
        self.assertIsNotNone(chosen)
        self.assertEqual(len(evaluations), 36)
        self.assertEqual((chosen["threshold_0"], chosen["threshold_1"]), (1.0, 1.0))

    def test_selection_is_deterministic_under_row_permutation(self) -> None:
        rows = policy_rows()
        first = POLICY.select_policy(rows)
        random.Random(999).shuffle(rows)
        second = POLICY.select_policy(rows)
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertEqual(first[2], second[2])

    def test_no_feasible_pair_fails_closed(self) -> None:
        rows = policy_rows(n_per_class=80, error_every=1)
        status, chosen, evaluations = POLICY.select_policy(rows)
        self.assertEqual(status, "policy-infeasible")
        self.assertIsNone(chosen)
        self.assertEqual(len(evaluations), 36)

    def test_sparse_policy_class_fails_closed(self) -> None:
        rows = policy_rows(n_per_class=20, error_every=1000)
        status, chosen, _ = POLICY.select_policy(rows)
        self.assertEqual(status, "policy-infeasible")
        self.assertIsNone(chosen)


class LineageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.roles = {
            **{f"d{i}": "dev" for i in range(6)},
            "p0": "policy",
            "c0": "conformal",
            "t0": "test",
        }

    def test_honest_oof_chain_passes(self) -> None:
        fits = [
            LINEAGE.FitNode("base", "base", frozenset({"d0", "d1", "d2"})),
            LINEAGE.FitNode(
                "cal", "platt", frozenset({"d3", "d4"}), ("base",)
            ),
        ]
        predictions = [
            LINEAGE.PredictionNode("pred_d5", "d5", "dev", ("cal",)),
            LINEAGE.PredictionNode("pred_t0", "t0", "test", ("cal",)),
        ]
        resolved = LINEAGE.validate_prediction_lineage(fits, predictions, self.roles)
        self.assertNotIn("d5", resolved["pred_d5"])
        self.assertEqual(resolved["pred_t0"], frozenset({"d0", "d1", "d2", "d3", "d4"}))

    def test_self_leakage_is_rejected(self) -> None:
        fits = [LINEAGE.FitNode("base", "base", frozenset({"d0", "d1"}))]
        predictions = [
            LINEAGE.PredictionNode("pred_d1", "d1", "dev", ("base",))
        ]
        with self.assertRaisesRegex(ValueError, "self/OOF leakage"):
            LINEAGE.validate_prediction_lineage(fits, predictions, self.roles)

    def test_outer_role_leakage_is_rejected(self) -> None:
        fits = [LINEAGE.FitNode("base", "base", frozenset({"d0", "p0"}))]
        predictions = [
            LINEAGE.PredictionNode("pred_t0", "t0", "test", ("base",))
        ]
        with self.assertRaisesRegex(ValueError, "outer-role leakage"):
            LINEAGE.validate_prediction_lineage(fits, predictions, self.roles)

    def test_lineage_cycle_is_rejected(self) -> None:
        fits = [
            LINEAGE.FitNode("a", "base", frozenset(), ("b",)),
            LINEAGE.FitNode("b", "stacker", frozenset(), ("a",)),
        ]
        with self.assertRaisesRegex(ValueError, "cycle"):
            LINEAGE.resolve_fit_rows(fits)


class EndpointEligibilityTests(unittest.TestCase):
    def test_all_cells_pass_yields_primary_candidate(self) -> None:
        cleaning = {
            "endpoint": "Tox21_test",
            "clean_class_0_n": 900,
            "clean_class_1_n": 500,
        }
        rows = [
            {
                "endpoint": "Tox21_test",
                "allocation": "50_20_15_15",
                "track": f"track_{i // 5}",
                "primary_count_gate": "pass",
            }
            for i in range(15)
        ]
        status, candidate, _, passing, total = ELIGIBILITY.decide(cleaning, rows)
        self.assertEqual(status, "primary_candidate")
        self.assertEqual(candidate, "freeze1_primary_candidate")
        self.assertEqual((passing, total), (15, 15))

    def test_small_class_is_calibration_limited_before_tracks(self) -> None:
        cleaning = {
            "endpoint": "Tox21_test",
            "clean_class_0_n": 900,
            "clean_class_1_n": 200,
        }
        status, candidate, _, _, _ = ELIGIBILITY.decide(cleaning, [])
        self.assertEqual(status, "calibration-limited")
        self.assertEqual(candidate, "freeze1_calibration_limited")


class Tox21ManifestTests(unittest.TestCase):
    def test_all_endpoint_manifests_reconcile_and_lock_source(self) -> None:
        manifest_dir = P2 / "data" / "manifests" / "racer_c"
        paths = sorted(manifest_dir.glob("Tox21_*_cleaning.json"))
        self.assertEqual(len(paths), 12)
        archive_hashes = set()
        member_hashes = set()
        for path in paths:
            row = json.loads(path.read_text(encoding="utf-8"))
            source_n = int(row["source_archive_rows"])
            observed_n = int(row["endpoint_observed_label_rows"])
            standardized_n = int(row["standardized_candidate_rows"])
            clean_n = int(row["clean_unique_structures"])
            duplicate_n = int(row["duplicate_rows_aggregated"])
            conflict_groups = int(row["conflicting_structure_groups_excluded"])
            self.assertEqual(source_n, 11764)
            self.assertEqual(observed_n + int(row["missing_label_rows"]), source_n)
            self.assertEqual(
                int(row["clean_class_0_n"]) + int(row["clean_class_1_n"]), clean_n
            )
            self.assertLessEqual(standardized_n, observed_n)
            conflict_rows = standardized_n - clean_n - duplicate_n
            self.assertGreaterEqual(conflict_rows, conflict_groups)
            for field in (
                "cleaned_byte_sha256",
                "rejections_byte_sha256",
                "role_input_byte_sha256",
                "cleaning_script_byte_sha256",
            ):
                self.assertRegex(str(row[field]), re.compile(r"^[0-9a-f]{64}$"))
            archive_hashes.add(row["source_archive_sha256"])
            member_hashes.add(row["source_member_sha256"])
        self.assertEqual(
            archive_hashes,
            {"024a3ae2690bcd4a593e6e0b10b455470b9bcb1d8f299dd36f220a250181517b"},
        )
        self.assertEqual(
            member_hashes,
            {"d66e1f9ec945ee528b1bea6e49af9c10d0bad546c2b304eb96004c8228824206"},
        )

    def test_seed99_cpu_smoke_never_scores_outer_roles(self) -> None:
        path = P2 / "results" / "racer_c_phase2_preflight" / "seed99_cpu_lineage_smoke.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(row["status"], "pass_development_only_cpu_smoke")
        self.assertEqual(row["seed"], 99)
        self.assertEqual(row["outer_oof_prediction_count"], row["dev_n"])
        self.assertEqual(row["outer_role_labels_used_by_trainer"], ["dev"])
        self.assertFalse(row["policy_conformal_test_predictions_generated"])
        self.assertFalse(row["performance_metrics_computed"])
        script = P2 / "scripts" / "racer_c" / "run_seed99_cpu_lineage_smoke.py"
        self.assertEqual(
            row["script_sha256"], hashlib.sha256(script.read_bytes()).hexdigest()
        )

    def test_committed_endpoint_decision_matches_predeclared_gate(self) -> None:
        rows = ELIGIBILITY.read_csv(
            P2
            / "results"
            / "racer_c_phase2_preflight"
            / "endpoint_eligibility_decision.csv"
        )
        by_status: dict[str, set[str]] = {}
        for row in rows:
            by_status.setdefault(row["eligibility_status"], set()).add(row["endpoint"])
            self.assertEqual(row["selection_used_model_outputs"], "false")
        self.assertEqual(
            by_status["primary_candidate"],
            {"Tox21_NR_ER", "Tox21_SR_ARE", "Tox21_SR_MMP"},
        )
        self.assertEqual(
            by_status["track_limited_secondary"],
            {"Tox21_NR_AhR", "Tox21_SR_p53"},
        )
        self.assertEqual(len(by_status["calibration-limited"]), 7)


if __name__ == "__main__":
    unittest.main()
