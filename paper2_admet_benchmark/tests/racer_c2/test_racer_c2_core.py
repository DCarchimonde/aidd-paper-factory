from __future__ import annotations

import csv
import importlib.util
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
P2 = ROOT / "paper2_admet_benchmark"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_module("core", P2 / "scripts" / "racer_c2" / "core.py")
AUDIT = load_module(
    "racer_c2_development_audit",
    P2 / "scripts" / "racer_c2" / "run_development_audit.py",
)
RETROSPECTIVE = load_module(
    "racer_c2_retrospective_extension",
    P2 / "scripts" / "racer_c2" / "run_retrospective_extension.py",
)


class CandidateLabelScoreTests(unittest.TestCase):
    def test_candidate_features_are_label_specific_without_hard_class(self) -> None:
        probabilities = np.asarray([[0.20, 0.40], [0.80, 0.60]])
        reliability = np.asarray([[0.1, 0.2], [0.3, 0.4]])
        zero = CORE.candidate_feature_matrix(probabilities, reliability, 0)
        one = CORE.candidate_feature_matrix(probabilities, reliability, 1)
        np.testing.assert_allclose(zero[:, :2], probabilities)
        np.testing.assert_allclose(one[:, :2], 1.0 - probabilities)
        np.testing.assert_allclose(zero[:, 2:], reliability)
        np.testing.assert_allclose(one[:, 2:], reliability)

    def test_zero_tilts_exactly_reproduce_safe_baseline(self) -> None:
        probability = np.asarray([0.1, 0.4, 0.6, 0.9])
        risk = np.asarray([0.0, 0.2, 0.8, 1.0])
        arbitrary_counterfactual = np.asarray(
            [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]]
        )
        configuration = CORE.ScoreConfiguration(1.5, 0.0, 0.0)
        observed = CORE.compose_candidate_scores(
            probability, risk, arbitrary_counterfactual, configuration
        )
        expected = CORE.candidate_nonconformity(
            CORE.attenuated_probability(probability, risk, 1.5)
        )
        np.testing.assert_array_equal(observed, expected)

    def test_candidate_label_tilts_have_declared_direction(self) -> None:
        scores = np.asarray([[0.4, 0.6], [0.4, 0.6]])
        risk = np.asarray([0.0, 1.0])
        observed = CORE.tilt_candidate_scores(scores, risk, 0.25, -0.10)
        np.testing.assert_array_equal(observed[0], scores[0])
        self.assertGreater(observed[1, 0], scores[1, 0])
        self.assertLess(observed[1, 1], scores[1, 1])

    def test_zero_tilt_is_bitwise_identity(self) -> None:
        scores = np.asarray([[0.123, 0.877], [0.456, 0.544]])
        observed = CORE.tilt_candidate_scores(
            scores, np.asarray([0.2, 0.9]), 0.0, 0.0
        )
        np.testing.assert_array_equal(observed, scores)

    def test_reference_percentiles_are_query_batch_invariant(self) -> None:
        reference = np.asarray([[0.1, 0.9], [0.2, 0.8], [0.2, 0.7], [0.8, 0.2]])
        query = np.asarray([[0.2, 0.75]])
        alone = CORE.reference_midrank_percentiles(reference, query)
        together = CORE.reference_midrank_percentiles(
            reference, np.vstack([query, [0.99, 0.01]])
        )[0:1]
        np.testing.assert_array_equal(alone, together)
        np.testing.assert_allclose(alone, [[0.5, 0.5]])

    def test_crossfit_scores_and_external_ensemble_are_complete(self) -> None:
        rng = np.random.default_rng(77)
        n = 180
        target = np.tile(np.asarray([0, 1], dtype=np.int8), n // 2)
        probability = np.column_stack(
            [np.clip(0.15 + 0.7 * target + rng.normal(0, 0.08, n), 0.01, 0.99)]
            * 5
        )
        reliability = rng.uniform(0.0, 1.0, size=(n, 4))
        folds = np.arange(n) % 3
        external_probability = probability[:12].copy()
        external_reliability = reliability[:12].copy()
        oof, external = CORE.crossfit_candidate_error_scores(
            probability,
            reliability,
            target,
            folds,
            2701,
            external_probability,
            external_reliability,
        )
        self.assertEqual(oof.shape, (n, 2))
        self.assertIsNotNone(external)
        self.assertEqual(external.shape, (12, 2))
        self.assertTrue(np.isfinite(oof).all())
        self.assertTrue(np.isfinite(external).all())


class ConformalAndCertificateTests(unittest.TestCase):
    def test_finite_quantile_and_small_cell_infinity(self) -> None:
        self.assertEqual(CORE.finite_sample_quantile(np.arange(9), 0.10), 8.0)
        self.assertTrue(math.isinf(CORE.finite_sample_quantile(np.arange(8), 0.10)))

    def test_candidate_label_mondrian_sets_match_hand_calculation(self) -> None:
        calibration_scores = np.asarray(
            [
                [0.1, 0.8],
                [0.2, 0.7],
                [0.3, 0.6],
                [0.8, 0.1],
                [0.7, 0.2],
                [0.6, 0.3],
            ]
        )
        targets = np.asarray([0, 0, 0, 1, 1, 1])
        thresholds = CORE.mondrian_thresholds(calibration_scores, targets, 0.50)
        self.assertEqual(thresholds, {0: 0.2, 1: 0.2})
        query = np.asarray([[0.2, 0.2], [0.2, 0.8], [0.8, 0.2], [0.8, 0.8]])
        np.testing.assert_array_equal(
            CORE.prediction_sets(query, thresholds),
            np.asarray([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=bool),
        )

    def test_final_set_metrics_use_true_class_denominators(self) -> None:
        targets = np.asarray([0, 0, 0, 1, 1, 1])
        sets = np.asarray(
            [[1, 0], [0, 1], [1, 1], [0, 1], [1, 0], [0, 0]], dtype=bool
        )
        row = CORE.set_metric_record(targets, sets)
        self.assertEqual(row["class_0_n"], 3)
        self.assertEqual(row["class_0_correct_singleton_n"], 1)
        self.assertEqual(row["class_0_wrong_singleton_n"], 1)
        self.assertEqual(row["class_1_correct_singleton_n"], 1)
        self.assertEqual(row["class_1_wrong_singleton_n"], 1)
        self.assertAlmostEqual(row["macro_csy"], 1.0 / 3.0)

    def test_perfect_large_final_sets_are_certified_without_grid_penalty(self) -> None:
        targets = np.asarray([0] * 200 + [1] * 200)
        sets = np.column_stack([targets == 0, targets == 1])
        result = CORE.certify_final_sets(
            targets,
            sets,
            CORE.ActionCertificateConstraints(
                wrong_singleton_ceiling=0.10,
                coverage_floor=0.90,
                critical_csy_floor=0.80,
            ),
        )
        self.assertEqual(result["status"], "certified")
        self.assertEqual(result["selection_grid_test_count"], 0)
        self.assertEqual(result["tested_constraint_count"], 5)

    def test_wrong_singleton_certificate_fails_closed(self) -> None:
        targets = np.asarray([0] * 100 + [1] * 100)
        sets = np.column_stack([targets == 0, targets == 1])
        sets[:30] = [False, True]
        result = CORE.certify_final_sets(
            targets,
            sets,
            CORE.ActionCertificateConstraints(
                wrong_singleton_ceiling=0.10,
                coverage_floor=None,
            ),
        )
        self.assertEqual(result["status"], "certificate-failed-closed")
        self.assertFalse(result["classes"]["0"]["checks"]["wrong_singleton"])

    def test_disabled_constraints_do_not_enter_multiplicity(self) -> None:
        targets = np.asarray([0] * 200 + [1] * 200)
        sets = np.column_stack([targets == 0, targets == 1])
        result = CORE.certify_final_sets(
            targets,
            sets,
            CORE.ActionCertificateConstraints(
                coverage_floor=None,
                wrong_singleton_ceiling=0.10,
                empty_exposure_ceiling=None,
                critical_csy_floor=None,
            ),
        )
        self.assertEqual(result["tested_constraint_count"], 2)
        self.assertAlmostEqual(result["simultaneous_alpha"], 0.025)


class DevelopmentSelectionTests(unittest.TestCase):
    @staticmethod
    def rows() -> list[dict[str, object]]:
        rows = []
        for cell in ("a", "b"):
            for t_max, gamma_0, gamma_1, blend, csy in (
                (1.0, 0.0, 0.0, 0.0, 0.50),
                (1.5, 0.0, 0.0, 0.0, 0.55),
                (1.5, 0.1, -0.1, 0.0, 0.55),
            ):
                rows.append(
                    {
                        "cell": cell,
                        "t_max": t_max,
                        "gamma_0": gamma_0,
                        "gamma_1": gamma_1,
                        "counterfactual_blend": blend,
                        "baseline_macro_csy": 0.50,
                        "candidate_macro_csy": csy,
                        "baseline_class_0_coverage": 0.90,
                        "baseline_class_1_coverage": 0.90,
                        "candidate_class_0_coverage": 0.89,
                        "candidate_class_1_coverage": 0.89,
                    }
                )
        return rows

    def test_selection_is_deterministic_and_prefers_safe_tie(self) -> None:
        rows = self.rows()
        first, first_summary = CORE.select_development_configuration(rows, 0.02, 0.85)
        random.Random(999).shuffle(rows)
        second, second_summary = CORE.select_development_configuration(
            rows, 0.02, 0.85
        )
        self.assertEqual(first, CORE.ScoreConfiguration(1.5, 0.0, 0.0))
        self.assertEqual(first, second)
        self.assertEqual(first_summary, second_summary)

    def test_incomplete_candidate_cannot_win(self) -> None:
        rows = self.rows()
        rows = [
            row
            for row in rows
            if not (
                row["cell"] == "b"
                and row["t_max"] == 1.5
                and row["gamma_0"] == 0.1
                and row["gamma_1"] == -0.1
            )
        ]
        selected, summary = CORE.select_development_configuration(rows, 0.02, 0.85)
        self.assertEqual(selected, CORE.ScoreConfiguration(1.5, 0.0, 0.0))
        incomplete = next(
            row
            for row in summary
            if row["t_max"] == 1.5
            and row["gamma_0"] == 0.1
            and row["gamma_1"] == -0.1
        )
        self.assertFalse(incomplete["complete"])
        self.assertFalse(incomplete["feasible"])

    def test_absolute_cell_coverage_floor_is_fail_closed(self) -> None:
        rows = self.rows()
        for row in rows:
            if row["gamma_0"] == 0.1:
                row["candidate_class_1_coverage"] = 0.84
        selected, summary = CORE.select_development_configuration(rows, 0.02, 0.85)
        self.assertEqual(selected, CORE.ScoreConfiguration(1.5, 0.0, 0.0))
        rejected = next(row for row in summary if row["gamma_0"] == 0.1)
        self.assertFalse(rejected["feasible"])
        self.assertEqual(rejected["class_1_candidate_minimum_coverage"], 0.84)


class RetrospectiveExtensionTests(unittest.TestCase):
    def test_cell_name_contract(self) -> None:
        self.assertEqual(
            RETROSPECTIVE.parse_cell_name(
                "Tox21_NR_AhR__strict_scaffold__seed103"
            ),
            ("Tox21_NR_AhR", "strict_scaffold", 103),
        )
        with self.assertRaises(ValueError):
            RETROSPECTIVE.parse_cell_name("malformed")

    def test_source_output_separation_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "v1"
            source.mkdir()
            RETROSPECTIVE.assert_separate_output(source, Path(temporary) / "v2")
            with self.assertRaises(ValueError):
                RETROSPECTIVE.assert_separate_output(source, source)
            with self.assertRaises(ValueError):
                RETROSPECTIVE.assert_separate_output(source, source / "new")

    def test_test_label_recovery_requires_complete_old_method_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test_predictions.csv"
            rows = [
                {
                    "structure_id": structure_id,
                    "target": target,
                    "method": method,
                    "state": "Accept(0)",
                }
                for structure_id, target in (("a", 0), ("b", 1))
                for method in ("old_a", "old_b")
            ]
            RETROSPECTIVE.write_csv(
                path, rows, ("structure_id", "target", "method", "state")
            )
            labels, methods = RETROSPECTIVE.recover_test_labels(path, 2)
            self.assertEqual(labels, {"a": 0, "b": 1})
            self.assertEqual(methods, ("old_a", "old_b"))
            rows.pop()
            RETROSPECTIVE.write_csv(
                path, rows, ("structure_id", "target", "method", "state")
            )
            with self.assertRaises(RuntimeError):
                RETROSPECTIVE.recover_test_labels(path, 2)

    def test_cluster_bootstrap_is_deterministic(self) -> None:
        rows = [
            {"cluster": "a", "difference": 0.1},
            {"cluster": "a", "difference": 0.2},
            {"cluster": "b", "difference": -0.1},
            {"cluster": "b", "difference": 0.0},
        ]
        first = RETROSPECTIVE.bootstrap_cluster_mean_ci(rows, 250, 77, 0.95)
        second = RETROSPECTIVE.bootstrap_cluster_mean_ci(rows, 250, 77, 0.95)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], first[1])

    def test_metric_row_uses_final_set_states(self) -> None:
        targets = np.asarray([0, 0, 1, 1])
        sets = np.asarray(
            [[1, 0], [1, 1], [0, 1], [1, 0]], dtype=bool
        )
        row = RETROSPECTIVE.metric_row(
            "endpoint",
            "track",
            101,
            "cell",
            "method",
            "role",
            "test",
            targets,
            sets,
            {0: 0.5, 1: 0.6},
        )
        self.assertEqual(row["singleton_n"], 3)
        self.assertAlmostEqual(row["singleton_rate"], 0.75)
        self.assertAlmostEqual(row["ambiguity_rate"], 0.25)
        self.assertEqual(row["class_1_wrong_singleton_n"], 1)

    def test_fixed_retrospective_lock_preserves_disclosure_and_fallback(self) -> None:
        lock = RETROSPECTIVE.load_lock(
            P2 / "configs" / "racer_c2" / "retrospective_extension_lock_v0.yaml"
        )
        self.assertFalse(lock["confirmatory_claim_authorized"])
        fallback = next(
            row
            for row in lock["fixed_method_family"]
            if row["method"] == "Stacking_Mondrian_fallback"
        )
        self.assertEqual(
            RETROSPECTIVE.configuration(fallback),
            CORE.ScoreConfiguration(1.0, 0.0, 0.0, 0.0),
        )

    def test_windows_increment_runner_reuses_v1_without_retraining(self) -> None:
        wrapper = (
            P2 / "scripts" / "racer_c2" / "run_racer_c2_increment.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("source_cell_count -ne 60", wrapper)
        self.assertIn("source_method_cell_count -ne 540", wrapper)
        self.assertIn("base_models_retrained -ne $false", wrapper)
        self.assertIn("old_methods_rerun -ne $false", wrapper)
        self.assertIn("run_retrospective_extension.py", wrapper)


class DevelopmentFirewallTests(unittest.TestCase):
    def test_cell_track_parser_is_explicit(self) -> None:
        self.assertEqual(
            AUDIT.cell_track(Path("Tox21_NR_AhR__strict_scaffold__seed101")),
            "strict_scaffold",
        )
        with self.assertRaises(ValueError):
            AUDIT.cell_track(Path("ambiguous-cell-name"))

    def test_reader_materializes_only_development_labels(self) -> None:
        fields = sorted(AUDIT.REQUIRED_COLUMNS)
        rows = []
        for role, target, fold in (
            ("dev", "1", "0"),
            ("policy", "outer-policy-secret", ""),
            ("conformal", "outer-conformal-secret", ""),
            ("test", "outer-test-secret", ""),
        ):
            row = {field: "0.2" for field in fields}
            row.update(
                {
                    "structure_id": f"id-{role}",
                    "role": role,
                    "target": target,
                    "meta_fold": fold,
                }
            )
            rows.append(row)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw_predictions.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            development, counts = AUDIT.read_development_rows(path)
        self.assertEqual([row["target"] for row in development], ["1"])
        self.assertEqual(counts, {"dev": 1, "policy": 1, "conformal": 1, "test": 1})

    def test_protocol_and_config_keep_v1_test_out_of_confirmation(self) -> None:
        config = (
            P2 / "configs" / "racer_c2" / "development_lock_v0.yaml"
        ).read_text(encoding="utf-8")
        protocol = (
            P2 / "protocols" / "racer_c2_prospective_protocol_draft.md"
        ).read_text(encoding="utf-8")
        specification = (
            P2 / "docs" / "racer_c2_algorithm_specification_v0.1.md"
        ).read_text(encoding="utf-8")
        self.assertIn("development_only_not_frozen", config)
        self.assertIn("prohibited_as_confirmatory", config)
        self.assertIn("No RACER-C2 result may be described as prospective", protocol)
        self.assertIn("can never become its confirmatory panel", specification)
        self.assertIn("selection_grid_test_count=0", specification)


if __name__ == "__main__":
    unittest.main()
