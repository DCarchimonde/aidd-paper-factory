from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]
P2 = ROOT / "paper2_admet_benchmark"
SCRIPT_DIR = P2 / "scripts" / "racer_c"
sys.path.insert(0, str(SCRIPT_DIR))

import racer_c_production_core as core


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


runner = load("production_runner_for_tests", SCRIPT_DIR / "run_confirmatory_racer_c.py")


class ProductionCoreTests(unittest.TestCase):
    def test_finite_sample_quantile_and_infinite_small_cell(self) -> None:
        self.assertEqual(core.finite_sample_quantile(np.arange(9) / 10, 0.10), 0.8)
        self.assertEqual(core.finite_sample_quantile(np.arange(8) / 10, 0.10), float("inf"))

    def test_candidate_label_nonconformity(self) -> None:
        values = core.nonconformity(np.asarray([0.2, 0.8]))
        np.testing.assert_allclose(values, [[0.2, 0.8], [0.8, 0.2]])
        sets = core.prediction_sets(np.asarray([0.2, 0.8]), {0: 0.3, 1: 0.3})
        self.assertEqual(core.state_labels(sets), ["Accept(0)", "Accept(1)"])

    def test_gate_and_metric_denominators(self) -> None:
        states = ["Accept(0)", "Ambiguous", "Accept(1)", "Defer-risk/domain"]
        metrics = core.metric_record(np.asarray([0, 0, 1, 1]), states)
        self.assertEqual(metrics["class_0_csy"], 0.5)
        self.assertEqual(metrics["class_0_coverage"], 1.0)
        self.assertEqual(metrics["class_1_gate_retention"], 0.5)
        self.assertEqual(metrics["macro_csy"], 0.5)

    def test_midrank_is_class_specific_and_deterministic(self) -> None:
        result = core.class_midrank_percentiles(
            np.asarray([0.1, 0.3, 0.2, 0.4]),
            np.asarray([0, 0, 1, 1]),
            np.asarray([0.2, 0.3]),
            np.asarray([0, 1]),
        )
        np.testing.assert_allclose(result, [0.5, 0.5])

    def test_rcp_multiplier_is_candidate_label_specific(self) -> None:
        rng = np.random.default_rng(7)
        features = rng.normal(size=(100, 4))
        probabilities = np.clip(0.2 + 0.6 * rng.random(100), 1e-3, 1 - 1e-3)
        targets = np.asarray([0, 1] * 50)
        transform = core.fit_rcp_transform(features, probabilities, targets, 0.10, 11)
        multipliers = transform.multipliers(features[:4])
        self.assertEqual(multipliers.shape, (4, 2))
        self.assertTrue(np.isfinite(multipliers).all())
        self.assertTrue((multipliers > 0).all())


class ProductionRunnerTests(unittest.TestCase):
    def test_lock_defines_exact_primary_scope(self) -> None:
        lock = yaml.safe_load(
            (P2 / "configs" / "racer_c" / "production_lock_v1.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(lock["primary_endpoints"]), 4)
        self.assertEqual(len(lock["tracks"]), 3)
        self.assertEqual(lock["main_split_seeds"], [101, 102, 103, 104, 105])
        self.assertEqual(lock["anchor_extra_split_seeds"], [106, 107, 108, 109, 110])
        self.assertEqual(
            len(lock["primary_endpoints"])
            * len(lock["tracks"])
            * len(lock["main_split_seeds"]),
            60,
        )
        self.assertEqual(len(lock["core_methods"]), 9)

    def test_ecfp_components_are_calibrated_before_block_average(self) -> None:
        calibrated = np.asarray(
            [
                [0.1, 0.2, 0.3, 0.4, 0.7, 0.8],
                [0.2, 0.4, 0.6, 0.8, 0.3, 0.1],
            ]
        )
        blocks = runner.calibrated_components_to_blocks(calibrated)
        np.testing.assert_allclose(blocks, [[0.25, 0.7, 0.8], [0.5, 0.3, 0.1]])

    def test_runner_requires_tag_review_and_complete_aggregate(self) -> None:
        source = (SCRIPT_DIR / "run_confirmatory_racer_c.py").read_text(encoding="utf-8")
        self.assertIn("paper2-racer-protocol-freeze-v1.0", source)
        self.assertIn("scientific_predictions_generated", source)
        self.assertIn("complete_confirmatory_primary_study", source)
        self.assertIn("incomplete_retained_failures", source)
        self.assertIn("aggregate completeness failure", source)
        self.assertIn("test_metrics_computed\": False", source)

    def test_protocol_tag_allows_only_recorded_post_freeze_repair(self) -> None:
        frozen = "a" * 40
        head = "b" * 40
        approved = "\n".join(sorted(runner.APPROVED_POST_FREEZE_PATHS))
        with mock.patch.object(
            runner,
            "git_output",
            side_effect=[head, frozen, "", approved],
        ):
            audit = runner.verify_protocol_tag(False)
        self.assertEqual(audit["head"], head)
        self.assertEqual(audit["tag_commit"], frozen)

        with mock.patch.object(
            runner,
            "git_output",
            side_effect=[head, frozen, "", "paper2_admet_benchmark/configs/racer_c/production_lock_v1.yaml"],
        ):
            with self.assertRaisesRegex(RuntimeError, "scientific or unrecorded files"):
                runner.verify_protocol_tag(False)

    def test_overnight_wrapper_is_keep_awake_resumable_and_fail_closed(self) -> None:
        source = (SCRIPT_DIR / "run_racer_c_overnight.ps1").read_text(encoding="utf-8")
        self.assertIn("SetThreadExecutionState", source)
        self.assertNotIn("[uint32]0x80000000", source)
        self.assertIn('[Convert]::ToUInt32("80000000", 16)', source)
        self.assertIn("paper2-racer-protocol-freeze-v1.0", source)
        self.assertIn("git merge-base --is-ancestor", source)
        self.assertIn("$AllowedPostFreezePaths", source)
        self.assertIn("protocol_deviations.md", source)
        self.assertIn("-Mode Full", source)
        self.assertIn("MaximumPasses = 3", source)
        self.assertIn("primary_cell_count -eq 60", source)
        self.assertIn("method_cell_count -eq 540", source)
        self.assertIn("failed_cell_count -eq 0", source)
        self.assertIn("do not delete the result directory", source)

    def test_review_rejects_wrong_prediction_flag(self) -> None:
        lock = {
            "formal_review_status": "pass_prediction_free_formal_freeze_review",
            "formal_review_required_cells": 60,
            "primary_endpoints": ["a", "b", "c", "d"],
            "tracks": ["random_grouped", "strict_scaffold", "similarity_cluster"],
            "main_split_seeds": [101, 102, 103, 104, 105],
            "outer_roles": {
                "dev": 0.50,
                "policy": 0.20,
                "conformal": 0.15,
                "test": 0.15,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.json"
            path.write_text(
                json.dumps(
                    {
                        "status": lock["formal_review_status"],
                        "scientific_predictions_generated": True,
                        "track_seed_cell_count": 60,
                        "primary_endpoints": lock["primary_endpoints"],
                        "tracks": lock["tracks"],
                        "main_split_seeds": lock["main_split_seeds"],
                        "allocation": "50_20_15_15",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "scientific_predictions_generated"):
                runner.verify_freeze_review(path, lock)

    def test_resume_requires_manifest_hash_and_row_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "raw_predictions.csv"
            runner.write_csv(table, [{"x": 1}, {"x": 2}], ["x"])
            runner.atomic_json(
                root / "lineage_manifest.json",
                {"status": "pass_transitive_outer_lineage"},
            )
            runner.atomic_json(
                root / "raw_manifest.json",
                {
                    "status": "complete_raw_predictions",
                    "n": 2,
                    "production_lock_sha256": "lock",
                    "raw_predictions_sha256": runner.sha256_file(table),
                    "lineage_manifest_sha256": runner.sha256_file(
                        root / "lineage_manifest.json"
                    ),
                },
            )
            self.assertTrue(runner.valid_cell_artifact(root, 2, "lock"))
            table.write_text("x\n1\n", encoding="utf-8")
            self.assertFalse(runner.valid_cell_artifact(root, 2, "lock"))

    def test_global_tmax_tie_chooses_smaller_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = root / "cell"
            cell.mkdir()
            rows = []
            for fold in range(3):
                for label in (0, 1):
                    for offset in range(20):
                        p = 0.1 if label == 0 else 0.9
                        rows.append(
                            {
                                "role": "dev",
                                "target": label,
                                "meta_fold": fold,
                                "stack_p": p,
                                "risk_percentile": (offset + 1) / 21,
                            }
                        )
            runner.write_csv(
                cell / "raw_predictions.csv",
                rows,
                ["role", "target", "meta_fold", "stack_p", "risk_percentile"],
            )
            lock = {
                "attenuation": {
                    "t_max_candidates": [1.0, 1.5, 2.0],
                    "coverage_shortfall_margin": 0.02,
                },
                "conformal": {"alpha": 0.10},
            }
            selected = runner.select_global_tmax(lock, root, [cell])
            self.assertEqual(selected["selected_t_max"], 1.0)

    def test_cell_raw_interruption_artifact_and_resume_without_gpu(self) -> None:
        rows = [
            {
                "endpoint": "Synthetic",
                "structure_id": f"s{i:04d}",
                "target": str(i % 2),
                "standardized_smiles": "C",
                "murcko_scaffold_id": f"scaffold_{i}",
                "similarity_cluster_id": f"cluster_{i}",
            }
            for i in range(240)
        ]
        ecfp = np.zeros((len(rows), 8), dtype=np.uint8)
        mol = np.zeros((len(rows), 4), dtype=float)
        lock = {
            "outer_roles": {"dev": 0.50, "policy": 0.20, "conformal": 0.15, "test": 0.15},
            "development_meta_folds": 3,
            "training_seed": 1701,
        }

        def fake_chain(train, predict, *args, **kwargs):
            n = len(predict)
            p = np.asarray([0.2 if int(rows[i]["target"]) == 0 else 0.8 for i in predict])
            blocks = np.column_stack([p, p, p])
            features = np.column_stack([np.ones(n), np.zeros(n), np.ones(n) * 0.1, np.ones(n) * 0.05])
            return {"blocks": blocks, "stack": p, "unrestricted": p, "bri": 1 - np.maximum(p, 1-p), "features": features}

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner, "fit_chain", side_effect=fake_chain
        ) as patched, mock.patch.object(runner, "sha256_file", wraps=runner.sha256_file) as sha:
            root = Path(directory)
            production_lock = root / "lock.yaml"
            production_lock.write_text("x: 1\n", encoding="utf-8")
            original_lock = runner.DEFAULT_PRODUCTION_LOCK
            runner.DEFAULT_PRODUCTION_LOCK = production_lock
            try:
                runner.run_cell_raw("Synthetic", "strict_scaffold", 101, rows, ecfp, mol, lock, {}, root)
                first_calls = patched.call_count
                self.assertEqual(first_calls, 3)
                cell = root / "cells" / runner.cell_id("Synthetic", "strict_scaffold", 101)
                self.assertTrue(runner.valid_cell_artifact(cell, len(rows), sha(production_lock)))
                runner.run_cell_raw("Synthetic", "strict_scaffold", 101, rows, ecfp, mol, lock, {}, root)
                self.assertEqual(patched.call_count, first_calls)
            finally:
                runner.DEFAULT_PRODUCTION_LOCK = original_lock

    def test_final_evaluation_produces_all_nine_methods_without_gpu(self) -> None:
        lock = yaml.safe_load(
            (P2 / "configs" / "racer_c" / "production_lock_v1.yaml").read_text(
                encoding="utf-8"
            )
        )
        rows = []
        for role in ("dev", "policy", "conformal", "test"):
            for label in (0, 1):
                for index in range(200):
                    probability = 0.1 if label == 0 else 0.9
                    rows.append(
                        {
                            "structure_id": f"{role}_{label}_{index}",
                            "role": role,
                            "target": "" if role == "test" else label,
                            "meta_fold": index % 3 if role == "dev" else "",
                            "ecfp_p": probability,
                            "dmpnn_p": probability,
                            "molformer_p": probability,
                            "heterogeneous_p": probability,
                            "stack_p": probability,
                            "unrestricted_p": probability,
                            "bri": 0.05,
                            "risk_percentile": (index + 1) / 201,
                            "absolute_margin": 2.0,
                            "disagreement": 0.0,
                            "ecfp_distance": 0.1,
                            "local_oof_brier_loss": 0.01,
                        }
                    )
        with tempfile.TemporaryDirectory() as directory:
            cell = Path(directory)
            runner.write_csv(cell / "raw_predictions.csv", rows, runner.RAW_FIELDS)
            sealed = {
                f"test_{label}_{index}": label
                for label in (0, 1)
                for index in range(200)
            }
            runner.evaluate_cell(
                cell, "Synthetic", "strict_scaffold", 101, 1, 1.5, lock, sealed
            )
            metrics = runner.read_table(cell / "metrics.csv")
            predictions = runner.read_table(cell / "test_predictions.csv")
            self.assertEqual(len(metrics), 9)
            self.assertEqual({row["method"] for row in metrics}, set(lock["core_methods"]))
            self.assertEqual(len(predictions), 9 * 400)
            manifest = json.loads((cell / "final_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete_final_evaluation")


if __name__ == "__main__":
    unittest.main()
