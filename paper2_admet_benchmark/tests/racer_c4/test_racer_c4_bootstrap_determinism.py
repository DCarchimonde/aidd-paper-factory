from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml


ROOT = Path(__file__).resolve().parents[3]
RUNNER_DIR = ROOT / "paper2_admet_benchmark" / "scripts" / "racer_c4"
sys.path.insert(0, str(RUNNER_DIR))

import run_prospective_racer_c4 as runner  # noqa: E402
import recompute_racer_c4_inference as recompute  # noqa: E402
from racer_c4_io import atomic_csv, atomic_json, sha256_file  # noqa: E402


ENDPOINTS = ("endpoint_zeta", "endpoint_alpha", "endpoint_mu", "endpoint_beta")
SEEDS = (211, 212, 213)


def synthetic_inputs() -> tuple[list[dict[str, object]], dict[str, dict[str, float]]]:
    predictions: list[dict[str, object]] = []
    labels: dict[str, dict[str, float]] = {}
    for endpoint_index, endpoint in enumerate(ENDPOINTS):
        sample_count = 18 + 2 * endpoint_index
        for position in range(sample_count):
            sample_id = f"{endpoint}_{position:02d}"
            target = int((position + endpoint_index) % 2)
            labels[sample_id] = {endpoint: float(target)}
        for seed_index, run_seed in enumerate(SEEDS):
            for method in (runner.METHOD_BASELINE, runner.METHOD_TAME):
                for position in range(sample_count):
                    sample_id = f"{endpoint}_{position:02d}"
                    target = int(labels[sample_id][endpoint])
                    include_0 = int(
                        (position + 2 * endpoint_index + seed_index) % 5 != 0
                    )
                    include_1 = int(
                        (2 * position + endpoint_index + seed_index) % 7 != 0
                    )
                    if method == runner.METHOD_TAME:
                        if target == 0 and (
                            position + endpoint_index + seed_index
                        ) % (3 + endpoint_index) == 0:
                            include_0 = 1
                        if target == 1 and (
                            position + 2 * endpoint_index + seed_index
                        ) % (6 - endpoint_index) == 0:
                            include_1 = 1
                    predictions.append(
                        {
                            "endpoint": endpoint,
                            "seed": run_seed,
                            "method": method,
                            "sample_id": sample_id,
                            "excluded_structure_overlap": 0,
                            "excluded_structure_invalid": 0,
                            "include_0": include_0,
                            "include_1": include_1,
                        }
                    )
    return predictions, labels


def probe_result() -> dict[str, object]:
    predictions, labels = synthetic_inputs()
    return runner.hierarchical_bootstrap_primary(
        predictions,
        labels,
        ENDPOINTS,
        SEEDS,
        repetitions=400,
        seed=44021,
    )


class RacerC4BootstrapDeterminismTests(unittest.TestCase):
    def test_bootstrap_is_invariant_to_python_hash_seed(self) -> None:
        outputs: list[str] = []
        for hash_seed in ("0", "1", "7", "41", "8675309"):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = hash_seed
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--probe"],
                cwd=ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            outputs.append(completed.stdout.strip())
        self.assertEqual(len(set(outputs)), 1, outputs)
        observed = json.loads(outputs[0])
        self.assertEqual(observed["bootstrap_seed"], 44021)
        self.assertEqual(observed["bootstrap_repetitions"], 400)
        self.assertEqual(
            observed["estimand"],
            "endpoint_seed_equal_mean_minimum_class_coverage_delta",
        )

    def test_bootstrap_does_not_traverse_sampled_endpoints_as_a_set(self) -> None:
        source = (RUNNER_DIR / "run_prospective_racer_c4.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("for endpoint in set(sampled_endpoints):", source)
        self.assertIn("for endpoint in dict.fromkeys(sampled_endpoints):", source)


class RacerC4InferenceRepairTests(unittest.TestCase):
    def test_repair_entrypoint_reuses_hash_bound_predictions_without_refitting(self) -> None:
        source = (
            RUNNER_DIR / "recompute_racer_c4_inference.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("run_panel(", source)
        self.assertNotIn("make_models(", source)
        self.assertNotIn(".fit(", source)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sealed = root / "sealed"
            source_root = root / "sources"
            output = root / "repair"
            sealed.mkdir()
            source_root.mkdir()

            lock = yaml.safe_load(runner.DEFAULT_LOCK.read_text(encoding="utf-8"))
            endpoint_order = list(lock["endpoint_order"])
            primary_endpoints = list(lock["primary_endpoints"])
            seeds = list(lock["roles"]["prospective_seeds"])
            sample_ids = [f"sample_{position:02d}" for position in range(20)]

            label_path = source_root / "synthetic_final_labels.csv"
            with label_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(["sample_id", *endpoint_order])
                for position, sample_id in enumerate(sample_ids):
                    writer.writerow(
                        [
                            sample_id,
                            *[
                                int((position + endpoint_order.index(endpoint)) % 2)
                                for endpoint in endpoint_order
                            ],
                        ]
                    )
            label_sha = hashlib.sha256(label_path.read_bytes()).hexdigest()
            lock["data_sources"]["final_epa_labels"]["filename"] = label_path.name
            lock["data_sources"]["final_epa_labels"]["sha256"] = label_sha
            lock_path = root / "lock.yaml"
            lock_path.write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")

            predictions: list[dict[str, object]] = []
            for endpoint_index, endpoint in enumerate(primary_endpoints):
                for seed_index, run_seed in enumerate(seeds):
                    for method in (runner.METHOD_BASELINE, runner.METHOD_TAME):
                        for position, sample_id in enumerate(sample_ids):
                            target = int((position + endpoint_order.index(endpoint)) % 2)
                            include_0 = int(
                                (position + endpoint_index + seed_index) % 4 != 0
                            )
                            include_1 = int(
                                (position + 2 * endpoint_index + seed_index) % 5 != 0
                            )
                            if method == runner.METHOD_TAME and target == 0:
                                include_0 = 1
                            predictions.append(
                                {
                                    "phase": "final",
                                    "endpoint": endpoint,
                                    "seed": run_seed,
                                    "method": method,
                                    "sample_id": sample_id,
                                    "structure_id": sample_id,
                                    "excluded_structure_overlap": 0,
                                    "excluded_structure_invalid": 0,
                                    "probability_1": 0.5,
                                    "include_0": include_0,
                                    "include_1": include_1,
                                    "active_transport_views": "",
                                    "failed_closed": 0,
                                }
                            )
            prediction_path = sealed / "sealed_final_predictions.csv"
            atomic_csv(prediction_path, predictions)
            parsed_labels = {
                sample_id: {
                    endpoint: float(
                        (position + endpoint_order.index(endpoint)) % 2
                    )
                    for endpoint in endpoint_order
                }
                for position, sample_id in enumerate(sample_ids)
            }
            original_metrics = runner.evaluate_predictions(predictions, parsed_labels)
            original_paired = runner.paired_rows(original_metrics)
            original_summary = runner.summarize_metrics(original_metrics)
            atomic_csv(sealed / "final_metrics.csv", original_metrics)
            atomic_csv(sealed / "final_paired.csv", original_paired)
            atomic_csv(sealed / "final_summary.csv", original_summary)
            promotion = {
                "status": "predictions_sealed_before_final_labels",
                "development_gate_passed": True,
                "final_labels_opened": False,
                "expected_final_label_sha256": label_sha,
                "sealed_predictions_sha256": sha256_file(prediction_path),
                "lock_sha256": sha256_file(lock_path),
                "git_head": "a" * 40,
                "prospective_seeds": seeds,
                "endpoint_scope": endpoint_order,
            }
            atomic_json(sealed / "promotion_record.json", promotion)
            original_inference = runner.hierarchical_bootstrap_primary(
                predictions,
                parsed_labels,
                primary_endpoints,
                seeds,
                repetitions=30,
                seed=44021,
            )
            primary_paired = [
                row
                for row in original_paired
                if row["endpoint"] in set(primary_endpoints)
                and row["method"] == runner.METHOD_TAME
            ]
            original_macro_delta = float(
                runner.np.mean(
                    [float(row["macro_csy_delta"]) for row in primary_paired]
                )
            )
            original_interpretation = (
                "coverage_gain_with_efficiency_noninferiority"
                if original_inference["estimate"] > 0.0
                and original_macro_delta
                >= float(lock["promotion_gate"]["macro_csy_noninferiority_margin"])
                else "negative_or_mixed_retain_without_retuning"
            )
            atomic_json(
                sealed / "final_report.json",
                {
                    "status": "complete_independent_final_epa_evaluation",
                    "primary_inference": original_inference,
                    "primary_mean_macro_csy_delta": original_macro_delta,
                    "result_interpretation": original_interpretation,
                },
            )
            atomic_json(
                sealed / "manifest.json",
                {
                    "final_metrics_sha256": sha256_file(sealed / "final_metrics.csv"),
                    "final_paired_sha256": sha256_file(sealed / "final_paired.csv"),
                    "final_summary_sha256": sha256_file(sealed / "final_summary.csv"),
                },
            )
            original_report_sha = sha256_file(sealed / "final_report.json")

            def short_bootstrap(*args: object, **kwargs: object) -> dict[str, object]:
                return runner.hierarchical_bootstrap_primary(
                    *args, **kwargs, repetitions=30
                )

            arguments = [
                "recompute_racer_c4_inference.py",
                "--lock",
                str(lock_path),
                "--source-root",
                str(source_root),
                "--sealed-output",
                str(sealed),
                "--output",
                str(output),
            ]
            with (
                mock.patch.object(recompute, "acquire_final_label_bytes", return_value=label_path),
                mock.patch.object(recompute, "git_head", return_value="b" * 40),
                mock.patch.object(
                    recompute,
                    "hierarchical_bootstrap_primary",
                    side_effect=short_bootstrap,
                ),
                mock.patch.object(sys, "argv", arguments),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(recompute.main(), 0)

            repaired_report = json.loads(
                (output / "final_report.json").read_text(encoding="utf-8")
            )
            repair_record = json.loads(
                (output / "repair_record.json").read_text(encoding="utf-8")
            )
            self.assertEqual(repaired_report["primary_cell_count"], 30)
            self.assertEqual(
                repaired_report["primary_inference"]["bootstrap_repetitions"], 30
            )
            self.assertFalse(repair_record["model_refit_performed"])
            self.assertFalse(repair_record["prediction_regeneration_performed"])
            self.assertEqual(
                repair_record["sealed_prediction_sha256"], sha256_file(prediction_path)
            )
            self.assertEqual(
                repair_record["original_report_sha256"], original_report_sha
            )
            self.assertEqual(
                sha256_file(sealed / "final_report.json"), original_report_sha
            )


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--probe":
        print(json.dumps(probe_result(), sort_keys=True, separators=(",", ":")))
    else:
        unittest.main()
