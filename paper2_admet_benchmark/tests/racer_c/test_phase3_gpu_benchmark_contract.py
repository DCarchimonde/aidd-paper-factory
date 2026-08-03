from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
P2 = ROOT / "paper2_admet_benchmark"
SCRIPT_DIR = P2 / "scripts" / "racer_c"
sys.path.insert(0, str(SCRIPT_DIR))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ENVIRONMENT = load_module(
    "racer_capture_gpu_environment", SCRIPT_DIR / "capture_gpu_environment.py"
)
PREPARE = load_module(
    "racer_prepare_gpu_benchmark", SCRIPT_DIR / "prepare_seed99_gpu_benchmark.py"
)
RUNNER = load_module(
    "racer_run_gpu_component_benchmark",
    SCRIPT_DIR / "run_seed99_gpu_component_benchmark.py",
)


class EnvironmentLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = P2 / "configs" / "racer_c" / "gpu_environment_lock.yaml"
        self.config = yaml.safe_load(self.path.read_text(encoding="utf-8"))

    def test_candidate_lock_is_exact_but_not_formally_frozen(self) -> None:
        self.assertEqual(
            self.config["status"],
            "candidate_pending_windows_rtx4060_verification",
        )
        self.assertEqual(self.config["platform"], "windows_amd64")
        self.assertEqual(
            self.config["gpu"]["device_name_contains"], "RTX 4060 Laptop GPU"
        )
        self.assertGreaterEqual(self.config["gpu"]["minimum_vram_gib"], 7.0)
        for value in self.config["packages"].values():
            self.assertRegex(str(value), r"^\d+\.\d+\.\d+$")
        self.assertEqual(
            self.config["molformer"]["revision"],
            "361063d0ad524ef77cf39b08469f6be770dc550f",
        )
        self.assertFalse(self.config["molformer"]["truncation"])
        self.assertEqual(
            self.config["molformer"]["overlength_action"],
            "fail_closed_before_any_fit",
        )

    def test_version_mismatch_fails_closed(self) -> None:
        failures = ENVIRONMENT.compare_versions(
            {"torch": "2.13.0", "chemprop": "2.3.0"},
            {"torch": "2.13.0+cu130", "chemprop": "2.2.4"},
        )
        self.assertEqual(failures, ["chemprop: expected 2.3.0, observed 2.2.4"])

    def test_windows_rtx4060_named_lock_matches_active_lock(self) -> None:
        windows = yaml.safe_load(
            (
                P2
                / "configs"
                / "racer_c"
                / "gpu_environment_windows_rtx4060.yaml"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(windows, self.config)

    def test_input_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            role = root / "role.csv"
            clean = root / "clean.csv"
            role.write_text("role\n", encoding="utf-8")
            clean.write_text("clean\n", encoding="utf-8")
            config = {
                "inputs": {
                    "role_input_sha256": "0" * 64,
                    "clean_sha256": "1" * 64,
                }
            }
            with self.assertRaisesRegex(RuntimeError, "input hash mismatch"):
                RUNNER.require_input_hashes(role, clean, config)

    def test_chemprop_prediction_column_is_not_hard_coded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            path.write_text(
                "smiles,structure_id,pred_0\nCC,s1,0.2\nCCC,s2,0.8\n",
                encoding="utf-8",
            )
            values, field = RUNNER.read_chemprop_probabilities(
                path,
                [
                    {"standardized_smiles": "CC", "structure_id": "s1"},
                    {"standardized_smiles": "CCC", "structure_id": "s2"},
                ],
            )
            self.assertEqual(field, "pred_0")
            self.assertEqual(values, [0.2, 0.8])


class BenchmarkPlanTests(unittest.TestCase):
    def test_group_folds_ignore_arbitrary_labels(self) -> None:
        rows = [
            {
                "structure_id": f"s{i}",
                "murcko_scaffold_id": f"g{i % 7}",
                "target": str(i % 2),
            }
            for i in range(70)
        ]
        changed = [dict(row, target=str((i * 3 + 1) % 2)) for i, row in enumerate(rows)]
        first = PREPARE.label_blind_group_folds(rows, "murcko_scaffold_id", 3, 99)
        second = PREPARE.label_blind_group_folds(changed, "murcko_scaffold_id", 3, 99)
        self.assertEqual(first, second)

    def test_committed_plan_is_development_only(self) -> None:
        path = (
            P2
            / "results"
            / "racer_c_phase3_preflight"
            / "seed99_gpu_benchmark_plan.json"
        )
        row = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(row["seed"], 99)
        self.assertEqual(row["trainer_label_roles"], ["dev"])
        self.assertFalse(row["policy_conformal_test_predictions_generated"])
        self.assertFalse(row["performance_metrics_permitted"])
        self.assertEqual(row["primary_endpoint_count"], 4)
        self.assertEqual(row["primary_endpoint_track_seed_cells"], 60)
        self.assertEqual(row["dmpnn_fit_jobs_per_endpoint_track_seed"], 9)
        self.assertEqual(len(row["jobs"]), 9)
        self.assertEqual(sum(job["stage"] == "outer_final" for job in row["jobs"]), 3)

    def test_chemprop_command_is_unweighted_seed99_gpu(self) -> None:
        config = yaml.safe_load(
            (P2 / "configs" / "racer_c" / "gpu_environment_lock.yaml").read_text(
                encoding="utf-8"
            )
        )
        train, predict = RUNNER.chemprop_commands(
            Path("train.csv"),
            Path("predict.csv"),
            Path("model"),
            Path("predictions.csv"),
            config,
        )
        self.assertIn("99", train)
        self.assertIn("gpu", train)
        self.assertNotIn("--class-balance", train)
        self.assertIn("--splits-column", train)
        self.assertIn("--model-paths", predict)


if __name__ == "__main__":
    unittest.main()
