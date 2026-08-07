from __future__ import annotations

import importlib.util
import json
import re
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
TOKEN_CONTRACT = load_module(
    "racer_molformer_token_contract", SCRIPT_DIR / "molformer_token_contract.py"
)
FREEZE_REVIEW = load_module(
    "racer_prepare_formal_freeze_review",
    SCRIPT_DIR / "prepare_formal_freeze_review.py",
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
        self.assertEqual(str(self.config["lock_version"]), "0.4")
        self.assertEqual(
            self.config["gpu"]["device_name_contains"], "RTX 4060 Laptop GPU"
        )
        self.assertGreaterEqual(self.config["gpu"]["minimum_vram_gib"], 7.0)
        self.assertEqual(self.config["gpu"]["minimum_driver_version"], "580.00")
        for value in self.config["packages"].values():
            self.assertRegex(str(value), r"^\d+\.\d+\.\d+$")
        self.assertEqual(
            self.config["molformer"]["revision"],
            "361063d0ad524ef77cf39b08469f6be770dc550f",
        )
        self.assertFalse(self.config["molformer"]["truncation"])
        self.assertEqual(
            self.config["molformer"]["max_tokens_including_special_tokens"], 202
        )
        self.assertEqual(
            self.config["molformer"]["tokenizer_json_sha256"],
            "3df1f2219653c44fac9fa03b7f788b372eb2544ecc176737bb9aca8411b471a5",
        )
        self.assertEqual(
            self.config["molformer"]["overlength_action"],
            "exclude_before_role_assignment_and_all_component_fits",
        )
        self.assertEqual(self.config["gpu"]["count"], 1)
        self.assertEqual(self.config["chemprop"]["accelerator"], "gpu")
        self.assertEqual(self.config["chemprop"]["devices"], "1")

    def test_token_domain_filter_is_exact_and_label_blind(self) -> None:
        role = [
            {
                "endpoint": "toy",
                "structure_id": "short",
                "target": "0",
                "murcko_scaffold_id": "g1",
                "similarity_cluster_id": "c1",
            },
            {
                "endpoint": "toy",
                "structure_id": "long",
                "target": "1",
                "murcko_scaffold_id": "g2",
                "similarity_cluster_id": "c2",
            },
        ]
        clean = [
            {
                "structure_id": "short",
                "source_record_id": "1",
                "standardized_smiles": "CC",
            },
            {
                "structure_id": "long",
                "source_record_id": "2",
                "standardized_smiles": "C" * 201,
            },
        ]
        first_role, first_clean, first_report = (
            TOKEN_CONTRACT.filter_model_eligible_rows(role, clean, self.config)
        )
        changed = [dict(row, target=str(1 - int(row["target"]))) for row in role]
        second_role, second_clean, second_report = (
            TOKEN_CONTRACT.filter_model_eligible_rows(changed, clean, self.config)
        )
        self.assertEqual([row["structure_id"] for row in first_role], ["short"])
        self.assertEqual([row["structure_id"] for row in first_clean], ["short"])
        self.assertEqual([row["structure_id"] for row in second_role], ["short"])
        self.assertEqual([row["structure_id"] for row in second_clean], ["short"])
        self.assertEqual(first_report, second_report)
        self.assertEqual(first_report["excluded_n"], 1)
        self.assertEqual(first_report["source_max_tokens_observed"], 203)
        self.assertEqual(first_report["eligible_max_tokens_observed"], 4)
        self.assertFalse(first_report["selection_uses_labels"])

    def test_runtime_tokenizer_mismatch_fails_closed(self) -> None:
        class FakeTokenizer:
            def __call__(self, smiles, **kwargs):
                return {"input_ids": [[0, 9, 1] for _ in smiles]}

        with self.assertRaisesRegex(RuntimeError, "runtime tokenizer differs"):
            TOKEN_CONTRACT.verify_runtime_tokenizer(
                [{"structure_id": "s1", "standardized_smiles": "CC"}],
                FakeTokenizer(),
                "standardized_smiles",
            )

    def test_version_mismatch_fails_closed(self) -> None:
        failures = ENVIRONMENT.compare_versions(
            {"torch": "2.13.0", "chemprop": "2.3.0"},
            {"torch": "2.13.0+cu130", "chemprop": "2.2.4"},
        )
        self.assertEqual(failures, ["chemprop: expected 2.3.0, observed 2.2.4"])

    def test_nvidia_driver_versions_are_compared_numerically(self) -> None:
        self.assertLess(
            ENVIRONMENT.numeric_version("576.80"),
            ENVIRONMENT.numeric_version("580.00"),
        )
        self.assertGreaterEqual(
            ENVIRONMENT.numeric_version("580.126.20"),
            ENVIRONMENT.numeric_version("580.00"),
        )
        with self.assertRaisesRegex(ValueError, "invalid numeric version"):
            ENVIRONMENT.numeric_version("not-a-driver")

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

    def test_python311_requirements_match_environment_lock(self) -> None:
        requirements_path = (
            P2 / "environment" / "racer_c_gpu_requirements.txt"
        )
        requirement_versions = {}
        for line in requirements_path.read_text(encoding="utf-8").splitlines():
            match = re.fullmatch(r"([A-Za-z0-9_-]+)==([0-9]+(?:\.[0-9]+)+)", line)
            if match:
                requirement_versions[match.group(1).lower()] = match.group(2)

        package_names = {
            "chemprop": "chemprop",
            "transformers": "transformers",
            "rdkit": "rdkit",
            "numpy": "numpy",
            "scipy": "scipy",
            "pandas": "pandas",
            "scikit-learn": "scikit-learn",
            "xgboost": "xgboost",
            "pyyaml": "pyyaml",
        }
        expected = {
            requirement: str(self.config["packages"][lock_name])
            for requirement, lock_name in package_names.items()
        }
        self.assertEqual(requirement_versions, expected)
        self.assertEqual(self.config["python"], "3.11.13")
        self.assertEqual(requirement_versions["scipy"], "1.17.1")
        self.assertEqual(requirement_versions["xgboost"], "3.2.0")

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
    def test_windows_one_click_runner_preserves_scientific_gates(self) -> None:
        path = SCRIPT_DIR / "run_racer_c_pipeline.ps1"
        source = path.read_text(encoding="utf-8")
        self.assertIn(
            '[ValidateSet("Validate", "Benchmark", "FreezeReview", "Full")]',
            source,
        )
        self.assertIn("capture_gpu_environment.py", source)
        self.assertIn("Invoke-WebRequest", source)
        self.assertIn("prepare_tox21_challenge.py", source)
        self.assertIn("build_similarity_clusters.py", source)
        self.assertIn("024a3ae2690bcd4a593e6e0b10b455470b9bcb1d8f299dd36f220a250181517b", source)
        self.assertIn("2a6217e66e3300e437d11fad68637b291526abc610c091effbbef4955d7d54a0", source)
        self.assertIn("edbe26eeee9cb9aa188e941f5884967b1775b3fe36d92349656a42b5b6bee900", source)
        self.assertIn("prepare_seed99_gpu_benchmark.py", source)
        self.assertIn("run_seed99_gpu_component_benchmark.py", source)
        self.assertIn("$Record.config_sha256", source)
        self.assertIn("$Record.script_sha256", source)
        self.assertIn("draft_pre_freeze", source)
        self.assertIn("run_confirmatory_racer_c.py", source)
        self.assertLess(
            source.index("draft_pre_freeze"),
            source.index("run_confirmatory_racer_c.py"),
        )

    def test_freeze_review_is_prediction_free_and_covers_all_primary_cells(self) -> None:
        self.assertEqual(
            FREEZE_REVIEW.PRIMARY_ENDPOINTS,
            (
                "Tox21_NR_AhR",
                "Tox21_NR_ER",
                "Tox21_SR_ARE",
                "Tox21_SR_MMP",
            ),
        )
        self.assertEqual(
            len(FREEZE_REVIEW.PRIMARY_ENDPOINTS)
            * len(FREEZE_REVIEW.TRACKS)
            * len(FREEZE_REVIEW.SEEDS),
            60,
        )
        semantics = FREEZE_REVIEW.load_primary_semantics(
            P2 / "protocols" / "endpoint_candidate_manifest.csv"
        )
        self.assertEqual(set(semantics), set(FREEZE_REVIEW.PRIMARY_ENDPOINTS))
        self.assertTrue(all(row["critical_class"] == "1" for row in semantics.values()))
        source = (SCRIPT_DIR / "run_racer_c_pipeline.ps1").read_text(encoding="utf-8")
        self.assertIn(
            '[ValidateSet("Validate", "Benchmark", "FreezeReview", "Full")]',
            source,
        )
        self.assertIn("prepare_formal_freeze_review.py", source)
        self.assertIn("scientific_predictions_generated", source)

    def test_freeze_review_recovers_all_primary_inputs_before_review(self) -> None:
        source = (SCRIPT_DIR / "run_racer_c_pipeline.ps1").read_text(encoding="utf-8")
        for endpoint in (
            "Tox21_NR_AhR",
            "Tox21_NR_ER",
            "Tox21_SR_ARE",
            "Tox21_SR_MMP",
        ):
            self.assertIn(f'"{endpoint}"', source)
        self.assertIn('$FreezePrimaryEndpoints -join ","', source)
        self.assertIn("--processed-dir $StagedProcessedDir", source)
        self.assertIn("--manifest-dir $StagedManifestDir", source)
        self.assertIn("replaced_input_backup", source)
        self.assertIn("rejections_byte_sha256", source)
        self.assertIn("role_input_byte_sha256", source)
        self.assertIn("similarity_cluster_status", source)

        staged_role_gate = (
            "Assert-LockedFile -Path $StagedRolePath -ExpectedSha256 $($Record.RoleSha256)"
        )
        live_role_write = (
            "Copy-Item -LiteralPath $StagedRolePath -Destination $Record.RolePath -Force"
        )
        self.assertIn(staged_role_gate, source)
        self.assertIn(live_role_write, source)
        self.assertLess(source.index(staged_role_gate), source.index(live_role_write))

    def test_freeze_review_does_not_accept_unclustered_role_hashes(self) -> None:
        source = (SCRIPT_DIR / "run_racer_c_pipeline.ps1").read_text(encoding="utf-8")
        self.assertNotIn(
            "41db4cbf9f1f4d704404950916e9d10d897f99b78b15ae2e183a3bedd31597ba",
            source,
        )
        self.assertIn(
            "461c99d4a658c5e1eaee3ad4159761e8ef2bc2fa9041386be8b68dff4461178b",
            (
                P2
                / "data"
                / "manifests"
                / "racer_c"
                / "Tox21_NR_AhR_cleaning.json"
            ).read_text(encoding="utf-8"),
        )

    def test_freeze_review_rejects_benchmark_with_scientific_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "lock.yaml"
            script_path = root / "runner.py"
            benchmark_path = root / "benchmark.json"
            config = {
                "platform": "windows_amd64",
                "packages": {"chemprop": "2.3.0"},
                "gpu": {"count": 1, "torch_cuda_build": "13.0"},
            }
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            script_path.write_text("# frozen runner\n", encoding="utf-8")
            train_seconds = 30.0
            row = {
                "status": "pass_gpu_component_benchmark",
                "seed": 99,
                "endpoint": "Tox21_NR_ER",
                "trainer_label_roles": ["dev"],
                "performance_metrics_computed": True,
                "policy_conformal_test_predictions_generated": False,
                "config_sha256": FREEZE_REVIEW.sha256_file(config_path),
                "script_sha256": FREEZE_REVIEW.sha256_file(script_path),
                "dev_n": 2926,
                "lineage_prediction_count": 976,
                "environment": {
                    "status": "pass",
                    "failures": [],
                    "platform": "windows_amd64",
                    "packages": {"chemprop": "2.3.0"},
                    "pip_freeze_sha256": "a" * 64,
                    "torch": {
                        "cuda_available": True,
                        "device_count": 1,
                        "cuda_build": "13.0",
                    },
                },
                "chemprop": {
                    "status": "pass_component_timing",
                    "predict_n": 976,
                    "finite_probability_count": 976,
                    "train_seconds": train_seconds,
                    "predict_seconds": 10.0,
                    "train_peak_gpu_memory_mib": 2000,
                },
                "molformer": {"n": 2926, "seconds": 7.0},
                "model_eligibility": {
                    "selection_uses_labels": False,
                    "source_n": 5855,
                    "eligible_n": 5852,
                    "excluded_n": 3,
                },
                "planning_projection": {
                    "projected_primary_dmpnn_gpu_hours": train_seconds * 6 * 60 / 3600,
                    "projected_with_20pct_rerun_reserve_gpu_hours": train_seconds
                    * 6
                    * 60
                    * 1.2
                    / 3600,
                },
            }
            benchmark_path.write_text(json.dumps(row), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "performance metrics were computed"):
                FREEZE_REVIEW.validate_seed99_benchmark(
                    benchmark_path, config_path, script_path, config
                )
            row["performance_metrics_computed"] = False
            benchmark_path.write_text(json.dumps(row), encoding="utf-8")
            reviewed = FREEZE_REVIEW.validate_seed99_benchmark(
                benchmark_path, config_path, script_path, config
            )
            self.assertEqual(reviewed["status"], "pass")
            self.assertEqual(reviewed["measured_chemprop_train_seconds"], train_seconds)

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
        self.assertEqual(train[train.index("--devices") + 1], "1")
        self.assertEqual(predict[predict.index("--devices") + 1], "1")
        self.assertNotIn("--class-balance", train)
        self.assertIn("--splits-column", train)
        self.assertIn("--model-paths", predict)

    def test_chemprop_zero_devices_fails_before_launch(self) -> None:
        config = yaml.safe_load(
            (P2 / "configs" / "racer_c" / "gpu_environment_lock.yaml").read_text(
                encoding="utf-8"
            )
        )
        config["chemprop"]["devices"] = "0"
        with self.assertRaisesRegex(ValueError, "positive device count"):
            RUNNER.chemprop_commands(
                Path("train.csv"),
                Path("predict.csv"),
                Path("model"),
                Path("predictions.csv"),
                config,
            )


if __name__ == "__main__":
    unittest.main()
