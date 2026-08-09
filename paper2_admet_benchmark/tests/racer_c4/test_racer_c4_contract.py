from __future__ import annotations

import hashlib
import json
import re
import subprocess
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
P2 = ROOT / "paper2_admet_benchmark"
LOCK_PATH = P2 / "configs" / "racer_c4" / "prospective_lock_v1.yaml"


class RacerC4ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lock = yaml.safe_load(LOCK_PATH.read_text(encoding="utf-8"))

    def test_final_source_is_hash_locked_and_sealed(self) -> None:
        self.assertEqual(
            self.lock["lock_status"],
            "candidate_frozen_before_final_epa_label_open",
        )
        labels = self.lock["data_sources"]["final_epa_labels"]
        self.assertRegex(labels["sha256"], r"^[0-9a-f]{64}$")
        self.assertIn("sealed", labels["role"])
        self.assertEqual(
            self.lock["promotion_gate"]["failed_gate_action"],
            "stop_before_final_label_download_or_parse",
        )

    def test_prospective_seeds_are_fresh_and_v1_panel_is_not_confirmatory(self) -> None:
        self.assertEqual(self.lock["roles"]["development_seeds"], [101, 102, 103, 104, 105])
        self.assertEqual(self.lock["roles"]["prospective_seeds"], [211, 212, 213, 214, 215])
        self.assertTrue(
            self.lock["roles"]["prospective_seeds_were_not_used_in_architecture_development"]
        )
        self.assertFalse(
            self.lock["development_decisions"]["known_v1_panel_used_for_prospective_claim"]
        )
        self.assertEqual(
            self.lock["roles"]["invalid_external_structure_action"],
            "retain_identity_full_set_and_exclude_from_domain_fit_and_evaluation",
        )

    def test_algorithm_has_structural_safety_contract(self) -> None:
        envelope = self.lock["envelope"]
        self.assertEqual(
            self.lock["transport"]["views"],
            ["physchem_descriptors", "component_and_stack_logits"],
        )
        self.assertEqual(self.lock["transport"]["diagnostic_views"], ["ecfp_bits"])
        self.assertEqual(envelope["protected_labels"], [0])
        self.assertEqual(envelope["quorum"], "all")
        self.assertEqual(envelope["minimum_active_views"], 2)
        self.assertTrue(envelope["always_contains_ordinary_mondrian_baseline"])
        self.assertEqual(envelope["baseline_empty_action"], "full_set")
        self.assertEqual(
            envelope["insufficient_transport_views_action"],
            "ordinary_baseline_with_empty_to_full",
        )
        self.assertEqual(envelope["new_singleton_creation"], "prohibited")

    def test_runner_binds_predictions_before_acquiring_final_labels(self) -> None:
        source = (
            P2 / "scripts" / "racer_c4" / "run_prospective_racer_c4.py"
        ).read_text(encoding="utf-8")
        promotion = source.index('promotion_path = args.output / "promotion_record.json"')
        write_promotion = source.index("atomic_json(promotion_path, promotion)", promotion)
        acquire_labels = source.index("acquire_final_label_bytes(lock, args.source_root)")
        parse_labels = source.index("open_final_labels_after_promotion(")
        self.assertLess(promotion, write_promotion)
        self.assertLess(write_promotion, acquire_labels)
        self.assertLess(acquire_labels, parse_labels)
        self.assertNotRegex(
            source[:write_promotion],
            re.compile(r"parse_final_label_text|open_final_labels_after_promotion\("),
        )
        run_cell_signature = source[
            source.index("def run_cell(") : source.index(") -> tuple", source.index("def run_cell("))
        ]
        self.assertNotIn("target_labels", run_cell_signature)
        self.assertNotIn("target_targets", run_cell_signature)

    def test_one_click_wrapper_defaults_to_known_user_environment(self) -> None:
        wrapper = (
            P2 / "scripts" / "racer_c4" / "run_racer_c4_overnight.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn('[string]$CondaEnv = "aidd_paper"', wrapper)
        self.assertIn("SetThreadExecutionState", wrapper)
        self.assertIn("--mode full --scope $Scope", wrapper)

    def test_one_click_wrapper_repairs_and_rechecks_locked_rdkit(self) -> None:
        wrapper = (
            P2 / "scripts" / "racer_c4" / "run_racer_c4_overnight.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn('$RequiredRdkitRuntime = "2026.03.4"', wrapper)
        self.assertIn('$RequiredRdkitDistribution = "2026.3.4"', wrapper)
        self.assertIn('"rdkit==$RequiredRdkitDistribution"', wrapper)
        self.assertIn("--no-deps", wrapper)
        self.assertIn('"--only-binary=:all:"', wrapper)
        preflight = wrapper.index("$RdkitBefore = Get-RdkitRuntimeVersion")
        repair = wrapper.index("Install-LockedRdkit", preflight)
        recheck = wrapper.index("$RdkitAfter = Get-RdkitRuntimeVersion", repair)
        tests = wrapper.index('Write-Host "`n==> Contract and numerical tests"')
        self.assertLess(preflight, repair)
        self.assertLess(repair, recheck)
        self.assertLess(recheck, tests)

    def test_committed_final_report_matches_sealed_integrity_record(self) -> None:
        result = P2 / "results" / "racer_c4_independent_final"
        report_path = result / "final_report.json"
        promotion_path = result / "promotion_record.json"
        manifest = json.loads(
            (result / "integrity_manifest.json").read_text(encoding="utf-8")
        )
        # A Windows checkout may materialize text blobs with CRLF. The sealed
        # integrity record binds the exact bytes committed to Git, not the
        # platform-specific working-tree representation.
        def committed_digest(path: Path) -> str:
            repository_path = path.relative_to(ROOT).as_posix()
            committed_bytes = subprocess.run(
                ["git", "cat-file", "blob", f"HEAD:{repository_path}"],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout
            return hashlib.sha256(committed_bytes).hexdigest()

        self.assertEqual(
            committed_digest(report_path), manifest["final_report_sha256"]
        )
        self.assertEqual(
            committed_digest(promotion_path), manifest["promotion_record_sha256"]
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
        self.assertTrue(report["predictions_sealed_before_final_labels"])
        self.assertTrue(report["final_labels_opened_after_promotion"])
        self.assertFalse(report["scientific_superiority_claim_authorized"])
        self.assertEqual(
            promotion["sealed_predictions_sha256"],
            manifest["sealed_predictions_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
