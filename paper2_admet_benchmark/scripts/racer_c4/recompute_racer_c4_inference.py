from __future__ import annotations

"""Recompute only RACER-C4 final inference from already sealed predictions.

This repair path never fits a model or rewrites the original sealed output.  It
verifies the prediction, lock, promotion, and final-label hashes before writing
a separate deterministic inference package.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]
P2 = ROOT / "paper2_admet_benchmark"
RUNNER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNNER_DIR))

from racer_c4_io import (  # noqa: E402
    acquire_final_label_bytes,
    atomic_csv,
    atomic_json,
    open_final_labels_after_promotion,
    sha256_file,
)
from run_prospective_racer_c4 import (  # noqa: E402
    DEFAULT_LOCK,
    METHOD_TAME,
    evaluate_predictions,
    hierarchical_bootstrap_primary,
    paired_rows,
    read_csv,
    summarize_metrics,
    validate_lock,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute deterministic RACER-C4 inference without refitting"
    )
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument(
        "--source-root", type=Path, default=ROOT / ".local" / "racer_c4_sources"
    )
    parser.add_argument(
        "--sealed-output", type=Path, default=ROOT / ".local" / "racer_c4_results"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".local" / "racer_c4_deterministic_inference",
    )
    return parser.parse_args()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    args = parse_args()
    lock = yaml.safe_load(args.lock.read_text(encoding="utf-8"))
    validate_lock(lock)

    prediction_path = args.sealed_output / "sealed_final_predictions.csv"
    promotion_path = args.sealed_output / "promotion_record.json"
    original_report_path = args.sealed_output / "final_report.json"
    original_manifest_path = args.sealed_output / "manifest.json"
    for required in (
        prediction_path,
        promotion_path,
        original_report_path,
        original_manifest_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(f"required completed-run artifact is missing: {required}")

    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    original_report = json.loads(original_report_path.read_text(encoding="utf-8"))
    original_manifest = json.loads(
        original_manifest_path.read_text(encoding="utf-8")
    )
    expected_prediction_sha = str(promotion.get("sealed_predictions_sha256", ""))
    observed_prediction_sha = sha256_file(prediction_path)
    if observed_prediction_sha != expected_prediction_sha:
        raise RuntimeError(
            "sealed prediction SHA256 mismatch: "
            f"expected={expected_prediction_sha} observed={observed_prediction_sha}"
        )
    expected_lock_sha = sha256_file(args.lock)
    if promotion.get("lock_sha256") != expected_lock_sha:
        raise RuntimeError("promotion record does not bind the current frozen lock")
    if list(promotion.get("prospective_seeds", [])) != list(
        lock["roles"]["prospective_seeds"]
    ):
        raise RuntimeError("promotion record prospective seeds differ from the lock")
    if not set(lock["primary_endpoints"]).issubset(
        set(promotion.get("endpoint_scope", []))
    ):
        raise RuntimeError("promotion record does not cover every primary endpoint")

    expected_label_sha = str(lock["data_sources"]["final_epa_labels"]["sha256"])
    final_label_path = acquire_final_label_bytes(lock, args.source_root)
    final_labels = open_final_labels_after_promotion(
        final_label_path,
        promotion_path,
        expected_label_sha,
        list(lock["endpoint_order"]),
    )
    predictions = read_csv(prediction_path)
    prediction_ids = {str(row["sample_id"]) for row in predictions}
    if prediction_ids != set(final_labels):
        raise RuntimeError(
            "sealed prediction/final-label identity mismatch: "
            f"missing={sorted(prediction_ids - set(final_labels))[:10]} "
            f"extra={sorted(set(final_labels) - prediction_ids)[:10]}"
        )

    final_metrics = evaluate_predictions(predictions, final_labels)
    final_paired = paired_rows(final_metrics)
    primary = set(lock["primary_endpoints"])
    primary_paired = [
        row
        for row in final_paired
        if row["endpoint"] in primary and row["method"] == METHOD_TAME
    ]
    primary_inference = hierarchical_bootstrap_primary(
        predictions,
        final_labels,
        list(lock["primary_endpoints"]),
        list(lock["roles"]["prospective_seeds"]),
    )
    mean_macro_delta = float(
        np.mean([float(row["macro_csy_delta"]) for row in primary_paired])
    )
    margin = float(lock["promotion_gate"]["macro_csy_noninferiority_margin"])
    report = {
        "status": "complete_independent_final_epa_evaluation",
        "algorithm": lock["algorithm"],
        "development_gate_passed": True,
        "predictions_sealed_before_final_labels": True,
        "final_labels_opened_after_promotion": True,
        "final_label_sha256": sha256_file(final_label_path),
        "primary_cell_count": len(primary_paired),
        "primary_inference": primary_inference,
        "primary_mean_macro_csy_delta": mean_macro_delta,
        "macro_csy_noninferiority_margin": margin,
        "result_interpretation": (
            "coverage_gain_with_efficiency_noninferiority"
            if primary_inference["estimate"] > 0.0 and mean_macro_delta >= margin
            else "negative_or_mixed_retain_without_retuning"
        ),
        "scientific_superiority_claim_authorized": False,
        "arbitrary_shift_coverage_guarantee_claim_authorized": False,
    }
    if primary_inference["estimate"] != original_report["primary_inference"]["estimate"]:
        raise RuntimeError("deterministic repair changed the primary point estimate")
    if mean_macro_delta != original_report["primary_mean_macro_csy_delta"]:
        raise RuntimeError("deterministic repair changed the MacroCSY estimate")
    if report["result_interpretation"] != original_report["result_interpretation"]:
        raise RuntimeError("deterministic repair changed the frozen interpretation")

    args.output.mkdir(parents=True, exist_ok=True)
    atomic_csv(args.output / "final_metrics.csv", final_metrics)
    atomic_csv(args.output / "final_paired.csv", final_paired)
    atomic_csv(args.output / "final_summary.csv", summarize_metrics(final_metrics))
    atomic_json(args.output / "final_report.json", report)
    unchanged_artifacts = {
        "final_metrics_sha256": args.output / "final_metrics.csv",
        "final_paired_sha256": args.output / "final_paired.csv",
        "final_summary_sha256": args.output / "final_summary.csv",
    }
    for manifest_field, repaired_path in unchanged_artifacts.items():
        observed = sha256_file(repaired_path)
        expected = str(original_manifest.get(manifest_field, ""))
        if observed != expected:
            raise RuntimeError(
                f"deterministic repair changed {manifest_field}: "
                f"expected={expected} observed={observed}"
            )
    repair_record = {
        "status": "complete_deterministic_inference_repair",
        "repair_scope": "bootstrap_rng_endpoint_traversal_only",
        "model_refit_performed": False,
        "prediction_regeneration_performed": False,
        "labels_or_predictions_changed": False,
        "sealed_prediction_sha256": observed_prediction_sha,
        "promotion_record_sha256": sha256_file(promotion_path),
        "final_label_sha256": sha256_file(final_label_path),
        "frozen_lock_sha256": expected_lock_sha,
        "original_report_sha256": sha256_file(original_report_path),
        "original_manifest_sha256": sha256_file(original_manifest_path),
        "repaired_report_sha256": sha256_file(args.output / "final_report.json"),
        "unchanged_scientific_artifact_hashes_verified": {
            field: sha256_file(path) for field, path in unchanged_artifacts.items()
        },
        "prediction_git_head": str(promotion.get("git_head", "")),
        "inference_repair_git_head": git_head(),
        "original_primary_inference": original_report.get("primary_inference"),
        "repaired_primary_inference": primary_inference,
    }
    atomic_json(args.output / "repair_record.json", repair_record)
    print(json.dumps({"report": report, "repair": repair_record}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
