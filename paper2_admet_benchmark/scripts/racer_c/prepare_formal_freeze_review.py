from __future__ import annotations

"""Create the prediction-free, multi-endpoint RACER-C freeze-review record.

This gate deliberately runs before the protocol tag.  It verifies the successful
seed-99 engineering benchmark, applies the immutable MoLFormer token-domain rule
to every primary endpoint before role allocation, and repeats the selected
count-only feasibility audit.  It never fits a predictor or reads a policy,
conformal, or test prediction.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Mapping

import yaml

from molformer_token_contract import (
    filter_model_eligible_rows,
    verify_runtime_tokenizer,
)
from role_feasibility import TRACK_GROUP_COLUMN, audit_one, read_csv, validate_role_input


ROOT = Path(__file__).resolve().parents[3]
P2 = ROOT / "paper2_admet_benchmark"
PRIMARY_ENDPOINTS = (
    "Tox21_NR_AhR",
    "Tox21_NR_ER",
    "Tox21_SR_ARE",
    "Tox21_SR_MMP",
)
FRACTIONS = {"dev": 0.50, "policy": 0.20, "conformal": 0.15, "test": 0.15}
ALLOCATION_ID = "50_20_15_15"
TRACKS = ("random_grouped", "strict_scaffold", "similarity_cluster")
SEEDS = (101, 102, 103, 104, 105)
EXPECTED_BENCHMARK_ENDPOINT = "Tox21_NR_ER"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_clean(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"clean endpoint is empty: {path}")
    return rows


def load_primary_semantics(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    by_endpoint = {row["endpoint"]: row for row in rows}
    if len(by_endpoint) != len(rows):
        raise ValueError("endpoint candidate manifest contains duplicate endpoints")
    missing = set(PRIMARY_ENDPOINTS) - set(by_endpoint)
    if missing:
        raise ValueError(f"endpoint candidate manifest is missing: {sorted(missing)}")
    selected: dict[str, dict[str, str]] = {}
    for endpoint in PRIMARY_ENDPOINTS:
        row = by_endpoint[endpoint]
        if row["eligibility_status"] != "primary_candidate":
            raise RuntimeError(f"{endpoint} is no longer a primary candidate")
        if row["data_source"] != "NCATS_Tox21_2014":
            raise RuntimeError(f"{endpoint} is outside the frozen Tox21 source family")
        if row["critical_class"] not in {"0", "1"}:
            raise RuntimeError(f"{endpoint} has no binary critical-class decision")
        if not row["critical_class_reason"] or not row["label_definition"]:
            raise RuntimeError(f"{endpoint} lacks frozen label semantics")
        selected[endpoint] = row
    return selected


def require_exact_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"{label} SHA256 mismatch: expected={expected} observed={observed} path={path}"
        )
    return observed


def validate_seed99_benchmark(
    benchmark_path: Path,
    config_path: Path,
    benchmark_script_path: Path,
    config: Mapping[str, object],
) -> dict[str, object]:
    if not benchmark_path.is_file():
        raise FileNotFoundError(f"missing completed seed-99 benchmark: {benchmark_path}")
    row = json.loads(benchmark_path.read_text(encoding="utf-8"))
    failures: list[str] = []

    def expect(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    expect(row.get("status") == "pass_gpu_component_benchmark", "benchmark status is not pass")
    expect(row.get("seed") == 99, "benchmark seed is not 99")
    expect(row.get("endpoint") == EXPECTED_BENCHMARK_ENDPOINT, "unexpected benchmark endpoint")
    expect(row.get("trainer_label_roles") == ["dev"], "trainer accessed a non-development role")
    expect(row.get("performance_metrics_computed") is False, "performance metrics were computed")
    expect(
        row.get("policy_conformal_test_predictions_generated") is False,
        "policy/conformal/test predictions were generated",
    )
    expect(row.get("config_sha256") == sha256_file(config_path), "benchmark/config byte hash mismatch")
    expect(
        row.get("script_sha256") == sha256_file(benchmark_script_path),
        "benchmark/runner byte hash mismatch",
    )
    environment = row.get("environment", {})
    expect(environment.get("status") == "pass", "environment audit did not pass")
    expect(environment.get("failures") == [], "environment audit contains failures")
    expect(environment.get("platform") == config["platform"], "platform differs from lock")
    packages = environment.get("packages", {})
    for name, version in config["packages"].items():
        observed_version = str(packages.get(name))
        expected_version = str(version)
        if name == "torch":
            observed_version = observed_version.split("+", 1)[0]
        expect(observed_version == expected_version, f"package mismatch: {name}")
    torch = environment.get("torch", {})
    expect(torch.get("cuda_available") is True, "CUDA was unavailable")
    expect(int(torch.get("device_count", 0)) == int(config["gpu"]["count"]), "GPU count mismatch")
    expect(str(torch.get("cuda_build")) == str(config["gpu"]["torch_cuda_build"]), "CUDA build mismatch")
    chemprop = row.get("chemprop", {})
    expect(chemprop.get("status") == "pass_component_timing", "Chemprop component did not pass")
    expect(
        chemprop.get("finite_probability_count") == chemprop.get("predict_n"),
        "Chemprop finite-probability count mismatch",
    )
    expect(row.get("lineage_prediction_count") == chemprop.get("predict_n"), "lineage count mismatch")
    molformer = row.get("molformer", {})
    expect(molformer.get("n") == row.get("dev_n"), "MoLFormer/dev row-count mismatch")
    eligibility = row.get("model_eligibility", {})
    expect(eligibility.get("selection_uses_labels") is False, "token eligibility used labels")
    expect(eligibility.get("source_n") == 5855, "unexpected NR-ER source count")
    expect(eligibility.get("eligible_n") == 5852, "unexpected NR-ER eligible count")
    expect(eligibility.get("excluded_n") == 3, "unexpected NR-ER exclusion count")
    projection = row.get("planning_projection", {})
    train_seconds = float(chemprop.get("train_seconds", math.nan))
    expected_hours = train_seconds * 6.0 * 60.0 / 3600.0
    expect(
        math.isclose(
            float(projection.get("projected_primary_dmpnn_gpu_hours", math.nan)),
            expected_hours,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        "engineering projection is inconsistent with measured training time",
    )
    if failures:
        raise RuntimeError("seed-99 benchmark review failed: " + "; ".join(failures))
    return {
        "status": "pass",
        "benchmark_file_sha256": sha256_file(benchmark_path),
        "config_sha256": row["config_sha256"],
        "script_sha256": row["script_sha256"],
        "environment_pip_freeze_sha256": environment["pip_freeze_sha256"],
        "measured_chemprop_train_seconds": train_seconds,
        "measured_chemprop_predict_seconds": float(chemprop["predict_seconds"]),
        "measured_chemprop_train_peak_gpu_memory_mib": chemprop["train_peak_gpu_memory_mib"],
        "measured_molformer_seconds": float(molformer["seconds"]),
        "projected_primary_dmpnn_gpu_hours": float(
            projection["projected_primary_dmpnn_gpu_hours"]
        ),
        "projected_with_20pct_rerun_reserve_gpu_hours": float(
            projection["projected_with_20pct_rerun_reserve_gpu_hours"]
        ),
        "scientific_interpretation": "engineering timing and lineage only; excluded from scientific outcomes",
    }


def endpoint_review(
    endpoint: str,
    processed_dir: Path,
    manifest_dir: Path,
    config: Mapping[str, object],
    tokenizer: object,
    minimum_retention: float,
    alpha: float,
    semantics: Mapping[str, str],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    clean_path = processed_dir / f"{endpoint}_clean.csv"
    role_path = processed_dir / "role_inputs" / f"{endpoint}_role_input.csv"
    manifest_path = manifest_dir / f"{endpoint}_cleaning.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing cleaning manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require_exact_hash(clean_path, manifest["cleaned_byte_sha256"], f"{endpoint} clean input")
    require_exact_hash(role_path, manifest["role_input_byte_sha256"], f"{endpoint} role input")

    role_rows = validate_role_input(read_csv(role_path))
    clean_rows = read_clean(clean_path)
    verify_runtime_tokenizer(clean_rows, tokenizer, str(config["molformer"]["input_column"]))
    eligible_role, eligible_clean, eligibility = filter_model_eligible_rows(
        role_rows, clean_rows, config
    )
    eligible_ids = {row["structure_id"] for row in eligible_clean}
    if {row["structure_id"] for row in eligible_role} != eligible_ids:
        raise AssertionError("eligible clean/role cohorts differ")

    cells: list[dict[str, object]] = []
    for track in TRACKS:
        for seed in SEEDS:
            summary, _ = audit_one(
                eligible_role,
                track,
                FRACTIONS,
                seed,
                minimum_retention=minimum_retention,
                alpha=alpha,
            )
            cells.append(dict(summary))
    passing = sum(row["primary_count_gate"] == "pass" for row in cells)
    counts = Counter(int(row["target"]) for row in eligible_role)
    return (
        {
            "endpoint": endpoint,
            "critical_class": int(semantics["critical_class"]),
            "critical_class_reason": semantics["critical_class_reason"],
            "label_definition": semantics["label_definition"],
            "source_clean_n": len(clean_rows),
            "source_class_0_n": sum(int(row["target"]) == 0 for row in role_rows),
            "source_class_1_n": sum(int(row["target"]) == 1 for row in role_rows),
            "eligible_n": len(eligible_role),
            "eligible_class_0_n": counts[0],
            "eligible_class_1_n": counts[1],
            "model_eligibility": eligibility,
            "audited_track_seed_cells": len(cells),
            "passing_track_seed_cells": passing,
            "eligibility_status_after_model_domain": (
                "primary_freeze_ready" if passing == len(cells) else "freeze_blocked"
            ),
            "clean_input_sha256": sha256_file(clean_path),
            "role_input_sha256": sha256_file(role_path),
        },
        cells,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=P2 / "configs" / "racer_c" / "gpu_environment_lock.yaml",
    )
    parser.add_argument(
        "--study-design",
        type=Path,
        default=P2 / "configs" / "racer_c" / "study_design.yaml",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=P2
        / "results"
        / "racer_c_phase3_preflight"
        / "seed99_gpu_component_benchmark_windows_rtx4060.json",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=P2 / "data" / "processed" / "racer_c",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=P2 / "data" / "manifests" / "racer_c",
    )
    parser.add_argument(
        "--endpoint-manifest",
        type=Path,
        default=P2 / "protocols" / "endpoint_candidate_manifest.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=P2
        / "results"
        / "racer_c_phase4_freeze_review"
        / "formal_freeze_review_windows_rtx4060.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    study = yaml.safe_load(args.study_design.read_text(encoding="utf-8"))
    if study["protocol_status"] != "draft_pre_freeze":
        raise RuntimeError("freeze review must be completed while protocol is draft_pre_freeze")
    if study["selected_after_count_precision_audit"] != FRACTIONS:
        raise RuntimeError("selected outer-role allocation differs from freeze-review contract")
    if tuple(study["tracks"]) != TRACKS or tuple(study["main_split_seeds"]) != SEEDS:
        raise RuntimeError("track or seed contract differs from freeze-review contract")
    semantics = load_primary_semantics(args.endpoint_manifest)

    benchmark_script = Path(__file__).with_name("run_seed99_gpu_component_benchmark.py")
    benchmark = validate_seed99_benchmark(
        args.benchmark, args.config, benchmark_script, config
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["molformer"]["model_id"],
        revision=config["molformer"]["revision"],
        trust_remote_code=bool(config["molformer"]["trust_remote_code"]),
    )
    endpoint_rows: list[dict[str, object]] = []
    cell_rows: list[dict[str, object]] = []
    for endpoint in PRIMARY_ENDPOINTS:
        summary, cells = endpoint_review(
            endpoint,
            args.processed_dir,
            args.manifest_dir,
            config,
            tokenizer,
            minimum_retention=float(study["gate"]["minimum_planned_retention"]),
            alpha=float(study["conformal"]["alpha_primary"]),
            semantics=semantics[endpoint],
        )
        endpoint_rows.append(summary)
        cell_rows.extend(cells)

    primary_ready = all(
        row["eligibility_status_after_model_domain"] == "primary_freeze_ready"
        for row in endpoint_rows
    )
    result = {
        "status": "pass_prediction_free_formal_freeze_review" if primary_ready else "fail_closed",
        "protocol_status_at_review": study["protocol_status"],
        "scientific_predictions_generated": False,
        "performance_metrics_computed": False,
        "selection_uses_model_outputs": False,
        "primary_endpoints": list(PRIMARY_ENDPOINTS),
        "primary_endpoint_count": len(PRIMARY_ENDPOINTS),
        "tracks": list(TRACKS),
        "main_split_seeds": list(SEEDS),
        "allocation": ALLOCATION_ID,
        "critical_class_by_endpoint": {
            endpoint: int(semantics[endpoint]["critical_class"])
            for endpoint in PRIMARY_ENDPOINTS
        },
        "seed99_engineering_benchmark": benchmark,
        "endpoint_reviews": endpoint_rows,
        "track_seed_cell_count": len(cell_rows),
        "track_seed_cells_sha256": stable_sha256(cell_rows),
        "study_design_sha256": sha256_file(args.study_design),
        "endpoint_manifest_sha256": sha256_file(args.endpoint_manifest),
        "environment_candidate_lock_sha256": sha256_file(args.config),
        "review_script_sha256": sha256_file(Path(__file__)),
        "next_gate": "explicit_user_approval_for_protocol_freeze_tag",
    }
    if len(cell_rows) != len(PRIMARY_ENDPOINTS) * len(TRACKS) * len(SEEDS):
        raise AssertionError("formal freeze review did not audit all 60 primary cells")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    if not primary_ready:
        raise RuntimeError(
            "one or more endpoints failed after the label-blind model-domain gate; "
            "do not freeze or run confirmatory seeds"
        )
    print(f"FORMAL FREEZE REVIEW COMPLETE: {args.output}")
    print("No model was fit and no policy/conformal/test prediction was generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
