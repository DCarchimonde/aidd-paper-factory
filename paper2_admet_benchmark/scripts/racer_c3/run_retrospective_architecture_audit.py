from __future__ import annotations

"""Apply the fixed RACER-C3 candidate to the already-open v1 panel.

This command is intentionally retrospective.  It reads known v1 test outcomes,
performs no prospective selection, writes no prediction-level table, and marks
every output as unauthorized for a scientific superiority claim.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from racer_c3_core import (
    PROBABILITY_COLUMNS,
    RISK_COLUMNS,
    crossfit_candidate_correctness,
    fallback_scores,
    frontier_scores,
    mondrian_thresholds,
    prediction_membership,
    routed_scores,
    set_metrics,
    symmetric_frontier_gate,
)


ROOT = Path(__file__).resolve().parents[3]
P2 = ROOT / "paper2_admet_benchmark"
DEFAULT_LOCK = P2 / "configs" / "racer_c3" / "development_lock_v0.yaml"
DEFAULT_OUTPUT = P2 / "results" / "racer_c3_development"
METHOD_BASELINE = "RACER-C_v1_no_gate_recomputed"
METHOD_FRONTIER = "RACER-C3_frontier_expert_always"
METHOD_ROUTED = "RACER-C3_routed"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def load_known_cell(directory: Path) -> tuple[str, str, int, pd.DataFrame]:
    endpoint, track, seed_text = directory.name.split("__")
    seed = int(seed_text.removeprefix("seed"))
    raw = pd.read_csv(
        directory / "raw_predictions.csv", dtype={"structure_id": str}
    )
    known = pd.read_csv(
        directory / "test_predictions.csv",
        usecols=["structure_id", "target"],
        dtype={"structure_id": str},
    ).drop_duplicates("structure_id")
    target_by_id = known.set_index("structure_id")["target"]
    test = raw.role.eq("test")
    raw.loc[test, "target"] = raw.loc[test, "structure_id"].map(target_by_id)
    if raw.target.isna().any():
        raise RuntimeError(f"target recovery failed for {directory.name}")
    raw["target"] = raw.target.astype(np.int8)
    return endpoint, track, seed, raw


def add_scaffolds(frame: pd.DataFrame, endpoint: str, role_input_dir: Path) -> pd.DataFrame:
    role = pd.read_csv(
        role_input_dir / f"{endpoint}_role_input.csv",
        usecols=["structure_id", "murcko_scaffold_id"],
        dtype={"structure_id": str, "murcko_scaffold_id": str},
    )
    merged = frame.merge(role, on="structure_id", how="left", validate="one_to_one")
    if merged.murcko_scaffold_id.fillna("").str.strip().eq("").any():
        raise RuntimeError(f"blank scaffold identifier for {endpoint}")
    return merged


def evaluate_cell(
    directory: Path,
    role_input_dir: Path,
    lock: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    endpoint, track, seed, frame = load_known_cell(directory)
    frame = add_scaffolds(frame, endpoint, role_input_dir)
    target = frame.target.to_numpy(np.int8)
    probability = frame.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
    risk = frame.loc[:, RISK_COLUMNS].to_numpy(float)
    risk_percentile = frame.risk_percentile.to_numpy(float)
    development = frame.role.eq("dev").to_numpy()
    conformal = frame.role.eq("conformal").to_numpy()
    test = frame.role.eq("test").to_numpy()
    union = conformal | test

    expert_lock = lock["candidate_experts"]
    model_lock = expert_lock["class_1"]
    fitted = crossfit_candidate_correctness(
        probability[development],
        risk[development],
        target[development],
        frame.loc[development, "meta_fold"].to_numpy(int),
        seed=seed + 3000,
        c_value=float(model_lock["C"]),
    )
    correctness = np.empty((len(frame), 2), dtype=float)
    correctness[development] = fitted.oof_correctness
    correctness[~development] = fitted.ensemble.predict_correctness(
        probability[~development], risk[~development]
    )
    temperature_lock = expert_lock["risk_temperature"]
    baseline = fallback_scores(
        frame.stack_p.to_numpy(float),
        risk_percentile,
        t_max=float(lock["fallback"]["t_max"]),
    )
    frontier = frontier_scores(
        probability,
        risk,
        correctness,
        risk_percentile,
        t_max_0=float(temperature_lock["class_0_t_max"]),
        t_max_1=float(temperature_lock["class_1_t_max"]),
    )
    route_lock = lock["frontier_route"]
    decision = symmetric_frontier_gate(
        frame.loc[development, "murcko_scaffold_id"].astype(str).tolist(),
        frame.loc[union, "murcko_scaffold_id"].astype(str).tolist(),
        frame.loc[union, "ecfp_distance"].to_numpy(float),
        overlap_fraction_max=float(route_lock["overlap_fraction_max"]),
        median_ecfp_distance_min=float(route_lock["median_ecfp_distance_min"]),
        minimum_valid_union_n=int(route_lock["minimum_valid_union_n"]),
    )
    routed = routed_scores(baseline, frontier, decision)

    rows: list[dict[str, object]] = []
    conformal_lock = lock["conformal"]
    fallback_alpha = {
        int(label): float(value)
        for label, value in conformal_lock["fallback_alpha_by_class"].items()
    }
    frontier_alpha = {
        int(label): float(value)
        for label, value in conformal_lock["frontier_alpha_by_class"].items()
    }
    for method, scores, alpha in (
        (METHOD_BASELINE, baseline, fallback_alpha),
        (METHOD_FRONTIER, frontier, frontier_alpha),
        (METHOD_ROUTED, routed, frontier_alpha if decision.active else fallback_alpha),
    ):
        thresholds = mondrian_thresholds(scores[conformal], target[conformal], alpha)
        membership = prediction_membership(scores[test], thresholds)
        rows.append(
            {
                "endpoint": endpoint,
                "track": track,
                "seed": seed,
                "method": method,
                "route_active": decision.active,
                **set_metrics(target[test], membership),
            }
        )

    stored = pd.read_csv(directory / "metrics.csv")
    stored_value = float(
        stored.loc[stored.method.eq("RACER_score_no_gate"), "macro_csy"].iloc[0]
    )
    recomputed = float(rows[0]["macro_csy"])
    if not np.isclose(stored_value, recomputed, rtol=0.0, atol=1.0e-12):
        raise RuntimeError(
            f"v1 fallback identity failed for {directory.name}: "
            f"stored={stored_value} recomputed={recomputed}"
        )
    gate = {
        "endpoint": endpoint,
        "track": track,
        "seed": seed,
        "route_active": decision.active,
        "overlap_fraction": decision.overlap_fraction,
        "median_ecfp_distance": decision.median_ecfp_distance,
        "valid_union_n": decision.valid_union_n,
        "reason": decision.reason,
        "v1_fallback_identity_pass": True,
    }
    return rows, gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-package", type=Path, required=True)
    parser.add_argument("--role-input-dir", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    lock = yaml.safe_load(args.lock.read_text(encoding="utf-8"))
    if lock["lock_status"] != "retrospective_architecture_candidate_not_freeze_ready":
        raise RuntimeError("unexpected RACER-C3 development lock status")
    cell_root = args.analysis_package / "cells"
    directories = sorted(path for path in cell_root.iterdir() if path.is_dir())
    if len(directories) != 60:
        raise RuntimeError(f"expected 60 v1 cells, found {len(directories)}")

    rows: list[dict[str, object]] = []
    gates: list[dict[str, object]] = []
    for position, directory in enumerate(directories, start=1):
        cell_rows, gate = evaluate_cell(directory, args.role_input_dir, lock)
        rows.extend(cell_rows)
        gates.append(gate)
        if position % 10 == 0:
            print(f"processed {position}/{len(directories)}", flush=True)

    detail = pd.DataFrame(rows)
    gate_frame = pd.DataFrame(gates)
    baseline = detail[detail.method.eq(METHOD_BASELINE)][
        ["endpoint", "track", "seed", "macro_csy"]
    ].rename(columns={"macro_csy": "baseline_macro_csy"})
    paired = detail.merge(baseline, on=["endpoint", "track", "seed"], how="left")
    paired["macro_csy_delta"] = paired.macro_csy - paired.baseline_macro_csy
    summary = (
        paired.groupby(["method", "track"])
        .agg(
            cell_count=("macro_csy", "size"),
            mean_macro_csy=("macro_csy", "mean"),
            mean_macro_csy_delta=("macro_csy_delta", "mean"),
            mean_class_0_coverage=("class_0_coverage", "mean"),
            mean_class_1_coverage=("class_1_coverage", "mean"),
        )
        .reset_index()
    )
    routed = paired[paired.method.eq(METHOD_ROUTED)]
    frontier = paired[paired.method.eq(METHOD_FRONTIER)]
    report = {
        "status": "known_v1_panel_retrospective_architecture_only",
        "algorithm": "RACER-C3",
        "cell_count": 60,
        "known_test_labels_used": True,
        "test_labels_used_for_current_architecture_search": True,
        "scientific_superiority_claim_authorized": False,
        "prospective_test_predictions_authorized": False,
        "fallback_identity_pass_count": int(gate_frame.v1_fallback_identity_pass.sum()),
        "frontier_activation_by_track": {
            str(track): int(group.route_active.sum())
            for track, group in gate_frame.groupby("track")
        },
        "routed_delta_pp_by_track": {
            str(track): 100.0 * float(group.macro_csy_delta.mean())
            for track, group in routed.groupby("track")
        },
        "frontier_always_delta_pp_by_track": {
            str(track): 100.0 * float(group.macro_csy_delta.mean())
            for track, group in frontier.groupby("track")
        },
        "routed_all_cell_delta_pp": 100.0 * float(routed.macro_csy_delta.mean()),
        "freeze_gate_passed": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_csv(args.output / "cell_metrics.csv", detail)
    atomic_csv(args.output / "gate_audit.csv", gate_frame)
    atomic_csv(args.output / "paired_deltas.csv", paired)
    atomic_csv(args.output / "summary.csv", summary)
    atomic_text(
        args.output / "report.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    manifest = {
        "status": "complete_retrospective_architecture_audit",
        "development_lock_sha256": sha256_file(args.lock),
        "analysis_package_integrity_sha256": sha256_file(
            args.analysis_package / "package_integrity.json"
        ),
        "cell_metrics_sha256": sha256_file(args.output / "cell_metrics.csv"),
        "gate_audit_sha256": sha256_file(args.output / "gate_audit.csv"),
        "paired_deltas_sha256": sha256_file(args.output / "paired_deltas.csv"),
        "summary_sha256": sha256_file(args.output / "summary.csv"),
        "report_sha256": sha256_file(args.output / "report.json"),
        "known_test_labels_used": True,
        "scientific_superiority_claim_authorized": False,
    }
    atomic_text(
        args.output / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
