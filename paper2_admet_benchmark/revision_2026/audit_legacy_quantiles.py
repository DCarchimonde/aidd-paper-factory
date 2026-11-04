"""Read-only audit of the historical Stage I classification quantile cap.

This writes a separate diagnostic, never edits historical predictions or results,
and never treats absent row-level files as evidence that no cell was affected.
The Stage II finite-sample function already uses +infinity correctly.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
GROUP = ["endpoint", "split_col", "model"]


def rank_for(n: int, alpha: float) -> int:
    if n < 0 or not 0 < alpha < 1:
        raise ValueError("n must be nonnegative and alpha in (0,1)")
    return math.ceil((n + 1) * (1 - alpha))


def thresholds(scores: np.ndarray, alpha: float) -> tuple[float | None, float]:
    values = np.sort(np.asarray(scores, dtype=float))
    if not np.isfinite(values).all():
        raise ValueError("nonfinite calibration scores")
    rank = rank_for(len(values), alpha)
    if not len(values):
        return None, math.inf
    return float(values[min(rank, len(values)) - 1]), (
        math.inf if rank > len(values) else float(values[rank - 1])
    )


def audit_predictions(frame: pd.DataFrame, alphas: list[float], mode: str) -> list[dict]:
    required = {*GROUP, "task_type", "role", "row_index", "y_true", "y_output"}
    if not required.issubset(frame):
        raise ValueError(f"missing prediction columns: {sorted(required - set(frame))}")
    frame = frame.loc[frame.task_type == "classification"].copy()
    if mode == "confirmatory":
        frame = frame.loc[frame.split_col.astype(str).str.startswith("split_confirm_")]
    if frame.empty:
        return []
    # The legacy model key includes imbalance regime; preserve an explicit regime
    # too if provided. Do not double count duplicated files or mixed replay runs.
    grouping = GROUP + (["imbalance_regime"] if "imbalance_regime" in frame else [])
    identity = grouping + ["role", "row_index"]
    if frame.duplicated(identity).any():
        raise ValueError("duplicate prediction identities: select one historical run, not duplicate files")
    for column in ("y_true", "y_output"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not frame.y_true.isin([0, 1]).all():
        raise ValueError("classification labels must be 0/1")
    if not np.isfinite(frame.y_output).all() or not frame.y_output.between(0, 1).all():
        raise ValueError("classification probabilities must be finite in [0,1]")
    rows = []
    for key, group in frame.groupby(grouping, sort=True, dropna=False):
        base = dict(zip(grouping, key))
        calibration = group.loc[group.role == "calibration"]
        test = group.loc[group.role == "test"]
        if calibration.empty or test.empty:
            rows.append({**base, "status": "missing_calibration_or_test", "n_calibration": len(calibration), "n_test": len(test)})
            continue
        pcal = np.clip(calibration.y_output.to_numpy(float), 1e-7, 1 - 1e-7)
        ptest = np.clip(test.y_output.to_numpy(float), 1e-7, 1 - 1e-7)
        ycal = calibration.y_true.to_numpy(int)
        ytest = test.y_true.to_numpy(int)
        class_counts = [int((ycal == c).sum()) for c in (0, 1)]
        for alpha in alphas:
            for class_label in ("marginal", 0, 1):
                if class_label == "marginal":
                    scores = np.where(ycal == 1, 1 - pcal, pcal)
                    query = np.column_stack([ptest, 1 - ptest])
                    true_mask = np.ones(len(test), bool)
                else:
                    label = int(class_label)
                    scores = (pcal if label == 0 else 1 - pcal)[ycal == label]
                    query = ptest if label == 0 else 1 - ptest
                    true_mask = ytest == label
                old, new = thresholds(scores, alpha)
                skipped = class_label != "marginal" and min(class_counts) == 0
                record = {
                    **base, "alpha": alpha, "calibration_class": class_label,
                    "n_calibration": len(scores), "rank": rank_for(len(scores), alpha),
                    "rank_exceeds_n": rank_for(len(scores), alpha) > len(scores),
                    "legacy_qhat": old, "corrected_qhat": new,
                    "n_test": len(test), "n_test_true_class": int(true_mask.sum()),
                    "status": "legacy_cell_skipped_zero_class" if skipped else "row_level_verified",
                    "candidate_memberships_changed": None, "true_labels_newly_covered": None,
                    "coverage_delta": None,
                }
                if not skipped and old is not None:
                    changed = (query > old) & (query <= new)
                    record["candidate_memberships_changed"] = int(changed.sum())
                    covered_changed = changed[np.arange(len(test)), ytest] if class_label == "marginal" else changed[true_mask]
                    record["true_labels_newly_covered"] = int(covered_changed.sum())
                    record["coverage_delta"] = float(covered_changed.mean()) if len(covered_changed) else None
                rows.append(record)
    return rows


def audit_summary(frame: pd.DataFrame, mode: str) -> list[dict]:
    required = {*GROUP, "alpha", "calibration_negative_count", "calibration_positive_count"}
    if not required.issubset(frame):
        raise ValueError(f"summary lacks class counts: {sorted(required - set(frame))}")
    if mode == "confirmatory":
        frame = frame.loc[frame.split_col.astype(str).str.startswith("split_confirm_")]
    if frame.duplicated(GROUP + ["alpha"]).any():
        raise ValueError("duplicate summary cells: select one historical summary")
    rows = []
    for record in frame.to_dict("records"):
        for label, field in ((0, "calibration_negative_count"), (1, "calibration_positive_count")):
            value = float(record[field])
            if not math.isfinite(value) or value < 0 or not value.is_integer():
                raise ValueError("calibration counts must be nonnegative integers")
            n = int(value)
            alpha = float(record["alpha"])
            rank = rank_for(n, alpha)
            rows.append({**{k: record[k] for k in GROUP}, "alpha": alpha,
                         "calibration_class": label, "n_calibration": n, "rank": rank,
                         "rank_exceeds_n": rank > n, "status": "counts_only_output_impact_unknown",
                         "candidate_memberships_changed": None, "true_labels_newly_covered": None})
    return rows


def run(prediction_dir: Path, summary: Path | None, output: Path, alphas: list[float], mode: str, pattern: str) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite an existing audit; choose a fresh --output directory")
    for alpha in alphas:
        rank_for(1, alpha)
    files = sorted(prediction_dir.glob(pattern)) if prediction_dir.is_dir() else []
    sources = files if files else ([summary] if summary and summary.is_file() else [])
    if files:
        rows = audit_predictions(pd.concat([pd.read_csv(p) for p in files], ignore_index=True), alphas, mode)
        status = "row_level_audit_complete" if rows else "missing_matching_prediction_rows"
    elif sources:
        rows = audit_summary(pd.read_csv(sources[0]), mode)
        status = "counts_only_actual_prediction_changes_unconfirmed" if rows else "missing_matching_summary_rows"
    else:
        rows = []
        status = "missing_inputs_cannot_determine_historical_impact"
    report = {
        "status": status, "mode": mode, "alphas_for_row_level_audit": alphas,
        "scope": "Stage I marginal and Mondrian classification only; no model refitting or historical overwrite",
        "historical_prediction_directory": str(prediction_dir), "prediction_pattern": pattern,
        "historical_summary": str(summary) if summary else None,
        "source_files": [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],
        "quantile_rows": len(rows),
        "rows_with_rank_exceeding_n": sum(bool(r.get("rank_exceeds_n")) for r in rows),
        "rows_with_verified_changed_membership": sum((r.get("candidate_memberships_changed") or 0) > 0 for r in rows),
        "unresolved_rows": sum(r["status"] != "row_level_verified" for r in rows),
        "historical_published_numbers_modified": False,
        "interpretation": "Counts-only or missing-input reports cannot establish zero impact. A changed threshold need not change any test prediction. At alpha=0.10, positive n<9 needs +infinity; n=0 was skipped by the old Mondrian runner.",
    }
    output.mkdir(parents=True, exist_ok=True)
    if rows:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (output / "quantile_cells.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, default=REPO / "paper2_admet_benchmark/results/predictions")
    parser.add_argument("--pattern", default="*_predictions.csv")
    parser.add_argument("--summary", type=Path, default=REPO / "paper2_admet_benchmark/results/conformal/mondrian_conformal_summary_confirmatory.csv")
    parser.add_argument("--output", type=Path, default=REPO / ".local/chemolab_revision/legacy_quantile_audit")
    parser.add_argument("--mode", choices=("confirmatory", "all"), default="confirmatory")
    parser.add_argument("--alphas", default="0.10,0.20")
    args = parser.parse_args()
    report = run(args.prediction_dir, args.summary, args.output, [float(v) for v in args.alphas.split(",")], args.mode, args.pattern)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 2 if report["status"].startswith("missing") else 0


if __name__ == "__main__":
    raise SystemExit(main())
