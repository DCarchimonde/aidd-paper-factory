from __future__ import annotations

"""Finalize selective-prediction analysis using matched partial-AURC support.

The existing selective curves are defined over retained coverage 0.10-1.00. This
script therefore reports partial AURC (pAURC), not full AURC. For classification
balanced error, selective and random curves are integrated only on the same finite
coverage points. This avoids comparing areas computed over different domains.

No models are retrained and no selective curves are recomputed.
"""

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper2_admet_benchmark"
SELECTIVE_DIR = PAPER_DIR / "results" / "selective"
TABLE_DIR = PAPER_DIR / "results" / "tables"

MODE = "confirmatory_full"
SPLIT_PATTERN = re.compile(r"split_confirm_(random|scaffold|cluster)_seed(\d+)$")
HEADLINE_COVERAGES = [0.10, 0.25, 0.50, 0.75, 1.00]


def add_split_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    extracted = out["split_col"].astype(str).str.extract(SPLIT_PATTERN)
    out["split_type"] = extracted[0]
    out["seed"] = pd.to_numeric(extracted[1], errors="coerce").astype("Int64")
    if out["split_type"].isna().any() or out["seed"].isna().any():
        bad = out.loc[
            out["split_type"].isna() | out["seed"].isna(), "split_col"
        ].drop_duplicates()
        raise ValueError(f"Unrecognized confirmatory split columns: {bad.tolist()}")
    return out


def matched_partial_area(
    group: pd.DataFrame,
    value_col: str,
    baseline_col: str,
) -> dict[str, float | int]:
    matched = (
        group[["actual_coverage", value_col, baseline_col]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values("actual_coverage")
        .drop_duplicates("actual_coverage", keep="last")
    )
    if len(matched) < 2:
        return {
            "n_matched_points": int(len(matched)),
            "coverage_min": math.nan,
            "coverage_max": math.nan,
            "coverage_span": math.nan,
            "selective_paurc": math.nan,
            "random_paurc": math.nan,
            "paurc_improvement_vs_random": math.nan,
            "selective_mean_risk_over_range": math.nan,
            "random_mean_risk_over_range": math.nan,
        }

    x = matched["actual_coverage"].to_numpy(float)
    y = matched[value_col].to_numpy(float)
    y_random = matched[baseline_col].to_numpy(float)
    coverage_min = float(x.min())
    coverage_max = float(x.max())
    span = coverage_max - coverage_min
    selective_area = float(np.trapezoid(y, x))
    random_area = float(np.trapezoid(y_random, x))
    return {
        "n_matched_points": int(len(matched)),
        "coverage_min": coverage_min,
        "coverage_max": coverage_max,
        "coverage_span": span,
        "selective_paurc": selective_area,
        "random_paurc": random_area,
        "paurc_improvement_vs_random": random_area - selective_area,
        "selective_mean_risk_over_range": selective_area / span if span > 0 else math.nan,
        "random_mean_risk_over_range": random_area / span if span > 0 else math.nan,
    }


def closest_row(group: pd.DataFrame, target: float) -> pd.Series:
    distances = (pd.to_numeric(group["target_coverage"], errors="coerce") - target).abs()
    return group.loc[distances.idxmin()]


def summarize_run(group: pd.DataFrame) -> dict[str, object]:
    first = group.iloc[0]
    task_type = str(first["task_type"])
    primary = matched_partial_area(group, "risk", "random_risk_mean")
    row: dict[str, object] = {
        "endpoint": str(first["endpoint"]),
        "task_type": task_type,
        "split_col": str(first["split_col"]),
        "model": str(first["model"]),
        "uncertainty_measure": str(first["uncertainty_measure"]),
        "primary_n_matched_points": primary["n_matched_points"],
        "primary_coverage_min": primary["coverage_min"],
        "primary_coverage_max": primary["coverage_max"],
        "primary_coverage_span": primary["coverage_span"],
        "primary_selective_paurc": primary["selective_paurc"],
        "primary_random_paurc": primary["random_paurc"],
        "primary_paurc_improvement_vs_random": primary[
            "paurc_improvement_vs_random"
        ],
        "primary_selective_mean_risk_over_range": primary[
            "selective_mean_risk_over_range"
        ],
        "primary_random_mean_risk_over_range": primary[
            "random_mean_risk_over_range"
        ],
    }

    for target in HEADLINE_COVERAGES:
        point = closest_row(group, target)
        tag = str(target).replace(".", "")
        row[f"actual_coverage_at_{tag}"] = float(point["actual_coverage"])
        row[f"risk_at_{tag}"] = float(point["risk"])
        row[f"random_risk_at_{tag}"] = float(point["random_risk_mean"])
        row[f"risk_improvement_at_{tag}"] = float(point["random_risk_mean"]) - float(
            point["risk"]
        )
        if task_type == "classification":
            row[f"positive_retention_at_{tag}"] = float(
                point["positive_retention_rate"]
            )
            row[f"negative_retention_at_{tag}"] = float(
                point["negative_retention_rate"]
            )
            row[f"class_balance_shift_at_{tag}"] = float(point["class_balance_shift"])

    if task_type == "classification":
        balanced = matched_partial_area(
            group,
            "balanced_error_rate",
            "random_balanced_error_mean",
        )
        row.update(
            {
                "balanced_n_matched_points": balanced["n_matched_points"],
                "balanced_coverage_min": balanced["coverage_min"],
                "balanced_coverage_max": balanced["coverage_max"],
                "balanced_coverage_span": balanced["coverage_span"],
                "balanced_selective_paurc": balanced["selective_paurc"],
                "balanced_random_paurc": balanced["random_paurc"],
                "balanced_paurc_improvement_vs_random": balanced[
                    "paurc_improvement_vs_random"
                ],
                "balanced_selective_mean_risk_over_range": balanced[
                    "selective_mean_risk_over_range"
                ],
                "balanced_random_mean_risk_over_range": balanced[
                    "random_mean_risk_over_range"
                ],
                "minimum_positive_retention": float(
                    pd.to_numeric(group["positive_retention_rate"], errors="coerce").min()
                ),
                "minimum_negative_retention": float(
                    pd.to_numeric(group["negative_retention_rate"], errors="coerce").min()
                ),
                "maximum_absolute_class_balance_shift": float(
                    pd.to_numeric(group["class_balance_shift"], errors="coerce").abs().max()
                ),
            }
        )
    return row


def summarize_values(values: pd.Series) -> dict[str, float | int]:
    x = (
        pd.to_numeric(values, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    n = int(len(x))
    if n == 0:
        return {
            "n_splits": 0,
            "mean": math.nan,
            "std": math.nan,
            "se": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
            "min": math.nan,
            "max": math.nan,
        }
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if n > 1 else math.nan
    se = std / math.sqrt(n) if n > 1 else math.nan
    critical = float(t.ppf(0.975, df=n - 1)) if n > 1 else math.nan
    margin = critical * se if n > 1 else math.nan
    return {
        "n_splits": n,
        "mean": mean,
        "std": std,
        "se": se,
        "ci95_low": mean - margin if n > 1 else math.nan,
        "ci95_high": mean + margin if n > 1 else math.nan,
        "min": float(x.min()),
        "max": float(x.max()),
    }


def aggregate_by_model(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "primary_paurc_improvement_vs_random",
        "primary_selective_mean_risk_over_range",
        "risk_improvement_at_05",
    ]
    if (runs["task_type"] == "classification").any():
        metrics.extend(
            [
                "balanced_paurc_improvement_vs_random",
                "balanced_selective_mean_risk_over_range",
                "positive_retention_at_01",
                "positive_retention_at_025",
                "positive_retention_at_05",
                "negative_retention_at_05",
                "class_balance_shift_at_05",
                "minimum_positive_retention",
                "maximum_absolute_class_balance_shift",
            ]
        )

    rows: list[dict[str, object]] = []
    group_cols = [
        "endpoint",
        "task_type",
        "split_type",
        "model",
        "uncertainty_measure",
    ]
    for keys, group in runs.groupby(group_cols, sort=True):
        base = dict(zip(group_cols, keys))
        expected = 5 if base["split_type"] == "cluster" else 10
        for metric in metrics:
            if metric not in group.columns:
                continue
            summary = summarize_values(group[metric])
            if int(summary["n_splits"]) == 0:
                continue
            if int(summary["n_splits"]) != expected:
                raise RuntimeError(
                    f"Incomplete aggregate for {base}, metric={metric}: "
                    f"expected {expected}, got {summary['n_splits']}"
                )
            rows.append({**base, "metric": metric, **summary})
    return pd.DataFrame(rows)


def descriptive_headline(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "primary_paurc_improvement_vs_random",
        "primary_selective_mean_risk_over_range",
        "risk_improvement_at_05",
        "balanced_paurc_improvement_vs_random",
        "positive_retention_at_01",
        "positive_retention_at_025",
        "positive_retention_at_05",
        "negative_retention_at_05",
        "class_balance_shift_at_05",
        "minimum_positive_retention",
        "maximum_absolute_class_balance_shift",
    ]
    rows: list[dict[str, object]] = []
    group_cols = [
        "endpoint",
        "task_type",
        "split_type",
        "uncertainty_measure",
    ]
    for keys, group in runs.groupby(group_cols, sort=True):
        base = dict(zip(group_cols, keys))
        expected_models = 8 if base["task_type"] == "classification" else 4
        expected_seeds = 5 if base["split_type"] == "cluster" else 10
        if group["model"].nunique() != expected_models:
            raise RuntimeError(f"Unexpected model count for {base}")
        if group["seed"].nunique() != expected_seeds:
            raise RuntimeError(f"Unexpected seed count for {base}")
        row: dict[str, object] = {
            **base,
            "models": int(group["model"].nunique()),
            "seeds": int(group["seed"].nunique()),
            "model_seed_cells": int(len(group)),
            "descriptive_average_across_models": True,
        }
        for metric in metrics:
            if metric in group.columns:
                values = pd.to_numeric(group[metric], errors="coerce")
                row[metric] = float(values.mean()) if values.notna().any() else math.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["endpoint", "split_type", "uncertainty_measure"]
    )


def main() -> None:
    curve_path = SELECTIVE_DIR / f"selective_prediction_v2_curves_{MODE}.csv"
    if not curve_path.exists():
        raise FileNotFoundError(curve_path)
    curves = pd.read_csv(curve_path, low_memory=False)
    if len(curves) != 19000:
        raise RuntimeError(f"Expected 19000 selective curve rows, found {len(curves)}")

    group_cols = [
        "endpoint",
        "task_type",
        "split_col",
        "model",
        "uncertainty_measure",
    ]
    run_rows = [summarize_run(group) for _, group in curves.groupby(group_cols, sort=True)]
    runs = add_split_metadata(pd.DataFrame(run_rows))
    if len(runs) != 1000:
        raise RuntimeError(f"Expected 1000 selective run summaries, found {len(runs)}")

    by_model = aggregate_by_model(runs)
    headline = descriptive_headline(runs)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    runs_path = TABLE_DIR / "selective_prediction_matched_paurc_runs.csv"
    model_path = TABLE_DIR / "selective_prediction_matched_paurc_by_model.csv"
    headline_path = TABLE_DIR / "selective_prediction_matched_paurc_headline.csv"
    runs.to_csv(runs_path, index=False)
    by_model.to_csv(model_path, index=False)
    headline.to_csv(headline_path, index=False)

    print("saved", runs_path)
    print("saved", model_path)
    print("saved", headline_path)
    print(
        "Matched selective-prediction finalization complete. "
        f"curve_rows={len(curves)}, runs={len(runs)}, "
        f"model_metrics={len(by_model)}, headline={len(headline)}."
    )


if __name__ == "__main__":
    main()
