from __future__ import annotations

"""Evaluate predictive reliability within high/medium/low chemical-domain bins.

This script merges test predictions with applicability-domain diagnostics and reports:
- classification performance by domain bin;
- regression performance by domain bin;
- conformal empirical coverage and uncertainty size by domain bin;
- class-conditional conformal coverage for classification;
- warnings for very small or single-class domain bins.

Inputs:
- results/predictions/*_test_predictions.csv
- results/applicability/applicability_details_<mode>.csv
- results/conformal/conformal_predictions_<mode>.csv

Outputs:
- results/applicability/domain_conditioned_performance_<mode>.csv
- results/applicability/domain_conditioned_conformal_<mode>.csv
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PAPER_DIR = ROOT / "paper2_admet_benchmark"
PREDICTION_DIR = PAPER_DIR / "results" / "predictions"
APPLICABILITY_DIR = PAPER_DIR / "results" / "applicability"
CONFORMAL_DIR = PAPER_DIR / "results" / "conformal"

MVP_ENDPOINTS = ["bbbp", "clintox", "esol", "lipophilicity"]
MVP_SPLITS = ["split_random_seed0", "split_scaffold"]
ALL_RANDOM_SPLITS = [f"split_random_seed{i}" for i in range(5)]
DOMAIN_ORDER = ["high_domain", "medium_domain", "low_domain"]
MIN_GROUP_N_WARNING = 30
MIN_CLASS_COUNT_WARNING = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute domain-conditioned reliability summaries.")
    parser.add_argument("--mode", choices=["mvp", "full"], default="mvp")
    parser.add_argument("--endpoints", default=None)
    parser.add_argument("--splits", default=None)
    return parser.parse_args()


def resolve_endpoints(mode: str, value: str | None) -> list[str] | None:
    if value:
        return [item.strip().lower() for item in value.split(",") if item.strip()]
    return MVP_ENDPOINTS if mode == "mvp" else None


def resolve_splits(mode: str, value: str | None) -> list[str]:
    if value:
        return [item.strip() for item in value.split(",") if item.strip()]
    return MVP_SPLITS if mode == "mvp" else ALL_RANDOM_SPLITS + ["split_scaffold"]


def load_test_predictions() -> pd.DataFrame:
    files = sorted(PREDICTION_DIR.glob("*_test_predictions.csv"))
    if not files:
        raise FileNotFoundError(f"No test prediction files found in {PREDICTION_DIR}")
    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["prediction_file"] = str(path.relative_to(ROOT))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def to_bool_series(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.lower().map({"true": True, "false": False})
    if mapped.isna().any():
        return series.astype(bool)
    return mapped.astype(bool)


def classification_metrics(group: pd.DataFrame) -> dict[str, float | int | bool]:
    y_true = group["y_true"].astype(int).to_numpy()
    y_score = np.clip(group["y_output"].astype(float).to_numpy(), 1e-7, 1 - 1e-7)
    y_pred = (y_score >= 0.5).astype(int)
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    out: dict[str, float | int | bool] = {
        "positive_count": positives,
        "negative_count": negatives,
        "positive_ratio": float(positives / len(y_true)) if len(y_true) else math.nan,
        "error_rate": float(np.mean(y_pred != y_true)) if len(y_true) else math.nan,
        "brier_score": float(brier_score_loss(y_true, y_score)) if len(y_true) else math.nan,
        "roc_auc": math.nan,
        "pr_auc": math.nan,
        "single_class_bin": positives == 0 or negatives == 0,
        "low_class_count_warning": min(positives, negatives) < MIN_CLASS_COUNT_WARNING,
    }
    if len(np.unique(y_true)) >= 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    return out


def regression_metrics(group: pd.DataFrame) -> dict[str, float]:
    y_true = group["y_true"].astype(float).to_numpy()
    y_pred = group["y_output"].astype(float).to_numpy()
    residual = y_true - y_pred
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(residual.mean()),
    }


def filter_mode(
    df: pd.DataFrame,
    endpoints: list[str] | None,
    splits: list[str],
) -> pd.DataFrame:
    out = df.copy()
    if endpoints is not None:
        out = out[out["endpoint"].isin(endpoints)].copy()
    return out[out["split_col"].isin(splits)].copy()


def compute_domain_performance(
    predictions: pd.DataFrame,
    applicability: pd.DataFrame,
) -> pd.DataFrame:
    merge_cols = ["endpoint", "task_type", "split_col", "row_index"]
    ad_cols = merge_cols + [
        "domain_bin",
        "max_tanimoto_to_train",
        "unseen_scaffold",
    ]
    merged = predictions.merge(
        applicability[ad_cols],
        on=merge_cols,
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        raise ValueError("Prediction/applicability merge produced zero rows.")

    rows = []
    group_cols = ["endpoint", "task_type", "split_col", "model", "domain_bin"]
    for keys, group in merged.groupby(group_cols, sort=True):
        endpoint, task_type, split_col, model, domain_bin = keys
        total_n = len(
            merged[
                (merged["endpoint"] == endpoint)
                & (merged["split_col"] == split_col)
                & (merged["model"] == model)
            ]
        )
        row = {
            "endpoint": endpoint,
            "task_type": task_type,
            "split_col": split_col,
            "model": model,
            "domain_bin": domain_bin,
            "n": int(len(group)),
            "fraction_within_test": float(len(group) / total_n) if total_n else math.nan,
            "mean_max_tanimoto": float(group["max_tanimoto_to_train"].mean()),
            "median_max_tanimoto": float(group["max_tanimoto_to_train"].median()),
            "unseen_scaffold_rate": float(group["unseen_scaffold"].mean()),
            "small_group_warning": len(group) < MIN_GROUP_N_WARNING,
        }
        if task_type == "classification":
            row.update(classification_metrics(group))
        else:
            row.update(regression_metrics(group))
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["domain_bin"] = pd.Categorical(out["domain_bin"], categories=DOMAIN_ORDER, ordered=True)
        out = out.sort_values(["endpoint", "split_col", "model", "domain_bin"]).reset_index(drop=True)
    return out


def compute_domain_conformal(
    conformal_predictions: pd.DataFrame,
    applicability: pd.DataFrame,
) -> pd.DataFrame:
    merge_cols = ["endpoint", "task_type", "split_col", "row_index"]
    ad_cols = merge_cols + [
        "domain_bin",
        "max_tanimoto_to_train",
        "unseen_scaffold",
    ]
    merged = conformal_predictions.merge(
        applicability[ad_cols],
        on=merge_cols,
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        raise ValueError("Conformal/applicability merge produced zero rows.")

    rows = []
    group_cols = ["endpoint", "task_type", "split_col", "model", "alpha", "domain_bin"]
    for keys, group in merged.groupby(group_cols, sort=True):
        endpoint, task_type, split_col, model, alpha, domain_bin = keys
        covered = to_bool_series(group["covered"])
        nominal = float(group["nominal_coverage"].iloc[0])
        empirical = float(covered.mean())
        row = {
            "endpoint": endpoint,
            "task_type": task_type,
            "split_col": split_col,
            "model": model,
            "alpha": float(alpha),
            "nominal_coverage": nominal,
            "domain_bin": domain_bin,
            "n": int(len(group)),
            "empirical_coverage": empirical,
            "coverage_gap": empirical - nominal,
            "absolute_coverage_gap": abs(empirical - nominal),
            "mean_max_tanimoto": float(group["max_tanimoto_to_train"].mean()),
            "unseen_scaffold_rate": float(group["unseen_scaffold"].mean()),
            "small_group_warning": len(group) < MIN_GROUP_N_WARNING,
        }
        if task_type == "classification":
            y_true = group["y_true"].astype(int)
            positives = int(y_true.sum())
            negatives = int(len(y_true) - positives)
            positive_mask = y_true == 1
            negative_mask = y_true == 0
            positive_coverage = float(covered[positive_mask].mean()) if positives > 0 else math.nan
            negative_coverage = float(covered[negative_mask].mean()) if negatives > 0 else math.nan
            row.update(
                {
                    "positive_count": positives,
                    "negative_count": negatives,
                    "single_class_bin": positives == 0 or negatives == 0,
                    "low_class_count_warning": min(positives, negatives) < MIN_CLASS_COUNT_WARNING,
                    "positive_coverage": positive_coverage,
                    "negative_coverage": negative_coverage,
                    "positive_coverage_gap": (
                        positive_coverage - nominal if positives > 0 else math.nan
                    ),
                    "negative_coverage_gap": (
                        negative_coverage - nominal if negatives > 0 else math.nan
                    ),
                    "class_conditional_coverage_gap": (
                        abs(positive_coverage - negative_coverage)
                        if positives > 0 and negatives > 0
                        else math.nan
                    ),
                    "mean_prediction_set_size": float(group["prediction_set_size"].mean()),
                    "empty_set_rate": float(to_bool_series(group["empty_set"]).mean()),
                    "ambiguous_set_rate": float(to_bool_series(group["ambiguous_set"]).mean()),
                }
            )
        else:
            row["mean_interval_width"] = float(group["interval_width"].mean())
            row["mean_absolute_error"] = float(group["absolute_error"].mean())
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["domain_bin"] = pd.Categorical(out["domain_bin"], categories=DOMAIN_ORDER, ordered=True)
        out = out.sort_values(["endpoint", "split_col", "model", "alpha", "domain_bin"]).reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()
    endpoints = resolve_endpoints(args.mode, args.endpoints)
    splits = resolve_splits(args.mode, args.splits)

    details_path = APPLICABILITY_DIR / f"applicability_details_{args.mode}.csv"
    conformal_path = CONFORMAL_DIR / f"conformal_predictions_{args.mode}.csv"
    if not details_path.exists():
        raise FileNotFoundError(f"Missing {details_path}. Run applicability analysis first.")
    if not conformal_path.exists():
        raise FileNotFoundError(f"Missing {conformal_path}. Run conformal analysis first.")

    predictions = filter_mode(load_test_predictions(), endpoints=endpoints, splits=splits)
    applicability = pd.read_csv(details_path)
    applicability = filter_mode(applicability[applicability["role"] == "test"].copy(), endpoints=endpoints, splits=splits)
    conformal_predictions = filter_mode(pd.read_csv(conformal_path), endpoints=endpoints, splits=splits)

    performance = compute_domain_performance(predictions, applicability)
    conformal = compute_domain_conformal(conformal_predictions, applicability)

    performance_path = APPLICABILITY_DIR / f"domain_conditioned_performance_{args.mode}.csv"
    conformal_out_path = APPLICABILITY_DIR / f"domain_conditioned_conformal_{args.mode}.csv"
    performance.to_csv(performance_path, index=False)
    conformal.to_csv(conformal_out_path, index=False)

    small_performance = int(performance.get("small_group_warning", pd.Series(dtype=bool)).fillna(False).sum())
    low_class_performance = int(performance.get("low_class_count_warning", pd.Series(dtype=bool)).fillna(False).sum())
    small_conformal = int(conformal.get("small_group_warning", pd.Series(dtype=bool)).fillna(False).sum())
    low_class_conformal = int(conformal.get("low_class_count_warning", pd.Series(dtype=bool)).fillna(False).sum())

    print("saved", performance_path)
    print("saved", conformal_out_path)
    print(
        "Domain-conditioned reliability analysis complete. "
        f"Small-group rows: performance={small_performance}, conformal={small_conformal}; "
        f"low-class-count rows: performance={low_class_performance}, conformal={low_class_conformal}."
    )


if __name__ == "__main__":
    main()
