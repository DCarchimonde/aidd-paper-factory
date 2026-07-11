from __future__ import annotations

"""Compute selective-prediction risk-coverage curves for Paper 2.

The script ranks test predictions from most to least trustworthy and evaluates
performance after retaining only a fraction of predictions.

Classification uncertainty rankings:
- one_minus_max_probability
- predictive_entropy
- one_minus_max_tanimoto_to_train

Regression uncertainty ranking:
- one_minus_max_tanimoto_to_train

Important:
- For binary classification, predictive entropy and one-minus-maximum probability
  are monotonic transformations of the same probability confidence. They therefore
  produce the same ranking; both are retained only as a documented sensitivity view.
- The canonical selective-prediction risk is classification error rate or regression
  RMSE. ROC-AUC/PR-AUC are supplementary because retained subsets can change class
  composition, especially for imbalanced endpoints such as ClinTox.
- Positive/negative retention rates are reported so apparent risk improvements cannot
  be mistaken for genuine reliability gains when minority-class samples are rejected.
- Symmetric split-conformal intervals have constant width within each
  endpoint/model/split, so interval width cannot meaningfully rank individual test
  samples in the current MVP protocol.

Inputs:
- results/predictions/*_test_predictions.csv
- results/applicability/applicability_details_<mode>.csv

Outputs:
- results/selective/selective_prediction_summary_<mode>.csv
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
OUTPUT_DIR = PAPER_DIR / "results" / "selective"

MVP_ENDPOINTS = ["bbbp", "clintox", "esol", "lipophilicity"]
MVP_SPLITS = ["split_random_seed0", "split_scaffold"]
ALL_RANDOM_SPLITS = [f"split_random_seed{i}" for i in range(5)]
RETAINED_FRACTIONS = [1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40]
MIN_CLASS_COUNT_WARNING = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Paper 2 selective-prediction curves.")
    parser.add_argument("--mode", choices=["mvp", "full"], default="mvp")
    parser.add_argument(
        "--endpoints",
        default=None,
        help="Optional comma-separated endpoints. Overrides mode defaults.",
    )
    parser.add_argument(
        "--splits",
        default=None,
        help="Optional comma-separated split columns. Overrides mode defaults.",
    )
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


def classification_metrics(group: pd.DataFrame) -> dict[str, float | int | bool]:
    y_true = group["y_true"].astype(int).to_numpy()
    y_score = np.clip(group["y_output"].astype(float).to_numpy(), 1e-7, 1 - 1e-7)
    y_pred = (y_score >= 0.5).astype(int)
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    error_rate = float(np.mean(y_pred != y_true))
    out: dict[str, float | int | bool] = {
        "risk": error_rate,
        "error_rate": error_rate,
        "positive_count": positives,
        "negative_count": negatives,
        "positive_ratio": float(positives / len(y_true)) if len(y_true) else math.nan,
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "roc_auc": math.nan,
        "pr_auc": math.nan,
        "single_class_subset": positives == 0 or negatives == 0,
        "low_class_count_warning": min(positives, negatives) < MIN_CLASS_COUNT_WARNING,
    }
    if len(np.unique(y_true)) >= 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    return out


def regression_metrics(group: pd.DataFrame) -> dict[str, float]:
    y_true = group["y_true"].astype(float).to_numpy()
    y_pred = group["y_output"].astype(float).to_numpy()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "risk": rmse,
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def add_uncertainty_columns(group: pd.DataFrame) -> pd.DataFrame:
    out = group.copy()
    out["one_minus_max_tanimoto_to_train"] = 1.0 - out["max_tanimoto_to_train"].astype(float)

    if str(out["task_type"].iloc[0]) == "classification":
        p1 = np.clip(out["y_output"].astype(float).to_numpy(), 1e-7, 1 - 1e-7)
        max_probability = np.maximum(p1, 1 - p1)
        entropy = -(p1 * np.log(p1) + (1 - p1) * np.log(1 - p1)) / np.log(2.0)
        out["one_minus_max_probability"] = 1.0 - max_probability
        out["predictive_entropy"] = entropy
    return out


def uncertainty_columns(task_type: str) -> list[str]:
    if task_type == "classification":
        return [
            "one_minus_max_probability",
            "predictive_entropy",
            "one_minus_max_tanimoto_to_train",
        ]
    return ["one_minus_max_tanimoto_to_train"]


def evaluate_curve(group: pd.DataFrame, uncertainty_col: str) -> list[dict]:
    ranked = group.sort_values(uncertainty_col, ascending=True).reset_index(drop=True)
    task_type = str(group["task_type"].iloc[0])
    rows = []

    if task_type == "classification":
        full_positive_count = int(ranked["y_true"].astype(int).sum())
        full_negative_count = int(len(ranked) - full_positive_count)
        full_metrics = classification_metrics(ranked)
    else:
        full_positive_count = 0
        full_negative_count = 0
        full_metrics = regression_metrics(ranked)
    full_risk = float(full_metrics["risk"])

    for retained_fraction in RETAINED_FRACTIONS:
        n_retain = max(1, int(math.ceil(len(ranked) * retained_fraction)))
        retained = ranked.iloc[:n_retain]
        metrics = (
            classification_metrics(retained)
            if task_type == "classification"
            else regression_metrics(retained)
        )
        risk = float(metrics["risk"])
        row = {
            "endpoint": str(group["endpoint"].iloc[0]),
            "task_type": task_type,
            "split_col": str(group["split_col"].iloc[0]),
            "model": str(group["model"].iloc[0]),
            "uncertainty_measure": uncertainty_col,
            "target_retained_fraction": retained_fraction,
            "n_total": int(len(ranked)),
            "n_retained": int(n_retain),
            "actual_retained_fraction": float(n_retain / len(ranked)),
            "mean_uncertainty_retained": float(retained[uncertainty_col].mean()),
            "max_uncertainty_retained": float(retained[uncertainty_col].max()),
            "full_risk": full_risk,
            "risk_reduction_vs_full": full_risk - risk,
            "relative_risk_reduction_vs_full": (
                (full_risk - risk) / full_risk if full_risk > 0 else math.nan
            ),
            **metrics,
        }
        if task_type == "classification":
            retained_positive_count = int(metrics["positive_count"])
            retained_negative_count = int(metrics["negative_count"])
            row.update(
                {
                    "full_positive_count": full_positive_count,
                    "full_negative_count": full_negative_count,
                    "positive_retention_rate": (
                        retained_positive_count / full_positive_count
                        if full_positive_count > 0
                        else math.nan
                    ),
                    "negative_retention_rate": (
                        retained_negative_count / full_negative_count
                        if full_negative_count > 0
                        else math.nan
                    ),
                    "class_balance_shift": (
                        float(metrics["positive_ratio"])
                        - float(full_metrics["positive_ratio"])
                    ),
                }
            )
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    details_path = APPLICABILITY_DIR / f"applicability_details_{args.mode}.csv"
    if not details_path.exists():
        raise FileNotFoundError(
            f"Missing {details_path}. Run 10_applicability_domain_analysis.py --mode {args.mode} first."
        )

    predictions = load_test_predictions()
    applicability = pd.read_csv(details_path)
    applicability = applicability[applicability["role"] == "test"].copy()

    endpoints = resolve_endpoints(args.mode, args.endpoints)
    splits = resolve_splits(args.mode, args.splits)
    if endpoints is not None:
        predictions = predictions[predictions["endpoint"].isin(endpoints)].copy()
        applicability = applicability[applicability["endpoint"].isin(endpoints)].copy()
    predictions = predictions[predictions["split_col"].isin(splits)].copy()
    applicability = applicability[applicability["split_col"].isin(splits)].copy()

    merge_cols = ["endpoint", "task_type", "split_col", "row_index"]
    ad_cols = merge_cols + ["max_tanimoto_to_train", "domain_bin", "unseen_scaffold"]
    merged = predictions.merge(
        applicability[ad_cols],
        on=merge_cols,
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        raise ValueError("Prediction/applicability merge produced zero rows.")

    rows = []
    group_cols = ["endpoint", "task_type", "split_col", "model"]
    for keys, group in merged.groupby(group_cols, sort=True):
        endpoint, task_type, split_col, model = keys
        print(f"selective endpoint={endpoint} split={split_col} model={model}")
        enriched = add_uncertainty_columns(group)
        for uncertainty_col in uncertainty_columns(task_type):
            rows.extend(evaluate_curve(enriched, uncertainty_col))

    summary = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / f"selective_prediction_summary_{args.mode}.csv"
    summary.to_csv(out_path, index=False)

    warning_count = int(summary.get("low_class_count_warning", pd.Series(dtype=bool)).fillna(False).sum())
    print("\nsaved", out_path)
    print(f"Selective-prediction analysis complete. Low-class-count rows: {warning_count}.")


if __name__ == "__main__":
    main()
