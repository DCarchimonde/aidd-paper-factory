from __future__ import annotations

"""Compute probability and confidence calibration diagnostics.

For binary classification this script reports two distinct calibration views:

1. probability calibration: mean predicted positive probability versus observed
   positive frequency;
2. confidence calibration: maximum class probability versus classification accuracy.

The two quantities answer different questions and must not be conflated. Regression
outputs retain residual summaries by predicted-value quantile bins.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PAPER_DIR = ROOT / "paper2_admet_benchmark"
PREDICTION_DIR = PAPER_DIR / "results" / "predictions"
CALIBRATION_DIR = PAPER_DIR / "results" / "calibration"
DEFAULT_BINS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute dual-view calibration diagnostics.")
    parser.add_argument("--mode", default="confirmatory_smoke_seed99")
    parser.add_argument("--pattern", default="*confirm*seed99*_predictions.csv")
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    return parser.parse_args()


def load_prediction_files(pattern: str) -> pd.DataFrame:
    files = sorted(PREDICTION_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No prediction files found in {PREDICTION_DIR} with pattern {pattern}"
        )
    frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path)
        frame["prediction_file"] = str(path.relative_to(ROOT))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def fixed_bin_rows(
    values: np.ndarray,
    observed: np.ndarray,
    n_bins: int,
    view: str,
) -> tuple[float, float, list[dict[str, object]]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    rows: list[dict[str, object]] = []
    n_total = len(values)

    for i in range(n_bins):
        left = float(edges[i])
        right = float(edges[i + 1])
        mask = (
            (values >= left) & (values <= right)
            if i == n_bins - 1
            else (values >= left) & (values < right)
        )
        n_bin = int(mask.sum())
        if n_bin:
            mean_value = float(values[mask].mean())
            mean_observed = float(observed[mask].mean())
            gap = abs(mean_observed - mean_value)
            ece += (n_bin / n_total) * gap
            mce = max(mce, gap)
        else:
            mean_value = math.nan
            mean_observed = math.nan
            gap = math.nan
        rows.append(
            {
                "calibration_view": view,
                "bin_id": i,
                "bin_left": left,
                "bin_right": right,
                "n_bin": n_bin,
                "fraction": float(n_bin / n_total) if n_total else math.nan,
                "mean_predicted_value": mean_value,
                "mean_observed_value": mean_observed,
                "calibration_gap": gap,
            }
        )
    return float(ece), float(mce), rows


def classification_calibration(
    group: pd.DataFrame,
    n_bins: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    y_true = group["y_true"].astype(int).to_numpy()
    p1 = np.clip(group["y_output"].astype(float).to_numpy(), 1e-7, 1 - 1e-7)
    y_pred = (p1 >= 0.5).astype(int)
    confidence = np.maximum(p1, 1 - p1)
    correctness = (y_pred == y_true).astype(float)

    ece_probability, mce_probability, probability_rows = fixed_bin_rows(
        values=p1,
        observed=y_true.astype(float),
        n_bins=n_bins,
        view="positive_probability",
    )
    ece_confidence, mce_confidence, confidence_rows = fixed_bin_rows(
        values=confidence,
        observed=correctness,
        n_bins=n_bins,
        view="classification_confidence",
    )

    summary: dict[str, object] = {
        "n": int(len(group)),
        "brier_score": float(brier_score_loss(y_true, p1)),
        "negative_log_likelihood": (
            float(log_loss(y_true, p1, labels=[0, 1]))
            if len(np.unique(y_true)) >= 2
            else math.nan
        ),
        "ece_probability": ece_probability,
        "mce_probability": mce_probability,
        "ece_confidence": ece_confidence,
        "mce_confidence": mce_confidence,
        "positive_rate": float(y_true.mean()),
        "mean_positive_probability": float(p1.mean()),
        "accuracy": float(correctness.mean()),
        "mean_confidence": float(confidence.mean()),
    }
    return summary, probability_rows + confidence_rows


def regression_calibration(
    group: pd.DataFrame,
    n_bins: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    y_true = group["y_true"].astype(float).to_numpy()
    y_pred = group["y_output"].astype(float).to_numpy()
    residual = y_true - y_pred
    abs_error = np.abs(residual)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(y_pred, quantiles))
    if len(edges) <= 2:
        edges = np.linspace(
            float(y_pred.min()),
            float(y_pred.max()),
            min(n_bins, max(2, len(y_pred))) + 1,
        )

    rows: list[dict[str, object]] = []
    for i in range(len(edges) - 1):
        left = float(edges[i])
        right = float(edges[i + 1])
        mask = (
            (y_pred >= left) & (y_pred <= right)
            if i == len(edges) - 2
            else (y_pred >= left) & (y_pred < right)
        )
        n_bin = int(mask.sum())
        if n_bin:
            rmse = float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))
            mae = float(mean_absolute_error(y_true[mask], y_pred[mask]))
            bias = float(residual[mask].mean())
            mean_true = float(y_true[mask].mean())
            mean_prediction = float(y_pred[mask].mean())
        else:
            rmse = mae = bias = mean_true = mean_prediction = math.nan
        rows.append(
            {
                "calibration_view": "regression_prediction_bin",
                "bin_id": i,
                "bin_left": left,
                "bin_right": right,
                "n_bin": n_bin,
                "fraction": float(n_bin / len(group)) if len(group) else math.nan,
                "rmse": rmse,
                "mae": mae,
                "bias": bias,
                "mean_true": mean_true,
                "mean_prediction": mean_prediction,
            }
        )

    summary: dict[str, object] = {
        "n": int(len(group)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(residual.mean()),
        "mean_abs_error": float(abs_error.mean()),
        "median_abs_error": float(np.median(abs_error)),
        "mean_prediction": float(y_pred.mean()),
        "mean_true": float(y_true.mean()),
    }
    return summary, rows


def main() -> None:
    args = parse_args()
    if args.bins < 2:
        raise ValueError("--bins must be at least 2")
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    predictions = load_prediction_files(args.pattern)

    group_cols = ["endpoint", "task_type", "split_col", "role", "model"]
    summary_rows: list[dict[str, object]] = []
    bin_rows: list[dict[str, object]] = []

    for keys, group in predictions.groupby(group_cols, sort=True):
        endpoint, task_type, split_col, role, model = keys
        print(
            f"calibration-v2 endpoint={endpoint} split={split_col} "
            f"role={role} model={model}"
        )
        if task_type == "classification":
            summary, rows = classification_calibration(group, n_bins=args.bins)
        else:
            summary, rows = regression_calibration(group, n_bins=args.bins)
        base = {
            "endpoint": endpoint,
            "task_type": task_type,
            "split_col": split_col,
            "role": role,
            "model": model,
            "n_bins": args.bins,
        }
        summary_rows.append({**base, **summary})
        bin_rows.extend({**base, **row} for row in rows)

    summary_df = pd.DataFrame(summary_rows)
    bins_df = pd.DataFrame(bin_rows)
    if summary_df.empty:
        raise RuntimeError("Calibration analysis produced zero rows; refusing to write empty outputs.")

    summary_path = CALIBRATION_DIR / f"calibration_summary_v2_{args.mode}.csv"
    bins_path = CALIBRATION_DIR / f"reliability_bins_v2_{args.mode}.csv"
    summary_df.to_csv(summary_path, index=False)
    bins_df.to_csv(bins_path, index=False)
    print("\nsaved", summary_path)
    print("saved", bins_path)
    print(f"Dual-view calibration analysis complete. Summary rows: {len(summary_df)}.")


if __name__ == "__main__":
    main()
