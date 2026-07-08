from __future__ import annotations

"""Compute calibration diagnostics for Paper 2 baseline predictions.

Classification outputs:
- Brier score
- negative log-likelihood
- expected calibration error (ECE)
- maximum calibration error (MCE)
- reliability-bin tables

Regression outputs:
- residual calibration-style summaries by predicted-value bins
- RMSE/MAE/bias by bin

Inputs:
- paper2_admet_benchmark/results/predictions/*_predictions.csv

Outputs:
- paper2_admet_benchmark/results/calibration/calibration_summary_<mode>.csv
- paper2_admet_benchmark/results/calibration/reliability_bins_<mode>.csv
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
    parser = argparse.ArgumentParser(description="Compute calibration diagnostics from saved predictions.")
    parser.add_argument(
        "--mode",
        default="mvp",
        help="Output suffix only, e.g. smoke, mvp, full.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help="Number of bins for ECE/reliability analysis.",
    )
    parser.add_argument(
        "--pattern",
        default="*_predictions.csv",
        help="Prediction-file glob pattern under results/predictions.",
    )
    return parser.parse_args()


def load_prediction_files(pattern: str) -> pd.DataFrame:
    files = sorted(PREDICTION_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No prediction files found in {PREDICTION_DIR} with pattern {pattern}")

    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["prediction_file"] = str(path.relative_to(ROOT))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def classification_calibration(group: pd.DataFrame, n_bins: int) -> tuple[dict, list[dict]]:
    y_true = group["y_true"].astype(int).to_numpy()
    y_score = np.clip(group["y_output"].astype(float).to_numpy(), 1e-7, 1 - 1e-7)
    y_pred = (y_score >= 0.5).astype(int)
    confidence = np.maximum(y_score, 1 - y_score)
    correctness = (y_pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_rows = []
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        left = bins[i]
        right = bins[i + 1]
        if i == n_bins - 1:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)
        n_bin = int(mask.sum())
        if n_bin == 0:
            avg_conf = math.nan
            acc = math.nan
            gap = math.nan
            pos_rate = math.nan
            mean_score = math.nan
        else:
            avg_conf = float(confidence[mask].mean())
            acc = float(correctness[mask].mean())
            gap = abs(acc - avg_conf)
            pos_rate = float(y_true[mask].mean())
            mean_score = float(y_score[mask].mean())
            ece += (n_bin / len(group)) * gap
            mce = max(mce, gap)
        bin_rows.append(
            {
                "bin_id": i,
                "bin_left": float(left),
                "bin_right": float(right),
                "n_bin": n_bin,
                "fraction": float(n_bin / len(group)) if len(group) else math.nan,
                "avg_confidence": avg_conf,
                "accuracy": acc,
                "calibration_gap": gap,
                "positive_rate": pos_rate,
                "mean_positive_probability": mean_score,
            }
        )

    summary = {
        "n": int(len(group)),
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "negative_log_likelihood": float(log_loss(y_true, y_score, labels=[0, 1])) if len(np.unique(y_true)) >= 2 else math.nan,
        "ece": float(ece),
        "mce": float(mce),
        "mean_confidence": float(confidence.mean()),
        "accuracy": float(correctness.mean()),
        "positive_rate": float(y_true.mean()),
        "mean_positive_probability": float(y_score.mean()),
    }
    return summary, bin_rows


def regression_calibration(group: pd.DataFrame, n_bins: int) -> tuple[dict, list[dict]]:
    y_true = group["y_true"].astype(float).to_numpy()
    y_pred = group["y_output"].astype(float).to_numpy()
    residual = y_true - y_pred
    abs_error = np.abs(residual)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(y_pred, quantiles))
    if len(edges) <= 2:
        edges = np.linspace(float(y_pred.min()), float(y_pred.max()), min(n_bins, max(2, len(y_pred))) + 1)

    bin_rows = []
    for i in range(len(edges) - 1):
        left = edges[i]
        right = edges[i + 1]
        if i == len(edges) - 2:
            mask = (y_pred >= left) & (y_pred <= right)
        else:
            mask = (y_pred >= left) & (y_pred < right)
        n_bin = int(mask.sum())
        if n_bin == 0:
            rmse = mae = bias = mean_true = mean_pred = math.nan
        else:
            rmse = float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))
            mae = float(mean_absolute_error(y_true[mask], y_pred[mask]))
            bias = float(residual[mask].mean())
            mean_true = float(y_true[mask].mean())
            mean_pred = float(y_pred[mask].mean())
        bin_rows.append(
            {
                "bin_id": i,
                "bin_left": float(left),
                "bin_right": float(right),
                "n_bin": n_bin,
                "fraction": float(n_bin / len(group)) if len(group) else math.nan,
                "rmse": rmse,
                "mae": mae,
                "bias": bias,
                "mean_true": mean_true,
                "mean_prediction": mean_pred,
            }
        )

    summary = {
        "n": int(len(group)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(residual.mean()),
        "mean_abs_error": float(abs_error.mean()),
        "median_abs_error": float(np.median(abs_error)),
        "mean_prediction": float(y_pred.mean()),
        "mean_true": float(y_true.mean()),
    }
    return summary, bin_rows


def main() -> None:
    args = parse_args()
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

    predictions = load_prediction_files(args.pattern)
    group_cols = ["endpoint", "task_type", "split_col", "role", "model"]

    summary_rows = []
    bin_rows_all = []
    for keys, group in predictions.groupby(group_cols, sort=True):
        endpoint, task_type, split_col, role, model = keys
        print(f"calibration endpoint={endpoint} split={split_col} role={role} model={model}")
        if task_type == "classification":
            summary, bin_rows = classification_calibration(group, n_bins=args.bins)
        else:
            summary, bin_rows = regression_calibration(group, n_bins=args.bins)

        base = {
            "endpoint": endpoint,
            "task_type": task_type,
            "split_col": split_col,
            "role": role,
            "model": model,
            "n_bins": args.bins,
        }
        summary_rows.append({**base, **summary})
        for row in bin_rows:
            bin_rows_all.append({**base, **row})

    summary_df = pd.DataFrame(summary_rows)
    bins_df = pd.DataFrame(bin_rows_all)

    summary_path = CALIBRATION_DIR / f"calibration_summary_{args.mode}.csv"
    bins_path = CALIBRATION_DIR / f"reliability_bins_{args.mode}.csv"
    summary_df.to_csv(summary_path, index=False)
    bins_df.to_csv(bins_path, index=False)

    print("\nsaved", summary_path)
    print("saved", bins_path)
    print("Calibration analysis complete.")


if __name__ == "__main__":
    main()
