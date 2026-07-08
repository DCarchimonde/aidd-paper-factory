from __future__ import annotations

"""Compute split-conformal diagnostics for Paper 2 predictions.

Classification:
- calibration nonconformity score: 1 - predicted probability of the true label
- test prediction set: labels whose nonconformity score is <= calibration threshold
- outputs empirical coverage and mean prediction-set size

Regression:
- calibration nonconformity score: absolute residual
- test interval: prediction +/- calibration threshold
- outputs empirical coverage and interval width

Important interpretation:
These are empirical split-conformal diagnostics under the chosen random/scaffold
calibration/test protocol. Do not describe them as theoretical guarantees under
chemical distribution shift.

Inputs:
- paper2_admet_benchmark/results/predictions/*_predictions.csv

Outputs:
- paper2_admet_benchmark/results/conformal/conformal_summary_<mode>.csv
- paper2_admet_benchmark/results/conformal/conformal_predictions_<mode>.csv
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PAPER_DIR = ROOT / "paper2_admet_benchmark"
PREDICTION_DIR = PAPER_DIR / "results" / "predictions"
CONFORMAL_DIR = PAPER_DIR / "results" / "conformal"

ALPHAS = [0.10, 0.20]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute split-conformal diagnostics from saved predictions.")
    parser.add_argument(
        "--mode",
        default="mvp",
        help="Output suffix only, e.g. smoke, mvp, full.",
    )
    parser.add_argument(
        "--pattern",
        default="*_predictions.csv",
        help="Prediction-file glob pattern under results/predictions.",
    )
    parser.add_argument(
        "--alphas",
        default=",".join(str(a) for a in ALPHAS),
        help="Comma-separated alpha values, e.g. 0.1,0.2.",
    )
    return parser.parse_args()


def parse_alphas(value: str) -> list[float]:
    alphas = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not alphas:
        raise ValueError("At least one alpha is required.")
    for alpha in alphas:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    return alphas


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


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile using the ceil((n+1)(1-alpha)) rule."""
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    if n == 0:
        raise ValueError("Cannot compute conformal quantile with zero calibration scores.")
    rank = int(math.ceil((n + 1) * (1 - alpha)))
    rank = min(max(rank, 1), n)
    return float(scores[rank - 1])


def classification_scores(df: pd.DataFrame) -> np.ndarray:
    y_true = df["y_true"].astype(int).to_numpy()
    p1 = np.clip(df["y_output"].astype(float).to_numpy(), 1e-7, 1 - 1e-7)
    p_true = np.where(y_true == 1, p1, 1 - p1)
    return 1 - p_true


def classification_prediction_sets(df: pd.DataFrame, qhat: float) -> pd.DataFrame:
    p1 = np.clip(df["y_output"].astype(float).to_numpy(), 1e-7, 1 - 1e-7)
    p0 = 1 - p1
    y_true = df["y_true"].astype(int).to_numpy()

    include_0 = (1 - p0) <= qhat
    include_1 = (1 - p1) <= qhat
    set_size = include_0.astype(int) + include_1.astype(int)
    covered = np.where(y_true == 0, include_0, include_1)

    out = df.copy()
    out["qhat"] = qhat
    out["include_label_0"] = include_0
    out["include_label_1"] = include_1
    out["prediction_set_size"] = set_size
    out["covered"] = covered.astype(bool)
    out["empty_set"] = set_size == 0
    out["singleton_set"] = set_size == 1
    out["ambiguous_set"] = set_size == 2
    return out


def regression_scores(df: pd.DataFrame) -> np.ndarray:
    y_true = df["y_true"].astype(float).to_numpy()
    y_pred = df["y_output"].astype(float).to_numpy()
    return np.abs(y_true - y_pred)


def regression_intervals(df: pd.DataFrame, qhat: float) -> pd.DataFrame:
    y_true = df["y_true"].astype(float).to_numpy()
    y_pred = df["y_output"].astype(float).to_numpy()
    lower = y_pred - qhat
    upper = y_pred + qhat
    covered = (y_true >= lower) & (y_true <= upper)

    out = df.copy()
    out["qhat"] = qhat
    out["interval_lower"] = lower
    out["interval_upper"] = upper
    out["interval_width"] = 2 * qhat
    out["covered"] = covered.astype(bool)
    out["absolute_error"] = np.abs(y_true - y_pred)
    return out


def summarize_conformal(
    endpoint: str,
    task_type: str,
    split_col: str,
    model: str,
    alpha: float,
    qhat: float,
    calibration_n: int,
    test_out: pd.DataFrame,
) -> dict:
    base = {
        "endpoint": endpoint,
        "task_type": task_type,
        "split_col": split_col,
        "model": model,
        "alpha": alpha,
        "nominal_coverage": 1 - alpha,
        "qhat": qhat,
        "n_calibration": int(calibration_n),
        "n_test": int(len(test_out)),
        "empirical_coverage": float(test_out["covered"].mean()) if len(test_out) else math.nan,
    }
    if task_type == "classification":
        base.update(
            {
                "mean_prediction_set_size": float(test_out["prediction_set_size"].mean()),
                "empty_set_rate": float(test_out["empty_set"].mean()),
                "singleton_rate": float(test_out["singleton_set"].mean()),
                "ambiguous_set_rate": float(test_out["ambiguous_set"].mean()),
            }
        )
    else:
        base.update(
            {
                "mean_interval_width": float(test_out["interval_width"].mean()),
                "median_absolute_error": float(test_out["absolute_error"].median()),
                "mean_absolute_error": float(test_out["absolute_error"].mean()),
            }
        )
    return base


def main() -> None:
    args = parse_args()
    alphas = parse_alphas(args.alphas)
    CONFORMAL_DIR.mkdir(parents=True, exist_ok=True)

    predictions = load_prediction_files(args.pattern)
    group_cols = ["endpoint", "task_type", "split_col", "model"]

    summary_rows = []
    prediction_rows = []

    for keys, group in predictions.groupby(group_cols, sort=True):
        endpoint, task_type, split_col, model = keys
        calibration = group[group["role"] == "calibration"].copy()
        test = group[group["role"] == "test"].copy()
        if calibration.empty or test.empty:
            print(f"[SKIP] endpoint={endpoint} split={split_col} model={model}: missing calibration or test predictions")
            continue

        print(f"conformal endpoint={endpoint} split={split_col} model={model}")
        if task_type == "classification":
            cal_scores = classification_scores(calibration)
        else:
            cal_scores = regression_scores(calibration)

        for alpha in alphas:
            qhat = conformal_quantile(cal_scores, alpha=alpha)
            if task_type == "classification":
                test_out = classification_prediction_sets(test, qhat=qhat)
            else:
                test_out = regression_intervals(test, qhat=qhat)

            test_out["alpha"] = alpha
            test_out["nominal_coverage"] = 1 - alpha
            summary_rows.append(
                summarize_conformal(
                    endpoint=endpoint,
                    task_type=task_type,
                    split_col=split_col,
                    model=model,
                    alpha=alpha,
                    qhat=qhat,
                    calibration_n=len(calibration),
                    test_out=test_out,
                )
            )
            prediction_rows.append(test_out)

    summary_df = pd.DataFrame(summary_rows)
    predictions_df = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()

    summary_path = CONFORMAL_DIR / f"conformal_summary_{args.mode}.csv"
    predictions_path = CONFORMAL_DIR / f"conformal_predictions_{args.mode}.csv"
    summary_df.to_csv(summary_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    print("\nsaved", summary_path)
    print("saved", predictions_path)
    print("Conformal analysis complete.")


if __name__ == "__main__":
    main()
