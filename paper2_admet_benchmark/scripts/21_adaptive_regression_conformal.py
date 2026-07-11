from __future__ import annotations

"""Locally adaptive split conformal intervals for regression endpoints.

The original calibration role is split once, using a deterministic RNG derived from
the split seed. One half fits an error-scale model from ECFP features to log absolute
residuals; the other half calibrates normalized residual scores. Test labels are never
used for fitting, scale estimation, or quantile selection.
"""

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper2_admet_benchmark"
PREDICTION_DIR = PAPER_DIR / "results" / "predictions"
PROCESSED_DIR = PAPER_DIR / "data" / "processed"
CONFORMAL_DIR = PAPER_DIR / "results" / "conformal"

DEFAULT_ALPHAS = [0.10, 0.20]
SCALE_TREES = 300
SCALE_MIN_LEAF = 20
SCALE_FIT_FRACTION = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run locally adaptive regression conformal analysis.")
    parser.add_argument("--mode", default="confirmatory_smoke_seed99")
    parser.add_argument("--pattern", default="*confirm*seed99*_predictions.csv")
    parser.add_argument("--endpoints", default=None)
    parser.add_argument("--splits", default=None)
    parser.add_argument("--alphas", default=",".join(str(x) for x in DEFAULT_ALPHAS))
    return parser.parse_args()


def parse_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_alphas(value: str) -> list[float]:
    out = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not out or any(not 0 < alpha < 1 for alpha in out):
        raise ValueError(f"Invalid alpha values: {out}")
    return out


def infer_seed(split_col: str) -> int:
    match = re.search(r"seed(\d+)$", split_col)
    return int(match.group(1)) if match else 0


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    if n == 0:
        raise ValueError("Cannot calibrate with zero normalized residuals.")
    rank = int(math.ceil((n + 1) * (1 - alpha)))
    rank = min(max(rank, 1), n)
    return float(scores[rank - 1])


def load_predictions(pattern: str) -> pd.DataFrame:
    files = sorted(PREDICTION_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No prediction files matched {pattern}")
    out = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    required = {"endpoint", "task_type", "split_col", "role", "model", "row_index", "y_true", "y_output"}
    missing = required - set(out.columns)
    if missing:
        raise KeyError(f"Prediction files missing columns: {sorted(missing)}")
    return out[out["task_type"] == "regression"].copy()


def filter_scope(df: pd.DataFrame, endpoints: list[str] | None, splits: list[str] | None) -> pd.DataFrame:
    out = df.copy()
    if endpoints is not None:
        out = out[out["endpoint"].str.lower().isin({x.lower() for x in endpoints})].copy()
    if splits is not None:
        out = out[out["split_col"].isin(splits)].copy()
    return out


def split_calibration(calibration: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(calibration) < 40:
        raise ValueError(f"Adaptive conformal requires at least 40 calibration rows, got {len(calibration)}")
    rng = np.random.default_rng(seed + 7200)
    order = rng.permutation(len(calibration))
    n_scale = int(round(len(calibration) * SCALE_FIT_FRACTION))
    n_scale = min(max(n_scale, 20), len(calibration) - 20)
    scale_fit = calibration.iloc[order[:n_scale]].copy()
    quantile_cal = calibration.iloc[order[n_scale:]].copy()
    return scale_fit, quantile_cal


def fit_scale_model(
    X: np.ndarray,
    scale_fit: pd.DataFrame,
    seed: int,
) -> tuple[RandomForestRegressor, float]:
    rows = scale_fit["row_index"].astype(int).to_numpy()
    residuals = np.abs(
        scale_fit["y_true"].astype(float).to_numpy()
        - scale_fit["y_output"].astype(float).to_numpy()
    )
    floor = max(float(np.median(residuals) * 0.10), 1e-3)
    target = np.log1p(residuals)
    model = RandomForestRegressor(
        n_estimators=SCALE_TREES,
        min_samples_leaf=SCALE_MIN_LEAF,
        max_features="sqrt",
        random_state=seed + 7200,
        n_jobs=-1,
    )
    model.fit(X[rows], target)
    return model, floor


def predict_scale(
    model: RandomForestRegressor,
    X: np.ndarray,
    rows: np.ndarray,
    floor: float,
) -> np.ndarray:
    scale = np.expm1(model.predict(X[rows])) + floor
    return np.clip(scale, floor, None)


def summarize(
    endpoint: str,
    split_col: str,
    model_name: str,
    alpha: float,
    test_out: pd.DataFrame,
    n_scale_fit: int,
    n_quantile_cal: int,
    floor: float,
    qhat: float,
) -> dict[str, object]:
    nominal = 1 - alpha
    empirical = float(test_out["covered"].mean())
    width = test_out["interval_width"].astype(float)
    error = test_out["absolute_error"].astype(float)
    predicted_scale = test_out["predicted_error_scale"].astype(float)
    correlation = (
        float(np.corrcoef(predicted_scale, error)[0, 1])
        if predicted_scale.nunique() > 1 and error.nunique() > 1
        else math.nan
    )
    return {
        "endpoint": endpoint,
        "task_type": "regression",
        "split_col": split_col,
        "model": model_name,
        "method": "split_calibration_adaptive_normalized",
        "alpha": alpha,
        "nominal_coverage": nominal,
        "n_scale_fit": n_scale_fit,
        "n_quantile_calibration": n_quantile_cal,
        "n_test": int(len(test_out)),
        "scale_floor": floor,
        "normalized_qhat": qhat,
        "empirical_coverage": empirical,
        "coverage_gap": empirical - nominal,
        "mean_interval_width": float(width.mean()),
        "median_interval_width": float(width.median()),
        "interval_width_std": float(width.std(ddof=1)),
        "mean_absolute_error": float(error.mean()),
        "scale_error_correlation": correlation,
    }


def main() -> None:
    args = parse_args()
    alphas = parse_alphas(args.alphas)
    predictions = filter_scope(
        load_predictions(args.pattern),
        parse_list(args.endpoints),
        parse_list(args.splits),
    )
    if predictions.empty:
        raise ValueError("No regression predictions remain after filtering.")

    CONFORMAL_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []

    for (endpoint, split_col, model_name), group in predictions.groupby(
        ["endpoint", "split_col", "model"], sort=True
    ):
        calibration = group[group["role"] == "calibration"].copy()
        test = group[group["role"] == "test"].copy()
        if calibration.empty or test.empty:
            raise ValueError(f"Missing calibration/test for {endpoint} {split_col} {model_name}")
        feature_path = PROCESSED_DIR / f"{endpoint}_features.npz"
        X = np.load(feature_path, allow_pickle=True)["X"].astype(np.float32)
        seed = infer_seed(split_col)
        scale_fit, quantile_cal = split_calibration(calibration, seed)
        scale_model, floor = fit_scale_model(X, scale_fit, seed)

        quantile_rows = quantile_cal["row_index"].astype(int).to_numpy()
        quantile_scale = predict_scale(scale_model, X, quantile_rows, floor)
        quantile_error = np.abs(
            quantile_cal["y_true"].astype(float).to_numpy()
            - quantile_cal["y_output"].astype(float).to_numpy()
        )
        normalized_scores = quantile_error / quantile_scale

        test_rows = test["row_index"].astype(int).to_numpy()
        test_scale = predict_scale(scale_model, X, test_rows, floor)
        y_true = test["y_true"].astype(float).to_numpy()
        y_pred = test["y_output"].astype(float).to_numpy()
        print(
            f"adaptive-regression endpoint={endpoint} split={split_col} model={model_name} "
            f"scale_n={len(scale_fit)} quantile_n={len(quantile_cal)}"
        )

        for alpha in alphas:
            qhat = conformal_quantile(normalized_scores, alpha)
            half_width = qhat * test_scale
            lower = y_pred - half_width
            upper = y_pred + half_width
            output = test.copy()
            output["method"] = "split_calibration_adaptive_normalized"
            output["alpha"] = alpha
            output["nominal_coverage"] = 1 - alpha
            output["predicted_error_scale"] = test_scale
            output["normalized_qhat"] = qhat
            output["interval_lower"] = lower
            output["interval_upper"] = upper
            output["interval_width"] = 2 * half_width
            output["absolute_error"] = np.abs(y_true - y_pred)
            output["covered"] = (y_true >= lower) & (y_true <= upper)
            summary_rows.append(
                summarize(
                    endpoint,
                    split_col,
                    model_name,
                    alpha,
                    output,
                    len(scale_fit),
                    len(quantile_cal),
                    floor,
                    qhat,
                )
            )
            detail_frames.append(output)

    summary = pd.DataFrame(summary_rows)
    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    if summary.empty or details.empty:
        raise RuntimeError("Adaptive regression conformal produced empty outputs.")
    summary_path = CONFORMAL_DIR / f"adaptive_regression_conformal_summary_{args.mode}.csv"
    detail_path = CONFORMAL_DIR / f"adaptive_regression_conformal_predictions_{args.mode}.csv"
    summary.to_csv(summary_path, index=False)
    details.to_csv(detail_path, index=False)
    print("saved", summary_path)
    print("saved", detail_path)
    print(f"Adaptive regression conformal complete. Summary rows: {len(summary)}")


if __name__ == "__main__":
    main()
