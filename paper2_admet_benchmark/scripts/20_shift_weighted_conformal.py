from __future__ import annotations

"""Transductive covariate-shift-weighted conformal diagnostics.

Density ratios between calibration and test covariates are estimated with a
cross-fitted logistic domain classifier using ECFP features only. Test labels are
never used. Weighted conformal quantiles include the test-point mass at infinity,
as required by the weighted split-conformal construction.

The method is an empirical shift-aware diagnostic when density ratios are estimated.
It must not be described as an exact finite-sample guarantee under arbitrary shift.
"""

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper2_admet_benchmark"
PREDICTION_DIR = PAPER_DIR / "results" / "predictions"
PROCESSED_DIR = PAPER_DIR / "data" / "processed"
CONFORMAL_DIR = PAPER_DIR / "results" / "conformal"

DEFAULT_ALPHAS = [0.10, 0.20]
WEIGHT_MIN = 0.05
WEIGHT_MAX = 20.0
DOMAIN_FOLDS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run covariate-shift-weighted conformal analysis.")
    parser.add_argument("--mode", default="confirmatory_smoke_seed99")
    parser.add_argument("--pattern", default="*confirm*seed99*_predictions.csv")
    parser.add_argument("--endpoints", default=None)
    parser.add_argument("--splits", default=None)
    parser.add_argument("--alphas", default=",".join(str(x) for x in DEFAULT_ALPHAS))
    parser.add_argument("--weight-min", type=float, default=WEIGHT_MIN)
    parser.add_argument("--weight-max", type=float, default=WEIGHT_MAX)
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


def load_predictions(pattern: str) -> pd.DataFrame:
    files = sorted(PREDICTION_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No prediction files matched {pattern} in {PREDICTION_DIR}")
    frames = [pd.read_csv(path) for path in files]
    out = pd.concat(frames, ignore_index=True)
    required = {"endpoint", "task_type", "split_col", "role", "model", "row_index", "y_true", "y_output"}
    missing = required - set(out.columns)
    if missing:
        raise KeyError(f"Prediction files missing columns: {sorted(missing)}")
    return out


def filter_scope(df: pd.DataFrame, endpoints: list[str] | None, splits: list[str] | None) -> pd.DataFrame:
    out = df.copy()
    if endpoints is not None:
        out = out[out["endpoint"].str.lower().isin({x.lower() for x in endpoints})].copy()
    if splits is not None:
        out = out[out["split_col"].isin(splits)].copy()
    return out


def cross_fitted_density_ratios(
    X_cal: np.ndarray,
    X_test: np.ndarray,
    seed: int,
    weight_min: float,
    weight_max: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    X_domain = np.vstack([X_cal, X_test]).astype(np.float32, copy=False)
    y_domain = np.concatenate([
        np.zeros(len(X_cal), dtype=int),
        np.ones(len(X_test), dtype=int),
    ])
    min_class = int(np.bincount(y_domain).min())
    n_splits = min(DOMAIN_FOLDS, min_class)
    if n_splits < 2:
        raise ValueError("Need at least two examples in each domain for cross-fitting.")

    oof_probability = np.empty(len(y_domain), dtype=float)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + 9100)
    for train_idx, valid_idx in cv.split(X_domain, y_domain):
        model = Pipeline(
            [
                ("scale", StandardScaler(with_mean=False)),
                (
                    "domain",
                    LogisticRegression(
                        C=1.0,
                        solver="liblinear",
                        max_iter=2000,
                        class_weight=None,
                        random_state=seed + 9100,
                    ),
                ),
            ]
        )
        model.fit(X_domain[train_idx], y_domain[train_idx])
        oof_probability[valid_idx] = model.predict_proba(X_domain[valid_idx])[:, 1]

    clipped_probability = np.clip(oof_probability, 1e-6, 1 - 1e-6)
    prior_correction = len(X_cal) / len(X_test)
    ratios = (clipped_probability / (1 - clipped_probability)) * prior_correction
    ratios = np.clip(ratios, weight_min, weight_max)
    cal_weights = ratios[: len(X_cal)]
    test_weights = ratios[len(X_cal) :]

    total_cal_weight = float(cal_weights.sum())
    ess = float(total_cal_weight**2 / np.square(cal_weights).sum())
    diagnostics = {
        "domain_classifier_auc": float(roc_auc_score(y_domain, oof_probability)),
        "calibration_weight_mean": float(cal_weights.mean()),
        "calibration_weight_min": float(cal_weights.min()),
        "calibration_weight_max": float(cal_weights.max()),
        "test_weight_mean": float(test_weights.mean()),
        "test_weight_min": float(test_weights.min()),
        "test_weight_max": float(test_weights.max()),
        "calibration_weight_ess": ess,
    }
    return cal_weights, test_weights, diagnostics


def weighted_test_quantiles(
    calibration_scores: np.ndarray,
    calibration_weights: np.ndarray,
    test_weights: np.ndarray,
    alpha: float,
) -> np.ndarray:
    order = np.argsort(calibration_scores, kind="mergesort")
    sorted_scores = np.asarray(calibration_scores, dtype=float)[order]
    sorted_weights = np.asarray(calibration_weights, dtype=float)[order]
    cumulative = np.cumsum(sorted_weights)
    total_calibration_weight = float(cumulative[-1])
    targets = (1 - alpha) * (total_calibration_weight + np.asarray(test_weights, dtype=float))
    positions = np.searchsorted(cumulative, targets, side="left")
    qhat = np.full(len(test_weights), np.inf, dtype=float)
    finite = positions < len(sorted_scores)
    qhat[finite] = sorted_scores[positions[finite]]
    return qhat


def classification_scores(calibration: pd.DataFrame) -> np.ndarray:
    y_true = calibration["y_true"].astype(int).to_numpy()
    p1 = np.clip(calibration["y_output"].astype(float).to_numpy(), 1e-7, 1 - 1e-7)
    p_true = np.where(y_true == 1, p1, 1 - p1)
    return 1 - p_true


def regression_scores(calibration: pd.DataFrame) -> np.ndarray:
    return np.abs(
        calibration["y_true"].astype(float).to_numpy()
        - calibration["y_output"].astype(float).to_numpy()
    )


def make_classification_output(test: pd.DataFrame, qhat: np.ndarray) -> pd.DataFrame:
    p1 = np.clip(test["y_output"].astype(float).to_numpy(), 1e-7, 1 - 1e-7)
    p0 = 1 - p1
    y_true = test["y_true"].astype(int).to_numpy()
    include_0 = (1 - p0) <= qhat
    include_1 = (1 - p1) <= qhat
    size = include_0.astype(int) + include_1.astype(int)
    covered = np.where(y_true == 0, include_0, include_1)
    out = test.copy()
    out["method"] = "density_ratio_weighted_lac"
    out["qhat_local"] = qhat
    out["include_label_0"] = include_0
    out["include_label_1"] = include_1
    out["prediction_set_size"] = size
    out["covered"] = covered.astype(bool)
    out["empty_set"] = size == 0
    out["singleton_set"] = size == 1
    out["ambiguous_set"] = size == 2
    out["infinite_qhat"] = ~np.isfinite(qhat)
    return out


def make_regression_output(test: pd.DataFrame, qhat: np.ndarray) -> pd.DataFrame:
    y_true = test["y_true"].astype(float).to_numpy()
    y_pred = test["y_output"].astype(float).to_numpy()
    lower = y_pred - qhat
    upper = y_pred + qhat
    covered = (y_true >= lower) & (y_true <= upper)
    out = test.copy()
    out["method"] = "density_ratio_weighted_absolute_residual"
    out["qhat_local"] = qhat
    out["interval_lower"] = lower
    out["interval_upper"] = upper
    out["interval_width"] = 2 * qhat
    out["covered"] = covered.astype(bool)
    out["absolute_error"] = np.abs(y_true - y_pred)
    out["infinite_qhat"] = ~np.isfinite(qhat)
    return out


def summarize(
    endpoint: str,
    task_type: str,
    split_col: str,
    model: str,
    alpha: float,
    output: pd.DataFrame,
    diagnostics: dict[str, float],
) -> dict[str, object]:
    nominal = 1 - alpha
    empirical = float(output["covered"].mean())
    row: dict[str, object] = {
        "endpoint": endpoint,
        "task_type": task_type,
        "split_col": split_col,
        "model": model,
        "method": str(output["method"].iloc[0]),
        "alpha": alpha,
        "nominal_coverage": nominal,
        "n_test": int(len(output)),
        "empirical_coverage": empirical,
        "coverage_gap": empirical - nominal,
        "median_local_qhat": float(np.median(output.loc[np.isfinite(output["qhat_local"]), "qhat_local"]))
        if np.isfinite(output["qhat_local"]).any()
        else math.inf,
        "infinite_qhat_rate": float(output["infinite_qhat"].mean()),
        **diagnostics,
    }
    if task_type == "classification":
        y_true = output["y_true"].astype(int)
        pos = y_true == 1
        neg = y_true == 0
        row.update(
            {
                "positive_count": int(pos.sum()),
                "negative_count": int(neg.sum()),
                "positive_coverage": float(output.loc[pos, "covered"].mean()) if pos.any() else math.nan,
                "negative_coverage": float(output.loc[neg, "covered"].mean()) if neg.any() else math.nan,
                "mean_prediction_set_size": float(output["prediction_set_size"].mean()),
                "empty_set_rate": float(output["empty_set"].mean()),
                "singleton_rate": float(output["singleton_set"].mean()),
                "ambiguous_set_rate": float(output["ambiguous_set"].mean()),
            }
        )
    else:
        finite_width = output.loc[np.isfinite(output["interval_width"]), "interval_width"]
        row.update(
            {
                "mean_interval_width": float(finite_width.mean()) if len(finite_width) else math.inf,
                "median_interval_width": float(finite_width.median()) if len(finite_width) else math.inf,
                "mean_absolute_error": float(output["absolute_error"].mean()),
            }
        )
    return row


def main() -> None:
    args = parse_args()
    if not 0 < args.weight_min <= args.weight_max:
        raise ValueError("Require 0 < weight_min <= weight_max")
    alphas = parse_alphas(args.alphas)
    predictions = filter_scope(
        load_predictions(args.pattern),
        parse_list(args.endpoints),
        parse_list(args.splits),
    )
    if predictions.empty:
        raise ValueError("No predictions remain after filtering.")

    CONFORMAL_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []

    for (endpoint, split_col), split_group in predictions.groupby(["endpoint", "split_col"], sort=True):
        feature_path = PROCESSED_DIR / f"{endpoint}_features.npz"
        if not feature_path.exists():
            raise FileNotFoundError(feature_path)
        X = np.load(feature_path, allow_pickle=True)["X"].astype(np.float32)

        reference_model = str(split_group["model"].iloc[0])
        reference = split_group[split_group["model"] == reference_model]
        cal_reference = reference[reference["role"] == "calibration"].drop_duplicates("row_index")
        test_reference = reference[reference["role"] == "test"].drop_duplicates("row_index")
        cal_rows = cal_reference["row_index"].astype(int).to_numpy()
        test_rows = test_reference["row_index"].astype(int).to_numpy()
        seed = infer_seed(split_col)
        cal_weights, test_weights, diagnostics = cross_fitted_density_ratios(
            X[cal_rows], X[test_rows], seed, args.weight_min, args.weight_max
        )
        cal_weight_map = dict(zip(cal_rows.tolist(), cal_weights.tolist()))
        test_weight_map = dict(zip(test_rows.tolist(), test_weights.tolist()))
        print(
            f"weighted-conformal endpoint={endpoint} split={split_col} "
            f"domain_auc={diagnostics['domain_classifier_auc']:.3f} ess={diagnostics['calibration_weight_ess']:.1f}"
        )

        for model, model_group in split_group.groupby("model", sort=True):
            calibration = model_group[model_group["role"] == "calibration"].copy()
            test = model_group[model_group["role"] == "test"].copy()
            if calibration.empty or test.empty:
                raise ValueError(f"Missing calibration/test for {endpoint} {split_col} {model}")
            calibration_weights = calibration["row_index"].astype(int).map(cal_weight_map).to_numpy(float)
            local_test_weights = test["row_index"].astype(int).map(test_weight_map).to_numpy(float)
            if np.isnan(calibration_weights).any() or np.isnan(local_test_weights).any():
                raise ValueError("Density-ratio weight mapping produced missing values.")
            task_type = str(model_group["task_type"].iloc[0])
            scores = classification_scores(calibration) if task_type == "classification" else regression_scores(calibration)

            for alpha in alphas:
                qhat = weighted_test_quantiles(scores, calibration_weights, local_test_weights, alpha)
                output = (
                    make_classification_output(test, qhat)
                    if task_type == "classification"
                    else make_regression_output(test, qhat)
                )
                output["alpha"] = alpha
                output["nominal_coverage"] = 1 - alpha
                output["density_ratio_weight"] = local_test_weights
                summary_rows.append(
                    summarize(endpoint, task_type, split_col, model, alpha, output, diagnostics)
                )
                detail_frames.append(output)

    summary = pd.DataFrame(summary_rows)
    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    if summary.empty or details.empty:
        raise RuntimeError("Shift-weighted conformal produced empty outputs.")
    summary_path = CONFORMAL_DIR / f"shift_weighted_conformal_summary_{args.mode}.csv"
    detail_path = CONFORMAL_DIR / f"shift_weighted_conformal_predictions_{args.mode}.csv"
    summary.to_csv(summary_path, index=False)
    details.to_csv(detail_path, index=False)
    print("saved", summary_path)
    print("saved", detail_path)
    print(f"Shift-weighted conformal complete. Summary rows: {len(summary)}")


if __name__ == "__main__":
    main()
