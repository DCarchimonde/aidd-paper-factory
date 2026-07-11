from __future__ import annotations

"""Run class-conditional (Mondrian) split conformal analysis for classification.

For each endpoint/model/split and each candidate class c, this script estimates a
separate calibration threshold from calibration examples whose true class is c:

    score_i = 1 - p_hat(y_i | x_i)

A candidate label c is included in a test prediction set when:

    1 - p_hat(c | x) <= qhat_c

This construction targets class-conditional coverage under the usual exchangeability
assumptions within each class. Under scaffold or other chemical distribution shift,
results must still be described as empirical diagnostics rather than unconditional
theoretical guarantees.

Inputs:
- results/predictions/*_predictions.csv

Outputs:
- results/conformal/mondrian_conformal_summary_<mode>.csv
- results/conformal/mondrian_conformal_predictions_<mode>.csv
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

DEFAULT_ALPHAS = [0.10, 0.20]
MVP_ENDPOINTS = ["bbbp", "clintox"]
MVP_SPLITS = ["split_random_seed0", "split_scaffold"]
CONFIRMATORY_RANDOM_SEEDS = list(range(101, 111))
CONFIRMATORY_SPLITS = (
    [f"split_confirm_random_seed{seed}" for seed in CONFIRMATORY_RANDOM_SEEDS]
    + [f"split_confirm_scaffold_seed{seed}" for seed in CONFIRMATORY_RANDOM_SEEDS]
    + [f"split_confirm_cluster_seed{seed}" for seed in range(101, 106)]
)
MIN_CLASS_CALIBRATION_N = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run class-conditional Mondrian conformal analysis.")
    parser.add_argument("--mode", choices=["mvp", "confirmatory", "full"], default="mvp")
    parser.add_argument("--pattern", default="*_predictions.csv")
    parser.add_argument("--alphas", default=",".join(str(x) for x in DEFAULT_ALPHAS))
    parser.add_argument("--endpoints", default=None, help="Optional comma-separated endpoint names.")
    parser.add_argument("--splits", default=None, help="Optional comma-separated split columns.")
    return parser.parse_args()


def parse_csv_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_alphas(value: str) -> list[float]:
    out = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not out:
        raise ValueError("At least one alpha is required.")
    if any(not 0 < alpha < 1 for alpha in out):
        raise ValueError(f"All alpha values must lie in (0, 1), got {out}")
    return out


def default_endpoints(mode: str) -> list[str] | None:
    return MVP_ENDPOINTS if mode == "mvp" else None


def default_splits(mode: str) -> list[str] | None:
    if mode == "mvp":
        return MVP_SPLITS
    if mode == "confirmatory":
        return CONFIRMATORY_SPLITS
    return None


def load_predictions(pattern: str) -> pd.DataFrame:
    files = sorted(PREDICTION_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No prediction files found in {PREDICTION_DIR} with pattern {pattern}")
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["prediction_file"] = str(path.relative_to(ROOT))
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    required = {"endpoint", "task_type", "split_col", "role", "model", "row_index", "y_true", "y_output"}
    missing = required - set(out.columns)
    if missing:
        raise KeyError(f"Prediction files are missing required columns: {sorted(missing)}")
    return out


def filter_scope(
    predictions: pd.DataFrame,
    mode: str,
    endpoints_arg: str | None,
    splits_arg: str | None,
) -> pd.DataFrame:
    out = predictions[predictions["task_type"] == "classification"].copy()
    endpoints = parse_csv_list(endpoints_arg)
    splits = parse_csv_list(splits_arg)
    if endpoints is None:
        endpoints = default_endpoints(mode)
    if splits is None:
        splits = default_splits(mode)
    if endpoints is not None:
        endpoint_set = {item.lower() for item in endpoints}
        out = out[out["endpoint"].str.lower().isin(endpoint_set)].copy()
    if splits is not None:
        out = out[out["split_col"].isin(splits)].copy()
    return out


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    if n == 0:
        raise ValueError("Cannot compute a conformal quantile from zero scores.")
    rank = int(math.ceil((n + 1) * (1 - alpha)))
    rank = min(max(rank, 1), n)
    return float(scores[rank - 1])


def class_scores(calibration: pd.DataFrame, class_label: int) -> np.ndarray:
    class_rows = calibration[calibration["y_true"].astype(int) == class_label]
    p1 = np.clip(class_rows["y_output"].astype(float).to_numpy(), 1e-7, 1 - 1e-7)
    probability_of_class = p1 if class_label == 1 else 1 - p1
    return 1 - probability_of_class


def make_prediction_sets(test: pd.DataFrame, qhat_0: float, qhat_1: float) -> pd.DataFrame:
    p1 = np.clip(test["y_output"].astype(float).to_numpy(), 1e-7, 1 - 1e-7)
    p0 = 1 - p1
    y_true = test["y_true"].astype(int).to_numpy()

    include_0 = (1 - p0) <= qhat_0
    include_1 = (1 - p1) <= qhat_1
    set_size = include_0.astype(int) + include_1.astype(int)
    covered = np.where(y_true == 0, include_0, include_1)

    out = test.copy()
    out["method"] = "mondrian_lac"
    out["qhat_0"] = qhat_0
    out["qhat_1"] = qhat_1
    out["include_label_0"] = include_0
    out["include_label_1"] = include_1
    out["prediction_set_size"] = set_size
    out["covered"] = covered.astype(bool)
    out["empty_set"] = set_size == 0
    out["singleton_set"] = set_size == 1
    out["ambiguous_set"] = set_size == 2
    return out


def summarize(
    endpoint: str,
    split_col: str,
    model: str,
    alpha: float,
    qhat_0: float,
    qhat_1: float,
    calibration: pd.DataFrame,
    test_out: pd.DataFrame,
) -> dict[str, object]:
    y_cal = calibration["y_true"].astype(int)
    y_test = test_out["y_true"].astype(int)
    positives = y_test == 1
    negatives = y_test == 0

    positive_coverage = float(test_out.loc[positives, "covered"].mean()) if positives.any() else math.nan
    negative_coverage = float(test_out.loc[negatives, "covered"].mean()) if negatives.any() else math.nan
    nominal = 1 - alpha
    empirical = float(test_out["covered"].mean())

    return {
        "endpoint": endpoint,
        "task_type": "classification",
        "split_col": split_col,
        "model": model,
        "method": "mondrian_lac",
        "alpha": alpha,
        "nominal_coverage": nominal,
        "n_calibration": int(len(calibration)),
        "calibration_negative_count": int((y_cal == 0).sum()),
        "calibration_positive_count": int((y_cal == 1).sum()),
        "n_test": int(len(test_out)),
        "test_negative_count": int(negatives.sum()),
        "test_positive_count": int(positives.sum()),
        "qhat_0": qhat_0,
        "qhat_1": qhat_1,
        "empirical_coverage": empirical,
        "coverage_gap": empirical - nominal,
        "negative_coverage": negative_coverage,
        "positive_coverage": positive_coverage,
        "negative_coverage_gap": negative_coverage - nominal if negatives.any() else math.nan,
        "positive_coverage_gap": positive_coverage - nominal if positives.any() else math.nan,
        "class_conditional_coverage_gap": (
            abs(positive_coverage - negative_coverage)
            if positives.any() and negatives.any()
            else math.nan
        ),
        "mean_prediction_set_size": float(test_out["prediction_set_size"].mean()),
        "empty_set_rate": float(test_out["empty_set"].mean()),
        "singleton_rate": float(test_out["singleton_set"].mean()),
        "ambiguous_set_rate": float(test_out["ambiguous_set"].mean()),
        "low_negative_calibration_warning": int((y_cal == 0).sum()) < MIN_CLASS_CALIBRATION_N,
        "low_positive_calibration_warning": int((y_cal == 1).sum()) < MIN_CLASS_CALIBRATION_N,
    }


def main() -> None:
    args = parse_args()
    alphas = parse_alphas(args.alphas)
    CONFORMAL_DIR.mkdir(parents=True, exist_ok=True)

    predictions = filter_scope(
        load_predictions(args.pattern),
        mode=args.mode,
        endpoints_arg=args.endpoints,
        splits_arg=args.splits,
    )
    if predictions.empty:
        raise ValueError("No classification predictions remain after applying the requested scope.")

    summary_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    skipped = 0

    group_cols = ["endpoint", "split_col", "model"]
    for keys, group in predictions.groupby(group_cols, sort=True):
        endpoint, split_col, model = keys
        calibration = group[group["role"] == "calibration"].copy()
        test = group[group["role"] == "test"].copy()
        if calibration.empty or test.empty:
            print(f"[SKIP] endpoint={endpoint} split={split_col} model={model}: missing calibration/test")
            skipped += 1
            continue

        negative_scores = class_scores(calibration, class_label=0)
        positive_scores = class_scores(calibration, class_label=1)
        if len(negative_scores) == 0 or len(positive_scores) == 0:
            print(
                f"[SKIP] endpoint={endpoint} split={split_col} model={model}: "
                f"calibration class counts n0={len(negative_scores)}, n1={len(positive_scores)}"
            )
            skipped += 1
            continue

        print(
            f"mondrian endpoint={endpoint} split={split_col} model={model} "
            f"cal_n0={len(negative_scores)} cal_n1={len(positive_scores)}"
        )
        for alpha in alphas:
            qhat_0 = conformal_quantile(negative_scores, alpha)
            qhat_1 = conformal_quantile(positive_scores, alpha)
            test_out = make_prediction_sets(test, qhat_0=qhat_0, qhat_1=qhat_1)
            test_out["alpha"] = alpha
            test_out["nominal_coverage"] = 1 - alpha
            summary_rows.append(
                summarize(
                    endpoint=endpoint,
                    split_col=split_col,
                    model=model,
                    alpha=alpha,
                    qhat_0=qhat_0,
                    qhat_1=qhat_1,
                    calibration=calibration,
                    test_out=test_out,
                )
            )
            prediction_rows.append(test_out)

    summary = pd.DataFrame(summary_rows)
    prediction_details = (
        pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    )

    summary_path = CONFORMAL_DIR / f"mondrian_conformal_summary_{args.mode}.csv"
    predictions_path = CONFORMAL_DIR / f"mondrian_conformal_predictions_{args.mode}.csv"
    summary.to_csv(summary_path, index=False)
    prediction_details.to_csv(predictions_path, index=False)

    print("\nsaved", summary_path)
    print("saved", predictions_path)
    print(
        "Mondrian conformal analysis complete. "
        f"Groups analyzed: {len(summary.groupby(group_cols)) if not summary.empty else 0}; skipped: {skipped}."
    )


if __name__ == "__main__":
    main()
