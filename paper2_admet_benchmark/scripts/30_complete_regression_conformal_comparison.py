from __future__ import annotations

"""Create complete confirmatory regression conformal comparisons.

The script consumes the already frozen row-level outputs for marginal absolute-residual,
density-ratio-weighted, and locally adaptive normalized conformal intervals. It does
not retrain models or recompute conformal thresholds.

Outputs include:
1. one run-level row per endpoint/split/model/method at alpha=0.1;
2. model-specific cross-seed means, SDs, and 95% t intervals;
3. an 18-row descriptive headline table averaged across the four frozen regressors;
4. paired treatment-minus-marginal effects within the same endpoint/split/seed/model;
5. a 12-row descriptive paired-effect headline with counts of model-specific CIs.
"""

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper2_admet_benchmark"
CONFORMAL_DIR = PAPER_DIR / "results" / "conformal"
TABLE_DIR = PAPER_DIR / "results" / "tables"

SPLIT_PATTERN = re.compile(r"split_confirm_(random|scaffold|cluster)_seed(\d+)$")
METHOD_ORDER = [
    "marginal_absolute_residual",
    "density_ratio_weighted_absolute_residual",
    "split_calibration_adaptive_normalized",
]
CORE_METRICS = [
    "empirical_coverage",
    "coverage_gap",
    "absolute_coverage_gap",
    "mean_interval_width",
    "median_interval_width",
    "interval_width_std",
    "interval_width_cv",
    "mean_absolute_error",
    "infinite_interval_rate",
]
ADAPTIVITY_METRICS = ["width_error_pearson", "width_error_spearman"]
PAIRED_METRICS = [
    "empirical_coverage",
    "absolute_coverage_gap",
    "mean_interval_width",
    "median_interval_width",
    "interval_width_std",
    "interval_width_cv",
    "infinite_interval_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build complete confirmatory regression conformal comparisons."
    )
    parser.add_argument("--alpha", type=float, default=0.10)
    return parser.parse_args()


def to_bool(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.lower().map({"true": True, "false": False})
    if mapped.notna().all():
        return mapped.astype(bool)
    return series.astype(bool)


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise RuntimeError(f"Required table is empty: {path}")
    return frame


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


def safe_correlation(width: np.ndarray, error: np.ndarray, kind: str) -> float:
    finite = np.isfinite(width) & np.isfinite(error)
    x = width[finite]
    y = error[finite]
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return math.nan
    if kind == "pearson":
        return float(np.corrcoef(x, y)[0, 1])
    if kind == "spearman":
        return float(spearmanr(x, y).statistic)
    raise ValueError(kind)


def summarize_group(
    group: pd.DataFrame,
    method: str,
    alpha: float,
) -> dict[str, object]:
    covered = to_bool(group["covered"])
    width = pd.to_numeric(group["interval_width"], errors="coerce").to_numpy(float)
    error = pd.to_numeric(group["absolute_error"], errors="coerce").to_numpy(float)
    finite_width = width[np.isfinite(width)]
    if len(finite_width) == 0:
        mean_width = median_width = width_std = width_cv = math.inf
    else:
        mean_width = float(np.mean(finite_width))
        median_width = float(np.median(finite_width))
        width_std = float(np.std(finite_width, ddof=1)) if len(finite_width) > 1 else 0.0
        width_cv = width_std / mean_width if mean_width > 0 else math.nan

    nominal = 1 - alpha
    empirical = float(covered.mean())
    return {
        "endpoint": str(group["endpoint"].iloc[0]),
        "split_col": str(group["split_col"].iloc[0]),
        "model": str(group["model"].iloc[0]),
        "method": method,
        "alpha": alpha,
        "nominal_coverage": nominal,
        "n_test": int(len(group)),
        "empirical_coverage": empirical,
        "coverage_gap": empirical - nominal,
        "absolute_coverage_gap": abs(empirical - nominal),
        "mean_interval_width": mean_width,
        "median_interval_width": median_width,
        "interval_width_std": width_std,
        "interval_width_cv": width_cv,
        "mean_absolute_error": float(np.nanmean(error)),
        "infinite_interval_rate": float((~np.isfinite(width)).mean()),
        "width_error_pearson": safe_correlation(width, error, "pearson"),
        "width_error_spearman": safe_correlation(width, error, "spearman"),
    }


def summarize_method(path: Path, method: str, alpha: float) -> pd.DataFrame:
    frame = read_required(path)
    required = {
        "endpoint",
        "task_type",
        "split_col",
        "model",
        "alpha",
        "covered",
        "interval_width",
        "absolute_error",
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"{path.name} missing columns: {sorted(missing)}")

    frame = frame[
        (frame["task_type"] == "regression")
        & np.isclose(pd.to_numeric(frame["alpha"], errors="coerce"), alpha)
    ].copy()
    if frame.empty:
        raise RuntimeError(f"No regression alpha={alpha} rows in {path}")

    rows = [
        summarize_group(group, method=method, alpha=alpha)
        for _, group in frame.groupby(["endpoint", "split_col", "model"], sort=True)
    ]
    out = add_split_metadata(pd.DataFrame(rows))
    if len(out) != 200:
        raise RuntimeError(f"{method}: expected 200 run-level rows, found {len(out)}")
    return out


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
    rows: list[dict[str, object]] = []
    group_cols = ["endpoint", "split_type", "model", "method", "alpha"]
    for keys, group in runs.groupby(group_cols, sort=True):
        base = dict(zip(group_cols, keys))
        expected = 5 if base["split_type"] == "cluster" else 10
        for metric in CORE_METRICS + ADAPTIVITY_METRICS:
            summary = summarize_values(group[metric])
            if metric in CORE_METRICS and int(summary["n_splits"]) != expected:
                raise RuntimeError(
                    f"Incomplete model-level aggregate for {base}, metric={metric}: "
                    f"expected {expected}, got {summary['n_splits']}"
                )
            if int(summary["n_splits"]) == 0:
                continue
            rows.append({**base, "metric": metric, **summary})
    out = pd.DataFrame(rows)
    expected_core_rows = 2 * 3 * 4 * 3 * len(CORE_METRICS)
    actual_core_rows = int(out["metric"].isin(CORE_METRICS).sum())
    if actual_core_rows != expected_core_rows:
        raise RuntimeError(
            f"Expected {expected_core_rows} core model-level metric rows, found {actual_core_rows}"
        )
    return out


def descriptive_headline(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["endpoint", "split_type", "method", "alpha"]
    for keys, group in runs.groupby(group_cols, sort=True):
        base = dict(zip(group_cols, keys))
        expected_seeds = 5 if base["split_type"] == "cluster" else 10
        row: dict[str, object] = {
            **base,
            "models": int(group["model"].nunique()),
            "seeds": int(group["seed"].nunique()),
            "model_seed_cells": int(len(group)),
            "descriptive_model_average": True,
        }
        if row["models"] != 4 or row["seeds"] != expected_seeds:
            raise RuntimeError(f"Incomplete headline group: {row}")
        for metric in CORE_METRICS + ADAPTIVITY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            row[metric] = float(values.mean()) if values.notna().any() else math.nan
            row[f"valid_cells_{metric}"] = int(values.notna().sum())
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out) != 18:
        raise RuntimeError(f"Expected 18 headline rows, found {len(out)}")
    out["method"] = pd.Categorical(out["method"], METHOD_ORDER, ordered=True)
    return out.sort_values(["endpoint", "split_type", "method"]).reset_index(drop=True)


def paired_differences(
    runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    key_cols = ["endpoint", "split_col", "split_type", "seed", "model", "alpha"]
    indexed = runs.set_index(key_cols + ["method"])
    comparisons = [
        (
            "shift_weighted_minus_marginal",
            "density_ratio_weighted_absolute_residual",
            "marginal_absolute_residual",
        ),
        (
            "adaptive_minus_marginal",
            "split_calibration_adaptive_normalized",
            "marginal_absolute_residual",
        ),
    ]

    paired_rows: list[dict[str, object]] = []
    for keys in runs[key_cols].drop_duplicates().itertuples(index=False, name=None):
        base = dict(zip(key_cols, keys))
        for comparison, treatment, control in comparisons:
            treatment_row = indexed.loc[keys + (treatment,)]
            control_row = indexed.loc[keys + (control,)]
            if int(treatment_row["n_test"]) != int(control_row["n_test"]):
                raise RuntimeError(f"Mismatched test size for {base}, {comparison}")
            row: dict[str, object] = {**base, "comparison": comparison}
            for metric in PAIRED_METRICS:
                row[f"delta_{metric}"] = float(treatment_row[metric]) - float(
                    control_row[metric]
                )
            paired_rows.append(row)

    paired = pd.DataFrame(paired_rows)
    if len(paired) != 400:
        raise RuntimeError(f"Expected 400 paired run rows, found {len(paired)}")

    aggregate_rows: list[dict[str, object]] = []
    group_cols = ["endpoint", "split_type", "model", "comparison", "alpha"]
    for keys, group in paired.groupby(group_cols, sort=True):
        base = dict(zip(group_cols, keys))
        expected = 5 if base["split_type"] == "cluster" else 10
        for metric in PAIRED_METRICS:
            summary = summarize_values(group[f"delta_{metric}"])
            if int(summary["n_splits"]) != expected:
                raise RuntimeError(
                    f"Incomplete paired aggregate for {base}, metric={metric}"
                )
            aggregate_rows.append(
                {
                    **base,
                    "metric": metric,
                    "delta_definition": "treatment_minus_marginal",
                    **summary,
                }
            )
    aggregate = pd.DataFrame(aggregate_rows)
    expected_aggregate = 2 * 3 * 4 * 2 * len(PAIRED_METRICS)
    if len(aggregate) != expected_aggregate:
        raise RuntimeError(
            f"Expected {expected_aggregate} paired model metrics, found {len(aggregate)}"
        )

    headline_rows: list[dict[str, object]] = []
    for keys, group in aggregate.groupby(
        ["endpoint", "split_type", "comparison", "alpha"], sort=True
    ):
        endpoint, split_type, comparison, alpha = keys
        if group["model"].nunique() != 4:
            raise RuntimeError(f"Expected four regressors for paired headline {keys}")
        row: dict[str, object] = {
            "endpoint": endpoint,
            "split_type": split_type,
            "comparison": comparison,
            "alpha": alpha,
            "models": 4,
            "descriptive_average_across_models": True,
        }
        for metric in PAIRED_METRICS:
            metric_group = group[group["metric"] == metric]
            if len(metric_group) != 4:
                raise RuntimeError(f"Expected four model rows for {keys}, metric={metric}")
            means = pd.to_numeric(metric_group["mean"], errors="coerce")
            lows = pd.to_numeric(metric_group["ci95_low"], errors="coerce")
            highs = pd.to_numeric(metric_group["ci95_high"], errors="coerce")
            row[f"mean_delta_{metric}"] = float(means.mean())
            row[f"models_{metric}_delta_positive"] = int((means > 0).sum())
            row[f"models_{metric}_delta_negative"] = int((means < 0).sum())
            row[f"models_{metric}_ci_strictly_positive"] = int((lows > 0).sum())
            row[f"models_{metric}_ci_strictly_negative"] = int((highs < 0).sum())
        headline_rows.append(row)
    headline = pd.DataFrame(headline_rows)
    if len(headline) != 12:
        raise RuntimeError(f"Expected 12 paired headline rows, found {len(headline)}")
    headline = headline.sort_values(
        ["endpoint", "split_type", "comparison"]
    ).reset_index(drop=True)
    return paired, aggregate, headline


def main() -> None:
    args = parse_args()
    if not 0 < args.alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    alpha_tag = str(args.alpha).replace(".", "")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    method_frames = [
        summarize_method(
            CONFORMAL_DIR / "conformal_predictions_confirmatory_full.csv",
            method="marginal_absolute_residual",
            alpha=args.alpha,
        ),
        summarize_method(
            CONFORMAL_DIR / "shift_weighted_conformal_predictions_confirmatory_full.csv",
            method="density_ratio_weighted_absolute_residual",
            alpha=args.alpha,
        ),
        summarize_method(
            CONFORMAL_DIR / "adaptive_regression_conformal_predictions_confirmatory_full.csv",
            method="split_calibration_adaptive_normalized",
            alpha=args.alpha,
        ),
    ]
    runs = pd.concat(method_frames, ignore_index=True)
    if len(runs) != 600:
        raise RuntimeError(f"Expected 600 complete run rows, found {len(runs)}")

    key_cols = ["endpoint", "split_col", "model"]
    support = runs.groupby(key_cols)["method"].nunique()
    if not (support == 3).all():
        bad = support[support != 3]
        raise RuntimeError(f"Incomplete method support: {bad.to_dict()}")

    by_model = aggregate_by_model(runs)
    headline = descriptive_headline(runs)
    paired, paired_aggregate, paired_headline = paired_differences(runs)

    runs_path = TABLE_DIR / f"regression_conformal_runs_alpha{alpha_tag}_complete.csv"
    model_path = TABLE_DIR / f"regression_conformal_by_model_alpha{alpha_tag}.csv"
    headline_path = TABLE_DIR / f"headline_regression_conformal_alpha{alpha_tag}_complete.csv"
    paired_path = TABLE_DIR / f"paired_regression_conformal_runs_alpha{alpha_tag}.csv"
    paired_model_path = TABLE_DIR / f"paired_regression_conformal_by_model_alpha{alpha_tag}.csv"
    paired_headline_path = TABLE_DIR / f"paired_regression_effects_headline_alpha{alpha_tag}.csv"

    runs.to_csv(runs_path, index=False)
    by_model.to_csv(model_path, index=False)
    headline.to_csv(headline_path, index=False)
    paired.to_csv(paired_path, index=False)
    paired_aggregate.to_csv(paired_model_path, index=False)
    paired_headline.to_csv(paired_headline_path, index=False)

    print("saved", runs_path)
    print("saved", model_path)
    print("saved", headline_path)
    print("saved", paired_path)
    print("saved", paired_model_path)
    print("saved", paired_headline_path)
    print(
        "Complete regression conformal comparison generated: "
        f"runs={len(runs)}, model_metrics={len(by_model)}, headline={len(headline)}, "
        f"paired_runs={len(paired)}, paired_metrics={len(paired_aggregate)}, "
        f"paired_headline={len(paired_headline)}."
    )


if __name__ == "__main__":
    main()
