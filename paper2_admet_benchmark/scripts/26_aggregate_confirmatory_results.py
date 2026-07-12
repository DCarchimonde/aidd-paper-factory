from __future__ import annotations

"""Aggregate frozen confirmatory results across repeated split seeds."""

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper2_admet_benchmark"
METRIC_DIR = PAPER_DIR / "results" / "metrics"
CALIBRATION_DIR = PAPER_DIR / "results" / "calibration"
CONFORMAL_DIR = PAPER_DIR / "results" / "conformal"
APPLICABILITY_DIR = PAPER_DIR / "results" / "applicability"
SELECTIVE_DIR = PAPER_DIR / "results" / "selective"
TABLE_DIR = PAPER_DIR / "results" / "tables"

SPLIT_PATTERN = re.compile(r"split_confirm_(random|scaffold|cluster)_seed(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate confirmatory results across split seeds."
    )
    parser.add_argument("--mode", default="confirmatory_full")
    return parser.parse_args()


def add_split_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    extracted = out["split_col"].astype(str).str.extract(SPLIT_PATTERN)
    out["split_type"] = extracted[0]
    out["seed"] = pd.to_numeric(extracted[1], errors="coerce").astype("Int64")
    if out["split_type"].isna().any() or out["seed"].isna().any():
        bad = out.loc[
            out["split_type"].isna() | out["seed"].isna(), "split_col"
        ].drop_duplicates()
        raise ValueError(f"Unrecognized confirmatory split columns: {bad.tolist()}")
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


def aggregate_table(
    source_name: str,
    df: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str],
) -> list[dict[str, object]]:
    out = add_split_metadata(df)
    effective_groups = [col for col in group_cols if col in out.columns]
    if "split_type" not in effective_groups:
        effective_groups.append("split_type")

    rows: list[dict[str, object]] = []
    for keys, group in out.groupby(effective_groups, dropna=False, sort=True):
        keys_tuple = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(effective_groups, keys_tuple))
        for metric in metric_cols:
            if metric not in group.columns:
                continue
            summary = summarize_values(group[metric])
            if int(summary["n_splits"]) == 0:
                continue
            rows.append(
                {
                    "source_table": source_name,
                    **base,
                    "metric": metric,
                    **summary,
                }
            )
    return rows


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise RuntimeError(f"Required result table is empty: {path}")
    return df


def main() -> None:
    args = parse_args()
    mode = args.mode
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    configs: list[tuple[str, pd.DataFrame, list[str], list[str]]] = []

    baseline = read_required(METRIC_DIR / "baseline_metrics_confirmatory_full.csv")
    baseline = baseline[baseline["role"] == "test"].copy()
    configs.append(
        (
            "baseline_test",
            baseline,
            [
                "endpoint",
                "task_type",
                "base_model",
                "model",
                "imbalance_regime",
                "split_type",
            ],
            [
                "roc_auc",
                "pr_auc",
                "balanced_accuracy",
                "brier_score",
                "negative_log_likelihood",
                "rmse",
                "mae",
                "r2",
                "fit_seconds",
            ],
        )
    )

    calibration = read_required(
        CALIBRATION_DIR / f"calibration_summary_v2_{mode}.csv"
    )
    calibration = calibration[calibration["role"] == "test"].copy()
    configs.append(
        (
            "calibration_test",
            calibration,
            ["endpoint", "task_type", "model", "split_type"],
            [
                "brier_score",
                "negative_log_likelihood",
                "ece_probability",
                "mce_probability",
                "ece_confidence",
                "mce_confidence",
                "rmse",
                "mae",
                "bias",
            ],
        )
    )

    marginal = read_required(CONFORMAL_DIR / f"conformal_summary_{mode}.csv")
    marginal["method"] = np.where(
        marginal["task_type"] == "classification",
        "marginal_lac",
        "marginal_absolute_residual",
    )
    configs.append(
        (
            "marginal_conformal",
            marginal,
            ["endpoint", "task_type", "model", "method", "alpha", "split_type"],
            [
                "empirical_coverage",
                "mean_prediction_set_size",
                "empty_set_rate",
                "singleton_rate",
                "ambiguous_set_rate",
                "mean_interval_width",
                "mean_absolute_error",
            ],
        )
    )

    mondrian = read_required(
        CONFORMAL_DIR / "mondrian_conformal_summary_confirmatory.csv"
    )
    configs.append(
        (
            "mondrian_conformal",
            mondrian,
            ["endpoint", "task_type", "model", "method", "alpha", "split_type"],
            [
                "empirical_coverage",
                "positive_coverage",
                "negative_coverage",
                "class_conditional_coverage_gap",
                "mean_prediction_set_size",
                "empty_set_rate",
                "singleton_rate",
                "ambiguous_set_rate",
            ],
        )
    )

    weighted = read_required(
        CONFORMAL_DIR / f"shift_weighted_conformal_summary_{mode}.csv"
    )
    configs.append(
        (
            "shift_weighted_conformal",
            weighted,
            ["endpoint", "task_type", "model", "method", "alpha", "split_type"],
            [
                "domain_classifier_auc",
                "calibration_weight_ess",
                "empirical_coverage",
                "positive_coverage",
                "negative_coverage",
                "mean_prediction_set_size",
                "ambiguous_set_rate",
                "mean_interval_width",
                "infinite_qhat_rate",
            ],
        )
    )

    adaptive = read_required(
        CONFORMAL_DIR / f"adaptive_regression_conformal_summary_{mode}.csv"
    )
    configs.append(
        (
            "adaptive_regression_conformal",
            adaptive,
            ["endpoint", "task_type", "model", "method", "alpha", "split_type"],
            [
                "empirical_coverage",
                "mean_interval_width",
                "median_interval_width",
                "interval_width_std",
                "scale_error_correlation",
                "mean_absolute_error",
            ],
        )
    )

    domain_perf = read_required(
        APPLICABILITY_DIR / f"domain_conditioned_performance_{mode}.csv"
    )
    configs.append(
        (
            "domain_conditioned_performance",
            domain_perf,
            [
                "endpoint",
                "task_type",
                "model",
                "domain_bin",
                "split_type",
            ],
            [
                "fraction_within_test",
                "mean_max_tanimoto",
                "unseen_scaffold_rate",
                "error_rate",
                "brier_score",
                "roc_auc",
                "pr_auc",
                "rmse",
                "mae",
                "bias",
            ],
        )
    )

    all_domain_conformal = read_required(
        APPLICABILITY_DIR / f"domain_conditioned_all_conformal_{mode}.csv"
    )
    configs.append(
        (
            "domain_conditioned_all_conformal",
            all_domain_conformal,
            [
                "endpoint",
                "task_type",
                "model",
                "method",
                "alpha",
                "domain_bin",
                "split_type",
            ],
            [
                "empirical_coverage",
                "absolute_coverage_gap",
                "positive_coverage",
                "negative_coverage",
                "class_conditional_coverage_gap",
                "mean_prediction_set_size",
                "empty_set_rate",
                "ambiguous_set_rate",
                "mean_interval_width",
                "interval_width_std",
                "infinite_interval_rate",
            ],
        )
    )

    selective = read_required(
        SELECTIVE_DIR / f"selective_prediction_v2_aurc_{mode}.csv"
    )
    configs.append(
        (
            "selective_prediction_aurc",
            selective,
            [
                "endpoint",
                "task_type",
                "model",
                "uncertainty_measure",
                "split_type",
            ],
            [
                "aurc_primary_risk",
                "aurc_random_mean",
                "aurc_improvement_vs_random",
                "aurc_balanced_error",
                "aurc_random_balanced_error_mean",
            ],
        )
    )

    rows: list[dict[str, object]] = []
    for source_name, df, group_cols, metric_cols in configs:
        source_rows = aggregate_table(source_name, df, group_cols, metric_cols)
        rows.extend(source_rows)
        print(f"aggregated {source_name}: {len(source_rows)} metric rows")

    aggregate = pd.DataFrame(rows)
    if aggregate.empty:
        raise RuntimeError("Confirmatory aggregation produced zero rows.")

    preferred_cols = [
        "source_table",
        "endpoint",
        "task_type",
        "split_type",
        "base_model",
        "model",
        "imbalance_regime",
        "method",
        "alpha",
        "domain_bin",
        "uncertainty_measure",
        "metric",
        "n_splits",
        "mean",
        "std",
        "se",
        "ci95_low",
        "ci95_high",
        "min",
        "max",
    ]
    for col in preferred_cols:
        if col not in aggregate.columns:
            aggregate[col] = np.nan
    aggregate = aggregate[preferred_cols].sort_values(
        [
            "source_table",
            "endpoint",
            "split_type",
            "model",
            "method",
            "alpha",
            "domain_bin",
            "uncertainty_measure",
            "metric",
        ],
        na_position="last",
    )

    output_path = TABLE_DIR / "confirmatory_aggregate_long.csv"
    aggregate.to_csv(output_path, index=False)

    integrity = (
        aggregate.groupby(["source_table", "split_type"], dropna=False)["n_splits"]
        .agg(["min", "max", "count"])
        .reset_index()
    )
    integrity_path = TABLE_DIR / "confirmatory_aggregate_integrity.csv"
    integrity.to_csv(integrity_path, index=False)

    print("saved", output_path)
    print("saved", integrity_path)
    print(f"Confirmatory aggregation complete. Rows: {len(aggregate)}.")


if __name__ == "__main__":
    main()
