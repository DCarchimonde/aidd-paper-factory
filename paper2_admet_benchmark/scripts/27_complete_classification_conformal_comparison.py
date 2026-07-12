from __future__ import annotations

"""Create complete classification conformal comparisons from row-level outputs.

This script fixes an intentional limitation of the original marginal summary table:
class-conditional coverage was not recorded there. It recomputes the same common
metrics from row-level prediction outputs for marginal, Mondrian, and density-ratio
weighted conformal methods, then produces:

1. one run-level row per endpoint/split/model/method;
2. model-specific cross-seed mean, SD, and 95% t intervals;
3. an 18-row descriptive headline table averaged across the eight frozen model/regime
   combinations (not treated as inferentially independent replicates);
4. paired method-minus-marginal differences within the same seed/model.

No model is retrained and no conformal threshold is recomputed.
"""

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper2_admet_benchmark"
CONFORMAL_DIR = PAPER_DIR / "results" / "conformal"
TABLE_DIR = PAPER_DIR / "results" / "tables"

SPLIT_PATTERN = re.compile(r"split_confirm_(random|scaffold|cluster)_seed(\d+)$")
METRICS = [
    "empirical_coverage",
    "positive_coverage",
    "negative_coverage",
    "class_conditional_coverage_gap",
    "mean_prediction_set_size",
    "empty_set_rate",
    "singleton_rate",
    "ambiguous_set_rate",
]
METHOD_ORDER = ["marginal_lac", "density_ratio_weighted_lac", "mondrian_lac"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build complete classification conformal comparison tables."
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


def summarize_group(group: pd.DataFrame, method: str, alpha: float) -> dict[str, object]:
    covered = to_bool(group["covered"])
    y_true = group["y_true"].astype(int)
    positive = y_true == 1
    negative = y_true == 0
    positive_coverage = float(covered[positive].mean()) if positive.any() else math.nan
    negative_coverage = float(covered[negative].mean()) if negative.any() else math.nan

    return {
        "endpoint": str(group["endpoint"].iloc[0]),
        "split_col": str(group["split_col"].iloc[0]),
        "model": str(group["model"].iloc[0]),
        "method": method,
        "alpha": alpha,
        "nominal_coverage": 1 - alpha,
        "n_test": int(len(group)),
        "positive_count": int(positive.sum()),
        "negative_count": int(negative.sum()),
        "empirical_coverage": float(covered.mean()),
        "positive_coverage": positive_coverage,
        "negative_coverage": negative_coverage,
        "class_conditional_coverage_gap": (
            abs(positive_coverage - negative_coverage)
            if positive.any() and negative.any()
            else math.nan
        ),
        "mean_prediction_set_size": float(
            pd.to_numeric(group["prediction_set_size"], errors="coerce").mean()
        ),
        "empty_set_rate": float(to_bool(group["empty_set"]).mean()),
        "singleton_rate": float(to_bool(group["singleton_set"]).mean()),
        "ambiguous_set_rate": float(to_bool(group["ambiguous_set"]).mean()),
    }


def summarize_method(path: Path, method: str, alpha: float) -> pd.DataFrame:
    frame = read_required(path)
    required = {
        "endpoint",
        "task_type",
        "split_col",
        "model",
        "alpha",
        "y_true",
        "covered",
        "prediction_set_size",
        "empty_set",
        "singleton_set",
        "ambiguous_set",
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"{path.name} missing columns: {sorted(missing)}")

    frame = frame[
        (frame["task_type"] == "classification")
        & np.isclose(pd.to_numeric(frame["alpha"], errors="coerce"), alpha)
    ].copy()
    if frame.empty:
        raise RuntimeError(f"No classification alpha={alpha} rows in {path}")

    rows = [
        summarize_group(group, method=method, alpha=alpha)
        for _, group in frame.groupby(["endpoint", "split_col", "model"], sort=True)
    ]
    out = add_split_metadata(pd.DataFrame(rows))
    if len(out) != 400:
        raise RuntimeError(f"{method}: expected 400 run-level rows, found {len(out)}")
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
        for metric in METRICS:
            summary = summarize_values(group[metric])
            if int(summary["n_splits"]) != expected:
                raise RuntimeError(
                    f"Incomplete model-level aggregate for {base}, metric={metric}: "
                    f"expected {expected}, got {summary['n_splits']}"
                )
            rows.append({**base, "metric": metric, **summary})
    out = pd.DataFrame(rows)
    expected_rows = 2 * 3 * 8 * 3 * len(METRICS)
    if len(out) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} model-level metric rows, found {len(out)}"
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
        if row["models"] != 8 or row["seeds"] != expected_seeds:
            raise RuntimeError(f"Incomplete headline group: {row}")
        for metric in METRICS:
            row[metric] = float(pd.to_numeric(group[metric], errors="coerce").mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out) != 18:
        raise RuntimeError(f"Expected 18 headline rows, found {len(out)}")
    out["method"] = pd.Categorical(out["method"], METHOD_ORDER, ordered=True)
    return out.sort_values(["endpoint", "split_type", "method"]).reset_index(drop=True)


def paired_differences(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    key_cols = ["endpoint", "split_col", "split_type", "seed", "model", "alpha"]
    indexed = runs.set_index(key_cols + ["method"])
    comparisons = [
        ("mondrian_minus_marginal", "mondrian_lac", "marginal_lac"),
        (
            "shift_weighted_minus_marginal",
            "density_ratio_weighted_lac",
            "marginal_lac",
        ),
    ]
    paired_rows: list[dict[str, object]] = []
    for keys in runs[key_cols].drop_duplicates().itertuples(index=False, name=None):
        base = dict(zip(key_cols, keys))
        for comparison, treatment, control in comparisons:
            treatment_row = indexed.loc[keys + (treatment,)]
            control_row = indexed.loc[keys + (control,)]
            row: dict[str, object] = {**base, "comparison": comparison}
            for metric in METRICS:
                row[f"delta_{metric}"] = float(treatment_row[metric]) - float(
                    control_row[metric]
                )
            paired_rows.append(row)
    paired = pd.DataFrame(paired_rows)
    if len(paired) != 800:
        raise RuntimeError(f"Expected 800 paired rows, found {len(paired)}")

    aggregate_rows: list[dict[str, object]] = []
    group_cols = ["endpoint", "split_type", "model", "comparison", "alpha"]
    for keys, group in paired.groupby(group_cols, sort=True):
        base = dict(zip(group_cols, keys))
        expected = 5 if base["split_type"] == "cluster" else 10
        for metric in METRICS:
            delta_col = f"delta_{metric}"
            summary = summarize_values(group[delta_col])
            if int(summary["n_splits"]) != expected:
                raise RuntimeError(
                    f"Incomplete paired aggregate for {base}, metric={metric}"
                )
            aggregate_rows.append(
                {**base, "metric": metric, "delta_definition": "treatment_minus_marginal", **summary}
            )
    aggregate = pd.DataFrame(aggregate_rows)
    return paired, aggregate


def main() -> None:
    args = parse_args()
    if not 0 < args.alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    alpha_tag = str(args.alpha).replace(".", "")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    method_frames = [
        summarize_method(
            CONFORMAL_DIR / "conformal_predictions_confirmatory_full.csv",
            method="marginal_lac",
            alpha=args.alpha,
        ),
        summarize_method(
            CONFORMAL_DIR / "mondrian_conformal_predictions_confirmatory.csv",
            method="mondrian_lac",
            alpha=args.alpha,
        ),
        summarize_method(
            CONFORMAL_DIR / "shift_weighted_conformal_predictions_confirmatory_full.csv",
            method="density_ratio_weighted_lac",
            alpha=args.alpha,
        ),
    ]
    runs = pd.concat(method_frames, ignore_index=True)
    if len(runs) != 1200:
        raise RuntimeError(f"Expected 1200 complete run rows, found {len(runs)}")

    by_model = aggregate_by_model(runs)
    headline = descriptive_headline(runs)
    paired, paired_aggregate = paired_differences(runs)

    runs_path = TABLE_DIR / f"classification_conformal_runs_alpha{alpha_tag}_complete.csv"
    model_path = TABLE_DIR / f"classification_conformal_by_model_alpha{alpha_tag}.csv"
    headline_path = TABLE_DIR / f"headline_classification_conformal_alpha{alpha_tag}_complete.csv"
    paired_path = TABLE_DIR / f"paired_classification_conformal_runs_alpha{alpha_tag}.csv"
    paired_aggregate_path = TABLE_DIR / f"paired_classification_conformal_by_model_alpha{alpha_tag}.csv"

    runs.to_csv(runs_path, index=False)
    by_model.to_csv(model_path, index=False)
    headline.to_csv(headline_path, index=False)
    paired.to_csv(paired_path, index=False)
    paired_aggregate.to_csv(paired_aggregate_path, index=False)

    print("saved", runs_path)
    print("saved", model_path)
    print("saved", headline_path)
    print("saved", paired_path)
    print("saved", paired_aggregate_path)
    print(
        "Complete classification conformal comparison generated: "
        f"runs={len(runs)}, model_metrics={len(by_model)}, "
        f"headline={len(headline)}, paired_runs={len(paired)}, "
        f"paired_metrics={len(paired_aggregate)}."
    )


if __name__ == "__main__":
    main()
