from __future__ import annotations

"""Compare all frozen conformal methods within chemical-domain bins."""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper2_admet_benchmark"
CONFORMAL_DIR = PAPER_DIR / "results" / "conformal"
APPLICABILITY_DIR = PAPER_DIR / "results" / "applicability"
DOMAIN_ORDER = ["high_domain", "medium_domain", "low_domain"]
MIN_GROUP_N_WARNING = 30
MIN_CLASS_COUNT_WARNING = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute domain-conditioned summaries for all conformal methods."
    )
    parser.add_argument("--mode", default="confirmatory_full")
    return parser.parse_args()


def to_bool(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.lower().map({"true": True, "false": False})
    return mapped.astype(bool) if not mapped.isna().any() else series.astype(bool)


def read_details(mode: str) -> pd.DataFrame:
    paths = [
        CONFORMAL_DIR / f"conformal_predictions_{mode}.csv",
        CONFORMAL_DIR / "mondrian_conformal_predictions_confirmatory.csv",
        CONFORMAL_DIR / f"shift_weighted_conformal_predictions_{mode}.csv",
        CONFORMAL_DIR / f"adaptive_regression_conformal_predictions_{mode}.csv",
    ]
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, low_memory=False)
        if "method" not in frame.columns:
            frame["method"] = np.where(
                frame["task_type"].astype(str) == "classification",
                "marginal_lac",
                "marginal_absolute_residual",
            )
        frame["source_file"] = path.name
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True, sort=False)
    required = {
        "endpoint",
        "task_type",
        "split_col",
        "model",
        "row_index",
        "alpha",
        "nominal_coverage",
        "covered",
        "method",
    }
    missing = required - set(out.columns)
    if missing:
        raise KeyError(f"Conformal details missing columns: {sorted(missing)}")
    return out


def summarize_group(keys: tuple, group: pd.DataFrame) -> dict[str, object]:
    endpoint, task_type, split_col, model, method, alpha, domain_bin = keys
    covered = to_bool(group["covered"])
    nominal = float(group["nominal_coverage"].iloc[0])
    empirical = float(covered.mean())
    row: dict[str, object] = {
        "endpoint": endpoint,
        "task_type": task_type,
        "split_col": split_col,
        "model": model,
        "method": method,
        "alpha": float(alpha),
        "nominal_coverage": nominal,
        "domain_bin": domain_bin,
        "n": int(len(group)),
        "empirical_coverage": empirical,
        "coverage_gap": empirical - nominal,
        "absolute_coverage_gap": abs(empirical - nominal),
        "mean_max_tanimoto": float(group["max_tanimoto_to_train"].mean()),
        "unseen_scaffold_rate": float(group["unseen_scaffold"].mean()),
        "small_group_warning": len(group) < MIN_GROUP_N_WARNING,
    }

    if task_type == "classification":
        y_true = group["y_true"].astype(int)
        positives = y_true == 1
        negatives = y_true == 0
        positive_n = int(positives.sum())
        negative_n = int(negatives.sum())
        positive_coverage = (
            float(covered[positives].mean()) if positive_n > 0 else math.nan
        )
        negative_coverage = (
            float(covered[negatives].mean()) if negative_n > 0 else math.nan
        )
        row.update(
            {
                "positive_count": positive_n,
                "negative_count": negative_n,
                "single_class_bin": positive_n == 0 or negative_n == 0,
                "low_class_count_warning": min(positive_n, negative_n)
                < MIN_CLASS_COUNT_WARNING,
                "positive_coverage": positive_coverage,
                "negative_coverage": negative_coverage,
                "positive_coverage_gap": (
                    positive_coverage - nominal if positive_n > 0 else math.nan
                ),
                "negative_coverage_gap": (
                    negative_coverage - nominal if negative_n > 0 else math.nan
                ),
                "class_conditional_coverage_gap": (
                    abs(positive_coverage - negative_coverage)
                    if positive_n > 0 and negative_n > 0
                    else math.nan
                ),
                "mean_prediction_set_size": float(
                    group["prediction_set_size"].astype(float).mean()
                ),
                "empty_set_rate": float(to_bool(group["empty_set"]).mean()),
                "singleton_rate": float(to_bool(group["singleton_set"]).mean()),
                "ambiguous_set_rate": float(to_bool(group["ambiguous_set"]).mean()),
            }
        )
    else:
        width = group["interval_width"].astype(float)
        finite_width = width[np.isfinite(width)]
        row.update(
            {
                "mean_interval_width": (
                    float(finite_width.mean()) if len(finite_width) else math.inf
                ),
                "median_interval_width": (
                    float(finite_width.median()) if len(finite_width) else math.inf
                ),
                "interval_width_std": (
                    float(finite_width.std(ddof=1)) if len(finite_width) > 1 else 0.0
                ),
                "infinite_interval_rate": float((~np.isfinite(width)).mean()),
                "mean_absolute_error": float(
                    np.abs(
                        group["y_true"].astype(float)
                        - group["y_output"].astype(float)
                    ).mean()
                ),
            }
        )
    return row


def main() -> None:
    args = parse_args()
    applicability_path = (
        APPLICABILITY_DIR / f"applicability_details_{args.mode}.csv"
    )
    if not applicability_path.exists():
        raise FileNotFoundError(applicability_path)

    conformal = read_details(args.mode)
    applicability = pd.read_csv(applicability_path, low_memory=False)
    applicability = applicability[applicability["role"] == "test"].copy()
    merge_cols = ["endpoint", "task_type", "split_col", "row_index"]
    ad_cols = merge_cols + [
        "domain_bin",
        "max_tanimoto_to_train",
        "unseen_scaffold",
    ]
    merged = conformal.merge(
        applicability[ad_cols],
        on=merge_cols,
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        raise RuntimeError("All-method conformal/applicability merge produced zero rows.")

    rows: list[dict[str, object]] = []
    group_cols = [
        "endpoint",
        "task_type",
        "split_col",
        "model",
        "method",
        "alpha",
        "domain_bin",
    ]
    for keys, group in merged.groupby(group_cols, sort=True):
        rows.append(summarize_group(keys, group))

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("All-method domain-conditioned summary is empty.")
    out["domain_bin"] = pd.Categorical(
        out["domain_bin"], categories=DOMAIN_ORDER, ordered=True
    )
    out = out.sort_values(
        ["endpoint", "split_col", "model", "method", "alpha", "domain_bin"]
    ).reset_index(drop=True)

    output_path = (
        APPLICABILITY_DIR
        / f"domain_conditioned_all_conformal_{args.mode}.csv"
    )
    out.to_csv(output_path, index=False)
    print("saved", output_path)
    print(
        "All-method domain-conditioned conformal analysis complete. "
        f"Rows: {len(out)}; methods: {out['method'].nunique()}."
    )


if __name__ == "__main__":
    main()
