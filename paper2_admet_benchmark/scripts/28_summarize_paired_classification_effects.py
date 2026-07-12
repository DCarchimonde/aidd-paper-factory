from __future__ import annotations

"""Summarize paired classification conformal effects without treating models as replicates.

Consumes the model-specific cross-seed paired estimates produced by script 27 and
creates one descriptive row per endpoint/split/comparison. The table reports the
average of the eight frozen model/regime-specific mean deltas and counts how many
model-specific 95% confidence intervals lie entirely above or below zero.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "paper2_admet_benchmark" / "results" / "tables"
INPUT = TABLE_DIR / "paired_classification_conformal_by_model_alpha01.csv"
OUTPUT = TABLE_DIR / "paired_classification_effects_headline_alpha01.csv"

KEY_METRICS = [
    "empirical_coverage",
    "positive_coverage",
    "negative_coverage",
    "class_conditional_coverage_gap",
    "mean_prediction_set_size",
    "ambiguous_set_rate",
    "empty_set_rate",
]


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT)
    required = {
        "endpoint",
        "split_type",
        "model",
        "comparison",
        "metric",
        "n_splits",
        "mean",
        "ci95_low",
        "ci95_high",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    df = df[df["metric"].isin(KEY_METRICS)].copy()
    rows: list[dict[str, object]] = []

    for keys, group in df.groupby(
        ["endpoint", "split_type", "comparison"], sort=True
    ):
        endpoint, split_type, comparison = keys
        expected_splits = 5 if split_type == "cluster" else 10
        models = sorted(group["model"].dropna().unique())
        if len(models) != 8:
            raise RuntimeError(
                f"Expected 8 frozen model/regime combinations for {keys}, got {len(models)}"
            )
        if not (pd.to_numeric(group["n_splits"], errors="coerce") == expected_splits).all():
            raise RuntimeError(f"Incomplete seed support in paired summary for {keys}")

        row: dict[str, object] = {
            "endpoint": endpoint,
            "split_type": split_type,
            "comparison": comparison,
            "models": len(models),
            "expected_splits_per_model": expected_splits,
            "descriptive_average_across_models": True,
        }

        for metric in KEY_METRICS:
            metric_group = group[group["metric"] == metric].copy()
            if len(metric_group) != 8:
                raise RuntimeError(
                    f"Expected 8 model rows for {keys}, metric={metric}; got {len(metric_group)}"
                )
            means = pd.to_numeric(metric_group["mean"], errors="coerce")
            lows = pd.to_numeric(metric_group["ci95_low"], errors="coerce")
            highs = pd.to_numeric(metric_group["ci95_high"], errors="coerce")
            row[f"mean_delta_{metric}"] = float(means.mean())
            row[f"models_delta_positive"] = int((means > 0).sum())
            row[f"models_delta_negative"] = int((means < 0).sum())
            row[f"models_ci_strictly_positive"] = int((lows > 0).sum())
            row[f"models_ci_strictly_negative"] = int((highs < 0).sum())

        rows.append(row)

    out = pd.DataFrame(rows)
    if len(out) != 12:
        raise RuntimeError(f"Expected 12 paired headline rows, found {len(out)}")

    out = out.sort_values(["endpoint", "split_type", "comparison"]).reset_index(drop=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)
    print("saved", OUTPUT)
    print(f"Paired classification effect summary complete. Rows: {len(out)}.")


if __name__ == "__main__":
    main()
