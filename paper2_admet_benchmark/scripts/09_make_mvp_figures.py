from __future__ import annotations

"""Generate first-pass MVP figures for Paper 2.

Inputs:
- results/metrics/baseline_metrics_mvp.csv
- results/calibration/calibration_summary_mvp.csv
- results/conformal/conformal_summary_mvp.csv

Outputs:
- figures/fig2_random_vs_scaffold_performance_mvp.png
- figures/fig3_ece_by_split_mvp.png
- figures/fig4_conformal_coverage_mvp.png
- figures/fig5_conformal_width_or_set_size_mvp.png

These are draft analysis figures. Final manuscript figures may be redesigned later.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PAPER_DIR = ROOT / "paper2_admet_benchmark"
METRIC_DIR = PAPER_DIR / "results" / "metrics"
CALIBRATION_DIR = PAPER_DIR / "results" / "calibration"
CONFORMAL_DIR = PAPER_DIR / "results" / "conformal"
FIGURE_DIR = PAPER_DIR / "figures"


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def save_current_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("saved", path)


def pretty_split(split_col: str) -> str:
    if split_col == "split_random_seed0":
        return "Random"
    if split_col == "split_scaffold":
        return "Scaffold"
    return split_col.replace("split_", "")


def fig2_random_vs_scaffold_performance(metrics: pd.DataFrame) -> None:
    test = metrics[metrics["role"] == "test"].copy()
    test = test[test["split_col"].isin(["split_random_seed0", "split_scaffold"])]

    rows = []
    for _, row in test.iterrows():
        if row["task_type"] == "classification":
            value = row.get("roc_auc", np.nan)
            metric = "ROC-AUC"
        else:
            value = row.get("rmse", np.nan)
            metric = "RMSE"
        rows.append(
            {
                "endpoint": row["endpoint"],
                "model": row["model"],
                "split": pretty_split(row["split_col"]),
                "metric": metric,
                "value": value,
            }
        )
    plot_df = pd.DataFrame(rows).dropna(subset=["value"])

    for metric_name in ["ROC-AUC", "RMSE"]:
        part = plot_df[plot_df["metric"] == metric_name].copy()
        if part.empty:
            continue
        labels = [f"{r.endpoint}\n{r.model}" for r in part.drop_duplicates(["endpoint", "model"]).itertuples()]
        x = np.arange(len(labels))
        width = 0.38

        plt.figure(figsize=(max(8, len(labels) * 0.7), 5))
        for offset, split in [(-width / 2, "Random"), (width / 2, "Scaffold")]:
            values = []
            for label in labels:
                endpoint, model = label.split("\n")
                val = part[(part["endpoint"] == endpoint) & (part["model"] == model) & (part["split"] == split)]["value"]
                values.append(float(val.iloc[0]) if not val.empty else np.nan)
            plt.bar(x + offset, values, width, label=split)

        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel(metric_name)
        plt.title(f"MVP test performance under random and scaffold splits ({metric_name})")
        plt.legend()
        suffix = "classification" if metric_name == "ROC-AUC" else "regression"
        save_current_figure(FIGURE_DIR / f"fig2_random_vs_scaffold_performance_{suffix}_mvp.png")


def fig3_ece_by_split(calibration: pd.DataFrame) -> None:
    cls = calibration[
        (calibration["task_type"] == "classification")
        & (calibration["role"] == "test")
        & (calibration["split_col"].isin(["split_random_seed0", "split_scaffold"]))
    ].copy()
    if cls.empty:
        return

    labels = [f"{r.endpoint}\n{r.model}" for r in cls.drop_duplicates(["endpoint", "model"]).itertuples()]
    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(max(8, len(labels) * 0.7), 5))
    for offset, split_col in [(-width / 2, "split_random_seed0"), (width / 2, "split_scaffold")]:
        values = []
        for label in labels:
            endpoint, model = label.split("\n")
            val = cls[(cls["endpoint"] == endpoint) & (cls["model"] == model) & (cls["split_col"] == split_col)]["ece"]
            values.append(float(val.iloc[0]) if not val.empty else np.nan)
        plt.bar(x + offset, values, width, label=pretty_split(split_col))

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("ECE")
    plt.title("MVP classification calibration error on test sets")
    plt.legend()
    save_current_figure(FIGURE_DIR / "fig3_ece_by_split_mvp.png")


def fig4_conformal_coverage(conformal: pd.DataFrame) -> None:
    part = conformal[conformal["split_col"].isin(["split_random_seed0", "split_scaffold"])].copy()
    if part.empty:
        return

    labels = [f"{r.endpoint}\n{r.model}\nα={r.alpha}" for r in part.drop_duplicates(["endpoint", "model", "alpha"]).itertuples()]
    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(max(10, len(labels) * 0.5), 5))
    for offset, split_col in [(-width / 2, "split_random_seed0"), (width / 2, "split_scaffold")]:
        values = []
        nominal = []
        for label in labels:
            endpoint, model, alpha_text = label.split("\n")
            alpha = float(alpha_text.replace("α=", ""))
            val = part[(part["endpoint"] == endpoint) & (part["model"] == model) & (part["alpha"] == alpha) & (part["split_col"] == split_col)]
            if val.empty:
                values.append(np.nan)
                nominal.append(np.nan)
            else:
                values.append(float(val["empirical_coverage"].iloc[0]))
                nominal.append(float(val["nominal_coverage"].iloc[0]))
        plt.bar(x + offset, values, width, label=pretty_split(split_col))

    # Draw nominal reference lines for common alpha levels.
    for coverage in sorted(part["nominal_coverage"].dropna().unique()):
        plt.axhline(float(coverage), linestyle="--", linewidth=1)

    plt.xticks(x, labels, rotation=60, ha="right")
    plt.ylabel("Empirical coverage")
    plt.title("MVP conformal empirical coverage on test sets")
    plt.legend()
    save_current_figure(FIGURE_DIR / "fig4_conformal_coverage_mvp.png")


def fig5_conformal_width_or_set_size(conformal: pd.DataFrame) -> None:
    for task_type, value_col, ylabel, filename in [
        ("classification", "mean_prediction_set_size", "Mean prediction-set size", "fig5_prediction_set_size_mvp.png"),
        ("regression", "mean_interval_width", "Mean interval width", "fig5_interval_width_mvp.png"),
    ]:
        part = conformal[
            (conformal["task_type"] == task_type)
            & (conformal["alpha"] == 0.1)
            & (conformal["split_col"].isin(["split_random_seed0", "split_scaffold"]))
        ].copy()
        if part.empty or value_col not in part.columns:
            continue

        labels = [f"{r.endpoint}\n{r.model}" for r in part.drop_duplicates(["endpoint", "model"]).itertuples()]
        x = np.arange(len(labels))
        width = 0.38

        plt.figure(figsize=(max(8, len(labels) * 0.7), 5))
        for offset, split_col in [(-width / 2, "split_random_seed0"), (width / 2, "split_scaffold")]:
            values = []
            for label in labels:
                endpoint, model = label.split("\n")
                val = part[(part["endpoint"] == endpoint) & (part["model"] == model) & (part["split_col"] == split_col)][value_col]
                values.append(float(val.iloc[0]) if not val.empty else np.nan)
            plt.bar(x + offset, values, width, label=pretty_split(split_col))

        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel(ylabel)
        plt.title(f"MVP conformal uncertainty size at nominal 90% coverage ({task_type})")
        plt.legend()
        save_current_figure(FIGURE_DIR / filename)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(require_file(METRIC_DIR / "baseline_metrics_mvp.csv"))
    calibration = pd.read_csv(require_file(CALIBRATION_DIR / "calibration_summary_mvp.csv"))
    conformal = pd.read_csv(require_file(CONFORMAL_DIR / "conformal_summary_mvp.csv"))

    fig2_random_vs_scaffold_performance(metrics)
    fig3_ece_by_split(calibration)
    fig4_conformal_coverage(conformal)
    fig5_conformal_width_or_set_size(conformal)

    print("MVP figure generation complete.")


if __name__ == "__main__":
    main()
