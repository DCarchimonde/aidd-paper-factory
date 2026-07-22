from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper1_leakage_benchmark"
TABLE_DIR = PAPER_DIR / "results" / "tables"
FIGURE_DIR = PAPER_DIR / "results" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = TABLE_DIR / "paper1_robustness20_summary.csv"
NULL_PATH = TABLE_DIR / "paper1_balanced_split_null_summary.csv"

MODEL_LABELS = {
    "LogisticRegression": "Logistic regression",
    "RandomForest": "Random forest",
    "XGBoost": "XGBoost",
    "Ridge": "Ridge",
}


def draw_effect_panel(ax, frame: pd.DataFrame, title: str) -> None:
    frame = frame.copy()
    frame["label"] = frame["dataset"] + " — " + frame["model"].map(MODEL_LABELS)
    frame = frame.sort_values(["dataset", "model"], ascending=[False, False]).reset_index(drop=True)

    y = np.arange(len(frame))
    mean = frame["gap_reduction_mean"].to_numpy(dtype=float)
    low = frame["gap_reduction_ci95_low"].to_numpy(dtype=float)
    high = frame["gap_reduction_ci95_high"].to_numpy(dtype=float)
    xerr = np.vstack([mean - low, high - mean])

    significant = frame["wilcoxon_p_holm"].to_numpy(dtype=float) < 0.05
    markers = ["o" if sig else "s" for sig in significant]

    for i, marker in enumerate(markers):
        ax.errorbar(
            mean[i],
            y[i],
            xerr=xerr[:, i : i + 1],
            fmt=marker,
            capsize=3,
            markersize=5,
            linewidth=1.2,
        )

    ax.axvline(0.0, linewidth=1.0, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(frame["label"])
    ax.set_xlabel("Ordinary minus target-balanced generalization gap")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)

    xmin, xmax = ax.get_xlim()
    span = xmax - xmin
    ax.text(
        xmin + 0.02 * span,
        len(frame) - 0.35,
        "Balanced gap larger",
        ha="left",
        va="top",
        fontsize=8,
    )
    ax.text(
        xmax - 0.02 * span,
        len(frame) - 0.35,
        "Balanced gap smaller",
        ha="right",
        va="top",
        fontsize=8,
    )
    ax.text(
        0.01,
        0.01,
        "Circles: Holm-adjusted p < 0.05; squares: not significant",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
    )


def main() -> None:
    robustness = pd.read_csv(SUMMARY_PATH)
    null_summary = pd.read_csv(NULL_PATH)

    classification = robustness[robustness["task_type"] == "classification"].copy()
    regression = robustness[robustness["task_type"] == "regression"].copy()

    fig = plt.figure(figsize=(12.0, 9.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.82, 1.18])

    ax_a = fig.add_subplot(grid[0, :])
    ordered = null_summary.sort_values("balanced_improvement_percentile_mean", ascending=True)
    y = np.arange(len(ordered))
    percentile = 100.0 * ordered["balanced_improvement_percentile_mean"].to_numpy(dtype=float)
    minimum = 100.0 * ordered["balanced_improvement_percentile_min"].to_numpy(dtype=float)

    ax_a.barh(y, percentile, alpha=0.75)
    ax_a.scatter(minimum, y, marker="|", s=140, linewidths=2)
    ax_a.axvline(95.0, linewidth=1.0, linestyle="--")
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(ordered["dataset"])
    ax_a.set_xlim(0, 100)
    ax_a.set_xlabel("Random scaffold assignments with worse target balance (%)")
    ax_a.set_title(
        "A  Target-balanced scaffold assignments relative to the random-scaffold null",
        loc="left",
        fontweight="bold",
    )
    ax_a.grid(axis="x", alpha=0.25)

    for i, value in enumerate(percentile):
        ax_a.text(min(value + 1.2, 98.0), i, f"{value:.1f}%", va="center", fontsize=8)

    ax_a.text(
        0.995,
        0.03,
        "Bars: mean across 20 seeds; vertical marks: minimum across seeds",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )

    ax_b = fig.add_subplot(grid[1, 0])
    draw_effect_panel(ax_b, classification, "B  Classification datasets")

    ax_c = fig.add_subplot(grid[1, 1])
    draw_effect_panel(ax_c, regression, "C  Regression datasets")

    fig.suptitle(
        "Target balancing reliably reduces label shift but has heterogeneous predictive effects",
        fontsize=14,
        fontweight="bold",
    )

    png_path = FIGURE_DIR / "figure_robustness20_split_effects.png"
    pdf_path = FIGURE_DIR / "figure_robustness20_split_effects.pdf"
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("saved", png_path)
    print("saved", pdf_path)


if __name__ == "__main__":
    main()
