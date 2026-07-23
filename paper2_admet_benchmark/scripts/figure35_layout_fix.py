from __future__ import annotations

"""Final layout corrections for the Paper 2 main-figure package.

The frozen numerical content is unchanged.  This wrapper rebuilds the refined
figure package, then overwrites Figures 1 and 2 to correct two visual issues
identified in the compiled manuscript:

1. the connector arrow entering the third workflow box in Figure 1 crossed the
   first line of box text;
2. the small explanatory sentence below Figure 2 collided visually with the
   bottom legend after LaTeX scaling.  That sentence is already stated in the
   manuscript caption, so it is removed from the artwork.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figure34_refined as base


EXPECTED_STEMS = [
    "figure_1_confirmatory_design",
    "figure_2_performance_and_calibration",
    "figure_3_classification_conformal_tradeoff",
    "figure_4_applicability_domain_diagnostics",
    "figure_5_selective_prediction_diagnostics",
    "figure_6_regression_conformal_tradeoff",
]


def rebuild_workflow_figure() -> list[Path]:
    fig, ax = plt.subplots(figsize=(12.0, 6.25))
    ax.set_axis_off()

    top_boxes = [
        (
            0.045,
            0.72,
            0.255,
            0.18,
            "1. Data and frozen models\nFour ADMET endpoints\nECFP-based model families",
        ),
        (
            0.372,
            0.72,
            0.255,
            0.18,
            "2. Label-blind evaluation\nRandom · scaffold · cluster\nTrain / calibration / test",
        ),
        (
            0.699,
            0.72,
            0.255,
            0.18,
            "3. Confirmatory design\n10 random/scaffold seeds · 5 cluster seeds\nPaired within endpoint, model and seed",
        ),
    ]
    for x, y, width, height, text in top_boxes:
        patch = plt.Rectangle(
            (x, y),
            width,
            height,
            transform=ax.transAxes,
            facecolor=base.PALETTE["light_fill"],
            edgecolor=base.PALETTE["outline"],
            linewidth=1.35,
        )
        ax.add_patch(patch)
        ax.text(
            x + width / 2,
            y + height / 2,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11.0,
            linespacing=1.20,
        )

    # Stop arrowheads before they enter the boxes.  The previous version ended
    # the arrow at the box boundary and the head crossed the first text line.
    for start, end in [((0.300, 0.81), (0.372, 0.81)), ((0.627, 0.81), (0.699, 0.81))]:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords=ax.transAxes,
            arrowprops=dict(
                arrowstyle="-|>",
                lw=1.45,
                color=base.PALETTE["outline"],
                mutation_scale=11,
                shrinkA=6,
                shrinkB=12,
            ),
        )

    analysis_x, analysis_y, analysis_w, analysis_h = 0.18, 0.475, 0.64, 0.135
    analysis = plt.Rectangle(
        (analysis_x, analysis_y),
        analysis_w,
        analysis_h,
        transform=ax.transAxes,
        facecolor=base.PALETTE["light_fill_2"],
        edgecolor=base.PALETTE["outline"],
        linewidth=1.4,
    )
    ax.add_patch(analysis)
    ax.text(
        analysis_x + analysis_w / 2,
        analysis_y + analysis_h / 2,
        "Reliability-analysis modules\n"
        "Performance · calibration · applicability domain · conformal prediction · selective prediction",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.7,
        linespacing=1.30,
    )
    ax.annotate(
        "",
        xy=(0.50, analysis_y + analysis_h),
        xytext=(0.827, 0.72),
        xycoords=ax.transAxes,
        arrowprops=dict(
            arrowstyle="-|>",
            lw=1.4,
            color=base.PALETTE["outline"],
            mutation_scale=10,
            shrinkA=7,
            shrinkB=8,
        ),
    )

    rq_boxes = [
        (0.025, 0.145, 0.215, 0.205, "RQ1\nPerformance and\nprobability calibration", base.PALETTE["light_fill"]),
        (0.270, 0.145, 0.215, 0.205, "RQ2\nApplicability-domain\nrobustness", base.PALETTE["light_fill_2"]),
        (0.515, 0.145, 0.215, 0.205, "RQ3\nClass-conditional and\nsubgroup failure", base.PALETTE["light_fill_3"]),
        (0.760, 0.145, 0.215, 0.205, "RQ4\nCoverage–efficiency–\ninformativeness trade-offs", "#B3D6CC"),
    ]
    bus_y = 0.405
    ax.plot(
        [0.1325, 0.8675],
        [bus_y, bus_y],
        transform=ax.transAxes,
        color="#7A8B93",
        linewidth=1.2,
    )
    ax.plot(
        [0.50, 0.50],
        [analysis_y, bus_y],
        transform=ax.transAxes,
        color="#7A8B93",
        linewidth=1.2,
    )

    for x, y, width, height, text, fill in rq_boxes:
        patch = plt.Rectangle(
            (x, y),
            width,
            height,
            transform=ax.transAxes,
            facecolor=fill,
            edgecolor=base.PALETTE["outline"],
            linewidth=1.2,
        )
        ax.add_patch(patch)
        ax.text(
            x + width / 2,
            y + height / 2,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11.0,
            linespacing=1.22,
        )
        center_x = x + width / 2
        ax.annotate(
            "",
            xy=(center_x, y + height),
            xytext=(center_x, bus_y),
            xycoords=ax.transAxes,
            arrowprops=dict(
                arrowstyle="-|>",
                lw=1.05,
                color="#7A8B93",
                mutation_scale=9,
                shrinkA=1,
                shrinkB=7,
            ),
        )

    ax.set_title(
        "Confirmatory reliability-evaluation workflow",
        fontsize=15.5,
        fontweight="bold",
        pad=14,
    )
    fig.tight_layout()
    return base.save_figure(fig, "figure_1_confirmatory_design")


def rebuild_performance_figure() -> list[Path]:
    class_perf = base.read_required("table_rq1_classification_performance.csv").rename(
        columns={"split": "split_type"}
    )
    reg_perf = base.read_required("table_rq1_regression_performance.csv").rename(
        columns={"split": "split_type"}
    )
    class_cal = base.read_required("table_rq1_classification_calibration.csv").rename(
        columns={"split": "split_type"}
    )

    fig, axes = plt.subplots(2, 4, figsize=(15.4, 7.7))
    panels = [
        (axes[0, 0], class_perf, "roc_auc", ["bbbp", "clintox"], "ROC-AUC", "Discrimination", 0.5),
        (axes[0, 1], class_perf, "pr_auc", ["bbbp", "clintox"], "PR-AUC", "Precision–recall", None),
        (axes[0, 2], class_perf, "balanced_accuracy", ["bbbp", "clintox"], "Balanced accuracy", "Minority-sensitive accuracy", 0.5),
        (axes[0, 3], class_cal, "ece_probability", ["bbbp", "clintox"], "ECE", "Probability calibration error", None),
        (axes[1, 0], reg_perf, "rmse", ["esol", "lipophilicity"], "RMSE", "Regression error", None),
        (axes[1, 1], reg_perf, "mae", ["esol", "lipophilicity"], "MAE", "Absolute error", None),
        (axes[1, 2], reg_perf, "r2", ["esol", "lipophilicity"], r"$R^2$", "Explained variance", 0.0),
        (axes[1, 3], class_cal, "negative_log_likelihood", ["bbbp", "clintox"], "NLL", "Negative log-likelihood", None),
    ]
    for ax, frame, metric, endpoints, ylabel, title, baseline in panels:
        base.grouped_bars(ax, frame, metric, endpoints, ylabel, title, baseline)
    for label, ax in zip(list("ABCDEFGH"), axes.flat):
        base.panel_label(ax, label, x=-0.16, y=1.11)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.012),
    )
    fig.suptitle(
        "Predictive performance and probability calibration under chemical distribution shift",
        fontsize=15.5,
        fontweight="bold",
    )
    # The explanatory sentence previously printed below the panels is retained
    # in the LaTeX caption, where it remains legible after page scaling.
    fig.tight_layout(rect=(0, 0.065, 1, 0.94), h_pad=2.25, w_pad=1.8)
    return base.save_figure(fig, "figure_2_performance_and_calibration")


def refresh_manifest() -> Path:
    generated: list[Path] = []
    for stem in EXPECTED_STEMS:
        for suffix in (".pdf", ".png"):
            path = base.FIGURE_DIR / f"{stem}{suffix}"
            if not path.exists():
                raise FileNotFoundError(path)
            generated.append(path)

    manifest = pd.DataFrame(
        [
            {
                "file": path.relative_to(base.PAPER_DIR).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": base.sha256(path),
            }
            for path in sorted(generated)
        ]
    )
    manifest_path = base.FIGURE_DIR / "main_figure_integrity_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print("refreshed", manifest_path)
    return manifest_path


def main() -> None:
    base.main()
    rebuild_workflow_figure()
    rebuild_performance_figure()
    refresh_manifest()
    print("Final manuscript layout corrections complete.")


if __name__ == "__main__":
    main()
