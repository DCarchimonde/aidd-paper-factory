from __future__ import annotations

"""Safely refresh the final Paper 2 manuscript figures.

The final layout and claim-tone pass overwrites Figures 1--3. Figures 4--6 are
already frozen manuscript assets. A previous entry point unnecessarily invoked
the full figure builder first, which required a large row-level selective-curve
CSV even when the frozen Figure 5 PDF/PNG already existed. This wrapper:

1. performs a full rebuild when all raw dependencies are available;
2. otherwise reuses the existing frozen Figures 4--6;
3. always rebuilds the layout-corrected Figures 1 and 2;
4. always rebuilds Figure 3 from its frozen summary table so that its title and
   caption-facing wording remain aligned with the moderated manuscript claims;
5. validates and refreshes the 12-file figure integrity manifest.

No model fitting or statistical re-analysis is performed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import figure35_layout_fix as layout
import figure37_workflow_final_fix as workflow


REUSABLE_STEMS = [
    "figure_4_applicability_domain_diagnostics",
    "figure_5_selective_prediction_diagnostics",
    "figure_6_regression_conformal_tradeoff",
]


def missing_reusable_assets() -> list[Path]:
    missing: list[Path] = []
    for stem in REUSABLE_STEMS:
        for suffix in (".pdf", ".png"):
            path = layout.base.FIGURE_DIR / f"{stem}{suffix}"
            if not path.exists():
                missing.append(path)
    return missing


def rebuild_classification_conformal_figure() -> list[Path]:
    """Rebuild Figure 3 without changing any frozen numerical content."""

    base = layout.base
    frame = base.read_required("table_rq3_rq4_classification_conformal.csv")
    base.require_columns(
        frame,
        [
            "endpoint",
            "split_type",
            "method",
            "positive_coverage",
            "negative_coverage",
            "mean_prediction_set_size",
            "ambiguous_set_rate",
        ],
        "classification conformal table",
    )
    frame = frame.copy()
    frame["method_label"] = frame["method"].map(base.METHOD_LABELS)

    fig = plt.figure(figsize=(14.0, 10.0))
    grid = fig.add_gridspec(
        3,
        6,
        height_ratios=[1.0, 1.0, 0.72],
        hspace=0.42,
        wspace=0.42,
    )
    coverage_axes: list[plt.Axes] = []
    x = np.arange(len(base.METHODS_CLASS))
    width = 0.34

    for row, endpoint in enumerate(["bbbp", "clintox"]):
        for col, split in enumerate(base.SPLITS):
            ax = fig.add_subplot(grid[row, col * 2 : (col + 1) * 2])
            coverage_axes.append(ax)
            subset = frame[
                (frame["endpoint"] == endpoint)
                & (frame["split_type"] == split)
            ].set_index("method_label")
            positive = [
                float(subset.loc[method, "positive_coverage"])
                for method in base.METHODS_CLASS
            ]
            negative = [
                float(subset.loc[method, "negative_coverage"])
                for method in base.METHODS_CLASS
            ]
            ax.bar(
                x - width / 2,
                positive,
                width,
                color=base.PALETTE["positive"],
                edgecolor="white",
                label="Positive",
            )
            ax.bar(
                x + width / 2,
                negative,
                width,
                color=base.PALETTE["negative"],
                edgecolor="white",
                label="Negative",
            )
            ax.axhline(0.90, color="#6B7280", linestyle="--", linewidth=1.0)
            ax.set_xticks(x, ["Marginal", "Shift-\nweighted", "Mondrian"])
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", alpha=0.18)
            endpoint_label = "BBBP" if endpoint == "bbbp" else "ClinTox"
            ax.set_title(f"{endpoint_label} · {base.SPLIT_LABELS[split]}", pad=8)
            if col == 0:
                ax.set_ylabel("Empirical coverage")

    for label, ax in zip(list("ABCDEF"), coverage_axes):
        base.panel_label(ax, label, x=-0.16, y=1.11)

    columns: list[str] = []
    for split in base.SPLITS:
        for method in base.METHODS_CLASS:
            short_method = {
                "Marginal": "Marg.",
                "Shift-weighted": "Weighted",
                "Mondrian": "Mond.",
            }[method]
            columns.append(f"{base.SPLIT_SHORT[split]}\n{short_method}")

    set_matrix = np.full((2, 9), np.nan)
    ambiguity_matrix = np.full((2, 9), np.nan)
    for i, endpoint in enumerate(["bbbp", "clintox"]):
        col_idx = 0
        for split in base.SPLITS:
            subset = frame[
                (frame["endpoint"] == endpoint)
                & (frame["split_type"] == split)
            ].set_index("method_label")
            for method in base.METHODS_CLASS:
                set_matrix[i, col_idx] = float(
                    subset.loc[method, "mean_prediction_set_size"]
                )
                ambiguity_matrix[i, col_idx] = float(
                    subset.loc[method, "ambiguous_set_rate"]
                )
                col_idx += 1

    ax_set = fig.add_subplot(grid[2, 0:3])
    base.heatmap(
        ax_set,
        set_matrix,
        ["BBBP", "ClinTox"],
        columns,
        "Mean prediction-set size",
        vmin=0.9,
        vmax=1.8,
        cmap="BuGn",
        fontsize=7.5,
    )
    base.panel_label(ax_set, "G", x=-0.11, y=1.12)

    ax_amb = fig.add_subplot(grid[2, 3:6])
    base.heatmap(
        ax_amb,
        ambiguity_matrix,
        ["BBBP", "ClinTox"],
        columns,
        "Ambiguous two-label set rate",
        vmin=0.0,
        vmax=0.75,
        cmap="BuGn",
        fontsize=7.5,
    )
    base.panel_label(ax_amb, "H", x=-0.11, y=1.12)

    handles, labels = coverage_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.935),
    )
    fig.suptitle(
        "Mondrian improves class-conditional coverage at the cost of less informative prediction sets",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.02,
        "Dashed line: nominal 90% coverage. R/S/C denote random, scaffold and similarity-cluster splits.",
        ha="center",
        fontsize=8.8,
    )
    return base.save_figure(fig, "figure_3_classification_conformal_tradeoff")


def main() -> None:
    selective_source = (
        layout.base.SELECTIVE_DIR
        / "selective_prediction_v2_curves_confirmatory_full.csv"
    )

    if selective_source.exists():
        print("Raw selective curves found; rebuilding the complete six-figure package.")
        layout.base.main()
    else:
        missing = missing_reusable_assets()
        if missing:
            joined = "\n".join(str(path) for path in missing)
            raise FileNotFoundError(
                "The raw selective-curve source is absent and one or more frozen "
                "Figures 4--6 are also missing. Restore the frozen manuscript "
                "assets or regenerate the selective-prediction outputs before a "
                f"full rebuild. Missing assets:\n{joined}"
            )
        print(
            "Raw selective curves are absent; reusing frozen Figures 4--6 and "
            "rebuilding Figures 1--3 from frozen manuscript tables."
        )

    workflow.rebuild_workflow_figure()
    layout.rebuild_performance_figure()
    rebuild_classification_conformal_figure()
    layout.refresh_manifest()
    print("Safe final manuscript figure refresh complete.")


if __name__ == "__main__":
    main()
