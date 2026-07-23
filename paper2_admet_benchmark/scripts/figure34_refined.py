from __future__ import annotations

"""Refined publication-ready figure implementation for Paper 2.

No model fitting, threshold selection, or statistical re-analysis is performed.
The module reads frozen manuscript tables and already-computed selective curves,
then exports six vector PDF and high-resolution PNG figure stems.
"""

import hashlib
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper2_admet_benchmark"
ASSET_DIR = PAPER_DIR / "results" / "manuscript_assets"
TABLE_DIR = ASSET_DIR / "tables"
FIGURE_DIR = ASSET_DIR / "figures"
SELECTIVE_DIR = PAPER_DIR / "results" / "selective"

SPLITS = ["random", "scaffold", "cluster"]
SPLIT_LABELS = {"random": "Random", "scaffold": "Scaffold", "cluster": "Cluster"}
SPLIT_SHORT = {"random": "R", "scaffold": "S", "cluster": "C"}
METHODS_CLASS = ["Marginal", "Shift-weighted", "Mondrian"]
METHODS_REG = ["Marginal", "Shift-weighted", "Adaptive"]

PALETTE = {
    "random": "#2F8F9D",
    "scaffold": "#7BC8A4",
    "cluster": "#557C8A",
    "positive": "#2F8F9D",
    "negative": "#B8C5CC",
    "marginal": "#2F8F9D",
    "shift_weighted": "#7BC8A4",
    "mondrian": "#557C8A",
    "adaptive": "#AAB7B8",
    "coral": "#D9785F",
    "cream": "#F5F1E8",
    "dark_text": "#1F2937",
    "outline": "#5B6B73",
    "light_fill": "#EAF4F1",
    "light_fill_2": "#D8ECE7",
    "light_fill_3": "#C4E1D9",
}

METHOD_LABELS = {
    "marginal_lac": "Marginal",
    "density_ratio_weighted_lac": "Shift-weighted",
    "mondrian_lac": "Mondrian",
    "marginal_absolute_residual": "Marginal",
    "density_ratio_weighted_absolute_residual": "Shift-weighted",
    "split_calibration_adaptive_normalized": "Adaptive",
}
METHOD_COLORS = {
    "Marginal": PALETTE["marginal"],
    "Shift-weighted": PALETTE["shift_weighted"],
    "Mondrian": PALETTE["mondrian"],
    "Adaptive": PALETTE["adaptive"],
}

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "teal_cream_coral",
    ["#2B7A78", PALETTE["cream"], PALETTE["coral"]],
)
SPLIT_PATTERN = re.compile(r"split_confirm_(random|scaffold|cluster)_seed(\d+)$")

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": PALETTE["outline"],
        "axes.labelcolor": PALETTE["dark_text"],
        "text.color": PALETTE["dark_text"],
        "xtick.color": PALETTE["dark_text"],
        "ytick.color": PALETTE["dark_text"],
    }
)


def read_required(name: str) -> pd.DataFrame:
    path = TABLE_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise RuntimeError(f"Empty required table: {path}")
    return frame


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{name} missing columns: {missing}")


def panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.10) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
        clip_on=False,
    )


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = [FIGURE_DIR / f"{stem}.pdf", FIGURE_DIR / f"{stem}.png"]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cleanup_stale_outputs() -> None:
    stale_stems = [
        "figure_4_ad_and_selective_diagnostics",
        "figure_5_regression_conformal_tradeoff",
    ]
    for stem in stale_stems:
        for suffix in [".pdf", ".png"]:
            path = FIGURE_DIR / f"{stem}{suffix}"
            if path.exists():
                path.unlink()
                print("removed stale", path)


def grouped_bars(
    ax: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
    endpoint_order: list[str],
    ylabel: str,
    title: str,
    baseline: float | None = None,
) -> None:
    require_columns(frame, ["endpoint", "split_type", metric], title)
    width = 0.22
    x = np.arange(len(endpoint_order), dtype=float)
    for split, offset in zip(SPLITS, [-width, 0.0, width]):
        subset = frame[frame["split_type"] == split].set_index("endpoint")
        values = [float(subset.loc[endpoint, metric]) for endpoint in endpoint_order]
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=PALETTE[split],
            edgecolor="white",
            linewidth=0.8,
            label=SPLIT_LABELS[split],
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=7.5,
                rotation=90,
                color=PALETTE["dark_text"],
            )
    if baseline is not None:
        ax.axhline(baseline, linestyle="--", linewidth=1.0, color="#6B7280", alpha=0.8)
    labels = [endpoint.upper() if endpoint != "lipophilicity" else "Lipophilicity" for endpoint in endpoint_order]
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    ax.grid(axis="y", alpha=0.18)


def heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    *,
    vmin: float,
    vmax: float,
    center: float | None = None,
    cmap=DIVERGING_CMAP,
    colorbar: bool = True,
    fontsize: float = 8.0,
) -> None:
    norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax) if center is not None else None
    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        vmin=None if norm is not None else vmin,
        vmax=None if norm is not None else vmax,
        norm=norm,
    )
    ax.set_xticks(np.arange(len(col_labels)), col_labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_title(title, pad=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=fontsize)
    if colorbar:
        plt.colorbar(image, ax=ax, fraction=0.045, pad=0.03)


def fig1_design() -> list[Path]:
    fig, ax = plt.subplots(figsize=(11.8, 6.4))
    ax.set_axis_off()

    top_boxes = [
        (0.05, 0.72, 0.25, 0.17, "1. Data and frozen models\nFour ADMET endpoints\nECFP-based model families"),
        (0.375, 0.72, 0.25, 0.17, "2. Label-blind evaluation\nRandom · scaffold · cluster\nTrain / calibration / test"),
        (0.70, 0.72, 0.25, 0.17, "3. Confirmatory design\n10 random/scaffold seeds · 5 cluster seeds\nPaired within endpoint, model and seed"),
    ]
    for x, y, w, h, text in top_boxes:
        patch = plt.Rectangle(
            (x, y), w, h, transform=ax.transAxes,
            facecolor=PALETTE["light_fill"], edgecolor=PALETTE["outline"], linewidth=1.3
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, transform=ax.transAxes, ha="center", va="center", fontsize=10.5)

    for start, end in [((0.30, 0.805), (0.375, 0.805)), ((0.625, 0.805), (0.70, 0.805))]:
        ax.annotate("", xy=end, xytext=start, xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", lw=1.5, color=PALETTE["outline"]))

    analysis_x, analysis_y, analysis_w, analysis_h = 0.23, 0.49, 0.54, 0.12
    analysis = plt.Rectangle(
        (analysis_x, analysis_y), analysis_w, analysis_h, transform=ax.transAxes,
        facecolor=PALETTE["light_fill_2"], edgecolor=PALETTE["outline"], linewidth=1.4
    )
    ax.add_patch(analysis)
    ax.text(
        analysis_x + analysis_w / 2,
        analysis_y + analysis_h / 2,
        "Reliability-analysis modules: performance · calibration · applicability domain · conformal prediction · selective prediction",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.2,
    )
    ax.annotate("", xy=(0.50, analysis_y + analysis_h), xytext=(0.825, 0.72), xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", lw=1.5, color=PALETTE["outline"]))

    rq_boxes = [
        (0.03, 0.16, 0.21, 0.20, "RQ1\nPerformance and\nprobability calibration", PALETTE["light_fill"]),
        (0.275, 0.16, 0.21, 0.20, "RQ2\nApplicability-domain\nrobustness", PALETTE["light_fill_2"]),
        (0.52, 0.16, 0.21, 0.20, "RQ3\nClass-conditional and\nsubgroup failure", PALETTE["light_fill_3"]),
        (0.765, 0.16, 0.21, 0.20, "RQ4\nCoverage–efficiency–\ninformativeness trade-offs", "#B3D6CC"),
    ]
    bus_y = 0.425
    ax.plot([0.135, 0.87], [bus_y, bus_y], transform=ax.transAxes, color="#7A8B93", linewidth=1.2)
    ax.annotate("", xy=(0.50, bus_y), xytext=(0.50, analysis_y), xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="-", lw=1.2, color="#7A8B93"))
    for x, y, w, h, text, fill in rq_boxes:
        patch = plt.Rectangle(
            (x, y), w, h, transform=ax.transAxes,
            facecolor=fill, edgecolor=PALETTE["outline"], linewidth=1.2
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, transform=ax.transAxes, ha="center", va="center", fontsize=10.5)
        center_x = x + w / 2
        ax.annotate("", xy=(center_x, y + h), xytext=(center_x, bus_y), xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", lw=1.1, color="#7A8B93"))

    ax.set_title("Confirmatory reliability-evaluation workflow", fontsize=15, fontweight="bold", pad=14)
    fig.tight_layout()
    return save_figure(fig, "figure_1_confirmatory_design")


def fig2_performance_calibration() -> list[Path]:
    class_perf = read_required("table_rq1_classification_performance.csv").rename(columns={"split": "split_type"})
    reg_perf = read_required("table_rq1_regression_performance.csv").rename(columns={"split": "split_type"})
    class_cal = read_required("table_rq1_classification_calibration.csv").rename(columns={"split": "split_type"})

    fig, axes = plt.subplots(2, 4, figsize=(15.2, 7.5))
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
        grouped_bars(ax, frame, metric, endpoints, ylabel, title, baseline)
    for label, ax in zip(list("ABCDEFGH"), axes.flat):
        panel_label(ax, label, x=-0.16, y=1.11)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.015))
    fig.suptitle("Predictive performance and probability calibration under chemical distribution shift", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.045,
        "Bars are descriptive means across frozen model/regime combinations; model-specific cross-seed uncertainty is reported in supplementary tables.",
        ha="center",
        fontsize=8.8,
        style="italic",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94), h_pad=2.2, w_pad=1.7)
    return save_figure(fig, "figure_2_performance_and_calibration")


def fig3_classification_conformal() -> list[Path]:
    frame = read_required("table_rq3_rq4_classification_conformal.csv")
    require_columns(
        frame,
        ["endpoint", "split_type", "method", "positive_coverage", "negative_coverage", "mean_prediction_set_size", "ambiguous_set_rate"],
        "classification conformal table",
    )
    frame = frame.copy()
    frame["method_label"] = frame["method"].map(METHOD_LABELS)

    fig = plt.figure(figsize=(14.0, 10.0))
    grid = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.0, 0.72], hspace=0.42, wspace=0.42)
    coverage_axes: list[plt.Axes] = []
    x = np.arange(len(METHODS_CLASS))
    width = 0.34

    for row, endpoint in enumerate(["bbbp", "clintox"]):
        for col, split in enumerate(SPLITS):
            ax = fig.add_subplot(grid[row, col * 2:(col + 1) * 2])
            coverage_axes.append(ax)
            subset = frame[(frame["endpoint"] == endpoint) & (frame["split_type"] == split)].set_index("method_label")
            positive = [float(subset.loc[method, "positive_coverage"]) for method in METHODS_CLASS]
            negative = [float(subset.loc[method, "negative_coverage"]) for method in METHODS_CLASS]
            ax.bar(x - width / 2, positive, width, color=PALETTE["positive"], edgecolor="white", label="Positive")
            ax.bar(x + width / 2, negative, width, color=PALETTE["negative"], edgecolor="white", label="Negative")
            ax.axhline(0.90, color="#6B7280", linestyle="--", linewidth=1.0)
            ax.set_xticks(x, ["Marginal", "Shift-\nweighted", "Mondrian"])
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", alpha=0.18)
            endpoint_label = "BBBP" if endpoint == "bbbp" else "ClinTox"
            ax.set_title(f"{endpoint_label} · {SPLIT_LABELS[split]}", pad=8)
            if col == 0:
                ax.set_ylabel("Empirical coverage")

    for label, ax in zip(list("ABCDEF"), coverage_axes):
        panel_label(ax, label, x=-0.16, y=1.11)

    columns = []
    for split in SPLITS:
        for method in METHODS_CLASS:
            short_method = {"Marginal": "Marg.", "Shift-weighted": "Weighted", "Mondrian": "Mond."}[method]
            columns.append(f"{SPLIT_SHORT[split]}\n{short_method}")

    set_matrix = np.full((2, 9), np.nan)
    ambiguity_matrix = np.full((2, 9), np.nan)
    for i, endpoint in enumerate(["bbbp", "clintox"]):
        col_idx = 0
        for split in SPLITS:
            subset = frame[(frame["endpoint"] == endpoint) & (frame["split_type"] == split)].set_index("method_label")
            for method in METHODS_CLASS:
                set_matrix[i, col_idx] = float(subset.loc[method, "mean_prediction_set_size"])
                ambiguity_matrix[i, col_idx] = float(subset.loc[method, "ambiguous_set_rate"])
                col_idx += 1

    ax_set = fig.add_subplot(grid[2, 0:3])
    heatmap(
        ax_set, set_matrix, ["BBBP", "ClinTox"], columns, "Mean prediction-set size",
        vmin=0.9, vmax=1.8, cmap="BuGn", fontsize=7.5
    )
    panel_label(ax_set, "G", x=-0.11, y=1.12)

    ax_amb = fig.add_subplot(grid[2, 3:6])
    heatmap(
        ax_amb, ambiguity_matrix, ["BBBP", "ClinTox"], columns, "Ambiguous two-label set rate",
        vmin=0.0, vmax=0.75, cmap="BuGn", fontsize=7.5
    )
    panel_label(ax_amb, "H", x=-0.11, y=1.12)

    handles, labels = coverage_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.935))
    fig.suptitle("Mondrian repairs class-conditional coverage at the cost of less informative prediction sets", fontsize=15, fontweight="bold", y=0.985)
    fig.text(0.5, 0.02, "Dashed line: nominal 90% coverage. R/S/C denote random, scaffold and cluster splits.", ha="center", fontsize=8.8)
    return save_figure(fig, "figure_3_classification_conformal_tradeoff")


def fig4_ad_diagnostics() -> list[Path]:
    continuous = read_required("table_rq2_ad_continuous.csv")
    threshold = read_required("table_rq2_ad_threshold_sensitivity.csv")
    require_columns(
        continuous,
        ["endpoint", "split_type", "mean_risk_similarity_spearman", "mean_miscoverage_similarity_spearman"],
        "AD continuous table",
    )
    require_columns(
        threshold,
        ["endpoint", "split_type", "threshold", "mean_delta_risk_low_minus_high", "mean_delta_miscoverage_low_minus_high"],
        "AD threshold table",
    )

    endpoints = ["bbbp", "clintox", "esol", "lipophilicity"]
    endpoint_labels = ["BBBP", "ClinTox", "ESOL", "Lipophilicity"]
    risk_matrix = np.full((4, 3), np.nan)
    miscoverage_matrix = np.full((4, 3), np.nan)
    for i, endpoint in enumerate(endpoints):
        for j, split in enumerate(SPLITS):
            subset = continuous[(continuous["endpoint"] == endpoint) & (continuous["split_type"] == split)]
            if not subset.empty:
                risk_matrix[i, j] = float(subset["mean_risk_similarity_spearman"].iloc[0])
                miscoverage_matrix[i, j] = float(subset["mean_miscoverage_similarity_spearman"].iloc[0])

    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.2), constrained_layout=True)
    heatmap(
        axes[0, 0], risk_matrix, endpoint_labels, [SPLIT_LABELS[s] for s in SPLITS],
        "Similarity versus predictive risk\n(Spearman ρ)", vmin=-0.35, vmax=0.35, center=0.0
    )
    heatmap(
        axes[0, 1], miscoverage_matrix, endpoint_labels, [SPLIT_LABELS[s] for s in SPLITS],
        "Similarity versus conformal miscoverage\n(Spearman ρ)", vmin=-0.35, vmax=0.35, center=0.0
    )

    ax = axes[0, 2]
    for endpoint, linestyle in [("bbbp", "-"), ("lipophilicity", "--")]:
        for split in SPLITS:
            subset = threshold[(threshold["endpoint"] == endpoint) & (threshold["split_type"] == split)].sort_values("threshold")
            ax.plot(
                subset["threshold"], subset["mean_delta_risk_low_minus_high"],
                linestyle=linestyle, linewidth=2.1, color=PALETTE[split]
            )
    ax.axhline(0, color="#6B7280", linewidth=1.0)
    ax.set_title("Threshold sensitivity of low-domain risk", pad=9)
    ax.set_xlabel("Tanimoto threshold")
    ax.set_ylabel("Δ risk (low − high)")
    ax.grid(alpha=0.18)
    legend_handles = [
        Line2D([0], [0], color=PALETTE["random"], lw=2.1, label="Random"),
        Line2D([0], [0], color=PALETTE["scaffold"], lw=2.1, label="Scaffold"),
        Line2D([0], [0], color=PALETTE["cluster"], lw=2.1, label="Cluster"),
        Line2D([0], [0], color="#555555", lw=2.1, linestyle="-", label="BBBP"),
        Line2D([0], [0], color="#555555", lw=2.1, linestyle="--", label="Lipophilicity"),
    ]
    ax.legend(handles=legend_handles, frameon=False, ncol=2, fontsize=8, loc="best")

    for axis, endpoint, title in [
        (axes[1, 0], "bbbp", "BBBP: low-domain risk difference"),
        (axes[1, 1], "clintox", "ClinTox: low-domain risk difference"),
    ]:
        subset = threshold[threshold["endpoint"] == endpoint]
        thresholds = sorted(subset["threshold"].dropna().unique())
        matrix = np.full((3, len(thresholds)), np.nan)
        for i, split in enumerate(SPLITS):
            for j, cutoff in enumerate(thresholds):
                cell = subset[(subset["split_type"] == split) & np.isclose(subset["threshold"], cutoff)]
                if not cell.empty:
                    matrix[i, j] = float(cell["mean_delta_risk_low_minus_high"].iloc[0])
        heatmap(
            axis, matrix, [SPLIT_LABELS[s] for s in SPLITS], [f"{x:.1f}" for x in thresholds],
            title, vmin=-0.25, vmax=0.55, center=0.0, fontsize=7.5
        )
        axis.set_xlabel("Tanimoto threshold")

    ax = axes[1, 2]
    for split in SPLITS:
        subset = threshold[(threshold["endpoint"] == "lipophilicity") & (threshold["split_type"] == split)].sort_values("threshold")
        ax.plot(
            subset["threshold"], subset["mean_delta_miscoverage_low_minus_high"],
            marker="o", linewidth=2.1, color=PALETTE[split], label=SPLIT_LABELS[split]
        )
    ax.axhline(0, color="#6B7280", linewidth=1.0)
    ax.set_title("Lipophilicity: low-domain miscoverage", pad=9)
    ax.set_xlabel("Tanimoto threshold")
    ax.set_ylabel("Δ miscoverage (low − high)")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="best")

    for label, ax in zip(list("ABCDEF"), axes.flat):
        panel_label(ax, label, x=-0.17, y=1.14)
    fig.suptitle("Applicability-domain diagnostics are endpoint- and split-dependent", fontsize=15, fontweight="bold")
    return save_figure(fig, "figure_4_applicability_domain_diagnostics")


def load_clintox_retention_curves() -> pd.DataFrame:
    path = SELECTIVE_DIR / "selective_prediction_v2_curves_confirmatory_full.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    curves = pd.read_csv(path, low_memory=False)
    require_columns(
        curves,
        ["endpoint", "task_type", "split_col", "uncertainty_measure", "target_coverage", "positive_retention_rate"],
        "selective curves",
    )
    curves = curves[(curves["endpoint"] == "clintox") & (curves["task_type"] == "classification")].copy()
    extracted = curves["split_col"].astype(str).str.extract(SPLIT_PATTERN)
    curves["split_type"] = extracted[0]
    if curves["split_type"].isna().any():
        raise ValueError("Unrecognized split columns in selective curves")
    return (
        curves.groupby(["split_type", "uncertainty_measure", "target_coverage"], as_index=False)["positive_retention_rate"]
        .mean()
        .sort_values(["split_type", "uncertainty_measure", "target_coverage"])
    )


def fig5_selective_prediction() -> list[Path]:
    frame = read_required("table_rq2_rq3_selective_prediction.csv")
    require_columns(
        frame,
        ["endpoint", "split_type", "uncertainty_measure", "primary_paurc_improvement_vs_random", "balanced_paurc_improvement_vs_random", "risk_improvement_at_05"],
        "selective prediction table",
    )
    frame = frame.copy()
    frame["measure_label"] = frame["uncertainty_measure"].map(
        {"one_minus_max_probability": "Confidence", "one_minus_max_tanimoto_to_train": "Chemical similarity"}
    )
    class_frame = frame[frame["endpoint"].isin(["bbbp", "clintox"])].copy()
    regression_frame = frame[frame["endpoint"].isin(["esol", "lipophilicity"])].copy()

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.2))

    ax = axes[0, 0]
    positions = np.arange(6)
    labels = []
    ordinary = []
    balanced = []
    for endpoint in ["bbbp", "clintox"]:
        for split in SPLITS:
            subset = class_frame[
                (class_frame["endpoint"] == endpoint)
                & (class_frame["split_type"] == split)
                & (class_frame["measure_label"] == "Confidence")
            ]
            labels.append(f"{endpoint.upper() if endpoint == 'bbbp' else 'ClinTox'}\n{SPLIT_SHORT[split]}")
            ordinary.append(float(subset["primary_paurc_improvement_vs_random"].iloc[0]))
            balanced.append(float(subset["balanced_paurc_improvement_vs_random"].iloc[0]))
    width = 0.35
    ax.bar(positions - width / 2, ordinary, width, color=PALETTE["random"], label="Ordinary-risk gain")
    ax.bar(positions + width / 2, balanced, width, color=PALETTE["scaffold"], label="Balanced-risk gain")
    ax.axhline(0, color="#6B7280", linewidth=1.0)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Matched pAURC improvement")
    ax.set_title("Confidence-based selective prediction", pad=8)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    retention = load_clintox_retention_curves()
    for split in SPLITS:
        for measure, linestyle in [
            ("one_minus_max_probability", "-"),
            ("one_minus_max_tanimoto_to_train", "--"),
        ]:
            subset = retention[(retention["split_type"] == split) & (retention["uncertainty_measure"] == measure)]
            ax.plot(
                subset["target_coverage"], subset["positive_retention_rate"],
                color=PALETTE[split], linestyle=linestyle, linewidth=2.1,
            )
    ax.plot([0.1, 1.0], [0.1, 1.0], color="#6B7280", linestyle=":", linewidth=1.2)
    ax.set_xlim(0.1, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Retained coverage")
    ax.set_ylabel("Positive-class retention")
    ax.set_title("ClinTox minority retention across rejection levels", pad=8)
    ax.grid(alpha=0.18)
    retention_handles = [
        Line2D([0], [0], color=PALETTE["random"], lw=2.1, label="Random"),
        Line2D([0], [0], color=PALETTE["scaffold"], lw=2.1, label="Scaffold"),
        Line2D([0], [0], color=PALETTE["cluster"], lw=2.1, label="Cluster"),
        Line2D([0], [0], color="#555555", lw=2.1, linestyle="-", label="Confidence"),
        Line2D([0], [0], color="#555555", lw=2.1, linestyle="--", label="Chemical similarity"),
    ]
    ax.legend(handles=retention_handles, frameon=False, ncol=2, fontsize=8, loc="upper left")

    ax = axes[1, 0]
    values = []
    labels = []
    for endpoint, display in [("esol", "ESOL"), ("lipophilicity", "Lipo.")]:
        for split in SPLITS:
            subset = regression_frame[
                (regression_frame["endpoint"] == endpoint)
                & (regression_frame["split_type"] == split)
                & (regression_frame["measure_label"] == "Chemical similarity")
            ]
            values.append(float(subset["risk_improvement_at_05"].iloc[0]))
            labels.append(f"{display}\n{SPLIT_SHORT[split]}")
    ax.bar(np.arange(len(values)), values, color=PALETTE["cluster"], edgecolor="white")
    ax.axhline(0, color="#6B7280", linewidth=1.0)
    ax.set_xticks(np.arange(len(values)), labels)
    ax.set_ylabel("Risk improvement at 50% coverage")
    ax.set_title("Similarity-based selective prediction for regression", pad=8)
    ax.grid(axis="y", alpha=0.18)

    ax = axes[1, 1]
    x = np.arange(3)
    esol = []
    lipophilicity = []
    for split in SPLITS:
        esol_subset = regression_frame[
            (regression_frame["endpoint"] == "esol")
            & (regression_frame["split_type"] == split)
            & (regression_frame["measure_label"] == "Chemical similarity")
        ]
        lipophilicity_subset = regression_frame[
            (regression_frame["endpoint"] == "lipophilicity")
            & (regression_frame["split_type"] == split)
            & (regression_frame["measure_label"] == "Chemical similarity")
        ]
        esol.append(float(esol_subset["primary_paurc_improvement_vs_random"].iloc[0]))
        lipophilicity.append(float(lipophilicity_subset["primary_paurc_improvement_vs_random"].iloc[0]))
    ax.plot(x, esol, marker="o", linewidth=2.2, color=PALETTE["random"], label="ESOL")
    ax.plot(x, lipophilicity, marker="o", linewidth=2.2, color=PALETTE["cluster"], label="Lipophilicity")
    ax.axhline(0, color="#6B7280", linewidth=1.0)
    ax.set_xticks(x, [SPLIT_LABELS[s] for s in SPLITS])
    ax.set_ylabel("Primary pAURC improvement")
    ax.set_title("Endpoint dependence of similarity-based rejection", pad=8)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False)

    for label, ax in zip(list("ABCD"), axes.flat):
        panel_label(ax, label, x=-0.13, y=1.10)
    fig.suptitle("Selective prediction can reduce ordinary risk while disproportionately removing the minority class", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.015, "R/S/C denote random, scaffold and cluster splits; pAURC is integrated over retained coverage 0.10–1.00.", ha="center", fontsize=8.8)
    fig.tight_layout(rect=(0, 0.045, 1, 0.94), h_pad=2.5, w_pad=2.0)
    return save_figure(fig, "figure_5_selective_prediction_diagnostics")


def plot_regression_metric(
    ax: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
    methods: list[str],
    ylabel: str,
    title: str,
    baseline: float | None = None,
) -> None:
    x = np.arange(6, dtype=float)
    labels = [
        "ESOL\nRandom", "ESOL\nScaffold", "ESOL\nCluster",
        "Lipo.\nRandom", "Lipo.\nScaffold", "Lipo.\nCluster",
    ]
    width = 0.22 if len(methods) == 3 else 0.28
    offsets = np.linspace(-width * (len(methods) - 1) / 2, width * (len(methods) - 1) / 2, len(methods))
    for method, offset in zip(methods, offsets):
        values = []
        for endpoint in ["esol", "lipophilicity"]:
            for split in SPLITS:
                subset = frame[(frame["endpoint"] == endpoint) & (frame["split_type"] == split)].set_index("method_label")
                value = subset.loc[method, metric]
                values.append(float(value) if pd.notna(value) else np.nan)
        ax.bar(x + offset, values, width=width, color=METHOD_COLORS[method], edgecolor="white", label=method)
    if baseline is not None:
        ax.axhline(baseline, color="#6B7280", linestyle="--", linewidth=1.0)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    ax.grid(axis="y", alpha=0.18)


def fig6_regression_conformal() -> list[Path]:
    frame = read_required("table_rq4_regression_conformal.csv")
    require_columns(
        frame,
        ["endpoint", "split_type", "method", "empirical_coverage", "mean_interval_width", "interval_width_cv", "width_error_spearman"],
        "regression conformal table",
    )
    frame = frame.copy()
    frame["method_label"] = frame["method"].map(METHOD_LABELS)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2))
    plot_regression_metric(axes[0, 0], frame, "empirical_coverage", METHODS_REG, "Empirical coverage", "Coverage", 0.90)
    axes[0, 0].set_ylim(0.0, 1.0)
    plot_regression_metric(axes[0, 1], frame, "mean_interval_width", METHODS_REG, "Mean interval width", "Efficiency")
    plot_regression_metric(axes[1, 0], frame, "interval_width_cv", METHODS_REG, "Interval-width coefficient of variation", "Interval heterogeneity")
    plot_regression_metric(
        axes[1, 1], frame, "width_error_spearman", ["Shift-weighted", "Adaptive"],
        "Width–error Spearman ρ", "Width–error association for variable-width methods", 0.0
    )
    axes[1, 1].text(
        0.02, 0.95, "Marginal: N/A (constant interval width)",
        transform=axes[1, 1].transAxes, ha="left", va="top", fontsize=8.5, style="italic"
    )

    for label, ax in zip(list("ABCD"), axes.flat):
        panel_label(ax, label, x=-0.13, y=1.10)

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS[m]) for m in METHODS_REG]
    fig.legend(legend_handles, METHODS_REG, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Regression conformal methods show no universal coverage–efficiency advantage", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.055, 1, 0.94), h_pad=2.4, w_pad=2.0)
    return save_figure(fig, "figure_6_regression_conformal_tradeoff")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_stale_outputs()

    generated: list[Path] = []
    generated.extend(fig1_design())
    generated.extend(fig2_performance_calibration())
    generated.extend(fig3_classification_conformal())
    generated.extend(fig4_ad_diagnostics())
    generated.extend(fig5_selective_prediction())
    generated.extend(fig6_regression_conformal())

    manifest = pd.DataFrame(
        [
            {
                "file": path.relative_to(PAPER_DIR).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(generated)
        ]
    )
    manifest_path = FIGURE_DIR / "main_figure_integrity_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print("saved", manifest_path)
    print(f"Main figure package complete. figure_stems=6, exported_files={len(generated)}, manifest_rows={len(manifest)}.")


if __name__ == "__main__":
    main()
