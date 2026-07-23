from __future__ import annotations

"""Build publication-ready main figures from frozen manuscript result tables.

No model fitting, threshold selection, or statistical re-analysis is performed.
The script reads only the frozen manuscript-assets tables and exports vector PDF,
high-resolution PNG, and an integrity manifest.
"""

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper2_admet_benchmark"
ASSET_DIR = PAPER_DIR / "results" / "manuscript_assets"
TABLE_DIR = ASSET_DIR / "tables"
FIGURE_DIR = ASSET_DIR / "figures"

SPLITS = ["random", "scaffold", "cluster"]
SPLIT_LABELS = {"random": "Random", "scaffold": "Scaffold", "cluster": "Cluster"}
METHOD_LABELS = {
    "marginal_lac": "Marginal",
    "density_ratio_weighted_lac": "Shift-weighted",
    "mondrian_lac": "Mondrian",
    "marginal_absolute_residual": "Marginal",
    "density_ratio_weighted_absolute_residual": "Shift-weighted",
    "split_calibration_adaptive_normalized": "Adaptive",
}
COLORS = {
    "random": "#4C78A8",
    "scaffold": "#F58518",
    "cluster": "#54A24B",
    "Marginal": "#4C78A8",
    "Shift-weighted": "#F58518",
    "Mondrian": "#54A24B",
    "Adaptive": "#B279A2",
    "Positive": "#E45756",
    "Negative": "#72B7B2",
}

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def read_required(name: str) -> pd.DataFrame:
    path = TABLE_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise RuntimeError(f"Required table is empty: {path}")
    return frame


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{name} missing columns: {missing}")


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = [FIGURE_DIR / f"{stem}.pdf", FIGURE_DIR / f"{stem}.png"]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


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
    x = np.arange(len(endpoint_order), dtype=float)
    width = 0.23
    offsets = [-width, 0.0, width]
    for split, offset in zip(SPLITS, offsets):
        subset = frame[frame["split_type"] == split].set_index("endpoint")
        values = [float(subset.loc[endpoint, metric]) for endpoint in endpoint_order]
        ax.bar(x + offset, values, width=width, label=SPLIT_LABELS[split], color=COLORS[split])
    if baseline is not None:
        ax.axhline(baseline, linestyle="--", linewidth=1.0, color="black", alpha=0.65)
    labels = [endpoint.upper() if endpoint != "lipophilicity" else "Lipophilicity" for endpoint in endpoint_order]
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.22)


def figure_1_design() -> list[Path]:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.set_axis_off()
    boxes = [
        (0.03, 0.66, 0.16, 0.23, "Four ADMET endpoints\nBBBP · ClinTox\nESOL · Lipophilicity"),
        (0.23, 0.66, 0.17, 0.23, "Frozen molecular models\nECFP-based linear, RF,\nXGBoost and MLP/regressors"),
        (0.44, 0.66, 0.17, 0.23, "Label-blind splitting\nRandom · Scaffold · Cluster\nTrain / calibration / test"),
        (0.65, 0.66, 0.15, 0.23, "Confirmatory repeats\n10 random/scaffold seeds\n5 cluster seeds"),
        (0.84, 0.66, 0.13, 0.23, "Paired analysis\nSame endpoint, split,\nseed and model"),
        (0.08, 0.20, 0.17, 0.23, "RQ1\nPerformance and\nprobability calibration"),
        (0.29, 0.20, 0.17, 0.23, "RQ2\nApplicability-domain\nand threshold robustness"),
        (0.50, 0.20, 0.17, 0.23, "RQ3\nClass-conditional failure\nand selective retention"),
        (0.71, 0.20, 0.21, 0.23, "RQ4\nMarginal, Mondrian,\nshift-weighted and adaptive\nconformal trade-offs"),
    ]
    for x, y, w, h, text in boxes:
        patch = plt.Rectangle(
            (x, y), w, h, transform=ax.transAxes, facecolor="#F4F6F8",
            edgecolor="#44546A", linewidth=1.2
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, transform=ax.transAxes, ha="center", va="center", fontsize=10)
    for start, end in [
        ((0.19, 0.775), (0.23, 0.775)),
        ((0.40, 0.775), (0.44, 0.775)),
        ((0.61, 0.775), (0.65, 0.775)),
        ((0.80, 0.775), (0.84, 0.775)),
    ]:
        ax.annotate("", xy=end, xytext=start, xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", linewidth=1.3, color="#44546A"))
    for x in [0.165, 0.375, 0.585, 0.815]:
        ax.annotate("", xy=(x, 0.43), xytext=(0.74, 0.66), xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", linewidth=1.0, color="#7A7A7A"))
    ax.text(
        0.5, 0.06,
        "Frozen confirmatory analyses only; seed 99 was a technical smoke test and is excluded from scientific conclusions.",
        transform=ax.transAxes, ha="center", va="center", fontsize=10, fontstyle="italic",
    )
    ax.set_title("Confirmatory reliability-evaluation framework", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    return save_figure(fig, "figure_1_confirmatory_design")


def figure_2_performance() -> list[Path]:
    class_perf = read_required("table_rq1_classification_performance.csv").rename(columns={"split": "split_type"})
    reg_perf = read_required("table_rq1_regression_performance.csv").rename(columns={"split": "split_type"})
    calibration = read_required("table_rq1_classification_calibration.csv").rename(columns={"split": "split_type"})
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2))
    grouped_bars(axes[0, 0], class_perf, "roc_auc", ["bbbp", "clintox"], "ROC-AUC", "Classification discrimination", 0.5)
    grouped_bars(axes[0, 1], class_perf, "balanced_accuracy", ["bbbp", "clintox"], "Balanced accuracy", "Class-balanced accuracy", 0.5)
    grouped_bars(axes[0, 2], calibration, "ece_probability", ["bbbp", "clintox"], "ECE", "Probability calibration error")
    grouped_bars(axes[1, 0], reg_perf, "rmse", ["esol", "lipophilicity"], "RMSE", "Regression error")
    grouped_bars(axes[1, 1], reg_perf, "r2", ["esol", "lipophilicity"], r"$R^2$", "Explained variance", 0.0)
    grouped_bars(axes[1, 2], calibration, "negative_log_likelihood", ["bbbp", "clintox"], "NLL", "Negative log-likelihood")
    for label, ax in zip(list("ABCDEF"), axes.flat):
        panel_label(ax, label)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Predictive performance and calibration under chemical distribution shift", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    return save_figure(fig, "figure_2_performance_and_calibration")


def figure_3_classification_conformal() -> list[Path]:
    frame = read_required("table_rq3_rq4_classification_conformal.csv")
    require_columns(
        frame,
        ["endpoint", "split_type", "method", "positive_coverage", "negative_coverage", "ambiguous_set_rate"],
        "classification conformal table",
    )
    frame = frame.copy()
    frame["method_label"] = frame["method"].astype(str).map(METHOD_LABELS)
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.3), sharey="row")
    methods = ["Marginal", "Shift-weighted", "Mondrian"]
    x = np.arange(len(methods))
    width = 0.34
    for row, endpoint in enumerate(["bbbp", "clintox"]):
        for col, split in enumerate(SPLITS):
            ax = axes[row, col]
            sub = frame[(frame["endpoint"] == endpoint) & (frame["split_type"] == split)].set_index("method_label")
            pos = [float(sub.loc[m, "positive_coverage"]) for m in methods]
            neg = [float(sub.loc[m, "negative_coverage"]) for m in methods]
            ax.bar(x - width / 2, pos, width, label="Positive class", color=COLORS["Positive"])
            ax.bar(x + width / 2, neg, width, label="Negative class", color=COLORS["Negative"])
            ax.axhline(0.90, linestyle="--", color="black", linewidth=1.0, alpha=0.7)
            ax.set_xticks(x, methods, rotation=18, ha="right")
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", alpha=0.2)
            ax.set_title(f"{endpoint.upper()} · {SPLIT_LABELS[split]}")
            if col == 0:
                ax.set_ylabel("Empirical coverage")
            for i, method in enumerate(methods):
                ambiguity = float(sub.loc[method, "ambiguous_set_rate"])
                ax.text(i, 0.03, f"Ambig.\n{ambiguity:.2f}", ha="center", va="bottom", fontsize=8)
    for label, ax in zip(list("ABCDEF"), axes.flat):
        panel_label(ax, label)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Class-conditional coverage repair can sacrifice prediction-set informativeness", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.035, "Dashed line: nominal 90% coverage. Text reports ambiguous two-label prediction-set rate.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    return save_figure(fig, "figure_3_classification_conformal_tradeoff")


def heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    vmin: float,
    vmax: float,
) -> None:
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(col_labels)), col_labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(image, ax=ax, fraction=0.045, pad=0.03)


def figure_4_ad_and_selective() -> list[Path]:
    ad = read_required("table_rq2_ad_continuous.csv")
    selective = read_required("table_rq2_rq3_selective_prediction.csv")
    require_columns(ad, ["endpoint", "split_type", "mean_risk_similarity_spearman", "mean_miscoverage_similarity_spearman"], "AD table")
    require_columns(selective, ["endpoint", "split_type", "uncertainty_measure", "primary_paurc_improvement_vs_random", "balanced_paurc_improvement_vs_random", "positive_retention_at_05"], "selective table")
    row_order = ["BBBP", "ClinTox", "ESOL", "Lipophilicity"]
    endpoint_keys = ["bbbp", "clintox", "esol", "lipophilicity"]
    rho_risk = np.full((4, 3), np.nan)
    rho_mis = np.full((4, 3), np.nan)
    for i, endpoint in enumerate(endpoint_keys):
        for j, split in enumerate(SPLITS):
            sub = ad[(ad["endpoint"] == endpoint) & (ad["split_type"] == split)]
            if len(sub):
                rho_risk[i, j] = float(sub["mean_risk_similarity_spearman"].iloc[0])
                rho_mis[i, j] = float(sub["mean_miscoverage_similarity_spearman"].iloc[0])
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    heatmap(axes[0, 0], rho_risk, row_order, [SPLIT_LABELS[s] for s in SPLITS], "Similarity versus predictive risk (Spearman ρ)", -0.35, 0.35)
    heatmap(axes[0, 1], rho_mis, row_order, [SPLIT_LABELS[s] for s in SPLITS], "Similarity versus conformal miscoverage (Spearman ρ)", -0.35, 0.35)
    class_sel = selective[selective["endpoint"].isin(["bbbp", "clintox"])].copy()
    class_sel["measure"] = class_sel["uncertainty_measure"].map({"one_minus_max_probability": "Confidence", "one_minus_max_tanimoto_to_train": "Chemical similarity"})
    ax = axes[1, 0]
    positions = np.arange(6)
    labels = []
    primary_values = []
    balanced_values = []
    for endpoint in ["bbbp", "clintox"]:
        for split in SPLITS:
            sub = class_sel[(class_sel["endpoint"] == endpoint) & (class_sel["split_type"] == split) & (class_sel["measure"] == "Confidence")]
            labels.append(f"{endpoint.upper()}\n{SPLIT_LABELS[split]}")
            primary_values.append(float(sub["primary_paurc_improvement_vs_random"].iloc[0]))
            balanced_values.append(float(sub["balanced_paurc_improvement_vs_random"].iloc[0]))
    width = 0.36
    ax.bar(positions - width / 2, primary_values, width, label="Ordinary-risk pAURC gain", color=COLORS["random"])
    ax.bar(positions + width / 2, balanced_values, width, label="Balanced-risk pAURC gain", color=COLORS["scaffold"])
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Improvement versus random rejection")
    ax.set_title("Confidence-based selective prediction")
    ax.legend(frameon=False, loc="best")
    ax.grid(axis="y", alpha=0.2)
    ax = axes[1, 1]
    clintox = class_sel[class_sel["endpoint"] == "clintox"].copy()
    x = np.arange(len(SPLITS))
    conf = [float(clintox[(clintox["split_type"] == split) & (clintox["measure"] == "Confidence")]["positive_retention_at_05"].iloc[0]) for split in SPLITS]
    sim = [float(clintox[(clintox["split_type"] == split) & (clintox["measure"] == "Chemical similarity")]["positive_retention_at_05"].iloc[0]) for split in SPLITS]
    ax.bar(x - width / 2, conf, width, label="Confidence", color=COLORS["random"])
    ax.bar(x + width / 2, sim, width, label="Chemical similarity", color=COLORS["cluster"])
    ax.axhline(0.5, linestyle="--", color="black", linewidth=1.0, alpha=0.7)
    ax.set_xticks(x, [SPLIT_LABELS[s] for s in SPLITS])
    ax.set_ylim(0, 0.65)
    ax.set_ylabel("Positive-class retention at 50% coverage")
    ax.set_title("ClinTox minority-class retention")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    for label, axis in zip(list("ABCD"), axes.flat):
        panel_label(axis, label)
    fig.suptitle("Applicability-domain and selective-prediction diagnostics are endpoint dependent", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_figure(fig, "figure_4_ad_and_selective_diagnostics")


def figure_5_regression_conformal() -> list[Path]:
    frame = read_required("table_rq4_regression_conformal.csv")
    require_columns(frame, ["endpoint", "split_type", "method", "empirical_coverage", "mean_interval_width", "interval_width_cv", "width_error_spearman"], "regression conformal table")
    frame = frame.copy()
    frame["method_label"] = frame["method"].astype(str).map(METHOD_LABELS)
    methods = ["Marginal", "Shift-weighted", "Adaptive"]
    x = np.arange(len(methods))
    width = 0.22
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.5))
    metrics = [
        ("empirical_coverage", "Empirical coverage", "Coverage", 0.90),
        ("mean_interval_width", "Mean interval width", "Efficiency", None),
        ("interval_width_cv", "Interval-width coefficient of variation", "Interval heterogeneity", None),
        ("width_error_spearman", "Width–error Spearman ρ", "Error adaptivity", 0.0),
    ]
    for ax, (metric, ylabel, title, baseline) in zip(axes.flat, metrics):
        for endpoint_idx, endpoint in enumerate(["esol", "lipophilicity"]):
            sub_endpoint = frame[frame["endpoint"] == endpoint]
            for split_idx, split in enumerate(SPLITS):
                sub = sub_endpoint[sub_endpoint["split_type"] == split].set_index("method_label")
                values = [float(sub.loc[m, metric]) if pd.notna(sub.loc[m, metric]) else np.nan for m in methods]
                base = endpoint_idx * 4.0 + split_idx
                positions = base + x * width
                ax.bar(positions, values, width=width, color=[COLORS[m] for m in methods], alpha=0.95)
        if baseline is not None:
            ax.axhline(baseline, linestyle="--", color="black", linewidth=1.0, alpha=0.7)
        tick_positions = []
        tick_labels = []
        for endpoint_idx, endpoint in enumerate(["ESOL", "Lipophilicity"]):
            for split_idx, split in enumerate(SPLITS):
                tick_positions.append(endpoint_idx * 4.0 + split_idx + width)
                tick_labels.append(f"{endpoint}\n{SPLIT_LABELS[split]}")
        ax.set_xticks(tick_positions, tick_labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m]) for m in methods]
    fig.legend(handles, methods, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    for label, ax in zip(list("ABCD"), axes.flat):
        panel_label(ax, label)
    fig.suptitle("Regression conformal variants show no universal coverage–efficiency advantage", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    return save_figure(fig, "figure_5_regression_conformal_tradeoff")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    generated: list[Path] = []
    generated.extend(figure_1_design())
    generated.extend(figure_2_performance())
    generated.extend(figure_3_classification_conformal())
    generated.extend(figure_4_ad_and_selective())
    generated.extend(figure_5_regression_conformal())
    rows = []
    for path in sorted(generated):
        rows.append({"file": path.relative_to(PAPER_DIR).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = pd.DataFrame(rows)
    manifest_path = FIGURE_DIR / "main_figure_integrity_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print("saved", manifest_path)
    print(f"Main figure package complete. figure_stems=5, exported_files={len(generated)}, manifest_rows={len(manifest)}.")


if __name__ == "__main__":
    main()
