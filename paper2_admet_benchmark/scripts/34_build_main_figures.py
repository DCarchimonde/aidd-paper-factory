from __future__ import annotations

"""
Paper 2 — upgraded publication-style main figures.

This version is intentionally more visually structured than the first draft:
- 6 main figure stems instead of 5
- teal / mint / slate palette inspired by the user's reference style
- clearer panel hierarchy (A, B, C, ...)
- more separation between AD diagnostics and selective prediction
- no re-training or re-analysis; reads only frozen manuscript tables

Run:
    python paper2_admet_benchmark/scripts/34_build_main_figures.py
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
SPLIT_LABELS = {
    "random": "Random",
    "scaffold": "Scaffold",
    "cluster": "Cluster",
}

# user-liked style: teal / mint / slate / light gray
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
    "dark_text": "#1F2937",
    "light_fill": "#EAF4F1",
    "outline": "#5B6B73",
}

METHOD_LABELS = {
    "marginal_lac": "Marginal",
    "density_ratio_weighted_lac": "Shift-weighted",
    "mondrian_lac": "Mondrian",
    "marginal_absolute_residual": "Marginal",
    "density_ratio_weighted_absolute_residual": "Shift-weighted",
    "split_calibration_adaptive_normalized": "Adaptive",
}

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


def require_columns(frame: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(f"{name} missing columns: {missing}")


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
    )


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [FIGURE_DIR / f"{stem}.pdf", FIGURE_DIR / f"{stem}.png"]
    for path in outputs:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return outputs


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    offsets = [-width, 0, width]
    for split, offset in zip(SPLITS, offsets):
        sub = frame[frame["split_type"] == split].set_index("endpoint")
        vals = [float(sub.loc[e, metric]) for e in endpoint_order]
        ax.bar(
            x + offset,
            vals,
            width=width,
            color=PALETTE[split],
            edgecolor="white",
            linewidth=0.8,
            label=SPLIT_LABELS[split],
        )
    if baseline is not None:
        ax.axhline(baseline, linestyle="--", linewidth=1.0, color="#6B7280", alpha=0.8)
    labels = [e.upper() if e != "lipophilicity" else "Lipophilicity" for e in endpoint_order]
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.18)


def make_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    vmin: float,
    vmax: float,
    cmap: str = "BuGn",
    annotate: bool = True,
) -> None:
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(col_labels)), col_labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_title(title)
    if annotate:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(image, ax=ax, fraction=0.045, pad=0.03)


def fig1_design() -> list[Path]:
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    ax.set_axis_off()

    boxes = [
        (0.03, 0.70, 0.18, 0.18, "ADMET endpoints\nBBBP · ClinTox\nESOL · Lipophilicity"),
        (0.25, 0.70, 0.16, 0.18, "Frozen models\nLinear · RF\nXGBoost · MLP"),
        (0.45, 0.70, 0.18, 0.18, "Label-blind splits\nRandom · Scaffold · Cluster\nTrain / calibration / test"),
        (0.67, 0.70, 0.14, 0.18, "Confirmatory repeats\n10 random/scaffold\n5 cluster"),
        (0.84, 0.70, 0.13, 0.18, "Paired contrasts\nSame endpoint /\nmodel / seed"),
        (0.06, 0.24, 0.16, 0.20, "RQ1\nPerformance\nCalibration"),
        (0.27, 0.24, 0.16, 0.20, "RQ2\nApplicability domain\nThreshold robustness"),
        (0.48, 0.24, 0.16, 0.20, "RQ3\nClass-conditional\nfailure modes"),
        (0.69, 0.24, 0.22, 0.20, "RQ4\nMarginal vs Mondrian vs\nshift-weighted vs adaptive\nreliability trade-offs"),
    ]
    for x, y, w, h, text in boxes:
        rect = plt.Rectangle(
            (x, y), w, h,
            transform=ax.transAxes,
            facecolor=PALETTE["light_fill"],
            edgecolor=PALETTE["outline"],
            linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, transform=ax.transAxes, ha="center", va="center")

    arrows = [
        ((0.21, 0.79), (0.25, 0.79)),
        ((0.41, 0.79), (0.45, 0.79)),
        ((0.63, 0.79), (0.67, 0.79)),
        ((0.81, 0.79), (0.84, 0.79)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", lw=1.3, color=PALETTE["outline"]))

    for x in [0.14, 0.35, 0.56, 0.80]:
        ax.annotate("", xy=(x, 0.44), xytext=(0.74, 0.70), xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="#7A8B93"))

    ax.set_title("Confirmatory reliability-evaluation workflow", fontsize=14, fontweight="bold", pad=12)
    ax.text(
        0.5,
        0.08,
        "Scientific conclusions use only frozen confirmatory outputs; smoke-test runs are excluded from evidence.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9.5,
        style="italic",
    )
    fig.tight_layout()
    return save_figure(fig, "figure_1_confirmatory_design")


def fig2_performance_calibration() -> list[Path]:
    class_perf = read_required("table_rq1_classification_performance.csv").rename(columns={"split": "split_type"})
    reg_perf = read_required("table_rq1_regression_performance.csv").rename(columns={"split": "split_type"})
    class_cal = read_required("table_rq1_classification_calibration.csv").rename(columns={"split": "split_type"})

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.4))
    grouped_bars(axes[0, 0], class_perf, "roc_auc", ["bbbp", "clintox"], "ROC-AUC", "Classification discrimination", 0.5)
    grouped_bars(axes[0, 1], class_perf, "balanced_accuracy", ["bbbp", "clintox"], "Balanced accuracy", "Minority-sensitive accuracy", 0.5)
    grouped_bars(axes[0, 2], class_cal, "ece_probability", ["bbbp", "clintox"], "ECE", "Probability calibration error")
    grouped_bars(axes[1, 0], reg_perf, "rmse", ["esol", "lipophilicity"], "RMSE", "Regression error")
    grouped_bars(axes[1, 1], reg_perf, "r2", ["esol", "lipophilicity"], r"$R^2$", "Explained variance", 0.0)
    grouped_bars(axes[1, 2], class_cal, "negative_log_likelihood", ["bbbp", "clintox"], "NLL", "Negative log-likelihood")

    for lbl, ax in zip(list("ABCDEF"), axes.flat):
        panel_label(ax, lbl)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Predictive performance and probability calibration under chemical shift", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    return save_figure(fig, "figure_2_performance_and_calibration")


def fig3_classification_conformal() -> list[Path]:
    frame = read_required("table_rq3_rq4_classification_conformal.csv")
    require_columns(
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
    frame["method_label"] = frame["method"].map(METHOD_LABELS)

    methods = ["Marginal", "Shift-weighted", "Mondrian"]
    x = np.arange(len(methods))
    width = 0.34
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.6), sharey="row")

    for row, endpoint in enumerate(["bbbp", "clintox"]):
        for col, split in enumerate(SPLITS):
            ax = axes[row, col]
            sub = frame[(frame["endpoint"] == endpoint) & (frame["split_type"] == split)].set_index("method_label")
            pos = [float(sub.loc[m, "positive_coverage"]) for m in methods]
            neg = [float(sub.loc[m, "negative_coverage"]) for m in methods]

            ax.bar(x - width / 2, pos, width=width, color=PALETTE["positive"], edgecolor="white", label="Positive")
            ax.bar(x + width / 2, neg, width=width, color=PALETTE["negative"], edgecolor="white", label="Negative")
            ax.axhline(0.90, color="#6B7280", linestyle="--", linewidth=1.0)
            ax.set_xticks(x, methods, rotation=15, ha="right")
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", alpha=0.18)
            ax.set_title(f"{endpoint.upper()} · {SPLIT_LABELS[split]}")
            if col == 0:
                ax.set_ylabel("Empirical coverage")

            for i, method in enumerate(methods):
                size = float(sub.loc[method, "mean_prediction_set_size"])
                ambig = float(sub.loc[method, "ambiguous_set_rate"])
                ax.text(i, 0.05, f"set={size:.2f}\namb={ambig:.2f}", ha="center", va="bottom", fontsize=8)

    for lbl, ax in zip(list("ABCDEF"), axes.flat):
        panel_label(ax, lbl)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Mondrian repairs class-conditional coverage at the cost of larger and more ambiguous sets",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    return save_figure(fig, "figure_3_classification_conformal_tradeoff")


def fig4_ad_diagnostics() -> list[Path]:
    cont = read_required("table_rq2_ad_continuous.csv")
    thr = read_required("table_rq2_ad_threshold_sensitivity.csv")
    require_columns(
        cont,
        ["endpoint", "split_type", "mean_risk_similarity_spearman", "mean_miscoverage_similarity_spearman"],
        "AD continuous table",
    )
    require_columns(
        thr,
        ["endpoint", "split_type", "threshold", "mean_delta_risk_low_minus_high", "mean_delta_miscoverage_low_minus_high"],
        "AD threshold table",
    )

    endpoint_order = ["bbbp", "clintox", "esol", "lipophilicity"]
    endpoint_labels = ["BBBP", "ClinTox", "ESOL", "Lipophilicity"]

    risk_mat = np.full((4, 3), np.nan)
    mis_mat = np.full((4, 3), np.nan)
    for i, endpoint in enumerate(endpoint_order):
        for j, split in enumerate(SPLITS):
            sub = cont[(cont["endpoint"] == endpoint) & (cont["split_type"] == split)]
            if not sub.empty:
                risk_mat[i, j] = float(sub["mean_risk_similarity_spearman"].iloc[0])
                mis_mat[i, j] = float(sub["mean_miscoverage_similarity_spearman"].iloc[0])

    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.6))
    make_heatmap(
        axes[0, 0], risk_mat, endpoint_labels, [SPLIT_LABELS[s] for s in SPLITS],
        "Similarity vs predictive risk (Spearman ρ)", -0.35, 0.35, cmap="BuGn_r"
    )
    make_heatmap(
        axes[0, 1], mis_mat, endpoint_labels, [SPLIT_LABELS[s] for s in SPLITS],
        "Similarity vs conformal miscoverage (Spearman ρ)", -0.35, 0.35, cmap="BuGn_r"
    )

    ax = axes[0, 2]
    for endpoint, style in [("bbbp", "-"), ("lipophilicity", "--")]:
        for split in ["random", "scaffold", "cluster"]:
            sub = thr[(thr["endpoint"] == endpoint) & (thr["split_type"] == split)].sort_values("threshold")
            ax.plot(
                sub["threshold"],
                sub["mean_delta_risk_low_minus_high"],
                linestyle=style,
                linewidth=2.0,
                color=PALETTE[split],
                alpha=0.9,
                label=f"{endpoint.upper()} · {SPLIT_LABELS[split]}" if endpoint == "bbbp" else None,
            )
    ax.axhline(0, color="#6B7280", linewidth=1.0)
    ax.set_title("Threshold sensitivity: low-domain minus high-domain risk")
    ax.set_xlabel("Tanimoto threshold")
    ax.set_ylabel("Δ risk (low − high)")
    ax.grid(alpha=0.18)

    for ax, endpoint, title in [
        (axes[1, 0], "bbbp", "BBBP threshold effect"),
        (axes[1, 1], "clintox", "ClinTox threshold effect"),
    ]:
        subset = thr[thr["endpoint"] == endpoint].copy()
        thresholds = sorted(subset["threshold"].dropna().unique())
        mat = np.full((len(SPLITS), len(thresholds)), np.nan)
        for i, split in enumerate(SPLITS):
            for j, t in enumerate(thresholds):
                s = subset[(subset["split_type"] == split) & (subset["threshold"] == t)]
                if not s.empty:
                    mat[i, j] = float(s["mean_delta_risk_low_minus_high"].iloc[0])
        make_heatmap(
            ax,
            mat,
            [SPLIT_LABELS[s] for s in SPLITS],
            [f"{t:.1f}" for t in thresholds],
            title,
            vmin=-0.25,
            vmax=0.55,
            cmap="BuGn",
        )
        ax.set_xlabel("Threshold")

    ax = axes[1, 2]
    endpoint = "lipophilicity"
    for split in SPLITS:
        sub = thr[(thr["endpoint"] == endpoint) & (thr["split_type"] == split)].sort_values("threshold")
        ax.plot(
            sub["threshold"],
            sub["mean_delta_miscoverage_low_minus_high"],
            marker="o",
            linewidth=2.0,
            color=PALETTE[split],
            label=SPLIT_LABELS[split],
        )
    ax.axhline(0, color="#6B7280", linewidth=1.0)
    ax.set_title("Lipophilicity: miscoverage shift across thresholds")
    ax.set_xlabel("Tanimoto threshold")
    ax.set_ylabel("Δ miscoverage (low − high)")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="best")

    for lbl, ax in zip(list("ABCDEF"), axes.flat):
        panel_label(ax, lbl)

    fig.suptitle("Applicability-domain diagnostics are endpoint- and split-dependent", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_figure(fig, "figure_4_applicability_domain_diagnostics")


def fig5_selective_prediction() -> list[Path]:
    frame = read_required("table_rq2_rq3_selective_prediction.csv")
    require_columns(
        frame,
        [
            "endpoint",
            "split_type",
            "uncertainty_measure",
            "primary_paurc_improvement_vs_random",
            "balanced_paurc_improvement_vs_random",
            "risk_improvement_at_05",
        ],
        "selective prediction table",
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.4))
    frame = frame.copy()
    frame["measure_label"] = frame["uncertainty_measure"].map(
        {
            "one_minus_max_probability": "Confidence",
            "one_minus_max_tanimoto_to_train": "Chemical similarity",
        }
    )

    class_df = frame[frame["endpoint"].isin(["bbbp", "clintox"])].copy()
    pos = np.arange(6)
    width = 0.35
    primary_vals = []
    balanced_vals = []
    labels = []
    for endpoint in ["bbbp", "clintox"]:
        for split in SPLITS:
            sub = class_df[
                (class_df["endpoint"] == endpoint)
                & (class_df["split_type"] == split)
                & (class_df["measure_label"] == "Confidence")
            ]
            labels.append(f"{endpoint.upper()}\n{SPLIT_LABELS[split]}")
            primary_vals.append(float(sub["primary_paurc_improvement_vs_random"].iloc[0]))
            balanced_vals.append(float(sub["balanced_paurc_improvement_vs_random"].iloc[0]))

    ax = axes[0, 0]
    ax.bar(pos - width / 2, primary_vals, width, color=PALETTE["random"], label="Ordinary-risk gain")
    ax.bar(pos + width / 2, balanced_vals, width, color=PALETTE["scaffold"], label="Balanced-risk gain")
    ax.axhline(0, color="#6B7280", linewidth=1.0)
    ax.set_xticks(pos, labels)
    ax.set_ylabel("pAURC improvement vs random rejection")
    ax.set_title("Confidence-based selective prediction")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    width = 0.36
    x = np.arange(len(SPLITS))
    conf = []
    sim = []
    for split in SPLITS:
        conf_sub = class_df[
            (class_df["endpoint"] == "clintox")
            & (class_df["split_type"] == split)
            & (class_df["measure_label"] == "Confidence")
        ]
        sim_sub = class_df[
            (class_df["endpoint"] == "clintox")
            & (class_df["split_type"] == split)
            & (class_df["measure_label"] == "Chemical similarity")
        ]
        conf.append(float(conf_sub["positive_retention_at_05"].iloc[0]))
        sim.append(float(sim_sub["positive_retention_at_05"].iloc[0]))
    ax.bar(x - width / 2, conf, width, color=PALETTE["random"], label="Confidence")
    ax.bar(x + width / 2, sim, width, color=PALETTE["cluster"], label="Chemical similarity")
    ax.axhline(0.5, color="#6B7280", linestyle="--", linewidth=1.0)
    ax.set_xticks(x, [SPLIT_LABELS[s] for s in SPLITS])
    ax.set_ylabel("Positive retention at 50% coverage")
    ax.set_ylim(0, 0.65)
    ax.set_title("ClinTox minority retention")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False)

    reg_df = frame[frame["endpoint"].isin(["esol", "lipophilicity"])].copy()
    ax = axes[1, 0]
    vals = []
    labels = []
    for endpoint in ["esol", "lipophilicity"]:
        for split in SPLITS:
            sub = reg_df[
                (reg_df["endpoint"] == endpoint)
                & (reg_df["split_type"] == split)
                & (reg_df["measure_label"] == "Chemical similarity")
            ]
            labels.append(f"{endpoint.upper()}\n{SPLIT_LABELS[split]}")
            vals.append(float(sub["risk_improvement_at_05"].iloc[0]))
    ax.bar(np.arange(len(vals)), vals, color=PALETTE["cluster"], edgecolor="white")
    ax.axhline(0, color="#6B7280", linewidth=1.0)
    ax.set_xticks(np.arange(len(vals)), labels)
    ax.set_ylabel("Risk improvement at 50% retained coverage")
    ax.set_title("Similarity-based selective prediction for regression")
    ax.grid(axis="y", alpha=0.18)

    ax = axes[1, 1]
    x = np.arange(len(SPLITS))
    esol = []
    lipo = []
    for split in SPLITS:
        esol_sub = reg_df[
            (reg_df["endpoint"] == "esol")
            & (reg_df["split_type"] == split)
            & (reg_df["measure_label"] == "Chemical similarity")
        ]
        lipo_sub = reg_df[
            (reg_df["endpoint"] == "lipophilicity")
            & (reg_df["split_type"] == split)
            & (reg_df["measure_label"] == "Chemical similarity")
        ]
        esol.append(float(esol_sub["primary_paurc_improvement_vs_random"].iloc[0]))
        lipo.append(float(lipo_sub["primary_paurc_improvement_vs_random"].iloc[0]))
    ax.plot(x, esol, marker="o", linewidth=2.0, color=PALETTE["random"], label="ESOL")
    ax.plot(x, lipo, marker="o", linewidth=2.0, color=PALETTE["cluster"], label="Lipophilicity")
    ax.axhline(0, color="#6B7280", linewidth=1.0)
    ax.set_xticks(x, [SPLIT_LABELS[s] for s in SPLITS])
    ax.set_ylabel("Primary pAURC gain")
    ax.set_title("Endpoint dependence of similarity-based selective prediction")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False)

    for lbl, ax in zip(list("ABCD"), axes.flat):
        panel_label(ax, lbl)

    fig.suptitle("Selective prediction can reduce ordinary risk while erasing the minority class", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_figure(fig, "figure_5_selective_prediction_diagnostics")


def fig6_regression_conformal() -> list[Path]:
    frame = read_required("table_rq4_regression_conformal.csv")
    require_columns(
        frame,
        [
            "endpoint",
            "split_type",
            "method",
            "empirical_coverage",
            "mean_interval_width",
            "interval_width_cv",
            "width_error_spearman",
        ],
        "regression conformal table",
    )
    frame = frame.copy()
    frame["method_label"] = frame["method"].map(METHOD_LABELS)

    methods = ["Marginal", "Shift-weighted", "Adaptive"]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.8))
    metrics = [
        ("empirical_coverage", "Empirical coverage", "Coverage", 0.90),
        ("mean_interval_width", "Mean interval width", "Efficiency", None),
        ("interval_width_cv", "Interval-width coefficient of variation", "Heterogeneity", None),
        ("width_error_spearman", "Width–error Spearman ρ", "Error adaptivity", 0.0),
    ]

    for ax, (metric, ylabel, title, baseline) in zip(axes.flat, metrics):
        positions = []
        labels = []
        values = []
        colors = []
        base = 0
        for endpoint in ["esol", "lipophilicity"]:
            for split in SPLITS:
                sub = frame[(frame["endpoint"] == endpoint) & (frame["split_type"] == split)].set_index("method_label")
                for method in methods:
                    positions.append(base)
                    labels.append(f"{endpoint.upper()}\n{SPLIT_LABELS[split]}\n{method}")
                    values.append(float(sub.loc[method, metric]) if pd.notna(sub.loc[method, metric]) else np.nan)
                    if method == "Marginal":
                        colors.append(PALETTE["marginal"])
                    elif method == "Shift-weighted":
                        colors.append(PALETTE["shift_weighted"])
                    else:
                        colors.append(PALETTE["adaptive"])
                    base += 1
                base += 0.5
        ax.bar(positions, values, color=colors, edgecolor="white", linewidth=0.8)
        if baseline is not None:
            ax.axhline(baseline, color="#6B7280", linestyle="--", linewidth=1.0)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.18)
        tick_positions = [np.mean(positions[i:i+3]) for i in range(0, len(positions), 3)]
        tick_labels = []
        for endpoint in ["ESOL", "Lipophilicity"]:
            for split in SPLITS:
                tick_labels.append(f"{endpoint}\n{SPLIT_LABELS[split]}")
        ax.set_xticks(tick_positions, tick_labels, rotation=0)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["marginal"]),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["shift_weighted"]),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["adaptive"]),
    ]
    fig.legend(legend_handles, methods, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))

    for lbl, ax in zip(list("ABCD"), axes.flat):
        panel_label(ax, lbl)

    fig.suptitle("Regression conformal methods show no universal coverage–efficiency advantage", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    return save_figure(fig, "figure_6_regression_conformal_tradeoff")


def main() -> None:
    generated = []
    generated.extend(fig1_design())
    generated.extend(fig2_performance_calibration())
    generated.extend(fig3_classification_conformal())
    generated.extend(fig4_ad_diagnostics())
    generated.extend(fig5_selective_prediction())
    generated.extend(fig6_regression_conformal())

    manifest_rows = []
    for path in sorted(generated):
        manifest_rows.append(
            {
                "file": path.relative_to(PAPER_DIR).as_posix(),
                "bytes": path.stat().st_size if path.exists() else None,
                "sha256": sha256(path) if path.exists() else None,
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = FIGURE_DIR / "main_figure_integrity_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print("saved", manifest_path)
    print(f"Main figure package complete. figure_stems=6, exported_files={len(generated)}, manifest_rows={len(manifest)}.")


if __name__ == "__main__":
    main()
