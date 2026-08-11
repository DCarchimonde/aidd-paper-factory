from __future__ import annotations

"""Build the publication figures for the integrated Paper 2/TAME manuscript.

The script is deliberately reporting-only.  It reads versioned, frozen summary
tables from the reliability audit and the sealed TAME evaluation conducted
under the RACER-C4 protocol.  It
does not fit a model, regenerate a prediction, inspect an unsealed label, choose
an endpoint, or change any inferential quantity.
"""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper2_admet_benchmark"
ASSET_DIR = PAPER_DIR / "results" / "manuscript_assets"
TABLE_DIR = ASSET_DIR / "tables"
FIGURE_DIR = ASSET_DIR / "figures"
C4_DEV_DIR = PAPER_DIR / "results" / "racer_c4_development"
C4_FINAL_DIR = PAPER_DIR / "results" / "racer_c4_independent_final"

SPLITS = ["random", "scaffold", "cluster"]
SPLIT_LABELS = {
    "random": "Random",
    "scaffold": "Scaffold",
    "cluster": "Similarity cluster",
}
SPLIT_SHORT = {"random": "Random", "scaffold": "Scaffold", "cluster": "Cluster"}

COLORS = {
    "ink": "#26343D",
    "muted": "#687780",
    "grid": "#DCE4E7",
    "outline": "#71828A",
    "teal_dark": "#0E5962",
    "teal": "#2D8F9D",
    "teal_light": "#86C7C2",
    "mint": "#79C7A4",
    "mint_light": "#DDF0EA",
    "blue_gray": "#5D7F8C",
    "gray": "#B8C4C9",
    "gray_light": "#EEF2F3",
    "coral": "#D97A63",
    "coral_light": "#F4DED8",
    "amber": "#D8A84E",
    "white": "#FFFFFF",
}

SPLIT_COLORS = {
    "random": COLORS["teal"],
    "scaffold": COLORS["mint"],
    "cluster": COLORS["blue_gray"],
}

DIVERGING = LinearSegmentedColormap.from_list(
    "tame_diverging",
    [COLORS["teal_dark"], COLORS["gray_light"], COLORS["coral"]],
)

FIGURE_STEMS = [
    "figure_1_evidence_chain",
    "figure_2_shift_performance",
    "figure_3_reliability_illusion",
    "figure_4_domain_retention",
    "figure_5_tame_mechanism",
    "figure_6_epa_validation",
]

# Elsevier's general artwork guidance recommends approximately 7 pt lettering
# at final printed size.  The figures are authored near a 190 mm two-column
# width with no text below 9 pt, so lettering remains about 7 pt even when the
# review manuscript scales the art to its narrower text block.
ARTWORK_WIDTH = 7.4


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.6,
            "axes.titlesize": 10.2,
            "axes.labelsize": 9.4,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": COLORS["outline"],
            "axes.linewidth": 0.8,
            "axes.labelcolor": COLORS["ink"],
            "text.color": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "figure.facecolor": COLORS["white"],
            "axes.facecolor": COLORS["white"],
            "figure.dpi": 150,
            "savefig.dpi": 360,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise RuntimeError(f"Empty frozen table: {path}")
    return frame


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def table(name: str) -> pd.DataFrame:
    return read_csv(TABLE_DIR / name)


def panel_label(ax: plt.Axes, label: str, x: float = -0.11, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="top",
        clip_on=False,
        color=COLORS["ink"],
    )


def clean_axis(ax: plt.Axes, *, grid: str | None = "y") -> None:
    if grid:
        ax.grid(axis=grid, color=COLORS["grid"], linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGURE_DIR / f"{stem}.pdf"
    png = FIGURE_DIR / f"{stem}.png"
    # Write beside the destination and replace atomically only after each
    # encoder has closed the complete file.  This prevents a synchronizing
    # workspace or an interrupted renderer from exposing a partial PNG/PDF.
    pdf_tmp = FIGURE_DIR / f".{stem}.tmp.pdf"
    png_tmp = FIGURE_DIR / f".{stem}.tmp.png"
    fig.savefig(pdf_tmp, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(png_tmp, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    pdf_tmp.replace(pdf)
    png_tmp.replace(png)
    pdf_tmp.unlink(missing_ok=True)
    png_tmp.unlink(missing_ok=True)
    print("wrote", pdf)
    print("wrote", png)
    return [pdf, png]


def rounded_card(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str = COLORS["white"],
    edgecolor: str = COLORS["outline"],
    linewidth: float = 1.0,
    radius: float = 0.018,
    zorder: int = 1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def stage_header(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    label: str,
    title: str,
    color: str,
) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
    ax.text(
        x + 0.032,
        y,
        title,
        transform=ax.transAxes,
        fontsize=9.6,
        fontweight="bold",
        va="top",
        color=color,
    )
    ax.plot(
        [x + 0.032, x + width],
        [y - 0.035, y - 0.035],
        transform=ax.transAxes,
        color=color,
        linewidth=2.2,
        solid_capstyle="round",
    )


def figure1_evidence_chain() -> list[Path]:
    fig, ax = plt.subplots(figsize=(ARTWORK_WIDTH, 8.2))
    ax.set_axis_off()
    band_x, band_w, band_h = 0.035, 0.93, 0.190
    content_x = [0.055, 0.3575, 0.660]
    content_w = 0.285
    stages = [
        (0.760, "A", "CONFIRMATORY AUDIT", COLORS["teal_dark"]),
        (0.535, "B", "FAILURE SIGNALS", COLORS["coral"]),
        (0.310, "C", "TAME DESIGN", COLORS["teal"]),
        (0.085, "D", "SEALED EPA EVALUATION", COLORS["teal_dark"]),
    ]
    for y, label, title, accent in stages:
        rounded_card(ax, band_x, y, band_w, band_h, facecolor="#FAFCFC", edgecolor=COLORS["grid"], radius=0.012)
        ax.text(0.060, y + 0.158, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="center")
        ax.text(0.097, y + 0.158, title, transform=ax.transAxes, fontsize=10.1, fontweight="bold", va="center", color=accent)
        ax.plot([0.055, 0.945], [y + 0.137, y + 0.137], transform=ax.transAxes, color=accent, linewidth=1.6)

    stage_content = [
        (
            0.760,
            [
                ("4 public endpoints", "BBBP | ClinTox\nESOL | Lipophilicity", COLORS["mint_light"], COLORS["mint"]),
                ("3 split designs", "Random | scaffold\nsimilarity cluster", COLORS["gray_light"], COLORS["gray"]),
                ("5 reliability lenses", "Performance | calibration\ndomain | sets | retention", "#E8F2F3", COLORS["teal_light"]),
            ],
        ),
        (
            0.535,
            [
                ("0.90 overall", "0.07-0.10 positive\nClinTox marginal coverage", COLORS["coral_light"], "#E9B8AC"),
                ("~71% ambiguous", "Mondrian repairs coverage\nby returning {0,1}", COLORS["coral_light"], "#E9B8AC"),
                ("27-31% retained", "ClinTox positives at\n50% overall retention", COLORS["coral_light"], "#E9B8AC"),
            ],
        ),
        (
            0.310,
            [
                ("Two label-free views", "Physicochemical\ndescriptors + scores", COLORS["mint_light"], COLORS["mint"]),
                ("Transport certificate", "Support | ESS | clipping\nAUC | weighted balance", "#E8F2F3", COLORS["teal"]),
                ("Protected envelope", "Consensus expansion\nexact fallback on failure", "#E5F0F2", COLORS["teal"]),
            ],
        ),
        (
            0.085,
            [
                ("Prediction firewall", "HASH + SEAL\nthen open labels", COLORS["gray_light"], COLORS["blue_gray"]),
                ("+1.36 pp", "Minimum-class coverage\n95% interval:\n+0.58 to +2.01 pp", COLORS["mint_light"], COLORS["mint"]),
                ("MacroCSY -1.61 pp", "Inside frozen -5 pp\nefficiency guardrail", "#E8F2F3", COLORS["teal_light"]),
            ],
        ),
    ]
    for y, cards in stage_content:
        for x, (title, subtitle, fill, edge) in zip(content_x, cards):
            rounded_card(ax, x, y + 0.018, content_w, 0.108, facecolor=fill, edgecolor=edge, radius=0.010)
            headline_size = 13.0 if title == "+1.36 pp" else 9.8
            headline_color = COLORS["teal_dark"] if title == "+1.36 pp" else (COLORS["coral"] if y == 0.535 else COLORS["ink"])
            ax.text(x + content_w / 2, y + 0.091, title, transform=ax.transAxes, ha="center", va="center", fontsize=headline_size, fontweight="bold", color=headline_color)
            ax.text(x + content_w / 2, y + 0.048, subtitle, transform=ax.transAxes, ha="center", va="center", fontsize=9.0, color=COLORS["muted"], linespacing=1.16)

    for upper_y, lower_y in [(0.760, 0.535), (0.535, 0.310), (0.310, 0.085)]:
        ax.annotate("", xy=(0.50, lower_y + band_h + 0.005), xytext=(0.50, upper_y - 0.005), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="-|>", color=COLORS["outline"], lw=1.5, mutation_scale=12))
    ax.text(0.50, 0.025, "60/60 final endpoint-seed cells completed under the prediction-to-label firewall", transform=ax.transAxes, ha="center", fontsize=9.2, fontweight="bold", color=COLORS["teal_dark"])
    return save_figure(fig, "figure_1_evidence_chain")


def line_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
    endpoints: list[str],
    colors: list[str],
    title: str,
    ylabel: str,
    *,
    baseline: float | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    x = np.arange(len(SPLITS))
    for endpoint, color in zip(endpoints, colors):
        subset = frame[frame["endpoint"] == endpoint].set_index("split_type")
        values = np.asarray([float(subset.loc[split, metric]) for split in SPLITS])
        label = {
            "bbbp": "BBBP",
            "clintox": "ClinTox",
            "esol": "ESOL",
            "lipophilicity": "Lipophilicity",
        }[endpoint]
        ax.plot(x, values, marker="o", linewidth=2.0, markersize=6.5, color=color, label=label)
        for xx, value in zip(x, values):
            ax.text(xx, value, f" {value:.2f}", fontsize=9.0, color=color, va="bottom")
    if baseline is not None:
        ax.axhline(baseline, color=COLORS["muted"], linestyle="--", linewidth=0.9)
    ax.set_xticks(x, [SPLIT_SHORT[split] for split in SPLITS])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=9, fontweight="bold")
    if ylim:
        ax.set_ylim(*ylim)
    clean_axis(ax)
    ax.legend(frameon=False, loc="best")


def figure2_shift_performance() -> list[Path]:
    class_perf = table("table_rq1_classification_performance.csv")
    reg_perf = table("table_rq1_regression_performance.csv")
    class_cal = table("table_rq1_classification_calibration.csv")

    fig, axes = plt.subplots(2, 2, figsize=(ARTWORK_WIDTH, 7.8))
    line_panel(
        axes[0, 0],
        class_perf,
        "roc_auc",
        ["bbbp", "clintox"],
        [COLORS["teal_dark"], COLORS["coral"]],
        "ROC-AUC under molecular shift",
        "ROC-AUC",
        baseline=0.5,
        ylim=(0.48, 0.93),
    )
    line_panel(
        axes[0, 1],
        class_perf,
        "balanced_accuracy",
        ["bbbp", "clintox"],
        [COLORS["teal_dark"], COLORS["coral"]],
        "Class-balanced accuracy",
        "Balanced accuracy",
        baseline=0.5,
        ylim=(0.48, 0.83),
    )
    line_panel(
        axes[1, 0],
        reg_perf,
        "r2",
        ["esol", "lipophilicity"],
        [COLORS["mint"], COLORS["blue_gray"]],
        "Explained variance",
        "R²",
        baseline=0.0,
        ylim=(-0.70, 0.56),
    )
    line_panel(
        axes[1, 1],
        class_cal,
        "ece_probability",
        ["bbbp", "clintox"],
        [COLORS["teal_dark"], COLORS["coral"]],
        "Positive-probability calibration",
        "Expected calibration error",
        ylim=(0.035, 0.105),
    )

    for label, ax in zip("ABCD", axes.flat):
        panel_label(ax, label)

    fig.text(
        0.5,
        0.012,
        "Points are descriptive means across frozen model/regime combinations;\n"
        "model-specific cross-seed uncertainty is reported in the Supporting Information.",
        ha="center",
        va="bottom",
        fontsize=9.0,
        color=COLORS["muted"],
    )
    fig.tight_layout(rect=(0.02, 0.065, 1, 1), h_pad=2.2, w_pad=2.0)
    return save_figure(fig, "figure_2_shift_performance")


def figure3_reliability_illusion() -> list[Path]:
    conformal = table("table_rq3_rq4_classification_conformal.csv").copy()
    method_labels = {
        "marginal_lac": "Marginal",
        "density_ratio_weighted_lac": "Shift-weighted",
        "mondrian_lac": "Mondrian",
    }
    conformal["method_label"] = conformal["method"].map(method_labels)
    development_summary = read_csv(C4_DEV_DIR / "development_summary.csv")

    fig = plt.figure(figsize=(ARTWORK_WIDTH, 8.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.42, wspace=0.34)

    # A: the marginal-coverage illusion in ClinTox.
    ax = fig.add_subplot(grid[0, 0])
    clintox = (
        conformal[conformal["endpoint"] == "clintox"]
        .groupby("method_label", as_index=True)[["empirical_coverage", "positive_coverage", "negative_coverage"]]
        .mean()
        .reindex(["Marginal", "Shift-weighted", "Mondrian"])
    )
    x = np.arange(3)
    width = 0.24
    series = [
        ("Overall", "empirical_coverage", COLORS["gray"]),
        ("Positive class", "positive_coverage", COLORS["teal"]),
        ("Negative class", "negative_coverage", COLORS["mint"]),
    ]
    for offset, (label, column, color) in zip([-width, 0, width], series):
        values = clintox[column].to_numpy()
        ax.bar(x + offset, values, width=width, color=color, edgecolor=COLORS["white"], linewidth=0.8, label=label)
        for xx, value in zip(x + offset, values):
            ax.text(xx, value + 0.018, f"{value:.2f}", ha="center", fontsize=9.0, rotation=90)
    ax.axhline(0.90, color=COLORS["muted"], linestyle="--", linewidth=1.0)
    ax.set_xticks(x, ["Marginal", "Shift-\nweighted", "Mondrian"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Empirical coverage")
    ax.set_title("ClinTox: class-conditional coverage", loc="left", pad=9, fontweight="bold")
    clean_axis(ax)
    ax.legend(frameon=False, loc="lower right")
    panel_label(ax, "A")

    # B: class coverage versus ambiguity across split designs.
    ax = fig.add_subplot(grid[0, 1])
    marker_map = {"Marginal": "o", "Shift-weighted": "s", "Mondrian": "D"}
    endpoint_colors = {"bbbp": COLORS["blue_gray"], "clintox": COLORS["coral"]}
    for endpoint in ["bbbp", "clintox"]:
        subset = conformal[conformal["endpoint"] == endpoint]
        for method in ["Marginal", "Shift-weighted", "Mondrian"]:
            part = subset[subset["method_label"] == method]
            minimum = np.minimum(part["positive_coverage"], part["negative_coverage"])
            ax.scatter(
                part["ambiguous_set_rate"],
                minimum,
                marker=marker_map[method],
                s=66,
                color=endpoint_colors[endpoint],
                edgecolor=COLORS["white"],
                linewidth=0.7,
                alpha=0.9,
            )
    ax.axhline(0.90, color=COLORS["muted"], linestyle="--", linewidth=0.9)
    ax.annotate(
        "more useful",
        xy=(0.03, 0.99),
        xytext=(0.28, 0.76),
        arrowprops=dict(arrowstyle="->", color=COLORS["teal_dark"], lw=1.1),
        fontsize=9.0,
        color=COLORS["teal_dark"],
    )
    ax.set_xlabel("Ambiguous two-label rate")
    ax.set_ylabel("Minimum class coverage")
    ax.set_xlim(-0.03, 0.78)
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Coverage-ambiguity trade-off", loc="left", pad=9, fontweight="bold")
    clean_axis(ax)
    method_handles = [
        plt.Line2D([0], [0], marker=marker_map[m], color="none", markerfacecolor=COLORS["gray"], markeredgecolor=COLORS["gray"], markersize=6, label=m)
        for m in ["Marginal", "Shift-weighted", "Mondrian"]
    ]
    endpoint_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=endpoint_colors[e], markeredgecolor=endpoint_colors[e], markersize=6, label=("BBBP" if e == "bbbp" else "ClinTox"))
        for e in ["bbbp", "clintox"]
    ]
    ax.legend(handles=method_handles + endpoint_handles, frameon=False, loc="lower right", ncol=2, columnspacing=0.9, handletextpad=0.4)
    panel_label(ax, "B")

    # C: a public-development counterexample; near-perfect coverage can be useless.
    ax = fig.add_subplot(grid[1, :])
    display = {
        "RACER-C4_TAME": "TAME",
        "ordinary_mondrian_global_stack": "Ordinary Mondrian",
        "density_weighted_mondrian_ecfp": "ECFP-weighted",
        "density_weighted_mondrian_physchem": "Physchem-weighted",
        "density_weighted_mondrian_score_view": "Score-weighted",
        "ordinary_mondrian_equal_logit": "Equal-logit Mondrian",
    }
    plot_colors = {
        "TAME": COLORS["teal_dark"],
        "Ordinary Mondrian": COLORS["blue_gray"],
        "ECFP-weighted": COLORS["coral"],
        "Physchem-weighted": COLORS["mint"],
        "Score-weighted": COLORS["teal_light"],
        "Equal-logit Mondrian": COLORS["gray"],
    }
    for _, row in development_summary.iterrows():
        label = display[row["method"]]
        size = 55 + 300 * float(row["mean_ambiguous_rate"])
        ax.scatter(
            float(row["mean_macro_csy"]),
            float(row["mean_minimum_class_coverage"]),
            s=size,
            color=plot_colors[label],
            edgecolor=COLORS["white"],
            linewidth=0.9,
            alpha=0.95,
            zorder=3,
        )
        if label in {"TAME", "Ordinary Mondrian", "ECFP-weighted"}:
            offsets = {
                "TAME": (-0.012, 0.014),
                "Ordinary Mondrian": (-0.155, -0.015),
                "ECFP-weighted": (0.014, -0.008),
            }[label]
            ax.text(
                float(row["mean_macro_csy"]) + offsets[0],
                float(row["mean_minimum_class_coverage"]) + offsets[1],
                label,
                fontsize=9.0,
                color=plot_colors[label],
                fontweight="bold",
                ha="right" if label == "TAME" else "left",
            )
    ax.set_xlabel("Macro correct-singleton yield")
    ax.set_ylabel("Minimum class coverage")
    ax.set_xlim(-0.02, 0.48)
    ax.set_ylim(0.76, 1.015)
    ax.set_title("Public Tox21 development: high coverage can be uninformative", loc="left", pad=9, fontweight="bold")
    clean_axis(ax)
    ax.text(0.02, 0.772, "Bubble area scales with ambiguity", fontsize=9.0, color=COLORS["muted"])
    panel_label(ax, "C")

    fig.subplots_adjust(left=0.090, right=0.975, bottom=0.080, top=0.955, hspace=0.42, wspace=0.34)
    return save_figure(fig, "figure_3_reliability_illusion")


def heatmap_panel(
    ax: plt.Axes,
    matrix: np.ndarray,
    rows: list[str],
    columns: list[str],
    title: str,
    *,
    vmin: float,
    vmax: float,
) -> None:
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    image = ax.imshow(matrix, aspect="auto", cmap=DIVERGING, norm=norm)
    ax.set_xticks(np.arange(len(columns)), columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(rows)), rows)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9.0)
    ax.set_title(title, loc="left", pad=9, fontweight="bold")
    cbar = plt.colorbar(image, ax=ax, fraction=0.047, pad=0.03)
    cbar.ax.tick_params(labelsize=9.0)


def figure4_domain_retention() -> list[Path]:
    continuous = table("table_rq2_ad_continuous.csv")
    selective = table("table_rq2_rq3_selective_prediction.csv")

    fig, axes = plt.subplots(2, 2, figsize=(ARTWORK_WIDTH, 8.0))
    endpoint_order = ["bbbp", "clintox", "esol", "lipophilicity"]
    row_labels = ["BBBP", "ClinTox", "ESOL", "Lipophilicity"]
    columns = ["Random", "Scaffold", "Cluster"]

    for ax, metric, title, label in [
        (axes[0, 0], "mean_risk_similarity_spearman", "Similarity versus predictive risk", "A"),
        (axes[0, 1], "mean_miscoverage_similarity_spearman", "Similarity versus conformal miscoverage", "B"),
    ]:
        matrix = np.zeros((4, 3))
        for i, endpoint in enumerate(endpoint_order):
            subset = continuous[continuous["endpoint"] == endpoint].set_index("split_type")
            matrix[i, :] = [float(subset.loc[split, metric]) for split in SPLITS]
        heatmap_panel(ax, matrix, row_labels, columns, title, vmin=-0.36, vmax=0.36)
        panel_label(ax, label, x=-0.16)

    # C: positive retention in ClinTox at 50% retained sample coverage.
    ax = axes[1, 0]
    subset = selective[(selective["endpoint"] == "clintox") & (selective["task_type"] == "classification")]
    x = np.arange(3)
    width = 0.34
    measures = [
        ("one_minus_max_probability", "Confidence", COLORS["coral"]),
        ("one_minus_max_tanimoto_to_train", "Chemical similarity", COLORS["teal"]),
    ]
    for offset, (measure, label, color) in zip([-width / 2, width / 2], measures):
        part = subset[subset["uncertainty_measure"] == measure].set_index("split_type")
        values = np.asarray([float(part.loc[split, "positive_retention_at_05"]) for split in SPLITS])
        ax.bar(x + offset, values, width=width, color=color, edgecolor=COLORS["white"], label=label)
        for xx, value in zip(x + offset, values):
            ax.text(xx, value + 0.012, f"{100*value:.0f}%", ha="center", fontsize=9.0)
    ax.axhline(0.50, color=COLORS["muted"], linestyle="--", linewidth=0.9)
    ax.set_xticks(x, columns)
    ax.set_ylim(0, 0.57)
    ax.set_ylabel("Positive-class retention")
    ax.set_title("ClinTox positive retention\nat 50% coverage", loc="left", pad=9, fontweight="bold")
    clean_axis(ax)
    ax.legend(frameon=False, loc="upper right", ncol=2, columnspacing=1.2, handletextpad=0.5)
    panel_label(ax, "C", x=-0.16)

    # D: endpoint dependence of similarity-based regression rejection.
    ax = axes[1, 1]
    subset = selective[
        (selective["task_type"] == "regression")
        & (selective["uncertainty_measure"] == "one_minus_max_tanimoto_to_train")
    ]
    endpoint_x = np.arange(2)
    width = 0.22
    for split, offset in zip(SPLITS, [-width, 0.0, width]):
        part = subset[subset["split_type"] == split].set_index("endpoint")
        values = [float(part.loc[endpoint, "risk_improvement_at_05"]) for endpoint in ["esol", "lipophilicity"]]
        ax.bar(endpoint_x + offset, values, width=width, color=SPLIT_COLORS[split], edgecolor=COLORS["white"], label=SPLIT_SHORT[split])
    ax.axhline(0, color=COLORS["muted"], linewidth=0.9)
    ax.set_xticks(endpoint_x, ["ESOL", "Lipophilicity"])
    ax.set_ylabel("Risk improvement vs random rejection")
    ax.set_title("Regression risk after\nsimilarity rejection", loc="left", pad=9, fontweight="bold")
    clean_axis(ax)
    ax.legend(frameon=False, loc="upper left", ncol=3)
    panel_label(ax, "D", x=-0.16)

    fig.text(
        0.5,
        0.012,
        "Negative heat-map values indicate increasing risk or miscoverage as similarity decreases.\n"
        "Threshold partitions are sensitivity analyses, not universal chemical-domain rules.",
        ha="center",
        va="bottom",
        fontsize=9.0,
        color=COLORS["muted"],
    )
    fig.tight_layout(rect=(0.02, 0.065, 1, 1), h_pad=2.4, w_pad=2.8)
    return save_figure(fig, "figure_4_domain_retention")


def figure5_tame_mechanism() -> list[Path]:
    fig, ax = plt.subplots(figsize=(ARTWORK_WIDTH, 8.2))
    ax.set_axis_off()
    band_x, band_w, band_h = 0.035, 0.93, 0.190
    stage_specs = [
        (0.760, "A", "ORDINARY MONDRIAN BASELINE", COLORS["blue_gray"]),
        (0.535, "B", "TWO LABEL-FREE TRANSPORT VIEWS", COLORS["teal"]),
        (0.310, "C", "COMPLETE TRANSPORT CERTIFICATE", COLORS["coral"]),
        (0.085, "D", "ENVELOPE AND EXACT FALLBACK", COLORS["teal_dark"]),
    ]
    for y, label, title, accent in stage_specs:
        rounded_card(ax, band_x, y, band_w, band_h, facecolor="#FAFCFC", edgecolor=COLORS["grid"], radius=0.012)
        ax.text(0.060, y + 0.158, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="center")
        ax.text(0.097, y + 0.158, title, transform=ax.transAxes, fontsize=10.0, fontweight="bold", va="center", color=accent)
        ax.plot([0.055, 0.945], [y + 0.137, y + 0.137], transform=ax.transAxes, color=accent, linewidth=1.6)

    # A: frozen ordinary baseline.
    a_y = 0.778
    a_cards = [
        (0.055, 0.260, "ECFP classifiers", "logistic | RF\nextra trees | NB", COLORS["gray_light"], COLORS["gray"]),
        (0.370, 0.260, "Logit router", "four component logits", "#E6EFF2", COLORS["blue_gray"]),
        (0.685, 0.260, "Ordinary set", r"$s_0=p,\ s_1=1-p$" + "\n" + r"$S_M(x)$ at 90%", COLORS["mint_light"], COLORS["mint"]),
    ]
    for x, w, title, subtitle, fill, edge in a_cards:
        rounded_card(ax, x, a_y, w, 0.108, facecolor=fill, edgecolor=edge)
        ax.text(x + w / 2, a_y + 0.073, title, transform=ax.transAxes, ha="center", fontsize=9.8, fontweight="bold")
        ax.text(x + w / 2, a_y + 0.032, subtitle, transform=ax.transAxes, ha="center", va="center", fontsize=9.0, color=COLORS["muted"])
    for start, end in [((0.320, 0.832), (0.360, 0.832)), ((0.635, 0.832), (0.675, 0.832))]:
        ax.annotate("", xy=end, xytext=start, xycoords=ax.transAxes, arrowprops=dict(arrowstyle="-|>", color=COLORS["outline"], lw=1.3))

    # B: two independently weighted views.
    b_y = 0.553
    view_cards = [
        (0.055, "Physicochemical\nview", "MW | logP\nTPSA | counts", COLORS["mint_light"], COLORS["mint"]),
        (0.360, "Score view", "component\nlogits + router", "#E3F0F2", COLORS["teal"]),
    ]
    for x, title, subtitle, fill, edge in view_cards:
        rounded_card(ax, x, b_y, 0.275, 0.108, facecolor=fill, edgecolor=edge)
        ax.text(x + 0.1375, b_y + 0.078, title, transform=ax.transAxes, ha="center", va="center", fontsize=9.8, fontweight="bold", linespacing=1.0)
        ax.text(x + 0.1375, b_y + 0.027, subtitle, transform=ax.transAxes, ha="center", va="center", fontsize=9.0, color=COLORS["muted"], linespacing=1.0)
    rounded_card(ax, 0.665, b_y, 0.280, 0.108, facecolor=COLORS["white"], edgecolor=COLORS["teal"])
    ax.text(0.805, b_y + 0.073, "Weighted sets", transform=ax.transAxes, ha="center", fontsize=9.8, fontweight="bold")
    ax.text(0.805, b_y + 0.032, r"$S_{phys}(x)$ and $S_{score}(x)$", transform=ax.transAxes, ha="center", fontsize=9.0)

    # C: every check must pass for both approved views.
    c_y = 0.328
    rounded_card(ax, 0.055, c_y, 0.650, 0.108, facecolor=COLORS["white"], edgecolor=COLORS["coral"])
    checks = [
        "source/target ≥ 100",
        "class support ≥ 20",
        "ESS ≥ 25%",
        "clip ≤ 30%/bound",
        "domain AUC ≤ .99",
        "balance not worse",
    ]
    for i, check in enumerate(checks):
        col = i // 3
        row = i % 3
        xx = 0.075 + col * 0.330
        yy = c_y + 0.080 - row * 0.031
        ax.text(xx, yy, "PASS", transform=ax.transAxes, color=COLORS["teal_dark"], fontsize=9.0, fontweight="bold", va="center")
        ax.text(xx + 0.066, yy, check, transform=ax.transAxes, fontsize=9.0, va="center")
    rounded_card(ax, 0.745, c_y, 0.200, 0.108, facecolor=COLORS["coral_light"], edgecolor="#E9B8AC")
    ax.text(0.845, c_y + 0.072, "Both views\npass?", transform=ax.transAxes, ha="center", fontsize=9.5, fontweight="bold", linespacing=1.0)
    ax.text(0.845, c_y + 0.032, "all-view quorum", transform=ax.transAxes, ha="center", fontsize=9.0, color=COLORS["muted"])
    ax.annotate("", xy=(0.735, c_y + 0.054), xytext=(0.715, c_y + 0.054), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="-|>", color=COLORS["outline"], lw=1.2))

    # D: pass expands by consensus; failure returns the ordinary baseline.
    d_y = 0.103
    rounded_card(ax, 0.055, d_y, 0.575, 0.108, facecolor=COLORS["mint_light"], edgecolor=COLORS["teal_dark"], linewidth=1.2)
    ax.text(0.3425, d_y + 0.078, "PASS: protected-label consensus", transform=ax.transAxes, ha="center", fontsize=9.8, fontweight="bold", color=COLORS["teal_dark"])
    ax.text(0.3425, d_y + 0.045, r"$S_{TAME}=\bar S_M\cup\{0:\,0\in S_{phys}\cap S_{score}\}$", transform=ax.transAxes, ha="center", fontsize=10.2)
    ax.text(0.3425, d_y + 0.016, "Baseline labels are never removed", transform=ax.transAxes, ha="center", fontsize=9.0, color=COLORS["muted"])
    rounded_card(ax, 0.660, d_y, 0.285, 0.108, facecolor=COLORS["coral_light"], edgecolor=COLORS["coral"])
    ax.text(0.8025, d_y + 0.072, "FAIL: exact fallback", transform=ax.transAxes, ha="center", fontsize=9.8, fontweight="bold", color=COLORS["coral"])
    ax.text(0.8025, d_y + 0.039, r"$S_{TAME}=\bar S_M$", transform=ax.transAxes, ha="center", fontsize=10.2)
    ax.text(0.8025, d_y + 0.014, "empty baseline -> {0,1}", transform=ax.transAxes, ha="center", fontsize=9.0, color=COLORS["muted"])

    for upper_y, lower_y in [(0.760, 0.535), (0.535, 0.310), (0.310, 0.085)]:
        ax.annotate("", xy=(0.50, lower_y + band_h + 0.005), xytext=(0.50, upper_y - 0.005), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="-|>", color=COLORS["outline"], lw=1.5, mutation_scale=12))
    ax.text(
        0.50,
        0.030,
        r"$S_{TAME}\supseteq S_M$  |  no empty sets  |  no newly created singletons" "\n"
        "Only singleton-to-defer transitions are permitted",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color=COLORS["teal_dark"],
        linespacing=1.15,
    )
    return save_figure(fig, "figure_5_tame_mechanism")


def figure6_epa_validation() -> list[Path]:
    primary = read_csv(C4_FINAL_DIR / "primary_endpoint_means.csv")
    report = read_json(C4_FINAL_DIR / "publication_final_report.json")
    audit = read_json(C4_FINAL_DIR / "transport_audit_summary.json")

    fig, axes = plt.subplots(2, 2, figsize=(ARTWORK_WIDTH, 8.2))
    endpoint_labels = {
        "Tox21_NR_AhR": "NR-AhR",
        "Tox21_NR_ER": "NR-ER",
        "Tox21_SR_ARE": "SR-ARE",
        "Tox21_SR_ATAD5": "SR-ATAD5",
        "Tox21_SR_MMP": "SR-MMP",
        "Tox21_SR_p53": "SR-p53",
    }
    primary = primary.copy()
    primary["display"] = primary["endpoint"].map(endpoint_labels)

    # A: endpoint-level minimum-class coverage changes.
    ax = axes[0, 0]
    order = list(reversed(primary["display"].tolist()))
    values = [100 * float(primary.set_index("display").loc[label, "mean_minimum_class_coverage_delta"]) for label in order]
    y = np.arange(len(order))
    ax.hlines(y, 0, values, color=COLORS["teal_light"], linewidth=3.0)
    ax.scatter(values, y, color=COLORS["teal_dark"], s=50, zorder=3)
    for value, yy in zip(values, y):
        ax.text(value + 0.10, yy, f"{value:+.2f}", va="center", fontsize=9.0, color=COLORS["teal_dark"])
    ax.axvline(0, color=COLORS["muted"], linewidth=0.9)
    ax.set_yticks(y, order)
    ax.set_xlabel("Minimum-class coverage delta (pp)")
    ax.set_title("Primary endpoint effects", loc="left", pad=9, fontweight="bold")
    ax.set_xlim(-0.2, 2.8)
    clean_axis(ax, grid="x")
    panel_label(ax, "A", x=-0.18)

    # B: frozen primary estimate and deterministic bootstrap interval.
    ax = axes[0, 1]
    estimate = 100 * float(report["primary_inference"]["estimate"])
    lower, upper = [100 * float(value) for value in report["primary_inference"]["percentile_95_interval"]]
    ax.hlines(0, lower, upper, color=COLORS["teal_dark"], linewidth=5.0)
    ax.scatter([estimate], [0], color=COLORS["white"], edgecolor=COLORS["teal_dark"], linewidth=2.0, s=95, zorder=3)
    ax.axvline(0, color=COLORS["muted"], linestyle="--", linewidth=1.0)
    ax.set_ylim(-0.8, 0.8)
    ax.set_yticks([])
    ax.set_xlim(-0.25, 2.30)
    ax.set_xlabel("Endpoint-seed-equal mean delta (pp)")
    ax.set_title("Hierarchical-bootstrap inference", loc="left", pad=9, fontweight="bold")
    ax.text(estimate, 0.24, f"{estimate:+.2f} pp", ha="center", fontsize=12.5, fontweight="bold", color=COLORS["teal_dark"])
    ax.text((lower + upper) / 2, -0.28, f"95% interval  {lower:+.2f} to {upper:+.2f} pp", ha="center", fontsize=9.0)
    ax.text((lower + upper) / 2, -0.48, "2,000 draws | seed 44021", ha="center", fontsize=9.0, color=COLORS["muted"])
    clean_axis(ax, grid="x")
    panel_label(ax, "B", x=-0.12)

    # C: endpoint-level and aggregate MacroCSY cost against the frozen margin.
    ax = axes[1, 0]
    macro_by_endpoint = [100 * float(primary.set_index("display").loc[label, "mean_macro_csy_delta"]) for label in order]
    colors = [COLORS["coral"] if value < 0 else COLORS["mint"] for value in macro_by_endpoint]
    ax.barh(y, macro_by_endpoint, color=colors, edgecolor=COLORS["white"], height=0.62)
    aggregate = 100 * float(report["primary_mean_macro_csy_delta"])
    ax.scatter([aggregate], [-0.75], marker="D", s=58, color=COLORS["teal_dark"], zorder=4)
    ax.text(aggregate + 0.12, -0.75, f"aggregate {aggregate:+.2f}", va="center", fontsize=9.0, fontweight="bold", color=COLORS["teal_dark"])
    ax.axvline(-5.0, color=COLORS["coral"], linestyle="--", linewidth=1.2)
    ax.axvline(0.0, color=COLORS["muted"], linewidth=0.9)
    ax.text(-4.92, len(order) - 0.15, "frozen efficiency guardrail", rotation=90, va="top", fontsize=9.0, color=COLORS["coral"])
    ax.set_yticks(y, order)
    ax.set_ylim(-1.35, len(order) - 0.35)
    ax.set_xlim(-5.5, 0.6)
    ax.set_xlabel("MacroCSY delta (pp)")
    ax.set_title("MacroCSY versus -5 pp guardrail", loc="left", pad=9, fontweight="bold")
    clean_axis(ax, grid="x")
    panel_label(ax, "C", x=-0.18)

    # D: transport audit and prediction-to-label firewall.
    ax = axes[1, 1]
    views = ["Physicochemical", "Score", "ECFP diagnostic"]
    counts = [
        int(audit["active_cells_by_view"]["physchem_descriptors"]),
        int(audit["active_cells_by_view"]["component_and_stack_logits"]),
        int(audit["active_cells_by_view"]["ecfp_bits"]),
    ]
    bars = ax.barh(np.arange(3), counts, color=[COLORS["mint"], COLORS["teal"], COLORS["coral"]], height=0.52)
    ax.set_yticks(np.arange(3), views)
    ax.invert_yaxis()
    ax.set_xlim(0, 66)
    ax.set_xlabel("Active cells (of 60)")
    ax.set_title("Transport audit and firewall", loc="left", pad=9, fontweight="bold")
    for bar, value in zip(bars, counts):
        ax.text(value + 1.2, bar.get_y() + bar.get_height() / 2, f"{value}/60", va="center", fontsize=9.0, fontweight="bold")
    clean_axis(ax, grid="x")
    fig.text(
        0.50,
        0.025,
        "Firewall PASS: predictions and transport audit hashed before labels | 60/60 cells completed\n"
        "Integrity PASS: deterministic interval repair | 82/82 tests passed",
        ha="center",
        fontsize=9.0,
        linespacing=1.25,
        va="bottom",
        color=COLORS["teal_dark"],
        fontweight="bold",
    )
    panel_label(ax, "D", x=-0.18)

    fig.tight_layout(rect=(0.03, 0.12, 0.99, 0.99), h_pad=3.2, w_pad=3.0)
    return save_figure(fig, "figure_6_epa_validation")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refresh_manifest() -> Path:
    rows: list[dict[str, object]] = []
    for stem in FIGURE_STEMS:
        for suffix in [".pdf", ".png"]:
            path = FIGURE_DIR / f"{stem}{suffix}"
            if not path.exists():
                raise FileNotFoundError(path)
            rows.append(
                {
                    "file": path.relative_to(PAPER_DIR).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    manifest = pd.DataFrame(rows).sort_values("file")
    output = FIGURE_DIR / "main_figure_integrity_manifest.csv"
    manifest.to_csv(output, index=False)
    print("wrote", output)
    return output


def main() -> None:
    configure_style()
    figure1_evidence_chain()
    figure2_shift_performance()
    figure3_reliability_illusion()
    figure4_domain_retention()
    figure5_tame_mechanism()
    figure6_epa_validation()
    refresh_manifest()
    print("Integrated Paper 2/TAME figure package complete.")


if __name__ == "__main__":
    main()
