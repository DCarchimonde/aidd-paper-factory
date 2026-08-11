from __future__ import annotations

"""Build the publication figures for the integrated Paper 2/TAME manuscript.

The script is deliberately reporting-only.  It reads versioned, frozen summary
tables from the reliability audit and the sealed RACER-C4/TAME evaluation.  It
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


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.8,
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
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
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
    ax.text(x, y, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
    ax.text(
        x + 0.032,
        y,
        title,
        transform=ax.transAxes,
        fontsize=10.5,
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
    fig, ax = plt.subplots(figsize=(15.0, 6.4))
    ax.set_axis_off()

    xs = [0.018, 0.266, 0.514, 0.762]
    card_w = 0.218
    card_y = 0.16
    card_h = 0.73
    header_y = 0.84

    for x in xs:
        rounded_card(
            ax,
            x,
            card_y,
            card_w,
            card_h,
            facecolor="#FAFCFC",
            edgecolor=COLORS["grid"],
            linewidth=1.0,
            radius=0.016,
        )

    stage_header(ax, xs[0] + 0.012, header_y, card_w - 0.024, "A", "CONFIRMATORY AUDIT", COLORS["teal_dark"])
    stage_header(ax, xs[1] + 0.012, header_y, card_w - 0.024, "B", "FAILURE SIGNALS", COLORS["coral"])
    stage_header(ax, xs[2] + 0.012, header_y, card_w - 0.024, "C", "TAME DESIGN", COLORS["teal"])
    stage_header(ax, xs[3] + 0.012, header_y, card_w - 0.024, "D", "SEALED EPA TEST", COLORS["teal_dark"])

    # Stage A: audit ingredients.
    x = xs[0]
    for yy, title, subtitle, fill in [
        (0.675, "4 public endpoints", "BBBP | ClinTox | ESOL | Lipophilicity", COLORS["mint_light"]),
        (0.545, "3 label-blind split designs", "Random | scaffold | similarity cluster", COLORS["gray_light"]),
        (0.365, "5 reliability lenses", "Performance | calibration | domain\nconformal sets | selective retention", "#E8F2F3"),
    ]:
        rounded_card(ax, x + 0.022, yy, card_w - 0.044, 0.10 if yy != 0.365 else 0.145, facecolor=fill, edgecolor=fill)
        ax.text(x + card_w / 2, yy + (0.067 if yy != 0.365 else 0.100), title, transform=ax.transAxes, ha="center", va="center", fontsize=9.4, fontweight="bold")
        ax.text(x + card_w / 2, yy + (0.035 if yy != 0.365 else 0.052), subtitle, transform=ax.transAxes, ha="center", va="center", fontsize=7.8, color=COLORS["muted"], linespacing=1.2)
    ax.text(x + card_w / 2, 0.245, "Paired, frozen, repeated-seed evidence", transform=ax.transAxes, ha="center", fontsize=8.5, color=COLORS["teal_dark"], fontweight="bold")

    # Stage B: the three empirical contradictions that motivate the method.
    x = xs[1]
    signal_specs = [
        (0.635, "0.90 overall", "0.07-0.10 positive\nClinTox marginal coverage"),
        (0.455, "~71% ambiguous", "Mondrian repairs coverage\nby returning {0,1}"),
        (0.275, "27-31% retained", "ClinTox positives when\n50% of samples remain"),
    ]
    for yy, headline, subtitle in signal_specs:
        rounded_card(ax, x + 0.026, yy, card_w - 0.052, 0.135, facecolor=COLORS["coral_light"], edgecolor="#E9B8AC", linewidth=0.9)
        ax.text(x + card_w / 2, yy + 0.088, headline, transform=ax.transAxes, ha="center", va="center", fontsize=12.5, fontweight="bold", color=COLORS["coral"])
        ax.text(x + card_w / 2, yy + 0.040, subtitle, transform=ax.transAxes, ha="center", va="center", fontsize=7.9, color=COLORS["ink"], linespacing=1.15)

    # Stage C: method components.
    x = xs[2]
    rounded_card(ax, x + 0.024, 0.664, card_w - 0.048, 0.105, facecolor=COLORS["mint_light"], edgecolor=COLORS["mint"])
    ax.text(x + card_w / 2, 0.731, "Two label-free transport views", transform=ax.transAxes, ha="center", fontsize=9.3, fontweight="bold")
    ax.text(x + card_w / 2, 0.692, "Physicochemical descriptors  +  model scores", transform=ax.transAxes, ha="center", fontsize=7.7, color=COLORS["muted"])
    rounded_card(ax, x + 0.050, 0.515, card_w - 0.100, 0.090, facecolor="#E8F2F3", edgecolor=COLORS["teal"])
    ax.text(x + card_w / 2, 0.560, "Transport audit", transform=ax.transAxes, ha="center", va="center", fontsize=9.5, fontweight="bold", color=COLORS["teal_dark"])
    ax.annotate("", xy=(x + card_w / 2, 0.615), xytext=(x + card_w / 2, 0.655), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="-|>", color=COLORS["outline"], lw=1.1))
    rounded_card(ax, x + 0.024, 0.350, card_w - 0.048, 0.105, facecolor="#E5F0F2", edgecolor=COLORS["teal"])
    ax.text(x + card_w / 2, 0.416, "Baseline-containing consensus", transform=ax.transAxes, ha="center", fontsize=9.2, fontweight="bold")
    ax.text(x + card_w / 2, 0.378, "Expand uncertainty; never invent confidence", transform=ax.transAxes, ha="center", fontsize=7.7, color=COLORS["muted"])
    ax.annotate("", xy=(x + card_w / 2, 0.465), xytext=(x + card_w / 2, 0.505), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="-|>", color=COLORS["outline"], lw=1.1))
    ax.text(x + card_w / 2, 0.255, "Audit failure: ordinary baseline fallback", transform=ax.transAxes, ha="center", fontsize=8.2, color=COLORS["coral"], fontweight="bold")

    # Stage D: firewall and final result.
    x = xs[3]
    ax.text(x + card_w / 2, 0.725, "predictions", transform=ax.transAxes, ha="center", fontsize=8.0, color=COLORS["muted"])
    ax.text(x + card_w / 2, 0.674, "HASH + SEAL", transform=ax.transAxes, ha="center", fontsize=11.0, fontweight="bold", color=COLORS["teal_dark"])
    ax.text(x + card_w / 2, 0.623, "then open labels", transform=ax.transAxes, ha="center", fontsize=8.0, color=COLORS["muted"])
    ax.annotate("", xy=(x + card_w / 2, 0.635), xytext=(x + card_w / 2, 0.655), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="-|>", color=COLORS["outline"], lw=1.0))
    rounded_card(ax, x + 0.026, 0.405, card_w - 0.052, 0.155, facecolor=COLORS["mint_light"], edgecolor=COLORS["mint"])
    ax.text(x + card_w / 2, 0.505, "+1.36 pp", transform=ax.transAxes, ha="center", fontsize=18, fontweight="bold", color=COLORS["teal_dark"])
    ax.text(x + card_w / 2, 0.455, "minimum-class coverage", transform=ax.transAxes, ha="center", fontsize=8.8, fontweight="bold")
    ax.text(x + card_w / 2, 0.425, "95% CI +0.58 to +2.01 pp", transform=ax.transAxes, ha="center", fontsize=7.8, color=COLORS["muted"])
    ax.text(x + card_w / 2, 0.332, "MacroCSY  -1.61 pp", transform=ax.transAxes, ha="center", fontsize=9.2, fontweight="bold")
    ax.text(x + card_w / 2, 0.292, "within the frozen -5 pp non-inferiority bound", transform=ax.transAxes, ha="center", fontsize=7.6, color=COLORS["muted"])
    ax.text(x + card_w / 2, 0.225, "60/60 endpoint-seed cells complete", transform=ax.transAxes, ha="center", fontsize=8.3, color=COLORS["teal_dark"], fontweight="bold")

    # Arrows between the four stages.
    for left_x, right_x in zip(xs[:-1], xs[1:]):
        ax.annotate(
            "",
            xy=(right_x - 0.007, 0.52),
            xytext=(left_x + card_w + 0.007, 0.52),
            xycoords=ax.transAxes,
            arrowprops=dict(arrowstyle="-|>", color=COLORS["outline"], lw=1.5, mutation_scale=12),
        )

    ax.text(
        0.5,
        0.065,
        "DIAGNOSIS   >   DESIGN CONSTRAINTS   >   AUDITABLE METHOD   >   INDEPENDENT EVIDENCE",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=COLORS["teal_dark"],
    )
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
            ax.text(xx, value, f" {value:.2f}", fontsize=7.4, color=color, va="bottom")
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

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.4))
    line_panel(
        axes[0, 0],
        class_perf,
        "roc_auc",
        ["bbbp", "clintox"],
        [COLORS["teal_dark"], COLORS["coral"]],
        "Discrimination under chemical shift",
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
        r"$R^2$",
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
        "Points are descriptive means across frozen model/regime combinations; model-specific cross-seed uncertainty is reported in the Supporting Information.",
        ha="center",
        fontsize=8.2,
        color=COLORS["muted"],
    )
    fig.tight_layout(rect=(0.02, 0.04, 1, 1), h_pad=2.2, w_pad=2.0)
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

    fig = plt.figure(figsize=(14.4, 5.9))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.08, 1.05, 1.22], wspace=0.30)

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
            ax.text(xx, value + 0.018, f"{value:.2f}", ha="center", fontsize=7.2, rotation=90)
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
        fontsize=8.2,
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
    ax.legend(handles=method_handles + endpoint_handles, frameon=False, loc="lower right", ncol=1)
    panel_label(ax, "B")

    # C: a public-development counterexample; near-perfect coverage can be useless.
    ax = fig.add_subplot(grid[0, 2])
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
                "TAME": (0.010, 0.003),
                "Ordinary Mondrian": (-0.155, -0.015),
                "ECFP-weighted": (0.014, -0.008),
            }[label]
            ax.text(float(row["mean_macro_csy"]) + offsets[0], float(row["mean_minimum_class_coverage"]) + offsets[1], label, fontsize=8.0, color=plot_colors[label], fontweight="bold")
    ax.set_xlabel("Macro correct-singleton yield")
    ax.set_ylabel("Minimum class coverage")
    ax.set_xlim(-0.02, 0.48)
    ax.set_ylim(0.76, 1.015)
    ax.set_title("Public Tox21 development: high coverage can be uninformative", loc="left", pad=9, fontweight="bold")
    clean_axis(ax)
    ax.text(0.02, 0.772, "Bubble area scales with ambiguity", fontsize=8.0, color=COLORS["muted"])
    panel_label(ax, "C")

    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.14, top=0.90, wspace=0.30)
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
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8.3)
    ax.set_title(title, loc="left", pad=9, fontweight="bold")
    cbar = plt.colorbar(image, ax=ax, fraction=0.047, pad=0.03)
    cbar.ax.tick_params(labelsize=7.5)


def figure4_domain_retention() -> list[Path]:
    continuous = table("table_rq2_ad_continuous.csv")
    selective = table("table_rq2_rq3_selective_prediction.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.6))
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
            ax.text(xx, value + 0.012, f"{100*value:.0f}%", ha="center", fontsize=7.7)
    ax.axhline(0.50, color=COLORS["muted"], linestyle="--", linewidth=0.9)
    ax.set_xticks(x, columns)
    ax.set_ylim(0, 0.57)
    ax.set_ylabel("Positive-class retention")
    ax.set_title("ClinTox: who remains after 50% rejection?", loc="left", pad=9, fontweight="bold")
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
    ax.set_title("Similarity rejection is endpoint dependent", loc="left", pad=9, fontweight="bold")
    clean_axis(ax)
    ax.legend(frameon=False, loc="upper left", ncol=3)
    panel_label(ax, "D", x=-0.16)

    fig.text(
        0.5,
        0.012,
        "Negative heat-map values indicate increasing risk or miscoverage as similarity decreases. Threshold partitions are sensitivity analyses, not universal chemical-domain rules.",
        ha="center",
        fontsize=8.1,
        color=COLORS["muted"],
    )
    fig.tight_layout(rect=(0.02, 0.04, 1, 1), h_pad=2.1, w_pad=1.7)
    return save_figure(fig, "figure_4_domain_retention")


def figure5_tame_mechanism() -> list[Path]:
    fig, ax = plt.subplots(figsize=(15.0, 7.2))
    ax.set_axis_off()

    # A: ordinary baseline.
    stage_header(ax, 0.025, 0.91, 0.205, "A", "ORDINARY MONDRIAN", COLORS["blue_gray"])
    rounded_card(ax, 0.035, 0.705, 0.185, 0.125, facecolor=COLORS["gray_light"], edgecolor=COLORS["gray"])
    ax.text(0.1275, 0.778, "Four ECFP classifiers", transform=ax.transAxes, ha="center", fontsize=9.3, fontweight="bold")
    ax.text(0.1275, 0.735, "logistic | RF | extra trees | NB", transform=ax.transAxes, ha="center", fontsize=7.7, color=COLORS["muted"])
    rounded_card(ax, 0.060, 0.535, 0.135, 0.095, facecolor="#E6EFF2", edgecolor=COLORS["blue_gray"])
    ax.text(0.1275, 0.582, "Logit router", transform=ax.transAxes, ha="center", va="center", fontsize=9.4, fontweight="bold")
    rounded_card(ax, 0.035, 0.350, 0.185, 0.115, facecolor=COLORS["white"], edgecolor=COLORS["blue_gray"])
    ax.text(0.1275, 0.420, r"$s_0=p,\quad s_1=1-p$", transform=ax.transAxes, ha="center", fontsize=11.0)
    ax.text(0.1275, 0.379, "class-specific 90% thresholds", transform=ax.transAxes, ha="center", fontsize=7.9, color=COLORS["muted"])
    rounded_card(ax, 0.060, 0.185, 0.135, 0.090, facecolor=COLORS["mint_light"], edgecolor=COLORS["mint"])
    ax.text(0.1275, 0.230, r"baseline set $S_M(x)$", transform=ax.transAxes, ha="center", va="center", fontsize=9.6, fontweight="bold")
    for y1, y2 in [(0.695, 0.640), (0.525, 0.475), (0.340, 0.285)]:
        ax.annotate("", xy=(0.1275, y2), xytext=(0.1275, y1), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="-|>", color=COLORS["outline"], lw=1.1))

    # B: two approved transport views.
    stage_header(ax, 0.275, 0.91, 0.205, "B", "LABEL-FREE VIEWS", COLORS["teal"])
    view_specs = [
        (0.285, 0.640, "Physicochemical view", "MW | logP | TPSA | counts", COLORS["mint_light"], COLORS["mint"]),
        (0.285, 0.470, "Score view", "four component logits + router", "#E3F0F2", COLORS["teal"]),
    ]
    for x, y, title, subtitle, fill, edge in view_specs:
        rounded_card(ax, x, y, 0.185, 0.125, facecolor=fill, edgecolor=edge)
        ax.text(x + 0.0925, y + 0.078, title, transform=ax.transAxes, ha="center", fontsize=9.4, fontweight="bold")
        ax.text(x + 0.0925, y + 0.040, subtitle, transform=ax.transAxes, ha="center", fontsize=7.7, color=COLORS["muted"])
    rounded_card(ax, 0.310, 0.265, 0.135, 0.115, facecolor=COLORS["white"], edgecolor=COLORS["teal"])
    ax.text(0.3775, 0.340, "Weighted Mondrian", transform=ax.transAxes, ha="center", fontsize=9.0, fontweight="bold")
    ax.text(0.3775, 0.300, r"$S_{phys}(x)$   $S_{score}(x)$", transform=ax.transAxes, ha="center", fontsize=8.5)
    for x0 in [0.330, 0.425]:
        ax.annotate("", xy=(0.3775, 0.390), xytext=(x0, 0.460), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="-|>", color=COLORS["outline"], lw=1.0))

    # C: certificate before activation.
    stage_header(ax, 0.525, 0.91, 0.205, "C", "TRANSPORT CERTIFICATE", COLORS["coral"])
    rounded_card(ax, 0.535, 0.525, 0.185, 0.285, facecolor="#FAFCFC", edgecolor=COLORS["coral"])
    checks = [
        "source and target n >= 100",
        "per-class conformal n >= 20",
        "total and class ESS >= 25%",
        "clipping at each bound <= 30%",
        "domain AUC <= 0.99",
        "weighted balance not worse",
    ]
    for i, check in enumerate(checks):
        yy = 0.765 - i * 0.039
        ax.text(0.555, yy, "OK", transform=ax.transAxes, color=COLORS["teal_dark"], fontsize=7.0, fontweight="bold", va="center")
        ax.text(0.574, yy, check, transform=ax.transAxes, fontsize=7.8, va="center")
    rounded_card(ax, 0.555, 0.320, 0.145, 0.100, facecolor=COLORS["coral_light"], edgecolor="#E9B8AC")
    ax.text(0.6275, 0.373, "Both views pass?", transform=ax.transAxes, ha="center", va="center", fontsize=9.4, fontweight="bold")
    ax.text(0.6275, 0.340, "all-view quorum", transform=ax.transAxes, ha="center", fontsize=7.6, color=COLORS["muted"])
    ax.annotate("", xy=(0.6275, 0.430), xytext=(0.6275, 0.515), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="-|>", color=COLORS["outline"], lw=1.1))

    # D: envelope and exact fallback.
    stage_header(ax, 0.775, 0.91, 0.205, "D", "ENVELOPE + FALLBACK", COLORS["teal_dark"])
    rounded_card(ax, 0.785, 0.620, 0.185, 0.160, facecolor=COLORS["mint_light"], edgecolor=COLORS["teal_dark"], linewidth=1.2)
    ax.text(0.8775, 0.726, "Protected-label consensus", transform=ax.transAxes, ha="center", fontsize=9.6, fontweight="bold")
    ax.text(0.8775, 0.675, r"$S_{TAME}=S_M\cup\{0:\,0\in S_{phys}\cap S_{score}\}$", transform=ax.transAxes, ha="center", fontsize=9.0)
    ax.text(0.8775, 0.640, "baseline labels are never removed", transform=ax.transAxes, ha="center", fontsize=7.6, color=COLORS["muted"])
    rounded_card(ax, 0.785, 0.435, 0.185, 0.105, facecolor=COLORS["coral_light"], edgecolor=COLORS["coral"])
    ax.text(0.8775, 0.493, "Audit fails: ordinary fallback", transform=ax.transAxes, ha="center", fontsize=9.0, fontweight="bold", color=COLORS["coral"])
    ax.text(0.8775, 0.458, "baseline empty set becomes {0,1}", transform=ax.transAxes, ha="center", fontsize=7.6)
    rounded_card(ax, 0.785, 0.195, 0.185, 0.165, facecolor="#F7FAFA", edgecolor=COLORS["grid"])
    invariants = [
        r"$S_{TAME}\supseteq S_M$",
        "no empty prediction sets",
        "no newly created singletons",
    ]
    for i, item in enumerate(invariants):
        yy = 0.315 - i * 0.045
        ax.text(0.808, yy, "OK", transform=ax.transAxes, color=COLORS["teal_dark"], fontweight="bold", fontsize=7.0, va="center")
        ax.text(0.831, yy, item, transform=ax.transAxes, fontsize=8.4, va="center")

    # Main horizontal flow and failure branch.
    for start, end in [((0.225, 0.50), (0.273, 0.50)), ((0.475, 0.50), (0.523, 0.50)), ((0.725, 0.50), (0.773, 0.50))]:
        ax.annotate("", xy=end, xytext=start, xycoords=ax.transAxes, arrowprops=dict(arrowstyle="-|>", color=COLORS["outline"], lw=1.5, mutation_scale=12))
    ax.annotate("PASS", xy=(0.785, 0.695), xytext=(0.655, 0.300), xycoords=ax.transAxes, fontsize=7.5, color=COLORS["teal_dark"], fontweight="bold", arrowprops=dict(arrowstyle="-|>", color=COLORS["teal_dark"], lw=1.1, connectionstyle="arc3,rad=-0.20"))
    ax.annotate("FAIL", xy=(0.785, 0.490), xytext=(0.646, 0.275), xycoords=ax.transAxes, fontsize=7.5, color=COLORS["coral"], fontweight="bold", arrowprops=dict(arrowstyle="-|>", color=COLORS["coral"], lw=1.1, connectionstyle="arc3,rad=0.12"))

    ax.text(
        0.5,
        0.075,
        "TAME can turn a baseline singleton into an explicit defer, but it cannot create a new confident singleton.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9.2,
        fontweight="bold",
        color=COLORS["teal_dark"],
    )
    return save_figure(fig, "figure_5_tame_mechanism")


def figure6_epa_validation() -> list[Path]:
    primary = read_csv(C4_FINAL_DIR / "primary_endpoint_means.csv")
    report = read_json(C4_FINAL_DIR / "publication_final_report.json")
    audit = read_json(C4_FINAL_DIR / "transport_audit_summary.json")

    fig, axes = plt.subplots(2, 2, figsize=(12.7, 7.9))
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
        ax.text(value + 0.10, yy, f"{value:+.2f}", va="center", fontsize=8.0, color=COLORS["teal_dark"])
    ax.axvline(0, color=COLORS["muted"], linewidth=0.9)
    ax.set_yticks(y, order)
    ax.set_xlabel("Minimum-class coverage difference (percentage points)")
    ax.set_title("Six prespecified primary endpoints", loc="left", pad=9, fontweight="bold")
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
    ax.set_xlabel("Endpoint-seed-equal mean difference (percentage points)")
    ax.set_title("Primary hierarchical-bootstrap inference", loc="left", pad=9, fontweight="bold")
    ax.text(estimate, 0.24, f"{estimate:+.2f} pp", ha="center", fontsize=12.5, fontweight="bold", color=COLORS["teal_dark"])
    ax.text((lower + upper) / 2, -0.28, f"95% CI  {lower:+.2f} to {upper:+.2f} pp", ha="center", fontsize=8.8)
    ax.text((lower + upper) / 2, -0.48, "2,000 draws | seed 44021", ha="center", fontsize=7.7, color=COLORS["muted"])
    clean_axis(ax, grid="x")
    panel_label(ax, "B", x=-0.12)

    # C: endpoint-level and aggregate MacroCSY cost against the frozen margin.
    ax = axes[1, 0]
    macro_by_endpoint = [100 * float(primary.set_index("display").loc[label, "mean_macro_csy_delta"]) for label in order]
    colors = [COLORS["coral"] if value < 0 else COLORS["mint"] for value in macro_by_endpoint]
    ax.barh(y, macro_by_endpoint, color=colors, edgecolor=COLORS["white"], height=0.62)
    aggregate = 100 * float(report["primary_mean_macro_csy_delta"])
    ax.scatter([aggregate], [-0.75], marker="D", s=58, color=COLORS["teal_dark"], zorder=4)
    ax.text(aggregate + 0.12, -0.75, f"aggregate {aggregate:+.2f}", va="center", fontsize=8.0, fontweight="bold", color=COLORS["teal_dark"])
    ax.axvline(-5.0, color=COLORS["coral"], linestyle="--", linewidth=1.2)
    ax.axvline(0.0, color=COLORS["muted"], linewidth=0.9)
    ax.text(-4.92, len(order) - 0.15, "frozen non-inferiority bound", rotation=90, va="top", fontsize=7.5, color=COLORS["coral"])
    ax.set_yticks(y, order)
    ax.set_ylim(-1.35, len(order) - 0.35)
    ax.set_xlim(-5.5, 0.6)
    ax.set_xlabel("MacroCSY difference (percentage points)")
    ax.set_title("Efficiency cost remains inside the -5 pp bound", loc="left", pad=9, fontweight="bold")
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
    ax.set_xlabel("Active endpoint-seed cells (of 60)")
    ax.set_title("Label-blind transport audit and firewall", loc="left", pad=9, fontweight="bold")
    for bar, value in zip(bars, counts):
        ax.text(value + 1.2, bar.get_y() + bar.get_height() / 2, f"{value}/60", va="center", fontsize=8.3, fontweight="bold")
    clean_axis(ax, grid="x")
    ax.text(
        0.03,
        -0.30,
        "PASS  predictions and transport audit hashed before labels\n"
        "PASS  60/60 endpoint-seed cells completed\n"
        "PASS  82/82 tests after deterministic interval repair",
        transform=ax.transAxes,
        fontsize=8.1,
        linespacing=1.45,
        va="top",
        color=COLORS["teal_dark"],
        fontweight="bold",
    )
    panel_label(ax, "D", x=-0.18)

    fig.tight_layout(rect=(0.02, 0.08, 1, 1), h_pad=2.5, w_pad=2.0)
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
