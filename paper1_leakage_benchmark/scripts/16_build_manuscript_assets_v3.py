from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper1_leakage_benchmark"
TABLE_DIR = PAPER_DIR / "results" / "tables"
PARENT_DIR = PAPER_DIR / "results" / "parent_fragment_sensitivity_v3" / "tables"
FIG_DIR = PAPER_DIR / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY = TABLE_DIR / "primary_inference_summary_v3.csv"
SINGLETON = TABLE_DIR / "acyclic_singleton_sensitivity_v3.csv"
PARENT_COMPARISON = PARENT_DIR / "parent_fragment_vs_main_comparison_v3.csv"

MODEL_ORDER = {"LR": 0, "Ridge": 0, "RF": 1, "XGB": 2}
DATASET_ORDER = {
    "BACE": 0,
    "BBBP": 1,
    "ClinTox": 2,
    "HIV": 3,
    "ESOL": 4,
    "FreeSolv": 5,
}
DATASET_N = {
    "BACE": 1513,
    "BBBP": 1965,
    "ClinTox": 1442,
    "HIV": 41120,
    "ESOL": 1117,
    "FreeSolv": 642,
}
TEST_N = {
    "BACE": 303,
    "BBBP": 393,
    "ClinTox": 288,
    "HIV": 8224,
    "ESOL": 223,
    "FreeSolv": 128,
}
RAW_N = {
    "BACE": 1513,
    "BBBP": 2050,
    "ClinTox": 1484,
    "HIV": 41127,
    "ESOL": 1128,
    "FreeSolv": 642,
}
MULTICOMPONENT = {"BBBP": 105, "ClinTox": 14, "HIV": 3086}
SCAFFOLD_CHANGED = {"BBBP": 5, "ClinTox": 1, "HIV": 235}
SIM_LT_090 = {"BBBP": 18, "ClinTox": 5, "HIV": 640}
BUDGET_SINGLE_GROUP = {
    "ESOL": {3000: 0.034622, 5000: 0.018366, 10000: 0.010563, 20000: 0.001787},
    "FreeSolv": {3000: 1.076491, 5000: 1.044452, 10000: 0.787491, 20000: 0.686528},
}
BUDGET_SINGLETON = {
    "ESOL": {100: 0.003687, 300: 0.001535, 500: 0.000600, 1000: 0.000600, 3000: 0.000301, 5000: 0.000194},
    "FreeSolv": {100: 0.019467, 300: 0.006230, 500: 0.005645, 1000: 0.005416, 3000: 0.001609, 5000: 0.000471},
}

COLORS = {
    "navy": "#31546D",
    "teal": "#2A8C82",
    "teal_dark": "#19675F",
    "cyan": "#63B8C8",
    "orange": "#D58B43",
    "orange_dark": "#A96025",
    "sage": "#A9C7B8",
    "mint": "#DCECE6",
    "blue_pale": "#E7F0F4",
    "orange_pale": "#F7EBDD",
    "gray": "#6F7B82",
    "gray_mid": "#AEB8BD",
    "gray_light": "#EEF2F3",
    "ink": "#233139",
    "white": "#FFFFFF",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": COLORS["gray_mid"],
        "axes.labelcolor": COLORS["ink"],
        "text.color": COLORS["ink"],
        "xtick.color": COLORS["ink"],
        "ytick.color": COLORS["ink"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required result file: {path}")


def save(fig: plt.Figure, stem: str) -> None:
    pdf = FIG_DIR / f"{stem}.pdf"
    png = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(pdf)
    print(png)


def panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="top",
        color=COLORS["ink"],
    )


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str | None = None,
    fontsize: float = 9,
    weight: str = "normal",
    radius: float = 0.025,
    alpha: float = 1.0,
) -> FancyBboxPatch:
    edge = edgecolor or facecolor
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=1.0,
        edgecolor=edge,
        facecolor=facecolor,
        alpha=alpha,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=COLORS["ink"],
    )
    return patch


def arrow_axes(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = COLORS["gray"],
    lw: float = 1.3,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=lw,
            color=color,
        )
    )


def _sort_primary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dataset_order"] = out["dataset"].map(DATASET_ORDER)
    out["model_order"] = out["model"].map(MODEL_ORDER)
    return out.sort_values(["dataset_order", "model_order"], kind="mergesort").reset_index(drop=True)


def _singleton_frame() -> pd.DataFrame:
    require(SINGLETON)
    df = pd.read_csv(SINGLETON, keep_default_na=False).copy()
    if "mean_effect_positive_is_balanced_better" in df.columns:
        df = df.rename(columns={"mean_effect_positive_is_balanced_better": "mean_effect"})
    needed = {"dataset", "model", "mean_effect", "bootstrap_ci_low", "bootstrap_ci_high"}
    missing = needed.difference(df.columns)
    if missing:
        raise KeyError(f"Singleton sensitivity table missing columns: {sorted(missing)}")
    return df


def figure1_framework() -> None:
    """Journal-style multi-panel overview. Replaces the clipped black-and-white strip."""
    fig = plt.figure(figsize=(13.6, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.04], wspace=0.34, hspace=0.44)

    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "A")
    datasets = list(DATASET_N)
    values = [DATASET_N[d] for d in datasets]
    bar_colors = [
        COLORS["navy"] if d in {"BACE", "BBBP", "ClinTox", "HIV"} else COLORS["teal"]
        for d in datasets
    ]
    y = np.arange(len(datasets))
    ax.barh(y, values, color=bar_colors, height=0.66, edgecolor="white")
    ax.set_yticks(y, datasets)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Clean molecules (log scale)")
    ax.set_title("Audited molecular universe", loc="left", fontweight="bold")
    ax.grid(axis="x", color=COLORS["gray_light"], linewidth=0.8)
    for yi, val in zip(y, values):
        ax.text(val * 1.08, yi, f"{val:,}", va="center", fontsize=7.8)
    ax.legend(
        handles=[
            Rectangle((0, 0), 1, 1, facecolor=COLORS["navy"], label="Classification"),
            Rectangle((0, 0), 1, 1, facecolor=COLORS["teal"], label="Regression"),
        ],
        frameon=False,
        loc="lower right",
    )

    ax = fig.add_subplot(gs[0, 1])
    ax.set_axis_off()
    panel_label(ax, "B")
    ax.text(0.0, 1.02, "Exact-size paired perturbation", transform=ax.transAxes, fontweight="bold", fontsize=11)
    rounded_box(ax, (0.02, 0.69), 0.26, 0.16, "Target-blind\ncandidate pool", COLORS["blue_pale"], COLORS["navy"])
    arrow_axes(ax, (0.29, 0.77), (0.41, 0.77))
    rounded_box(ax, (0.42, 0.66), 0.24, 0.22, "Size-matched\nbaseline", COLORS["gray_light"], COLORS["gray_mid"], weight="bold")
    rounded_box(ax, (0.72, 0.66), 0.24, 0.22, "Target-balanced\ncounterpart", COLORS["mint"], COLORS["teal"], weight="bold")
    arrow_axes(ax, (0.66, 0.77), (0.71, 0.77), COLORS["teal"])
    rounded_box(ax, (0.22, 0.36), 0.56, 0.14, "same dataset · same seed · same candidate pool", COLORS["white"], COLORS["gray_mid"], fontsize=8.4)
    rounded_box(ax, (0.28, 0.16), 0.44, 0.12, "exactly the same test size", COLORS["orange_pale"], COLORS["orange"], fontsize=8.8, weight="bold")
    ax.text(0.5, 0.05, "Designed contrast: target-distribution mismatch", transform=ax.transAxes, ha="center", fontsize=8.3, color=COLORS["gray"])

    ax = fig.add_subplot(gs[0, 2])
    ax.set_axis_off()
    panel_label(ax, "C")
    ax.text(0.0, 1.02, "Split search is itself a protocol", transform=ax.transAxes, fontweight="bold", fontsize=11)
    rng_y = [0.73, 0.55, 0.37]
    labels = ["Small pool", "Larger pool", "Frozen production budget"]
    counts = [5, 9, 13]
    for yy, label, count in zip(rng_y, labels, counts):
        ax.text(0.01, yy + 0.035, label, transform=ax.transAxes, fontsize=8.4, va="center")
        x0 = 0.35
        for j in range(count):
            col = COLORS["gray_light"]
            ec = COLORS["gray_mid"]
            if j == count - 1:
                col = COLORS["mint"]
                ec = COLORS["teal"]
            rounded_box(ax, (x0 + j * 0.045, yy), 0.032, 0.07, "", col, ec, radius=0.008)
    ax.text(0.02, 0.16, "More candidates create more opportunities\nfor an extreme low-gap split.", transform=ax.transAxes, fontsize=9.0)
    rounded_box(ax, (0.12, 0.02), 0.75, 0.10, "Budget audited and frozen before model outcomes", COLORS["orange_pale"], COLORS["orange"], fontsize=8.5, weight="bold")

    ax = fig.add_subplot(gs[1, 0])
    ax.set_axis_off()
    panel_label(ax, "D")
    ax.text(0.0, 1.02, "Acyclic scaffold semantics", transform=ax.transAxes, fontweight="bold", fontsize=11)
    ax.text(0.19, 0.83, "single-group", transform=ax.transAxes, ha="center", fontsize=9, fontweight="bold")
    ax.text(0.74, 0.83, "singleton", transform=ax.transAxes, ha="center", fontsize=9, fontweight="bold")
    for k in range(4):
        yy = 0.64 - k * 0.13
        ax.plot([0.06, 0.15, 0.23, 0.31], [yy, yy + 0.04, yy - 0.02, yy + 0.03], color=COLORS["navy"], lw=2.0, transform=ax.transAxes, clip_on=False)
        ax.plot([0.60, 0.69, 0.77, 0.85], [yy, yy + 0.04, yy - 0.02, yy + 0.03], color=COLORS["teal"], lw=2.0, transform=ax.transAxes, clip_on=False)
        rounded_box(ax, (0.88, yy - 0.035), 0.09, 0.07, f"S{k+1}", COLORS["mint"], COLORS["teal"], fontsize=7.5, radius=0.01)
    rounded_box(ax, (0.04, 0.05), 0.32, 0.12, "one ACYCLIC\nscaffold identity", COLORS["blue_pale"], COLORS["navy"], fontsize=8.2)
    rounded_box(ax, (0.59, 0.05), 0.37, 0.12, "each acyclic molecule\ngets its own identity", COLORS["mint"], COLORS["teal"], fontsize=8.2)

    ax = fig.add_subplot(gs[1, 1])
    ax.set_axis_off()
    panel_label(ax, "E")
    ax.text(0.0, 1.02, "Disconnected-component representation", transform=ax.transAxes, fontweight="bold", fontsize=11)
    ax.text(0.20, 0.82, "source-faithful record", transform=ax.transAxes, ha="center", fontsize=8.8, fontweight="bold")
    centers = [(0.08, 0.60), (0.18, 0.64), (0.27, 0.56), (0.34, 0.68)]
    radii = [0.065, 0.05, 0.035, 0.025]
    colors = [COLORS["teal"], COLORS["cyan"], COLORS["orange"], COLORS["gray_mid"]]
    for (cx, cy), r, color in zip(centers, radii, colors):
        ax.add_patch(Circle((cx, cy), r, transform=ax.transAxes, facecolor=color, edgecolor="white", lw=1.0))
    arrow_axes(ax, (0.42, 0.61), (0.60, 0.61), COLORS["orange"], lw=1.8)
    ax.text(0.51, 0.67, "deterministic\nselection", transform=ax.transAxes, ha="center", fontsize=7.8, color=COLORS["gray"])
    ax.add_patch(Circle((0.76, 0.61), 0.105, transform=ax.transAxes, facecolor=COLORS["teal"], edgecolor="white", lw=1.2))
    ax.text(0.76, 0.82, "dominant fragment", transform=ax.transAxes, ha="center", fontsize=8.8, fontweight="bold")
    rounded_box(ax, (0.12, 0.20), 0.76, 0.16, "Representation choice can alter fingerprints,\nscaffold identities, duplicates, and point estimates.", COLORS["orange_pale"], COLORS["orange"], fontsize=8.5)

    ax = fig.add_subplot(gs[1, 2])
    ax.set_axis_off()
    panel_label(ax, "F")
    ax.text(0.0, 1.02, "Freeze → fit → paired inference", transform=ax.transAxes, fontweight="bold", fontsize=11)
    rounded_box(ax, (0.01, 0.67), 0.25, 0.16, "Molecule-level\nmanifest + hash", COLORS["blue_pale"], COLORS["navy"], fontsize=8.4, weight="bold")
    arrow_axes(ax, (0.27, 0.75), (0.37, 0.75))
    rounded_box(ax, (0.39, 0.67), 0.23, 0.16, "LR / Ridge\nRF · XGB", COLORS["gray_light"], COLORS["gray_mid"], fontsize=8.4, weight="bold")
    arrow_axes(ax, (0.63, 0.75), (0.72, 0.75))
    rounded_box(ax, (0.74, 0.67), 0.24, 0.16, "paired effect\nper partition", COLORS["mint"], COLORS["teal"], fontsize=8.4, weight="bold")
    rounded_box(ax, (0.04, 0.38), 0.25, 0.13, "20 unique\npartition pairs", COLORS["white"], COLORS["gray_mid"], fontsize=8.1)
    rounded_box(ax, (0.37, 0.38), 0.25, 0.13, "10,000 paired\nbootstrap draws", COLORS["white"], COLORS["gray_mid"], fontsize=8.1)
    rounded_box(ax, (0.70, 0.38), 0.25, 0.13, "Wilcoxon +\nHolm correction", COLORS["white"], COLORS["gray_mid"], fontsize=8.1)
    rounded_box(ax, (0.16, 0.08), 0.68, 0.15, "Claim stability is audited across protocol perturbations", COLORS["orange_pale"], COLORS["orange"], fontsize=8.8, weight="bold")

    fig.suptitle("Benchmark construction treated as a chemometric measurement process", fontsize=15, fontweight="bold", y=1.01)
    save(fig, "figure1_audit_framework_v3")


def _forest(ax: plt.Axes, df: pd.DataFrame, xlabel: str, panel: str) -> None:
    y = np.arange(len(df))
    means = df["mean_effect"].astype(float).to_numpy()
    lo = df["bootstrap_ci_low"].astype(float).to_numpy()
    hi = df["bootstrap_ci_high"].astype(float).to_numpy()
    labels = [f"{d} · {m}" for d, m in zip(df["dataset"], df["model"])]

    xmin = float(np.min(lo))
    xmax = float(np.max(hi))
    pad = max((xmax - xmin) * 0.06, 0.002)
    ax.axvspan(xmin - pad, 0.0, color=COLORS["orange_pale"], alpha=0.55, zorder=0)
    ax.axvspan(0.0, xmax + pad, color=COLORS["mint"], alpha=0.55, zorder=0)

    for yi, mean, low, high, inference in zip(y, means, lo, hi, df.get("inference", pd.Series([""] * len(df))).astype(str)):
        supported = "better" in inference.lower() and "inconclusive" not in inference.lower()
        color = COLORS["teal_dark"] if supported else COLORS["navy"]
        ax.errorbar(mean, yi, xerr=np.array([[mean - low], [high - mean]]), fmt="o", markersize=5.5 if supported else 4.8, capsize=3, linewidth=1.25, color=color, ecolor=color, zorder=3)
    ax.axvline(0.0, linewidth=1.0, linestyle="--", color=COLORS["gray"], zorder=2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=COLORS["gray_light"], linewidth=0.7, zorder=0)
    ax.set_xlim(xmin - pad, xmax + pad)
    panel_label(ax, panel, x=-0.10, y=1.06)


def figure2_primary() -> None:
    require(PRIMARY)
    df = _sort_primary(pd.read_csv(PRIMARY, keep_default_na=False))
    if len(df) != 18:
        raise AssertionError(f"Expected 18 primary cells; found {len(df)}")
    cls = df.loc[df["task_type"].eq("classification")].copy()
    reg = df.loc[df["task_type"].eq("regression")].copy()
    if len(cls) != 12 or len(reg) != 6:
        raise AssertionError("Primary task counts are not 12 classification + 6 regression")

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.7), gridspec_kw={"width_ratios": [1.2, 1.0]})
    _forest(axes[0], cls, "AUC effect: balanced − size-matched", "A")
    axes[0].set_title("Classification · 12 dataset–model cells", loc="left", fontweight="bold")
    axes[0].text(0.02, 1.01, "No cell met the pre-specified corrected decision rule", transform=axes[0].transAxes, fontsize=8.3, color=COLORS["gray"], va="bottom")

    _forest(axes[1], reg, "RMSE improvement: size-matched − balanced", "B")
    axes[1].set_title("Regression · primary single-group semantics", loc="left", fontweight="bold")
    axes[1].text(0.02, 1.01, "All six cells favored target balancing", transform=axes[1].transAxes, fontsize=8.3, color=COLORS["teal_dark"], va="bottom")

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["navy"], markeredgecolor=COLORS["navy"], label="Inconclusive"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["teal_dark"], markeredgecolor=COLORS["teal_dark"], label="Balanced better"),
    ]
    axes[1].legend(handles=legend, frameon=False, loc="lower right")
    fig.suptitle("Exact-size paired target-balance effects over 20 unique partition pairs", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "figure2_primary_effects_v3")


def figure3_acyclic() -> None:
    require(PRIMARY)
    primary = pd.read_csv(PRIMARY, keep_default_na=False)
    primary = primary.loc[primary["dataset"].isin(["ESOL", "FreeSolv"])].copy()
    singleton = _singleton_frame()
    if len(primary) != 6 or len(singleton) != 6:
        raise AssertionError("Expected six primary and six singleton regression cells")

    primary = primary[["dataset", "model", "mean_effect", "bootstrap_ci_low", "bootstrap_ci_high"]].copy()
    primary["semantics"] = "single-group (primary)"
    singleton = singleton[["dataset", "model", "mean_effect", "bootstrap_ci_low", "bootstrap_ci_high"]].copy()
    singleton["semantics"] = "singleton sensitivity"

    fig = plt.figure(figsize=(13.2, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.72, 1.4], hspace=0.36, wspace=0.22)
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.set_axis_off()
    panel_label(ax_top, "A", x=-0.03, y=1.03)
    ax_top.text(0.02, 0.91, "What changes when the empty Bemis–Murcko framework is interpreted differently?", transform=ax_top.transAxes, fontsize=11, fontweight="bold")

    rounded_box(ax_top, (0.07, 0.46), 0.29, 0.26, "single-group\nall acyclic molecules → one identity", COLORS["blue_pale"], COLORS["navy"], fontsize=10, weight="bold")
    rounded_box(ax_top, (0.64, 0.46), 0.29, 0.26, "singleton\neach acyclic molecule → own identity", COLORS["mint"], COLORS["teal"], fontsize=10, weight="bold")
    arrow_axes(ax_top, (0.38, 0.59), (0.61, 0.59), COLORS["orange"], lw=1.8)
    rounded_box(ax_top, (0.36, 0.12), 0.28, 0.16, "same endpoints · same models · same 20 seeds\nsame exact-size paired selection logic", COLORS["orange_pale"], COLORS["orange"], fontsize=8.2)

    for idx, dataset in enumerate(("ESOL", "FreeSolv")):
        ax = fig.add_subplot(gs[1, idx])
        panel_label(ax, "B" if idx == 0 else "C", x=-0.10, y=1.06)
        base_y = np.arange(3)
        for offset, semantics, marker, color in (
            (-0.10, "single-group (primary)", "o", COLORS["navy"]),
            (0.10, "singleton sensitivity", "s", COLORS["orange"]),
        ):
            frame = primary if "primary" in semantics else singleton
            vals = frame.loc[frame["dataset"].eq(dataset)].copy()
            vals["model_order"] = vals["model"].map(MODEL_ORDER)
            vals = vals.sort_values("model_order")
            means = vals["mean_effect"].astype(float).to_numpy()
            lo = vals["bootstrap_ci_low"].astype(float).to_numpy()
            hi = vals["bootstrap_ci_high"].astype(float).to_numpy()
            ax.errorbar(means, base_y + offset, xerr=np.vstack([means - lo, hi - means]), fmt=marker, capsize=3, linewidth=1.35, markersize=5.3, color=color, ecolor=color, label=semantics, zorder=3)
        ax.axvline(0.0, linestyle="--", linewidth=1.0, color=COLORS["gray"])
        ax.set_yticks(base_y, ["Ridge", "RF", "XGB"])
        ax.invert_yaxis()
        ax.set_title(dataset, loc="left", fontweight="bold")
        ax.set_xlabel("RMSE improvement: size-matched − balanced")
        ax.grid(axis="x", color=COLORS["gray_light"], linewidth=0.7)
        if dataset == "ESOL":
            ax.text(0.98, 0.04, "Primary gains attenuate", transform=ax.transAxes, ha="right", color=COLORS["orange_dark"], fontsize=8.5, fontweight="bold")
        else:
            ax.text(0.98, 0.04, "All three point estimates reverse sign", transform=ax.transAxes, ha="right", color=COLORS["orange_dark"], fontsize=8.5, fontweight="bold")

    axes_legend = fig.axes[-2]
    axes_legend.legend(frameon=False, loc="lower right")
    fig.suptitle("Regression effect depends strongly on the structural semantics assigned to acyclic molecules", fontsize=14, fontweight="bold", y=1.01)
    save(fig, "figure3_acyclic_sensitivity_v3")


def figure4_fragment() -> None:
    require(PARENT_COMPARISON)
    df = pd.read_csv(PARENT_COMPARISON, keep_default_na=False)
    if len(df) != 9:
        raise AssertionError(f"Expected nine dominant-fragment comparison cells; found {len(df)}")
    df["dataset_order"] = df["dataset"].map(DATASET_ORDER)
    df["model_order"] = df["model"].map(MODEL_ORDER)
    df = df.sort_values(["dataset_order", "model_order"], kind="mergesort").reset_index(drop=True)

    fig = plt.figure(figsize=(13.2, 8.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.92, 1.55], hspace=0.38, wspace=0.36)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_off()
    panel_label(ax, "A")
    ax.text(0.0, 1.00, "Representation perturbation", transform=ax.transAxes, fontsize=11, fontweight="bold")
    centers = [(0.08, 0.61), (0.19, 0.66), (0.29, 0.56), (0.36, 0.69)]
    radii = [0.072, 0.055, 0.038, 0.027]
    colors = [COLORS["teal"], COLORS["cyan"], COLORS["orange"], COLORS["gray_mid"]]
    for (cx, cy), r, color in zip(centers, radii, colors):
        ax.add_patch(Circle((cx, cy), r, transform=ax.transAxes, facecolor=color, edgecolor="white", lw=1.0))
    arrow_axes(ax, (0.44, 0.62), (0.63, 0.62), COLORS["orange"], lw=1.7)
    ax.add_patch(Circle((0.80, 0.62), 0.12, transform=ax.transAxes, facecolor=COLORS["teal"], edgecolor="white", lw=1.2))
    ax.text(0.21, 0.82, "source-faithful", transform=ax.transAxes, ha="center", fontsize=8.7, fontweight="bold")
    ax.text(0.80, 0.82, "dominant fragment", transform=ax.transAxes, ha="center", fontsize=8.7, fontweight="bold")
    rounded_box(ax, (0.10, 0.17), 0.78, 0.18, "not a lossless formatting step", COLORS["orange_pale"], COLORS["orange"], fontsize=9.0, weight="bold")

    for col, values, title in [
        (1, MULTICOMPONENT, "Multi-component records"),
        (2, SCAFFOLD_CHANGED, "Scaffold identities changed"),
    ]:
        ax = fig.add_subplot(gs[0, col])
        panel_label(ax, "B" if col == 1 else "C")
        names = ["BBBP", "ClinTox", "HIV"]
        arr = [values[n] for n in names]
        bars = ax.bar(names, arr, color=[COLORS["navy"], COLORS["cyan"], COLORS["teal"]], width=0.62)
        ax.set_yscale("log")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel("Count (log scale)")
        ax.grid(axis="y", color=COLORS["gray_light"], linewidth=0.7)
        for bar, val in zip(bars, arr):
            ax.text(bar.get_x() + bar.get_width() / 2, val * 1.16, f"{val:,}", ha="center", va="bottom", fontsize=8.2)

    ax = fig.add_subplot(gs[1, :])
    panel_label(ax, "D", x=-0.03, y=1.05)
    y = np.arange(len(df))
    main = df["main_mean_effect"].astype(float).to_numpy()
    parent = df["parent_mean_effect"].astype(float).to_numpy()
    for yi, x1, x2 in zip(y, main, parent):
        flipped = (x1 < 0 < x2) or (x2 < 0 < x1)
        line_color = COLORS["orange"] if flipped else COLORS["gray_mid"]
        ax.plot([x1, x2], [yi, yi], linewidth=1.7 if flipped else 1.1, color=line_color, zorder=1)
    ax.scatter(main, y, marker="o", s=38, color=COLORS["navy"], label="source-faithful primary", zorder=3)
    ax.scatter(parent, y, marker="s", s=38, color=COLORS["orange"], label="dominant-fragment sensitivity", zorder=3)
    ax.axvline(0.0, linestyle="--", linewidth=1.0, color=COLORS["gray"])
    ax.set_yticks(y, [f"{d} · {m}" for d, m in zip(df["dataset"], df["model"])])
    ax.invert_yaxis()
    ax.set_xlabel("AUC effect: balanced − size-matched")
    ax.set_title("Classification inference is stable while point-estimate direction is representation-sensitive", loc="left", fontweight="bold")
    ax.grid(axis="x", color=COLORS["gray_light"], linewidth=0.7)
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.99, 0.98, "7/9 mean-effect signs reverse\n9/9 sensitivity inferences remain inconclusive", transform=ax.transAxes, ha="right", va="top", fontsize=9.0, fontweight="bold", color=COLORS["orange_dark"], bbox=dict(boxstyle="round,pad=0.35", facecolor=COLORS["orange_pale"], edgecolor=COLORS["orange"], linewidth=0.8))

    fig.suptitle("Disconnected-component representation changes the benchmark without changing the corrected classification inference", fontsize=14, fontweight="bold", y=1.01)
    save(fig, "figure4_dominant_fragment_sensitivity_v3")


def figure5_budget_audit() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.0))
    panel_label(axes[0, 0], "A")
    panel_label(axes[0, 1], "B")
    panel_label(axes[1, 0], "C")
    panel_label(axes[1, 1], "D")

    for ax, dataset, color in [(axes[0, 0], "ESOL", COLORS["teal"]), (axes[0, 1], "FreeSolv", COLORS["navy"])]:
        data = BUDGET_SINGLE_GROUP[dataset]
        x = np.array(list(data.keys()), dtype=float)
        y = np.array(list(data.values()), dtype=float)
        ax.plot(x, y, marker="o", linewidth=2.0, color=color)
        ax.fill_between(x, y, alpha=0.08, color=color)
        ax.set_xscale("log")
        ax.set_title(f"{dataset} · single-group acyclic semantics", loc="left", fontweight="bold")
        ax.set_xlabel("Candidate budget")
        ax.set_ylabel("Mean balanced target gap")
        ax.grid(color=COLORS["gray_light"], linewidth=0.7)
        ax.annotate(f"{y[-1]:.4g} at 20,000", xy=(x[-1], y[-1]), xytext=(-78, 24), textcoords="offset points", arrowprops=dict(arrowstyle="->", color=color, lw=1.0), fontsize=8.4, color=color)

    ax = axes[1, 0]
    for dataset, color, marker in [("ESOL", COLORS["teal"], "o"), ("FreeSolv", COLORS["orange"], "s")]:
        data = BUDGET_SINGLETON[dataset]
        x = np.array(list(data.keys()), dtype=float)
        y = np.array(list(data.values()), dtype=float)
        rel = y / y[0]
        ax.plot(x, rel, marker=marker, linewidth=1.8, color=color, label=dataset)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Singleton sensitivity · relative target-gap reduction", loc="left", fontweight="bold")
    ax.set_xlabel("Candidate budget")
    ax.set_ylabel("Gap relative to 100-candidate value")
    ax.grid(color=COLORS["gray_light"], linewidth=0.7)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    names = list(TEST_N)
    vals = np.array([TEST_N[d] for d in names])
    x = np.arange(len(names))
    width = 0.33
    ax.bar(x - width / 2, vals, width, color=COLORS["navy"], label="Size-matched")
    ax.bar(x + width / 2, vals, width, color=COLORS["teal"], label="Target-balanced")
    ax.set_yscale("log")
    ax.set_xticks(x, names, rotation=20)
    ax.set_ylabel("Test molecules (log scale)")
    ax.set_title("Exact test-size pairing holds for every dataset", loc="left", fontweight="bold")
    ax.grid(axis="y", color=COLORS["gray_light"], linewidth=0.7)
    ax.legend(frameon=False, loc="upper left")
    for xi, val in zip(x, vals):
        ax.text(xi, val * 1.17, f"{val:,}", ha="center", fontsize=7.8)

    fig.suptitle("Candidate-search budget is a benchmark-construction parameter and was frozen before model fitting", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "figure5_candidate_budget_audit_v3")


def figure6_claim_stability() -> None:
    require(PRIMARY)
    require(PARENT_COMPARISON)
    primary = pd.read_csv(PRIMARY, keep_default_na=False)
    singleton = _singleton_frame()
    fragment = pd.read_csv(PARENT_COMPARISON, keep_default_na=False)

    cls = primary.loc[primary["task_type"].eq("classification")].copy()
    reg = primary.loc[primary["task_type"].eq("regression")].copy()
    supported_cls = int(cls.get("inference", pd.Series([""] * len(cls))).astype(str).str.lower().str.contains("balanced better").sum())
    supported_reg = int(reg.get("inference", pd.Series([""] * len(reg))).astype(str).str.lower().str.contains("balanced better").sum())
    cls_negative = int((cls["mean_effect"].astype(float) < 0).sum())
    fragment_flips = int((np.sign(fragment["main_mean_effect"].astype(float).to_numpy()) != np.sign(fragment["parent_mean_effect"].astype(float).to_numpy())).sum())
    free_singleton = singleton.loc[singleton["dataset"].eq("FreeSolv"), "mean_effect"].astype(float)
    free_reversals = int((free_singleton < 0).sum())

    fig = plt.figure(figsize=(13.2, 6.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.0], wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_off()
    panel_label(ax, "A", x=-0.04, y=1.02)
    ax.text(0.02, 0.95, "Primary classification", transform=ax.transAxes, fontsize=11, fontweight="bold")
    rounded_box(ax, (0.08, 0.71), 0.84, 0.16, f"{supported_cls}/12 supported\nbalanced advantage", COLORS["gray_light"], COLORS["gray_mid"], fontsize=12, weight="bold")
    ax.text(0.5, 0.62, f"{cls_negative}/12 mean effects were negative", transform=ax.transAxes, ha="center", fontsize=9.2, color=COLORS["gray"])
    for i in range(12):
        row, col = divmod(i, 4)
        xx = 0.17 + col * 0.20
        yy = 0.43 - row * 0.13
        eff = float(cls.iloc[i]["mean_effect"])
        color = COLORS["orange_pale"] if eff < 0 else COLORS["mint"]
        edge = COLORS["orange"] if eff < 0 else COLORS["teal"]
        ax.add_patch(Circle((xx, yy), 0.037, transform=ax.transAxes, facecolor=color, edgecolor=edge, lw=1.0))
    rounded_box(ax, (0.12, 0.05), 0.76, 0.12, "Inference: no reproducible classification gain", COLORS["blue_pale"], COLORS["navy"], fontsize=9.1, weight="bold")

    ax = fig.add_subplot(gs[0, 1])
    ax.set_axis_off()
    panel_label(ax, "B", x=-0.04, y=1.02)
    ax.text(0.02, 0.95, "Regression depends on scaffold semantics", transform=ax.transAxes, fontsize=11, fontweight="bold")
    rounded_box(ax, (0.05, 0.73), 0.40, 0.15, f"{supported_reg}/6 supported\nunder primary semantics", COLORS["mint"], COLORS["teal"], fontsize=10.5, weight="bold")
    rounded_box(ax, (0.55, 0.73), 0.40, 0.15, f"{free_reversals}/3 FreeSolv effects\nreverse under singleton", COLORS["orange_pale"], COLORS["orange"], fontsize=10.0, weight="bold")
    arrow_axes(ax, (0.46, 0.805), (0.54, 0.805), COLORS["orange"], lw=1.6)

    y_positions = [0.57, 0.46, 0.35, 0.24, 0.13, 0.02]
    reg_sorted = reg.copy()
    reg_sorted["dataset_order"] = reg_sorted["dataset"].map(DATASET_ORDER)
    reg_sorted["model_order"] = reg_sorted["model"].map(MODEL_ORDER)
    reg_sorted = reg_sorted.sort_values(["dataset_order", "model_order"])
    for yy, (_, row) in zip(y_positions, reg_sorted.iterrows()):
        dataset, model = row["dataset"], row["model"]
        p = float(row["mean_effect"])
        srow = singleton.loc[(singleton["dataset"] == dataset) & (singleton["model"] == model)]
        if len(srow) != 1:
            continue
        s = float(srow.iloc[0]["mean_effect"])
        ax.text(0.03, yy + 0.014, f"{dataset} · {model}", transform=ax.transAxes, fontsize=7.7, va="center")
        x1 = 0.54
        x2 = 0.84
        line_color = COLORS["orange"] if np.sign(p) != np.sign(s) else COLORS["gray_mid"]
        ax.plot([x1, x2], [yy, yy], transform=ax.transAxes, color=line_color, lw=1.2, clip_on=False)
        ax.scatter([x1], [yy], transform=ax.transAxes, color=COLORS["navy"], s=22, zorder=3, clip_on=False)
        ax.scatter([x2], [yy], transform=ax.transAxes, color=COLORS["orange"], marker="s", s=22, zorder=3, clip_on=False)
        ax.text(0.51, yy, f"{p:+.2f}", transform=ax.transAxes, ha="right", va="center", fontsize=7.3, color=COLORS["navy"])
        ax.text(0.87, yy, f"{s:+.2f}", transform=ax.transAxes, ha="left", va="center", fontsize=7.3, color=COLORS["orange_dark"])

    ax = fig.add_subplot(gs[0, 2])
    ax.set_axis_off()
    panel_label(ax, "C", x=-0.04, y=1.02)
    ax.text(0.02, 0.95, "Dominant-fragment sensitivity", transform=ax.transAxes, fontsize=11, fontweight="bold")
    rounded_box(ax, (0.08, 0.72), 0.84, 0.16, f"{fragment_flips}/9 point-estimate\ndirections reverse", COLORS["orange_pale"], COLORS["orange"], fontsize=12, weight="bold")
    rounded_box(ax, (0.08, 0.49), 0.84, 0.14, "9/9 corrected inferences\nremain inconclusive", COLORS["blue_pale"], COLORS["navy"], fontsize=10.5, weight="bold")
    for i in range(9):
        row, col = divmod(i, 3)
        xx = 0.25 + col * 0.25
        yy = 0.34 - row * 0.12
        m = float(fragment.iloc[i]["main_mean_effect"])
        p = float(fragment.iloc[i]["parent_mean_effect"])
        flipped = np.sign(m) != np.sign(p)
        ax.add_patch(FancyArrowPatch((xx - 0.055, yy), (xx + 0.055, yy), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=10, linewidth=1.5, color=COLORS["orange"] if flipped else COLORS["gray_mid"]))
    ax.text(0.5, 0.035, "Stable inference ≠ stable point estimate", transform=ax.transAxes, ha="center", fontsize=9.3, fontweight="bold", color=COLORS["orange_dark"])

    fig.suptitle("The scientific claim changes at different levels when benchmark-construction rules are perturbed", fontsize=14, fontweight="bold", y=1.01)
    save(fig, "figure6_claim_stability_map_v3")


def figureS1_dataset_construction() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.7))
    names = list(RAW_N)
    raw = np.array([RAW_N[d] for d in names], dtype=float)
    final = np.array([DATASET_N[d] for d in names], dtype=float)
    x = np.arange(len(names))
    width = 0.34
    ax.bar(x - width / 2, raw, width, label="Raw rows", color=COLORS["gray_mid"])
    ax.bar(x + width / 2, final, width, label="Final clean rows", color=COLORS["teal"])
    ax.set_yscale("log")
    ax.set_xticks(x, names)
    ax.set_ylabel("Rows (log scale)")
    ax.set_title("Audited raw-to-clean molecular-data construction", fontweight="bold")
    ax.grid(axis="y", color=COLORS["gray_light"], linewidth=0.7)
    ax.legend(frameon=False)
    for xi, r, f in zip(x, raw, final):
        removed = int(r - f)
        if removed > 0:
            ax.text(xi, max(r, f) * 1.16, f"−{removed}", ha="center", fontsize=8.0, color=COLORS["orange_dark"])
    save(fig, "figureS1_dataset_construction_v3")


def figureS2_budget_semantics() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0))
    for ax, dataset, panel in zip(axes, ("ESOL", "FreeSolv"), ("A", "B")):
        panel_label(ax, panel)
        for label, source, color, marker in [
            ("single-group", BUDGET_SINGLE_GROUP[dataset], COLORS["navy"], "o"),
            ("singleton", BUDGET_SINGLETON[dataset], COLORS["orange"], "s"),
        ]:
            x = np.array(list(source.keys()), dtype=float)
            y = np.array(list(source.values()), dtype=float)
            ax.plot(x, y, marker=marker, linewidth=1.8, color=color, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Candidate budget")
        ax.set_ylabel("Mean balanced target gap")
        ax.set_title(dataset, loc="left", fontweight="bold")
        ax.grid(color=COLORS["gray_light"], linewidth=0.7)
        ax.legend(frameon=False)
    fig.suptitle("Candidate-budget behavior differs under alternative acyclic-scaffold semantics", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "figureS2_budget_semantics_v3")


def figureS3_multicomponent_audit() -> None:
    metrics = ["Multi-component", "Scaffold changed", "Similarity < 0.90"]
    frames = [MULTICOMPONENT, SCAFFOLD_CHANGED, SIM_LT_090]
    names = ["BBBP", "ClinTox", "HIV"]
    matrix = np.array([[frame[name] for frame in frames] for name in names], dtype=float)
    normalized = np.log10(matrix + 1.0)

    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    image = ax.imshow(normalized, aspect="auto", cmap="GnBu")
    ax.set_xticks(np.arange(len(metrics)), metrics)
    ax.set_yticks(np.arange(len(names)), names)
    ax.set_title("Disconnected-component structural-audit summary", fontweight="bold")
    for i in range(len(names)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{int(matrix[i, j]):,}", ha="center", va="center", fontsize=10, color=COLORS["ink"])
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("log10(count + 1)")
    fig.tight_layout()
    save(fig, "figureS3_multicomponent_audit_v3")


def main() -> None:
    print("Building redesigned Paper 1 manuscript assets from frozen v3 result tables")
    figure1_framework()
    figure2_primary()
    figure3_acyclic()
    figure4_fragment()
    figure5_budget_audit()
    figure6_claim_stability()
    figureS1_dataset_construction()
    figureS2_budget_semantics()
    figureS3_multicomponent_audit()
    print("\nMANUSCRIPT ASSETS V3 VISUAL REFRESH COMPLETED")


if __name__ == "__main__":
    main()
