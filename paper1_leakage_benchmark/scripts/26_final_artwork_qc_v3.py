from __future__ import annotations

"""Final artwork-only QC for Paper 1.

This stage intentionally does not touch scientific results.  It redraws the four
remaining figures that showed minor text/panel-label collisions in the compiled
submission PDF: Figures 1, 4, 5, and 6.  The layouts are defined at the final
7.15-inch publication width so text is never rescued by post-hoc downscaling.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper1_leakage_benchmark"
TABLES = PAPER / "results" / "tables"
PARENT = PAPER / "results" / "parent_fragment_sensitivity_v3" / "tables"
FIG = PAPER / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

PRIMARY = TABLES / "primary_inference_summary_v3.csv"
PARENT_CMP = PARENT / "parent_fragment_vs_main_comparison_v3.csv"
COLLATERAL = TABLES / "q1_collateral_partition_diagnostics_v3.csv"
MEAN_ONLY = TABLES / "q1_mean_only_regression_summary_v3.csv"

DATASETS = ["BACE", "BBBP", "ClinTox", "HIV", "ESOL", "FreeSolv"]
CLS = ["BACE", "BBBP", "ClinTox", "HIV"]
REG = ["ESOL", "FreeSolv"]
N = {"BACE": 1513, "BBBP": 1965, "ClinTox": 1442, "HIV": 41120, "ESOL": 1117, "FreeSolv": 642}
TEST_N = {"BACE": 303, "BBBP": 393, "ClinTox": 288, "HIV": 8224, "ESOL": 223, "FreeSolv": 128}
SCAFF_CHANGED = {"BBBP": 5, "ClinTox": 1, "HIV": 235}
SIM090 = {"BBBP": 18, "ClinTox": 5, "HIV": 640}
CONFLICT = {"BBBP": 1, "ClinTox": 1, "HIV": 17}
MODEL_ORDER = {"LR": 0, "Ridge": 0, "RF": 1, "XGB": 2}
DATASET_ORDER = {d: i for i, d in enumerate(DATASETS)}

BUDGET_SINGLE = {
    "ESOL": {3000: 0.034622, 5000: 0.018366, 10000: 0.010563, 20000: 0.001787},
    "FreeSolv": {3000: 1.076491, 5000: 1.044452, 10000: 0.787491, 20000: 0.686528},
}
BUDGET_SINGLETON = {
    "ESOL": {100: 0.003687, 300: 0.001535, 500: 0.000600, 1000: 0.000600, 3000: 0.000301, 5000: 0.000194},
    "FreeSolv": {100: 0.019467, 300: 0.006230, 500: 0.005645, 1000: 0.005416, 3000: 0.001609, 5000: 0.000471},
}

C = {
    "ink": "#20313A", "navy": "#315B73", "navy2": "#24485D",
    "teal": "#2B8C82", "teal2": "#176B64", "orange": "#D58A43",
    "orange2": "#A85F28", "pale_blue": "#EAF1F5", "pale_teal": "#E8F3EF",
    "pale_orange": "#FBF0E5", "pale_gray": "#F4F6F7", "gray": "#6D7A81",
    "mid": "#B8C2C7", "grid": "#E5EAEC", "white": "#FFFFFF",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.4, "axes.titlesize": 9.0,
    "axes.labelsize": 8.4, "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
    "legend.fontsize": 6.7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": C["mid"], "text.color": C["ink"], "axes.labelcolor": C["ink"],
    "xtick.color": C["ink"], "ytick.color": C["ink"], "pdf.fonttype": 42,
    "ps.fonttype": 42, "figure.facecolor": "white", "savefig.facecolor": "white",
})


def need(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(FIG / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(FIG / f"{stem}.pdf")


def panel(ax: plt.Axes, letter: str, title: str, *, letter_x: float = -0.15, title_x: float = 0.02) -> None:
    """Place panel labels outside tick-label territory and titles separately."""
    ax.text(letter_x, 1.105, letter, transform=ax.transAxes, fontsize=10.8,
            fontweight="bold", ha="left", va="top", clip_on=False)
    ax.text(title_x, 1.055, title, transform=ax.transAxes, fontsize=8.7,
            fontweight="bold", ha="left", va="bottom", clip_on=False)


def box(ax: plt.Axes, xy, w, h, text, fc, ec, fs=7.0, bold=False):
    patch = FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.010,rounding_size=0.018",
        transform=ax.transAxes, facecolor=fc, edgecolor=ec, linewidth=0.9,
    )
    ax.add_patch(patch)
    return ax.text(xy[0] + w / 2, xy[1] + h / 2, text, transform=ax.transAxes,
                   ha="center", va="center", fontsize=fs,
                   fontweight="bold" if bold else "normal", linespacing=1.06)


def assert_no_text_overlap(fig: plt.Figure, left, right, label: str) -> None:
    """Fail the build if paired card labels overlap at final publication size."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if left.get_window_extent(renderer=renderer).overlaps(right.get_window_extent(renderer=renderer)):
        raise AssertionError(f"Artwork text collision detected: {label}")


def arrow(ax: plt.Axes, start, end, color=None) -> None:
    ax.add_patch(FancyArrowPatch(
        start, end, transform=ax.transAxes, arrowstyle="-|>", mutation_scale=10,
        linewidth=1.0, color=color or C["gray"],
    ))


def clean(ax: plt.Axes, axis="x") -> None:
    ax.grid(axis=axis, color=C["grid"], lw=0.7, zorder=0)


def primary_frame() -> pd.DataFrame:
    df = pd.read_csv(need(PRIMARY), keep_default_na=False)
    df["do"] = df["dataset"].map(DATASET_ORDER)
    df["mo"] = df["model"].map(MODEL_ORDER)
    return df.sort_values(["do", "mo"]).reset_index(drop=True)


def figure1() -> None:
    fig = plt.figure(figsize=(7.15, 5.25))
    gs = fig.add_gridspec(2, 2, wspace=0.38, hspace=0.48)

    ax = fig.add_subplot(gs[0, 0]); panel(ax, "A", "Audited molecular universe", letter_x=-0.10)
    y = np.arange(6); vals = [N[d] for d in DATASETS]
    cols = [C["navy"] if d in CLS else C["teal"] for d in DATASETS]
    ax.barh(y, vals, color=cols, height=0.56, zorder=2)
    ax.set_yticks(y, DATASETS); ax.invert_yaxis(); ax.set_xscale("log"); ax.set_xlabel("Clean molecules")
    clean(ax, "x")
    for yy, v in zip(y, vals):
        ax.text(v * 1.07, yy, f"{v:,}", va="center", fontsize=6.5)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=C["navy"], label="Classification"),
                       Rectangle((0, 0), 1, 1, color=C["teal"], label="Regression")],
              frameon=False, loc="lower right")

    ax = fig.add_subplot(gs[0, 1]); ax.set_axis_off(); panel(ax, "B", "Exact-size target-mean perturbation", letter_x=-0.08)
    box(ax, (0.25, 0.77), 0.50, 0.13, "Target-blind candidate pool", C["pale_blue"], C["navy"], 7.1, True)
    arrow(ax, (0.50, 0.76), (0.50, 0.65))
    box(ax, (0.20, 0.51), 0.60, 0.13, "Same realized test size", C["white"], C["mid"], 7.0, True)
    arrow(ax, (0.50, 0.50), (0.25, 0.38), C["gray"])
    arrow(ax, (0.50, 0.50), (0.75, 0.38), C["teal2"])
    b_left = box(ax, (0.02, 0.19), 0.40, 0.17, "Size-matched\ntarget-blind baseline", C["pale_gray"], C["mid"], 6.9, True)
    b_right = box(ax, (0.58, 0.19), 0.40, 0.17, "Target-mean-aware\nsame $n_{test}$", C["pale_teal"], C["teal"], 6.9, True)
    assert_no_text_overlap(fig, b_left, b_right, "Figure 1B paired selection cards")
    ax.text(0.50, 0.06, "Fixed before outcomes: seed · scaffold rule · candidate budget",
            transform=ax.transAxes, ha="center", fontsize=6.25, color=C["gray"])

    ax = fig.add_subplot(gs[1, 0]); ax.set_axis_off(); panel(ax, "C", "Pre-outcome freeze and inference", letter_x=-0.08)
    steps = [
        ("Budget\nfrozen", C["pale_orange"], C["orange"]),
        ("Manifest\n+ hash", C["pale_blue"], C["navy"]),
        ("Model\nfit", C["pale_gray"], C["mid"]),
        ("Paired\neffect", C["pale_teal"], C["teal"]),
    ]
    for i, (txt, fc, ec) in enumerate(steps):
        x = 0.01 + i * 0.247
        box(ax, (x, 0.62), 0.20, 0.18, txt, fc, ec, 6.8, True)
        if i < 3:
            arrow(ax, (x + 0.205, 0.71), (x + 0.238, 0.71))
    box(ax, (0.04, 0.30), 0.27, 0.13, "20 unique\npartition pairs", C["white"], C["mid"], 6.6)
    box(ax, (0.365, 0.30), 0.27, 0.13, "10,000 paired\nbootstrap draws", C["white"], C["mid"], 6.6)
    box(ax, (0.69, 0.30), 0.27, 0.13, "Wilcoxon +\nHolm", C["white"], C["mid"], 6.6)
    ax.text(0.50, 0.10, "Inferential $N$ = unique partition pairs, not model seeds",
            transform=ax.transAxes, ha="center", fontsize=6.45, color=C["gray"])

    ax = fig.add_subplot(gs[1, 1]); ax.set_axis_off(); panel(ax, "D", "Predeclared protocol sensitivities", letter_x=-0.08)
    d_left = box(ax, (0.01, 0.55), 0.36, 0.23, "Acyclic semantics\nsingle-group ↔\nsingleton", C["pale_blue"], C["navy"], 6.6, True)
    d_right = box(ax, (0.63, 0.55), 0.36, 0.23, "Record representation\nfull record ↔\nfragment", C["pale_orange"], C["orange"], 6.6, True)
    assert_no_text_overlap(fig, d_left, d_right, "Figure 1D sensitivity cards")
    arrow(ax, (0.19, 0.54), (0.40, 0.36), C["navy"])
    arrow(ax, (0.81, 0.54), (0.60, 0.36), C["orange"])
    box(ax, (0.25, 0.19), 0.50, 0.16, "Does the scientific claim survive?", C["pale_teal"], C["teal"], 7.0, True)
    ax.text(0.50, 0.07, "Disagreements are reported, not resolved post hoc.",
            transform=ax.transAxes, ha="center", fontsize=6.3, color=C["gray"])

    fig.suptitle("Benchmark construction as a controlled chemometric measurement process",
                 fontsize=10.6, fontweight="bold", y=0.995)
    save(fig, "figure1_audit_framework_v3")


def figure4() -> None:
    comp = pd.read_csv(need(PARENT_CMP), keep_default_na=False)
    main_col = next(c for c in ["main_mean_effect", "primary_mean_effect", "source_faithful_mean_effect"] if c in comp.columns)
    frag_col = next(c for c in ["parent_mean_effect", "fragment_mean_effect", "dominant_fragment_mean_effect"] if c in comp.columns)
    comp["do"] = comp["dataset"].map(DATASET_ORDER); comp["mo"] = comp["model"].map(MODEL_ORDER)
    comp = comp.sort_values(["do", "mo"])

    fig = plt.figure(figsize=(7.15, 5.35))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.72, 1.34], wspace=0.50, hspace=0.48)

    ax = fig.add_subplot(gs[0, 0]); ax.set_axis_off(); panel(ax, "A", "Representation perturbation", letter_x=-0.08)
    ax.scatter([0.14, 0.22, 0.30], [0.56, 0.62, 0.50], s=[75, 45, 28],
               c=[C["teal"], "#75C0C8", C["orange"]], transform=ax.transAxes, clip_on=False)
    arrow(ax, (0.39, 0.57), (0.60, 0.57), C["orange"])
    ax.scatter([0.75], [0.57], s=125, c=[C["teal"]], transform=ax.transAxes, clip_on=False)
    ax.text(0.22, 0.78, "source-faithful", transform=ax.transAxes, ha="center", fontsize=7.0, fontweight="bold")
    ax.text(0.75, 0.78, "dominant fragment", transform=ax.transAxes, ha="center", fontsize=7.0, fontweight="bold")
    box(ax, (0.14, 0.14), 0.72, 0.16, "algorithmic sensitivity; not a lossless formatting step",
        C["pale_orange"], C["orange"], 6.2, True)

    ax = fig.add_subplot(gs[0, 1])
    # Keep the B label farther left than the logarithmic y-tick labels.
    panel(ax, "B", "Structural consequences", letter_x=-0.20, title_x=0.02)
    x = np.arange(3); w = 0.24; ds = ["BBBP", "ClinTox", "HIV"]
    ax.bar(x - w, [SCAFF_CHANGED[d] for d in ds], w, color=C["navy"], label="Scaffold changed")
    ax.bar(x, [SIM090[d] for d in ds], w, color=C["orange"], label="Similarity < 0.90")
    ax.bar(x + w, [CONFLICT[d] for d in ds], w, color=C["teal"], label="Conflict groups")
    ax.set_yscale("symlog", linthresh=1); ax.set_xticks(x, ds); ax.set_ylabel("Count"); clean(ax, "y")
    ax.legend(frameon=False, ncol=1, loc="upper left")

    ax = fig.add_subplot(gs[1, :]); panel(ax, "C", "Effect direction is representation-sensitive", letter_x=-0.07)
    y = np.arange(len(comp)); a = comp[main_col].astype(float).to_numpy(); b = comp[frag_col].astype(float).to_numpy()
    labels = [f"{r.dataset} · {r.model}" for r in comp.itertuples(index=False)]
    for yy, x1, x2 in zip(y, a, b):
        ax.plot([x1, x2], [yy, yy], color=C["orange"], lw=1.0, zorder=1)
    ax.scatter(a, y, s=16, color=C["navy"], zorder=2)
    ax.scatter(b, y, s=18, marker="s", color=C["orange"], zorder=2)
    ax.axvline(0, color=C["gray"], ls="--", lw=0.8)
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlabel("AUC effect: balanced − size-matched"); clean(ax, "x")
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C["navy"], markeredgecolor=C["navy"], label="source-faithful primary"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=C["orange"], markeredgecolor=C["orange"], label="dominant-fragment sensitivity"),
    ], frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)

    fig.suptitle("Disconnected-component representation changes benchmark composition and point estimates",
                 fontsize=10.1, fontweight="bold", y=0.995)
    fig.subplots_adjust(top=0.87, bottom=0.16)
    save(fig, "figure4_dominant_fragment_sensitivity_v3")


def figure5() -> None:
    fig, axs = plt.subplots(2, 2, figsize=(7.15, 5.25),
                            gridspec_kw={"wspace": 0.50, "hspace": 0.56})
    fig.subplots_adjust(top=0.88, bottom=0.10)

    for j, ds in enumerate(REG):
        ax = axs[0, j]; panel(ax, "A" if j == 0 else "B", f"{ds} · single-group", letter_x=-0.14)
        x = np.array(list(BUDGET_SINGLE[ds]), dtype=float)
        y = np.array(list(BUDGET_SINGLE[ds].values()), dtype=float)
        color = C["teal"] if ds == "ESOL" else C["navy"]
        ax.plot(x, y, marker="o", ms=3.2, lw=1.2, color=color)
        ax.fill_between(x, y, 0, alpha=0.08, color=color)
        ax.axvline(20000, color=C["orange"], ls="--", lw=0.8)
        ax.annotate("frozen cap\n20,000", xy=(20000, y[-1]),
                    xytext=(13000, y[-1] + 0.25 * (max(y) - min(y))), fontsize=6.3,
                    color=C["orange2"], arrowprops=dict(arrowstyle="->", lw=0.7, color=C["orange2"]))
        ax.set_xlabel("Candidate budget"); ax.set_ylabel("Mean target-mean gap"); clean(ax, "both")

    ax = axs[1, 0]; panel(ax, "C", "Singleton budget trajectory", letter_x=-0.14)
    for ds, col, mark in [("ESOL", C["teal"], "o"), ("FreeSolv", C["orange"], "s")]:
        x = np.array(list(BUDGET_SINGLETON[ds]), dtype=float)
        y = np.array(list(BUDGET_SINGLETON[ds].values()), dtype=float)
        ax.plot(x, y / y[0], marker=mark, ms=3, lw=1.1, color=col, label=ds)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Candidate budget")
    ax.set_ylabel("Gap relative to 100-candidate value"); clean(ax, "both"); ax.legend(frameon=False)

    ax = axs[1, 1]; panel(ax, "D", "Exact test-size pairing", letter_x=-0.14)
    yy = np.arange(6); vals = np.array([TEST_N[d] for d in DATASETS], dtype=float)
    ax.scatter(vals, yy, s=26, color=C["navy"], label="size-matched", zorder=3)
    ax.scatter(vals, yy, s=15, marker="s", facecolor=C["white"], edgecolor=C["teal"],
               linewidth=1.0, label="target-balanced", zorder=4)
    ax.set_yticks(yy, DATASETS); ax.invert_yaxis(); ax.set_xscale("log"); ax.set_xlabel("Test molecules"); clean(ax, "x")
    ax.legend(frameon=False, loc="lower right")
    for x0, y0 in zip(vals, yy):
        ax.text(x0 * 1.08, y0, f"{int(x0):,}", va="center", fontsize=6.3)

    fig.suptitle("Candidate-search budget is a frozen benchmark-construction hyperparameter",
                 fontsize=10.2, fontweight="bold", y=0.985)
    save(fig, "figure5_candidate_budget_audit_v3")


def figure6() -> None:
    primary = primary_frame()
    mean_only = pd.read_csv(need(MEAN_ONLY), keep_default_na=False)
    collateral = pd.read_csv(need(COLLATERAL), keep_default_na=False)
    mean_only = mean_only.loc[mean_only["freeze_label"].eq("main_regression")].copy()

    fig = plt.figure(figsize=(7.15, 5.65))
    gs = fig.add_gridspec(2, 2, wspace=0.62, hspace=0.54)
    fig.subplots_adjust(top=0.87, bottom=0.11)

    ax = fig.add_subplot(gs[0, 0]); panel(ax, "A", "Mean-only control vs learned models", letter_x=-0.16)
    rows = []
    for ds in REG:
        m = mean_only.loc[mean_only["dataset"].eq(ds)].iloc[0]
        rows.append((ds, "Mean-only", float(m.mean_effect_size_minus_balanced_rmse), float(m.bootstrap_ci_low), float(m.bootstrap_ci_high), C["orange"]))
        for model in ["Ridge", "RF", "XGB"]:
            r = primary.loc[(primary["dataset"].eq(ds)) & (primary["model"].eq(model))].iloc[0]
            rows.append((ds, model, float(r.mean_effect), float(r.bootstrap_ci_low), float(r.bootstrap_ci_high), C["teal2"]))
    labels = []
    for yy, (ds, model, effect, lo, hi, color) in enumerate(rows):
        marker = "o" if model == "Mean-only" else "s"
        ax.errorbar(effect, yy, xerr=[[effect - lo], [hi - effect]], fmt=marker, ms=3.2,
                    color=color, lw=0.9, capsize=1.8, zorder=3)
        labels.append(f"{ds} · {model}")
    ax.axvline(0, color=C["gray"], ls="--", lw=0.8)
    ax.set_yticks(np.arange(len(labels)), labels); ax.invert_yaxis(); ax.set_xlabel("RMSE improvement: size − balanced"); clean(ax, "x")

    ax = fig.add_subplot(gs[0, 1]); panel(ax, "B", "Target-mean gap reduction", letter_x=-0.18)
    rng = np.random.default_rng(3)
    for i, ds in enumerate(DATASETS):
        g = collateral.loc[collateral["dataset"].eq(ds)].copy()
        s = g["size_abs_target_mean_gap"].to_numpy(float); b = g["balanced_abs_target_mean_gap"].to_numpy(float)
        ratio = np.divide(b, s, out=np.full_like(b, np.nan), where=s > 0)
        finite = ratio[np.isfinite(ratio)]
        floor = max(np.min(finite[finite > 0]) * 0.35, 1e-6) if np.any(finite > 0) else 1e-6
        plotted = np.where(finite > 0, finite, floor)
        x = np.full(len(plotted), i) + rng.normal(0, 0.040, len(plotted))
        ax.scatter(x, plotted, s=10, alpha=0.58, color=C["teal"], edgecolors="none")
        ax.scatter([i], [np.median(plotted)], s=28, marker="D", color=C["navy2"], zorder=4)
    ax.axhline(1, color=C["gray"], ls="--", lw=0.8)
    ax.set_yscale("log"); ax.set_xticks(range(6), DATASETS, rotation=25, ha="right")
    ax.set_ylabel("Balanced / size target-mean gap"); clean(ax, "y")

    ax = fig.add_subplot(gs[1, 0]); panel(ax, "C", "Largest-scaffold fraction", letter_x=-0.18)
    for i, ds in enumerate(DATASETS):
        g = collateral.loc[collateral["dataset"].eq(ds)]
        delta = g["delta_balanced_minus_size_largest_test_scaffold_fraction"].to_numpy(float)
        x = np.full(len(delta), i) + rng.normal(0, 0.040, len(delta))
        ax.scatter(x, delta, s=10, alpha=0.58, color=C["orange"], edgecolors="none")
        ax.scatter([i], [np.mean(delta)], s=28, marker="D", color=C["navy2"], zorder=4)
    ax.axhline(0, color=C["gray"], ls="--", lw=0.8)
    ax.set_xticks(range(6), DATASETS, rotation=25, ha="right"); ax.set_ylabel("Balanced − size"); clean(ax, "y")

    ax = fig.add_subplot(gs[1, 1]); panel(ax, "D", "Effective scaffold number", letter_x=-0.20)
    for i, ds in enumerate(DATASETS):
        g = collateral.loc[collateral["dataset"].eq(ds)]
        s = g["size_effective_test_scaffolds"].to_numpy(float); b = g["balanced_effective_test_scaffolds"].to_numpy(float)
        valid = (s > 0) & (b > 0) & np.isfinite(s) & np.isfinite(b)
        values = np.log2(b[valid] / s[valid])
        x = np.full(len(values), i) + rng.normal(0, 0.040, len(values))
        ax.scatter(x, values, s=10, alpha=0.58, color=C["navy"], edgecolors="none")
        ax.scatter([i], [np.mean(values)], s=28, marker="D", color=C["teal2"], zorder=4)
    ax.axhline(0, color=C["gray"], ls="--", lw=0.8)
    ax.set_xticks(range(6), DATASETS, rotation=25, ha="right"); ax.set_ylabel("log2(balanced / size)"); clean(ax, "y")

    fig.suptitle("Target-mean-aware selection changes RMSE difficulty and other benchmark properties",
                 fontsize=10.0, fontweight="bold", y=0.985)
    save(fig, "figure6_collateral_diagnostics_v3")


def main() -> None:
    for path in [PRIMARY, PARENT_CMP, COLLATERAL, MEAN_ONLY]:
        need(path)
    figure1(); figure4(); figure5(); figure6()
    print("FINAL ARTWORK QC: PASS")


if __name__ == "__main__":
    main()
