from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
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
SUPPORT = TABLES / "supporting_metric_effects_v3.csv"
SEED = TABLES / "q1_model_seed_summary_v3.csv"

DATASETS = ["BACE", "BBBP", "ClinTox", "HIV", "ESOL", "FreeSolv"]
CLS = ["BACE", "BBBP", "ClinTox", "HIV"]
REG = ["ESOL", "FreeSolv"]
N = {"BACE": 1513, "BBBP": 1965, "ClinTox": 1442, "HIV": 41120, "ESOL": 1117, "FreeSolv": 642}
MULTI = {"BBBP": 105, "ClinTox": 14, "HIV": 3086}
SCAFF_CHANGED = {"BBBP": 5, "ClinTox": 1, "HIV": 235}
SIM090 = {"BBBP": 18, "ClinTox": 5, "HIV": 640}
CONFLICT = {"BBBP": 1, "ClinTox": 1, "HIV": 17}
MODEL_ORDER = {"LR": 0, "Ridge": 0, "RF": 1, "XGB": 2}
DATASET_ORDER = {d: i for i, d in enumerate(DATASETS)}

C = {
    "ink": "#20313A", "navy": "#315B73", "navy2": "#24485D",
    "teal": "#2B8C82", "teal2": "#176B64", "orange": "#D58A43",
    "orange2": "#A85F28", "pale_blue": "#EAF1F5", "pale_teal": "#E8F3EF",
    "pale_orange": "#FBF0E5", "pale_gray": "#F4F6F7", "gray": "#6D7A81",
    "mid": "#B8C2C7", "grid": "#E5EAEC", "white": "#FFFFFF",
    "purple": "#7A6F9B", "olive": "#728B65",
}
DATASET_COLORS = {
    "BACE": C["navy"], "BBBP": C["teal"], "ClinTox": C["orange"], "HIV": C["purple"],
    "ESOL": C["navy"], "FreeSolv": C["teal"],
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
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(FIG / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(FIG / f"{stem}.pdf")


def panel(ax: plt.Axes, letter: str, title: str | None = None) -> None:
    ax.text(-0.08, 1.07, letter, transform=ax.transAxes, fontsize=10.9, fontweight="bold", va="top")
    if title:
        ax.text(0.04, 1.035, title, transform=ax.transAxes, fontsize=8.7, fontweight="bold", va="bottom")


def box(ax: plt.Axes, xy, w, h, text, fc, ec, fs=7.0, bold=False) -> None:
    patch = FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.010,rounding_size=0.018",
        transform=ax.transAxes, facecolor=fc, edgecolor=ec, linewidth=0.9,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + w / 2, xy[1] + h / 2, text, transform=ax.transAxes,
        ha="center", va="center", fontsize=fs,
        fontweight="bold" if bold else "normal", linespacing=1.08,
    )


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
    gs = fig.add_gridspec(2, 2, wspace=0.34, hspace=0.46)

    ax = fig.add_subplot(gs[0, 0]); panel(ax, "A", "Audited molecular universe")
    y = np.arange(6); vals = [N[d] for d in DATASETS]
    cols = [C["navy"] if d in CLS else C["teal"] for d in DATASETS]
    ax.barh(y, vals, color=cols, height=0.56, zorder=2)
    ax.set_yticks(y, DATASETS); ax.invert_yaxis(); ax.set_xscale("log"); ax.set_xlabel("Clean molecules")
    clean(ax, "x")
    for yy, v in zip(y, vals):
        ax.text(v * 1.07, yy, f"{v:,}", va="center", fontsize=6.5)
    ax.legend(
        handles=[Rectangle((0, 0), 1, 1, color=C["navy"], label="Classification"),
                 Rectangle((0, 0), 1, 1, color=C["teal"], label="Regression")],
        frameon=False, loc="lower right",
    )

    ax = fig.add_subplot(gs[0, 1]); ax.set_axis_off(); panel(ax, "B", "Exact-size target-mean perturbation")
    box(ax, (0.25, 0.76), 0.50, 0.13, "Target-blind candidate pool", C["pale_blue"], C["navy"], 7.2, True)
    arrow(ax, (0.50, 0.75), (0.50, 0.64))
    box(ax, (0.22, 0.50), 0.56, 0.13, "Candidates with the same realized test size", C["white"], C["mid"], 6.8, True)
    arrow(ax, (0.50, 0.49), (0.28, 0.36), C["gray"]); arrow(ax, (0.50, 0.49), (0.72, 0.36), C["teal2"])
    box(ax, (0.03, 0.19), 0.43, 0.16, "Size-matched baseline\n(target-blind choice)", C["pale_gray"], C["mid"], 6.9, True)
    box(ax, (0.54, 0.19), 0.43, 0.16, "Lowest target-mean gap\nat identical $n_{test}$", C["pale_teal"], C["teal"], 6.9, True)
    ax.text(0.50, 0.06, "Fixed before outcomes: seed · scaffold rule · candidate budget", transform=ax.transAxes,
            ha="center", fontsize=6.4, color=C["gray"])

    ax = fig.add_subplot(gs[1, 0]); ax.set_axis_off(); panel(ax, "C", "Pre-outcome freeze and inference")
    steps = [
        ("Budget\nfrozen", C["pale_orange"], C["orange"]),
        ("Manifest\n+ hash", C["pale_blue"], C["navy"]),
        ("Model\nfit", C["pale_gray"], C["mid"]),
        ("Paired\neffect", C["pale_teal"], C["teal"]),
    ]
    for i, (txt, fc, ec) in enumerate(steps):
        x = 0.01 + i * 0.247
        box(ax, (x, 0.62), 0.20, 0.18, txt, fc, ec, 6.9, True)
        if i < 3:
            arrow(ax, (x + 0.205, 0.71), (x + 0.238, 0.71))
    box(ax, (0.04, 0.30), 0.27, 0.13, "20 unique\npartition pairs", C["white"], C["mid"], 6.7)
    box(ax, (0.365, 0.30), 0.27, 0.13, "10,000 paired\nbootstrap draws", C["white"], C["mid"], 6.7)
    box(ax, (0.69, 0.30), 0.27, 0.13, "Wilcoxon +\nHolm", C["white"], C["mid"], 6.7)
    ax.text(0.50, 0.10, "Inferential $N$ = unique partition pairs, not model seeds", transform=ax.transAxes,
            ha="center", fontsize=6.6, color=C["gray"])

    ax = fig.add_subplot(gs[1, 1]); ax.set_axis_off(); panel(ax, "D", "Predeclared protocol sensitivities")
    box(ax, (0.03, 0.56), 0.42, 0.21, "Acyclic scaffold semantics\nsingle-group ↔ singleton", C["pale_blue"], C["navy"], 6.8, True)
    box(ax, (0.55, 0.56), 0.42, 0.21, "Molecular-record representation\nsource-faithful ↔ dominant fragment", C["pale_orange"], C["orange"], 6.5, True)
    arrow(ax, (0.24, 0.54), (0.40, 0.36), C["navy"]); arrow(ax, (0.76, 0.54), (0.60, 0.36), C["orange"])
    box(ax, (0.25, 0.19), 0.50, 0.16, "Does the scientific claim survive?", C["pale_teal"], C["teal"], 7.0, True)
    ax.text(0.50, 0.07, "Disagreements are reported, not resolved post hoc.", transform=ax.transAxes,
            ha="center", fontsize=6.4, color=C["gray"])

    fig.suptitle("Benchmark construction as a controlled chemometric measurement process",
                 fontsize=10.6, fontweight="bold", y=0.995)
    save(fig, "figure1_audit_framework_v3")


def forest(ax: plt.Axes, df: pd.DataFrame, task: str, title: str, letter: str) -> None:
    panel(ax, letter, title)
    y = np.arange(len(df)); eff = df["mean_effect"].to_numpy(float)
    lo = df["bootstrap_ci_low"].to_numpy(float); hi = df["bootstrap_ci_high"].to_numpy(float)
    labels = [f"{r.dataset} · {r.model}" for r in df.itertuples(index=False)]
    for i in range(len(df)):
        if i % 3 == 0:
            ax.axhspan(i - 0.48, min(i + 2.48, len(df) - 0.52), color=C["pale_gray"], zorder=0)
    colors = [C["teal2"] if str(x) == "target_balanced_better" else C["navy"] for x in df["inference_label"]]
    for yy, e, l, h, col in zip(y, eff, lo, hi, colors):
        ax.errorbar(e, yy, xerr=[[e - l], [h - e]], fmt="o", ms=3.2, lw=1.0,
                    capsize=2.0, color=col, zorder=3)
    ax.axvline(0, color=C["gray"], ls="--", lw=0.9)
    ax.set_yticks(y, labels); ax.invert_yaxis(); clean(ax, "x")
    ax.set_xlabel("AUC effect: balanced − size-matched" if task == "classification"
                  else "RMSE improvement: size-matched − balanced")


def figure2() -> None:
    df = primary_frame(); cls = df[df["task_type"].eq("classification")]; reg = df[df["task_type"].eq("regression")]
    fig, axs = plt.subplots(1, 2, figsize=(7.15, 4.45),
                            gridspec_kw={"width_ratios": [1.12, 0.88], "wspace": 0.47})
    forest(axs[0], cls, "classification", "Classification · 12 cells", "A")
    forest(axs[1], reg, "regression", "Regression · primary semantics", "B")
    axs[0].text(0.5, -0.18, "Classification: 0 / 12 supported", transform=axs[0].transAxes,
                ha="center", va="top", fontsize=7.0, fontweight="bold", color=C["navy2"],
                bbox=dict(boxstyle="round,pad=0.28", fc=C["pale_blue"], ec=C["mid"], lw=0.7))
    axs[1].text(0.5, -0.18, "Regression: 6 / 6 supported", transform=axs[1].transAxes,
                ha="center", va="top", fontsize=7.0, fontweight="bold", color=C["teal2"],
                bbox=dict(boxstyle="round,pad=0.28", fc=C["pale_teal"], ec=C["teal"], lw=0.7))
    fig.suptitle("Exact-size paired target-mean selection effects across 20 partition pairs",
                 fontsize=10.2, fontweight="bold", y=0.995)
    fig.subplots_adjust(bottom=0.18, top=0.87)
    save(fig, "figure2_primary_effects_v3")


def figure4() -> None:
    comp = pd.read_csv(need(PARENT_CMP), keep_default_na=False)
    main_col = next(c for c in ["main_mean_effect", "primary_mean_effect", "source_faithful_mean_effect"] if c in comp.columns)
    frag_col = next(c for c in ["parent_mean_effect", "fragment_mean_effect", "dominant_fragment_mean_effect"] if c in comp.columns)
    comp["do"] = comp["dataset"].map(DATASET_ORDER); comp["mo"] = comp["model"].map(MODEL_ORDER)
    comp = comp.sort_values(["do", "mo"])

    fig = plt.figure(figsize=(7.15, 5.35))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.72, 1.34], wspace=0.42, hspace=0.46)

    ax = fig.add_subplot(gs[0, 0]); ax.set_axis_off(); panel(ax, "A", "Representation perturbation")
    ax.scatter([0.14, 0.22, 0.30], [0.56, 0.62, 0.50], s=[75, 45, 28],
               c=[C["teal"], "#75C0C8", C["orange"]], transform=ax.transAxes, clip_on=False)
    arrow(ax, (0.39, 0.57), (0.60, 0.57), C["orange"])
    ax.scatter([0.75], [0.57], s=125, c=[C["teal"]], transform=ax.transAxes, clip_on=False)
    ax.text(0.22, 0.78, "source-faithful", transform=ax.transAxes, ha="center", fontsize=7.1, fontweight="bold")
    ax.text(0.75, 0.78, "dominant fragment", transform=ax.transAxes, ha="center", fontsize=7.1, fontweight="bold")
    box(ax, (0.15, 0.14), 0.70, 0.16, "algorithmic sensitivity; not a lossless formatting step",
        C["pale_orange"], C["orange"], 6.3, True)

    ax = fig.add_subplot(gs[0, 1]); panel(ax, "B", "Structural consequences")
    x = np.arange(3); w = 0.24; ds = ["BBBP", "ClinTox", "HIV"]
    ax.bar(x - w, [SCAFF_CHANGED[d] for d in ds], w, color=C["navy"], label="Scaffold changed")
    ax.bar(x, [SIM090[d] for d in ds], w, color=C["orange"], label="Similarity < 0.90")
    ax.bar(x + w, [CONFLICT[d] for d in ds], w, color=C["teal"], label="Conflict groups")
    ax.set_yscale("symlog", linthresh=1); ax.set_xticks(x, ds); ax.set_ylabel("Count"); clean(ax, "y")
    ax.legend(frameon=False, ncol=1, loc="upper left")

    ax = fig.add_subplot(gs[1, :]); panel(ax, "C", "Effect direction is representation-sensitive")
    y = np.arange(len(comp)); a = comp[main_col].astype(float).to_numpy(); b = comp[frag_col].astype(float).to_numpy()
    labels = [f"{r.dataset} · {r.model}" for r in comp.itertuples(index=False)]
    for yy, x1, x2 in zip(y, a, b):
        ax.plot([x1, x2], [yy, yy], color=C["orange"], lw=1.0, zorder=1)
    ax.scatter(a, y, s=16, color=C["navy"], zorder=2)
    ax.scatter(b, y, s=18, marker="s", color=C["orange"], zorder=2)
    ax.axvline(0, color=C["gray"], ls="--", lw=0.8)
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlabel("AUC effect: balanced − size-matched"); clean(ax, "x")
    ax.legend(
        handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=C["navy"], markeredgecolor=C["navy"], label="source-faithful primary"),
                 Line2D([0], [0], marker="s", color="none", markerfacecolor=C["orange"], markeredgecolor=C["orange"], label="dominant-fragment sensitivity")],
        frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
    )
    fig.suptitle("Disconnected-component representation changes benchmark composition and point estimates",
                 fontsize=10.1, fontweight="bold", y=0.995)
    fig.subplots_adjust(top=0.88, bottom=0.16)
    save(fig, "figure4_dominant_fragment_sensitivity_v3")


def figure6() -> None:
    primary = primary_frame()
    mean_only = pd.read_csv(need(MEAN_ONLY), keep_default_na=False)
    collateral = pd.read_csv(need(COLLATERAL), keep_default_na=False)
    mean_only = mean_only.loc[mean_only["freeze_label"].eq("main_regression")].copy()

    fig = plt.figure(figsize=(7.15, 5.65))
    gs = fig.add_gridspec(2, 2, wspace=0.54, hspace=0.50)
    fig.subplots_adjust(top=0.88, bottom=0.11)

    ax = fig.add_subplot(gs[0, 0]); panel(ax, "A", "Mean-only control vs learned models")
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
    ax.set_yticks(np.arange(len(labels)), labels); ax.invert_yaxis(); ax.set_xlabel("RMSE improvement: size − balanced")
    clean(ax, "x")

    ax = fig.add_subplot(gs[0, 1]); panel(ax, "B", "Target-mean gap reduction")
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

    ax = fig.add_subplot(gs[1, 0]); panel(ax, "C", "Largest-scaffold fraction")
    for i, ds in enumerate(DATASETS):
        g = collateral.loc[collateral["dataset"].eq(ds)]
        delta = g["delta_balanced_minus_size_largest_test_scaffold_fraction"].to_numpy(float)
        x = np.full(len(delta), i) + rng.normal(0, 0.040, len(delta))
        ax.scatter(x, delta, s=10, alpha=0.58, color=C["orange"], edgecolors="none")
        ax.scatter([i], [np.mean(delta)], s=28, marker="D", color=C["navy2"], zorder=4)
    ax.axhline(0, color=C["gray"], ls="--", lw=0.8)
    ax.set_xticks(range(6), DATASETS, rotation=25, ha="right"); ax.set_ylabel("Balanced − size"); clean(ax, "y")

    ax = fig.add_subplot(gs[1, 1]); panel(ax, "D", "Effective scaffold number")
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


def effect_matrix(df: pd.DataFrame, task: str, metrics: list[str], metric_labels: list[str]):
    sub = df.loc[df["task_type"].eq(task)].copy()
    sub["do"] = sub["dataset"].map(DATASET_ORDER); sub["mo"] = sub["model"].map(MODEL_ORDER)
    combos = sub[["dataset", "model", "do", "mo"]].drop_duplicates().sort_values(["do", "mo"])
    matrix, labels = [], []
    for r in combos.itertuples(index=False):
        labels.append(f"{r.dataset} · {r.model}"); row = []
        for metric in metrics:
            m = sub.loc[(sub["dataset"].eq(r.dataset)) & (sub["model"].eq(r.model)) & (sub["metric"].eq(metric))]
            if len(m) != 1:
                raise AssertionError(f"Missing supporting metric {task}/{r.dataset}/{r.model}/{metric}")
            value = float(m.iloc[0]["mean_effect_positive_is_balanced_better"])
            scale = float(sub.loc[sub["metric"].eq(metric), "mean_effect_positive_is_balanced_better"].abs().max())
            row.append(value / scale if scale > 0 else 0.0)
        matrix.append(row)
    return np.asarray(matrix), labels, metric_labels


def figure_s4() -> None:
    support = pd.read_csv(need(SUPPORT), keep_default_na=False)
    cls_mat, cls_rows, cls_cols = effect_matrix(
        support, "classification",
        ["roc_auc", "average_precision", "f1", "accuracy", "balanced_accuracy", "brier_score"],
        ["AUC", "AP", "F1", "Accuracy", "Bal. acc.", "Brier"],
    )
    reg_mat, reg_rows, reg_cols = effect_matrix(
        support, "regression", ["rmse", "mae", "r2"], ["RMSE", "MAE", "$R^2$"],
    )
    cmap = LinearSegmentedColormap.from_list("signed", [C["orange"], C["white"], C["teal"]])
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    fig, axs = plt.subplots(1, 2, figsize=(7.15, 4.45),
                            gridspec_kw={"width_ratios": [1.55, 0.75], "wspace": 0.40})
    im = axs[0].imshow(cls_mat, cmap=cmap, norm=norm, aspect="auto")
    axs[0].set_yticks(range(len(cls_rows)), cls_rows); axs[0].set_xticks(range(len(cls_cols)), cls_cols, rotation=25, ha="right")
    axs[0].set_title("A  Classification supporting metrics", loc="left", fontweight="bold", fontsize=8.7)
    axs[1].imshow(reg_mat, cmap=cmap, norm=norm, aspect="auto")
    axs[1].set_yticks(range(len(reg_rows)), reg_rows); axs[1].set_xticks(range(len(reg_cols)), reg_cols, rotation=25, ha="right")
    axs[1].set_title("B  Regression supporting metrics", loc="left", fontweight="bold", fontsize=8.7)
    for ax in axs:
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.025)
    cbar.set_label("Signed mean effect, normalized within metric")
    fig.suptitle("Supporting metrics reveal metric-dependent effect patterns without expanding the primary hypothesis family",
                 fontsize=9.3, fontweight="bold", y=0.99)
    fig.subplots_adjust(top=0.85, bottom=0.17)
    save(fig, "figureS4_supporting_metrics_v3")


def figure_s5() -> None:
    seed = pd.read_csv(need(SEED), keep_default_na=False)
    panels = [
        ("main_classification", "A  Primary classification"),
        ("main_regression", "B  Primary regression"),
        ("acyclic_singleton_sensitivity", "C  Singleton regression"),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(7.15, 3.75), gridspec_kw={"wspace": 0.40})
    for ax, (label, title) in zip(axs, panels):
        sub = seed.loc[seed["freeze_label"].eq(label)].copy()
        sub["do"] = sub["dataset"].map(DATASET_ORDER); sub["mo"] = sub["model"].map(MODEL_ORDER)
        combos = sub[["dataset", "model", "do", "mo"]].drop_duplicates().sort_values(["do", "mo"])
        datasets_here = []
        for r in combos.itertuples(index=False):
            g = sub.loc[(sub["dataset"].eq(r.dataset)) & (sub["model"].eq(r.model))].sort_values("model_seed")
            style = "-" if r.model == "RF" else "--"
            color = DATASET_COLORS[r.dataset]
            ax.plot(g["model_seed"], g["mean_effect"], marker="o", ms=2.8, lw=1.0,
                    linestyle=style, color=color)
            if r.dataset not in datasets_here:
                datasets_here.append(r.dataset)
        ax.axhline(0, color=C["gray"], ls=":", lw=0.8)
        ax.set_xticks([17, 29, 43]); ax.set_xlabel("Model seed"); ax.set_title(title, loc="left", fontweight="bold", fontsize=8.3)
        clean(ax, "y")
        dataset_handles = [Line2D([0], [0], color=DATASET_COLORS[d], lw=1.8, label=d) for d in datasets_here]
        ax.legend(handles=dataset_handles, frameon=False, fontsize=5.8, loc="best", handlelength=1.6, labelspacing=0.25)
    axs[0].set_ylabel("Mean paired effect over five partition seeds")
    model_handles = [
        Line2D([0], [0], color=C["ink"], lw=1.2, linestyle="-", label="RF"),
        Line2D([0], [0], color=C["ink"], lw=1.2, linestyle="--", label="XGB"),
    ]
    fig.legend(handles=model_handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.015))
    fig.suptitle("Predeclared RF/XGB stochastic-model sensitivity",
                 fontsize=9.6, fontweight="bold", y=0.99)
    fig.subplots_adjust(top=0.83, bottom=0.22)
    save(fig, "figureS5_model_seed_sensitivity_v3")


def main() -> None:
    for path in [PRIMARY, PARENT_CMP, COLLATERAL, MEAN_ONLY, SUPPORT, SEED]:
        need(path)
    figure1(); figure2(); figure4(); figure6(); figure_s4(); figure_s5()
    print("FINAL SUBMISSION FIGURE POLISH: PASS")


if __name__ == "__main__":
    main()
