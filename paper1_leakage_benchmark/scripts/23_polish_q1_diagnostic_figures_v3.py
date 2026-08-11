from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper1_leakage_benchmark"
TABLES = PAPER / "results" / "tables"
FIG = PAPER / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

PRIMARY = TABLES / "primary_inference_summary_v3.csv"
MEAN_ONLY = TABLES / "q1_mean_only_regression_summary_v3.csv"
COLLATERAL = TABLES / "q1_collateral_partition_diagnostics_v3.csv"
SUPPORT = TABLES / "supporting_metric_effects_v3.csv"
SEED = TABLES / "q1_model_seed_summary_v3.csv"

DATASETS = ["BACE", "BBBP", "ClinTox", "HIV", "ESOL", "FreeSolv"]
REG = ["ESOL", "FreeSolv"]
MODEL_ORDER = {"LR": 0, "Ridge": 0, "RF": 1, "XGB": 2}
DATASET_ORDER = {d: i for i, d in enumerate(DATASETS)}
C = {
    "ink": "#20313A", "navy": "#315B73", "navy2": "#24485D",
    "teal": "#2B8C82", "teal2": "#176B64", "orange": "#D58A43",
    "gray": "#6D7A81", "mid": "#B8C2C7", "grid": "#E5EAEC", "white": "#FFFFFF",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.3, "axes.titlesize": 9.0,
    "axes.labelsize": 8.3, "xtick.labelsize": 7.3, "ytick.labelsize": 7.3,
    "legend.fontsize": 6.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": C["mid"], "text.color": C["ink"], "axes.labelcolor": C["ink"],
    "xtick.color": C["ink"], "ytick.color": C["ink"], "pdf.fonttype": 42, "ps.fonttype": 42,
})


def need(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIG / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(FIG / f"{stem}.pdf")


def panel(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(-0.09, 1.07, letter, transform=ax.transAxes, fontsize=10.8, fontweight="bold", va="top")
    ax.text(0.0, 1.035, title, transform=ax.transAxes, fontsize=8.7, fontweight="bold", va="bottom")


def clean(ax: plt.Axes, axis: str = "x") -> None:
    ax.grid(axis=axis, color=C["grid"], lw=0.7, zorder=0)


def figure6() -> None:
    primary = pd.read_csv(need(PRIMARY), keep_default_na=False)
    mean_only = pd.read_csv(need(MEAN_ONLY), keep_default_na=False)
    collateral = pd.read_csv(need(COLLATERAL), keep_default_na=False)
    mean_only = mean_only.loc[mean_only["freeze_label"].eq("main_regression")].copy()

    fig = plt.figure(figsize=(7.15, 5.55))
    gs = fig.add_gridspec(2, 2, wspace=0.42, hspace=0.48)
    fig.subplots_adjust(top=0.88, bottom=0.10)

    ax = fig.add_subplot(gs[0, 0]); panel(ax, "A", "Mean-only control versus learned models")
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
    positive_values = []
    for i, ds in enumerate(DATASETS):
        g = collateral.loc[collateral["dataset"].eq(ds)].copy()
        s = g["size_abs_target_mean_gap"].to_numpy(float)
        b = g["balanced_abs_target_mean_gap"].to_numpy(float)
        ratio = np.divide(b, s, out=np.full_like(b, np.nan), where=s > 0)
        finite = ratio[np.isfinite(ratio)]
        positive_values.extend(finite[finite > 0].tolist())
        floor = max(np.min(finite[finite > 0]) * 0.35, 1e-6) if np.any(finite > 0) else 1e-6
        plotted = np.where(finite > 0, finite, floor)
        x = np.full(len(plotted), i) + rng.normal(0, 0.042, len(plotted))
        ax.scatter(x, plotted, s=10, alpha=0.58, color=C["teal"], edgecolors="none")
        ax.scatter([i], [np.median(plotted)], s=28, marker="D", color=C["navy2"], zorder=4)
    ax.axhline(1, color=C["gray"], ls="--", lw=0.8)
    ax.set_yscale("log"); ax.set_xticks(range(6), DATASETS, rotation=25, ha="right")
    ax.set_ylabel("Balanced / size target-mean gap"); clean(ax, "y")

    ax = fig.add_subplot(gs[1, 0]); panel(ax, "C", "Collateral change in largest-scaffold fraction")
    for i, ds in enumerate(DATASETS):
        g = collateral.loc[collateral["dataset"].eq(ds)]
        delta = g["delta_balanced_minus_size_largest_test_scaffold_fraction"].to_numpy(float)
        x = np.full(len(delta), i) + rng.normal(0, 0.042, len(delta))
        ax.scatter(x, delta, s=10, alpha=0.58, color=C["orange"], edgecolors="none")
        ax.scatter([i], [np.mean(delta)], s=28, marker="D", color=C["navy2"], zorder=4)
    ax.axhline(0, color=C["gray"], ls="--", lw=0.8)
    ax.set_xticks(range(6), DATASETS, rotation=25, ha="right"); ax.set_ylabel("Balanced − size")
    clean(ax, "y")

    ax = fig.add_subplot(gs[1, 1]); panel(ax, "D", "Collateral change in effective scaffold number")
    for i, ds in enumerate(DATASETS):
        g = collateral.loc[collateral["dataset"].eq(ds)]
        s = g["size_effective_test_scaffolds"].to_numpy(float)
        b = g["balanced_effective_test_scaffolds"].to_numpy(float)
        valid = (s > 0) & (b > 0) & np.isfinite(s) & np.isfinite(b)
        values = np.log2(b[valid] / s[valid])
        x = np.full(len(values), i) + rng.normal(0, 0.042, len(values))
        ax.scatter(x, values, s=10, alpha=0.58, color=C["navy"], edgecolors="none")
        ax.scatter([i], [np.mean(values)], s=28, marker="D", color=C["teal2"], zorder=4)
    ax.axhline(0, color=C["gray"], ls="--", lw=0.8)
    ax.set_xticks(range(6), DATASETS, rotation=25, ha="right"); ax.set_ylabel("log2(balanced / size)")
    clean(ax, "y")

    fig.suptitle("Target-mean-aware selection changes RMSE difficulty and other benchmark properties",
                 fontsize=10.0, fontweight="bold", y=0.985)
    save(fig, "figure6_collateral_diagnostics_v3")


def effect_matrix(df: pd.DataFrame, task: str, metrics: list[str], metric_labels: list[str]):
    sub = df.loc[df["task_type"].eq(task)].copy()
    sub["do"] = sub["dataset"].map(DATASET_ORDER)
    sub["mo"] = sub["model"].map(MODEL_ORDER)
    combos = sub[["dataset", "model", "do", "mo"]].drop_duplicates().sort_values(["do", "mo"])
    matrix = []
    row_labels = []
    for r in combos.itertuples(index=False):
        row_labels.append(f"{r.dataset} · {r.model}")
        values = []
        for metric in metrics:
            m = sub.loc[(sub["dataset"].eq(r.dataset)) & (sub["model"].eq(r.model)) & (sub["metric"].eq(metric))]
            if len(m) != 1:
                raise AssertionError(f"Missing supporting metric {task}/{r.dataset}/{r.model}/{metric}")
            value = float(m.iloc[0]["mean_effect_positive_is_balanced_better"])
            scale = float(sub.loc[sub["metric"].eq(metric), "mean_effect_positive_is_balanced_better"].abs().max())
            values.append(value / scale if scale > 0 else 0.0)
        matrix.append(values)
    return np.asarray(matrix), row_labels, metric_labels


def figure_s4() -> None:
    support = pd.read_csv(need(SUPPORT), keep_default_na=False)
    cls_metrics = ["roc_auc", "average_precision", "f1", "accuracy", "balanced_accuracy", "brier_score"]
    cls_labels = ["AUC", "AP", "F1", "Accuracy", "Bal. acc.", "Brier"]
    reg_metrics = ["rmse", "mae", "r2"]
    reg_labels = ["RMSE", "MAE", "$R^2$"]
    cls_mat, cls_rows, cls_cols = effect_matrix(support, "classification", cls_metrics, cls_labels)
    reg_mat, reg_rows, reg_cols = effect_matrix(support, "regression", reg_metrics, reg_labels)

    cmap = LinearSegmentedColormap.from_list("signed", [C["orange"], C["white"], C["teal"]])
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    fig, axs = plt.subplots(1, 2, figsize=(7.15, 4.45), gridspec_kw={"width_ratios": [1.55, 0.75], "wspace": 0.38})
    im = axs[0].imshow(cls_mat, cmap=cmap, norm=norm, aspect="auto")
    axs[0].set_yticks(range(len(cls_rows)), cls_rows); axs[0].set_xticks(range(len(cls_cols)), cls_cols, rotation=25, ha="right")
    axs[0].set_title("A  Classification supporting metrics", loc="left", fontweight="bold", fontsize=8.7)
    axs[1].imshow(reg_mat, cmap=cmap, norm=norm, aspect="auto")
    axs[1].set_yticks(range(len(reg_rows)), reg_rows); axs[1].set_xticks(range(len(reg_cols)), reg_cols, rotation=25, ha="right")
    axs[1].set_title("B  Regression supporting metrics", loc="left", fontweight="bold", fontsize=8.7)
    for ax in axs:
        for spine in ax.spines.values(): spine.set_visible(False)
    cbar = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.025)
    cbar.set_label("Signed mean effect, normalized within metric")
    fig.suptitle("Supporting metrics preserve direction without creating new primary hypothesis families",
                 fontsize=9.6, fontweight="bold", y=0.99)
    fig.subplots_adjust(top=0.86, bottom=0.17)
    save(fig, "figureS4_supporting_metrics_v3")


def figure_s5() -> None:
    seed = pd.read_csv(need(SEED), keep_default_na=False)
    panels = [
        ("main_classification", "A  Primary classification"),
        ("main_regression", "B  Primary regression"),
        ("acyclic_singleton_sensitivity", "C  Singleton regression"),
    ]
    # Matplotlib compatibility: spacing belongs in GridSpec, not Figure kwargs.
    fig, axs = plt.subplots(1, 3, figsize=(7.15, 3.65), gridspec_kw={"wspace": 0.38})
    for ax, (label, title) in zip(axs, panels):
        sub = seed.loc[seed["freeze_label"].eq(label)].copy()
        sub["do"] = sub["dataset"].map(DATASET_ORDER); sub["mo"] = sub["model"].map(MODEL_ORDER)
        combos = sub[["dataset", "model", "do", "mo"]].drop_duplicates().sort_values(["do", "mo"])
        for idx, r in enumerate(combos.itertuples(index=False)):
            g = sub.loc[(sub["dataset"].eq(r.dataset)) & (sub["model"].eq(r.model))].sort_values("model_seed")
            color = C["navy"] if idx % 2 == 0 else C["teal"]
            style = "-" if r.model == "RF" else "--"
            ax.plot(g["model_seed"], g["mean_effect"], marker="o", ms=2.8, lw=0.9,
                    linestyle=style, color=color, label=f"{r.dataset} · {r.model}")
        ax.axhline(0, color=C["gray"], ls=":", lw=0.8)
        ax.set_xticks([17, 29, 43]); ax.set_xlabel("Model seed"); ax.set_title(title, loc="left", fontweight="bold", fontsize=8.3)
        clean(ax, "y")
        ax.legend(frameon=False, fontsize=5.5, loc="best", handlelength=1.8, labelspacing=0.25)
    axs[0].set_ylabel("Mean paired effect over five partition seeds")
    fig.suptitle("Predeclared RF/XGB stochastic-model sensitivity", fontsize=9.6, fontweight="bold", y=0.99)
    fig.subplots_adjust(top=0.83, bottom=0.16)
    save(fig, "figureS5_model_seed_sensitivity_v3")


def main() -> None:
    for path in [PRIMARY, MEAN_ONLY, COLLATERAL, SUPPORT, SEED]:
        need(path)
    figure6(); figure_s4(); figure_s5()
    print("Q1 DIAGNOSTIC FIGURE POLISH: PASS")


if __name__ == "__main__":
    main()
