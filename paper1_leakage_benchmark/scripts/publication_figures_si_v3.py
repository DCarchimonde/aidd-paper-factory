from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

import publication_artwork_common_v3 as u

m21 = u.load("21_build_manuscript_assets_v3_round3.py", "paper1_si_base")
m25 = u.load("25_finalize_submission_figures_v3.py", "paper1_si_controls")


def figure_s1() -> None:
    u.configure()
    cleaning = pd.read_csv(m21.CLEAN, keep_default_na=False)
    fig, ax = plt.subplots(figsize=(u.WIDTH_IN, 2.65))
    fig.subplots_adjust(left=0.14, right=0.98, top=0.88, bottom=0.19)
    x = np.arange(len(m21.DATASETS)); raw = cleaning["raw_rows"].to_numpy(float)
    final = cleaning["final_clean_unique_molecules"].to_numpy(float)
    ax.bar(x, raw, width=0.62, color=u.C["pale_blue"], edgecolor=u.C["mid"], label="Raw rows")
    ax.bar(x, final, width=0.48, color=[u.C["navy"]] * 4 + [u.C["teal"]] * 2, label="Final unique molecules")
    ax.set_yscale("log"); ax.set_xticks(x, m21.DATASETS); ax.set_ylabel("Rows"); u.clean(ax, "y")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    for i, r in cleaning.iterrows():
        reduction = int(r.raw_rows - r.final_clean_unique_molecules)
        ax.annotate(f"-{reduction}" if reduction else "0", xy=(i, float(r.final_clean_unique_molecules)),
                    xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6.2, color=u.C["orange2"])
    ax.set_title("Audited raw-to-clean molecular-data construction", loc="left", fontweight="bold")
    u.save(fig, "figureS1_dataset_construction_v3")


def figure_s2() -> None:
    u.configure()
    fig, axs = plt.subplots(1, 2, figsize=(u.WIDTH_IN, 2.95))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.89, bottom=0.20, wspace=0.44)
    for j, ds in enumerate(m21.REG):
        ax = axs[j]; u.panel(ax, "A" if j == 0 else "B", ds)
        for vals, color, marker, label in [
            (m21.BUDGET_SINGLE[ds], u.C["navy"], "o", "single-group"),
            (m21.BUDGET_SINGLETON[ds], u.C["orange"], "s", "singleton")]:
            x = np.array(list(vals)); y = np.array(list(vals.values()))
            ax.plot(x, y, marker=marker, ms=3.0, lw=1.05, color=color, label=label)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Candidate budget")
        ax.set_ylabel("Mean target-mean gap"); u.clean(ax, "both"); ax.legend(frameon=False, loc="best")
    u.save(fig, "figureS2_budget_semantics_v3")


def figure_s3() -> None:
    u.configure()
    ds = ["BBBP", "ClinTox", "HIV"]
    vals = np.array([[m21.MULTI[d], m21.SCAFF_CHANGED[d], m21.SIM090[d], m21.CONFLICT[d]] for d in ds], float)
    logged = np.log10(vals + 1)
    cmap = LinearSegmentedColormap.from_list("audit", ["#FFF8DB", "#92D3B5", "#2E9DB6", "#153B73"])
    fig, ax = plt.subplots(figsize=(u.WIDTH_IN, 2.35))
    fig.subplots_adjust(left=0.15, right=0.82, top=0.85, bottom=0.19)
    im = ax.imshow(logged, cmap=cmap, aspect="auto")
    ax.set_xticks(range(4), ["Multi-component", "Scaffold changed", "Similarity < 0.90", "Conflict groups"])
    ax.set_yticks(range(3), ds)
    for i in range(3):
        for j in range(4):
            ax.text(j, i, f"{int(vals[i, j]):,}", ha="center", va="center", fontsize=7.2,
                    color="white" if logged[i, j] > 2.2 else u.C["ink"], fontweight="bold")
    cax = fig.add_axes([0.86, 0.19, 0.018, 0.66]); cb = fig.colorbar(im, cax=cax)
    cb.ax.set_title("log10\n(count+1)", fontsize=6.2, pad=3)
    ax.set_title("Disconnected-component structural audit", loc="left", fontweight="bold")
    u.save(fig, "figureS3_multicomponent_audit_v3")


def _effect_matrix(df, task, metrics, labels):
    sub = df[df["task_type"].eq(task)].copy()
    sub["do"] = sub["dataset"].map(m25.DATASET_ORDER); sub["mo"] = sub["model"].map(m25.MODEL_ORDER)
    combos = sub[["dataset", "model", "do", "mo"]].drop_duplicates().sort_values(["do", "mo"])
    matrix, rows = [], []
    for r in combos.itertuples(index=False):
        rows.append(f"{r.dataset} · {r.model}"); row = []
        for metric in metrics:
            item = sub[(sub["dataset"].eq(r.dataset)) & (sub["model"].eq(r.model)) & (sub["metric"].eq(metric))]
            value = float(item.iloc[0]["mean_effect_positive_is_balanced_better"])
            scale = float(sub[sub["metric"].eq(metric)]["mean_effect_positive_is_balanced_better"].abs().max())
            row.append(value / scale if scale > 0 else 0.0)
        matrix.append(row)
    return np.asarray(matrix), rows, labels


def figure_s4() -> None:
    u.configure()
    support = pd.read_csv(m25.SUPPORT, keep_default_na=False)
    cls_mat, cls_rows, cls_cols = _effect_matrix(support, "classification",
        ["roc_auc", "average_precision", "f1", "accuracy", "balanced_accuracy", "brier_score"],
        ["AUC", "AP", "F1", "Accuracy", "Bal. acc.", "Brier"])
    reg_mat, reg_rows, reg_cols = _effect_matrix(support, "regression", ["rmse", "mae", "r2"], ["RMSE", "MAE", "$R^2$"])
    cmap = LinearSegmentedColormap.from_list("signed", [u.C["orange"], u.C["white"], u.C["teal"]])
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    fig = plt.figure(figsize=(u.WIDTH_IN, 4.35)); gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 0.82])
    fig.subplots_adjust(left=0.23, right=0.96, top=0.92, bottom=0.19, hspace=0.50)
    ax = fig.add_subplot(gs[0, 0]); u.panel(ax, "A", "Classification supporting metrics")
    im = ax.imshow(cls_mat, cmap=cmap, norm=norm, aspect="auto")
    ax.set_yticks(range(len(cls_rows)), cls_rows); ax.set_xticks(range(len(cls_cols)), cls_cols)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax = fig.add_subplot(gs[1, 0]); u.panel(ax, "B", "Regression supporting metrics")
    ax.imshow(reg_mat, cmap=cmap, norm=norm, aspect="auto")
    ax.set_yticks(range(len(reg_rows)), reg_rows); ax.set_xticks(range(len(reg_cols)), reg_cols)
    for spine in ax.spines.values(): spine.set_visible(False)
    cax = fig.add_axes([0.34, 0.07, 0.32, 0.022]); cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Normalized signed effect", fontsize=6.4, labelpad=2)
    u.save(fig, "figureS4_supporting_metrics_v3")


def figure_s5() -> None:
    u.configure()
    seed = pd.read_csv(m25.SEED, keep_default_na=False)
    panels = [("main_classification", "A", "Primary classification"),
              ("main_regression", "B", "Primary regression"),
              ("acyclic_singleton_sensitivity", "C", "Singleton regression")]
    fig, axs = plt.subplots(3, 1, figsize=(u.WIDTH_IN, 5.25))
    fig.subplots_adjust(left=0.14, right=0.98, top=0.94, bottom=0.12, hspace=0.62)
    colors = {"BACE": u.C["navy"], "BBBP": u.C["teal"], "ClinTox": u.C["orange"],
              "HIV": u.C["purple"], "ESOL": u.C["navy"], "FreeSolv": u.C["teal"]}
    for ax, (freeze_label, letter, title) in zip(axs, panels):
        u.panel(ax, letter, title)
        sub = seed[seed["freeze_label"].eq(freeze_label)].copy()
        sub["do"] = sub["dataset"].map(m25.DATASET_ORDER); sub["mo"] = sub["model"].map(m25.MODEL_ORDER)
        combos = sub[["dataset", "model", "do", "mo"]].drop_duplicates().sort_values(["do", "mo"])
        datasets_here = []
        for r in combos.itertuples(index=False):
            g = sub[(sub["dataset"].eq(r.dataset)) & (sub["model"].eq(r.model))].sort_values("model_seed")
            ax.plot(g["model_seed"], g["mean_effect"], marker="o", ms=2.7, lw=1.0,
                    linestyle="-" if r.model == "RF" else "--", color=colors[r.dataset])
            if r.dataset not in datasets_here: datasets_here.append(r.dataset)
        ax.axhline(0, color=u.C["gray"], ls=":", lw=0.8); ax.set_xticks([17, 29, 43])
        ax.set_xlabel("Model seed"); ax.set_ylabel("Mean paired effect"); u.clean(ax, "y")
        handles = [Line2D([0], [0], color=colors[d], lw=1.6, label=d) for d in datasets_here]
        ax.legend(handles=handles, frameon=False, ncol=min(len(handles), 4), loc="upper right",
                  fontsize=5.7, handlelength=1.3, columnspacing=0.7)
    fig.legend(handles=[Line2D([0], [0], color=u.C["ink"], lw=1.2, linestyle="-", label="RF"),
                        Line2D([0], [0], color=u.C["ink"], lw=1.2, linestyle="--", label="XGB")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 0.015))
    u.save(fig, "figureS5_model_seed_sensitivity_v3")
