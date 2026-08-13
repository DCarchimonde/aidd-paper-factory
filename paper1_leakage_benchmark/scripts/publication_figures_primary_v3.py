from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import publication_artwork_common_v3 as u

m25 = u.load("25_finalize_submission_figures_v3.py", "paper1_primary_data")
m21 = u.load("21_build_manuscript_assets_v3_round3.py", "paper1_sensitivity_data")


def _forest(ax, df: pd.DataFrame, task: str, letter: str, title: str, result: str) -> None:
    u.panel(ax, letter, title)
    y = np.arange(len(df))
    eff = df["mean_effect"].to_numpy(float)
    lo = df["bootstrap_ci_low"].to_numpy(float)
    hi = df["bootstrap_ci_high"].to_numpy(float)
    labels = [f"{r.dataset} · {r.model}" for r in df.itertuples(index=False)]
    for i in range(len(df)):
        if i % 3 == 0:
            ax.axhspan(i - 0.48, min(i + 2.48, len(df) - 0.52), color=u.C["pale_gray"], zorder=0)
    colors = [u.C["teal2"] if str(x) == "target_balanced_better" else u.C["navy"] for x in df["inference_label"]]
    for yy, e, low, high, color in zip(y, eff, lo, hi, colors):
        ax.errorbar(e, yy, xerr=[[e - low], [high - e]], fmt="o", ms=3.1,
                    lw=0.95, capsize=1.7, color=color, zorder=3)
    ax.axvline(0, color=u.C["gray"], ls="--", lw=0.8)
    ax.set_yticks(y, labels); ax.invert_yaxis(); u.clean(ax, "x")
    ax.set_xlabel("AUC effect: balanced − size-matched" if task == "classification"
                  else "RMSE improvement: size-matched − balanced")
    ax.text(0.97, 0.03, result, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.4, fontweight="bold",
            color=u.C["navy2"] if task == "classification" else u.C["teal2"],
            bbox=dict(boxstyle="round,pad=0.22",
                      fc=u.C["pale_blue"] if task == "classification" else u.C["pale_teal"],
                      ec=u.C["mid"] if task == "classification" else u.C["teal"], lw=0.65))


def figure2() -> None:
    u.configure()
    df = m25.primary_frame()
    cls = df[df["task_type"].eq("classification")]
    reg = df[df["task_type"].eq("regression")]
    fig, axs = plt.subplots(1, 2, figsize=(u.WIDTH_IN, 3.65),
                            gridspec_kw={"width_ratios": [1.2, 0.9]})
    fig.subplots_adjust(left=0.22, right=0.98, top=0.91, bottom=0.17, wspace=0.58)
    _forest(axs[0], cls, "classification", "A", "Classification", "0 / 12 supported")
    _forest(axs[1], reg, "regression", "B", "Regression · primary semantics", "6 / 6 supported")
    u.save(fig, "figure2_primary_effects_v3")


def figure3() -> None:
    u.configure()
    primary = m21.primary_frame()
    single = pd.read_csv(m21.SINGLETON, keep_default_na=False).copy()
    if "mean_effect" not in single.columns and "mean_effect_positive_is_balanced_better" in single.columns:
        single = single.rename(columns={"mean_effect_positive_is_balanced_better": "mean_effect"})
    single["mo"] = single["model"].map(m21.MODEL_ORDER)

    fig = plt.figure(figsize=(u.WIDTH_IN, 3.95))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.58, 1.0])
    fig.subplots_adjust(left=0.17, right=0.98, top=0.93, bottom=0.16, wspace=0.50, hspace=0.62)

    ax = fig.add_subplot(gs[0, :]); ax.set_axis_off()
    u.panel(ax, "A", "Only the acyclic scaffold identity changes")
    u.card(ax, (0.08, 0.38), 0.34, 0.35, "Single-group\nall acyclic molecules\nshare one scaffold ID",
           u.C["pale_blue"], u.C["navy"], 6.5)
    u.arrow(ax, (0.44, 0.56), (0.56, 0.56), u.C["orange"])
    u.card(ax, (0.58, 0.38), 0.34, 0.35, "Singleton\neach acyclic molecule\nhas its own scaffold ID",
           u.C["pale_teal"], u.C["teal"], 6.5)
    ax.text(0.50, 0.12, "unchanged: endpoints · models · 20 partition seeds · exact-size pairing",
            transform=ax.transAxes, ha="center", fontsize=5.9, color=u.C["gray"])

    for j, ds in enumerate(m21.REG):
        ax = fig.add_subplot(gs[1, j])
        u.panel(ax, "B" if j == 0 else "C", ds)
        pp = primary[primary["dataset"].eq(ds)].copy()
        pp["mo"] = pp["model"].map(m21.MODEL_ORDER); pp = pp.sort_values("mo")
        ss = single[single["dataset"].eq(ds)].sort_values("mo")
        for k, model in enumerate(["Ridge", "RF", "XGB"]):
            pr = pp[pp["model"].eq(model)].iloc[0]
            sr = ss[ss["model"].eq(model)].iloc[0]
            ax.errorbar(float(pr.mean_effect), k,
                        xerr=[[float(pr.mean_effect - pr.bootstrap_ci_low)],
                              [float(pr.bootstrap_ci_high - pr.mean_effect)]],
                        fmt="o", ms=3.4, color=u.C["navy"], lw=0.9, capsize=1.8)
            ax.errorbar(float(sr.mean_effect), k,
                        xerr=[[float(sr.mean_effect - sr.bootstrap_ci_low)],
                              [float(sr.bootstrap_ci_high - sr.mean_effect)]],
                        fmt="s", ms=3.3, color=u.C["orange"], lw=0.9, capsize=1.8)
        ax.axvline(0, color=u.C["gray"], ls="--", lw=0.8)
        ax.set_yticks(np.arange(3), ["Ridge", "RF", "XGB"]); ax.invert_yaxis()
        ax.set_xlabel("RMSE improvement"); u.clean(ax, "x")

    fig.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=u.C["navy"], markeredgecolor=u.C["navy"], label="single-group (primary)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=u.C["orange"], markeredgecolor=u.C["orange"], label="singleton sensitivity")],
        loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.57, 0.015))
    u.save(fig, "figure3_acyclic_sensitivity_v3")
