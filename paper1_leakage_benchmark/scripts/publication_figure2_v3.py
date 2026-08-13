from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import publication_artwork_common_v3 as u

m = u.load("25_finalize_submission_figures_v3.py", "paper1_figure2_data")


def _forest(ax, df, task, letter, title):
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
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    u.clean(ax, "x")
    ax.set_xlabel("AUC effect (balanced - size)" if task == "classification" else "RMSE improvement")


def build():
    u.configure()
    df = m.primary_frame()
    cls = df[df["task_type"].eq("classification")]
    reg = df[df["task_type"].eq("regression")]
    fig, axs = plt.subplots(1, 2, figsize=(u.WIDTH_IN, 3.65), gridspec_kw={"width_ratios": [1.2, 0.9]})
    fig.subplots_adjust(left=0.22, right=0.98, top=0.91, bottom=0.17, wspace=0.60)
    _forest(axs[0], cls, "classification", "A", "Classification - 0/12")
    _forest(axs[1], reg, "regression", "B", "Regression - 6/6")
    u.save(fig, "figure2_primary_effects_v3")
