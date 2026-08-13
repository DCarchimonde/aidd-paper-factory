from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import publication_artwork_common_v3 as u

m = u.load("21_build_manuscript_assets_v3_round3.py", "paper1_fig1_data")


def build() -> None:
    u.configure()
    fig = plt.figure(figsize=(u.WIDTH_IN, 4.25))
    gs = fig.add_gridspec(2, 2)
    fig.subplots_adjust(left=0.13, right=0.965, top=0.945, bottom=0.09, wspace=0.45, hspace=0.50)

    ax = fig.add_subplot(gs[0, 0])
    u.panel(ax, "A", "Audited molecular universe")
    y = np.arange(len(m.DATASETS)); vals = [m.N[d] for d in m.DATASETS]
    cols = [u.C["navy"] if d in m.CLS else u.C["teal"] for d in m.DATASETS]
    ax.barh(y, vals, color=cols, height=0.56, zorder=2)
    ax.set_yticks(y, m.DATASETS); ax.invert_yaxis(); ax.set_xscale("log")
    # Keep the displayed log decade range intentional; this also prevents
    # unused off-view tick artists from crowding the fixed-size panel.
    ax.set_xlim(5e2, 8e4)
    ax.set_xlabel("Clean molecules"); u.clean(ax, "x")
    for yy, value in zip(y, vals):
        ax.text(value * 1.05, yy, f"{value:,}", va="center", fontsize=6.3)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=u.C["navy"], label="Classification"),
                       Rectangle((0, 0), 1, 1, color=u.C["teal"], label="Regression")],
              frameon=False, loc="lower right", handlelength=1.2)

    ax = fig.add_subplot(gs[0, 1]); ax.set_axis_off()
    u.panel(ax, "B", "Exact-size paired selection")
    u.card(ax, (0.17, 0.73), 0.66, 0.16, "Target-blind\ncandidate pool", u.C["pale_blue"], u.C["navy"])
    u.arrow(ax, (0.50, 0.72), (0.50, 0.60))
    u.card(ax, (0.22, 0.47), 0.56, 0.14, "Same test-set size", u.C["white"], u.C["mid"])
    u.arrow(ax, (0.50, 0.46), (0.28, 0.33)); u.arrow(ax, (0.50, 0.46), (0.72, 0.33), u.C["teal2"])
    u.card(ax, (0.02, 0.16), 0.43, 0.17, "Size-matched\nbaseline", u.C["pale_gray"], u.C["mid"])
    u.card(ax, (0.55, 0.16), 0.43, 0.17, "Lower target-mean\ngap", u.C["pale_teal"], u.C["teal"])
    ax.text(0.50, 0.04, "seed · scaffold rule · search budget frozen pre-outcome",
            transform=ax.transAxes, ha="center", fontsize=5.5, color=u.C["gray"])

    ax = fig.add_subplot(gs[1, 0]); ax.set_axis_off()
    u.panel(ax, "C", "Freeze before model outcomes")
    steps = [("Freeze\nbudget", u.C["pale_orange"], u.C["orange"]),
             ("Store\nmanifest", u.C["pale_blue"], u.C["navy"]),
             ("Fit\nmodels", u.C["pale_gray"], u.C["mid"]),
             ("Paired\ninference", u.C["pale_teal"], u.C["teal"])]
    for i, (label, fc, ec) in enumerate(steps):
        x = 0.01 + i * 0.247
        u.card(ax, (x, 0.58), 0.20, 0.20, label, fc, ec, 6.1)
        if i < 3: u.arrow(ax, (x + 0.205, 0.68), (x + 0.238, 0.68))
    ax.text(0.50, 0.36, "20 unique partition pairs", transform=ax.transAxes,
            ha="center", fontsize=6.5, fontweight="bold")
    ax.text(0.50, 0.21, "10,000 paired bootstraps · Wilcoxon + Holm", transform=ax.transAxes,
            ha="center", fontsize=6.0)
    ax.text(0.50, 0.07, "Inferential N = partition pairs, not model seeds", transform=ax.transAxes,
            ha="center", fontsize=5.8, color=u.C["gray"])

    ax = fig.add_subplot(gs[1, 1]); ax.set_axis_off()
    u.panel(ax, "D", "Protocol sensitivities")
    u.card(ax, (0.09, 0.65), 0.82, 0.16, "Acyclic semantics\nsingle-group vs singleton",
           u.C["pale_blue"], u.C["navy"])
    u.card(ax, (0.09, 0.41), 0.82, 0.16, "Record representation\nfull record vs fragment",
           u.C["pale_orange"], u.C["orange"])
    u.arrow(ax, (0.50, 0.39), (0.50, 0.28), u.C["teal2"])
    u.card(ax, (0.23, 0.12), 0.54, 0.15, "Does the claim survive?", u.C["pale_teal"], u.C["teal"])
    ax.text(0.50, 0.02, "Report disagreement; do not resolve it post hoc.",
            transform=ax.transAxes, ha="center", fontsize=5.5, color=u.C["gray"])
    u.save(fig, "figure1_audit_framework_v3")
