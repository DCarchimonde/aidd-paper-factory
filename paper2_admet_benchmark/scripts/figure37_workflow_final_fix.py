from __future__ import annotations

"""Final isolated correction for the Paper 2 workflow figure.

The previous arrow endpoint was already outside the third box, but the long
centered seed-description line extended visually beyond the box boundary after
manuscript scaling.  This version shortens and wraps that text, widens the third
box slightly, and leaves a visible white gap before the arrowhead.
"""

from pathlib import Path

import matplotlib.pyplot as plt

import figure35_layout_fix as layout


def rebuild_workflow_figure() -> list[Path]:
    base = layout.base
    fig, ax = plt.subplots(figsize=(12.0, 6.25))
    ax.set_axis_off()

    top_boxes = [
        (
            0.040,
            0.715,
            0.255,
            0.19,
            "1. Data and frozen models\nFour ADMET endpoints\nECFP-based model families",
        ),
        (
            0.360,
            0.715,
            0.255,
            0.19,
            "2. Label-blind evaluation\nRandom · scaffold · cluster\nTrain / calibration / test",
        ),
        (
            0.675,
            0.715,
            0.285,
            0.19,
            "3. Confirmatory design\n10 random/scaffold seeds\n5 cluster seeds; paired contrasts",
        ),
    ]

    for x, y, width, height, text in top_boxes:
        patch = plt.Rectangle(
            (x, y),
            width,
            height,
            transform=ax.transAxes,
            facecolor=base.PALETTE["light_fill"],
            edgecolor=base.PALETTE["outline"],
            linewidth=1.35,
        )
        ax.add_patch(patch)
        ax.text(
            x + width / 2,
            y + height / 2,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11.0,
            linespacing=1.20,
        )

    # Arrowheads terminate well inside the white inter-box gaps.
    for start, end in [
        ((0.299, 0.81), (0.346, 0.81)),
        ((0.619, 0.81), (0.658, 0.81)),
    ]:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords=ax.transAxes,
            arrowprops=dict(
                arrowstyle="->",
                lw=1.45,
                color=base.PALETTE["outline"],
                mutation_scale=11,
                shrinkA=0,
                shrinkB=0,
            ),
        )

    analysis_x, analysis_y, analysis_w, analysis_h = 0.18, 0.475, 0.64, 0.135
    analysis = plt.Rectangle(
        (analysis_x, analysis_y),
        analysis_w,
        analysis_h,
        transform=ax.transAxes,
        facecolor=base.PALETTE["light_fill_2"],
        edgecolor=base.PALETTE["outline"],
        linewidth=1.4,
    )
    ax.add_patch(analysis)
    ax.text(
        analysis_x + analysis_w / 2,
        analysis_y + analysis_h / 2,
        "Reliability-analysis modules\n"
        "Performance · calibration · applicability domain · conformal prediction · selective prediction",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.7,
        linespacing=1.30,
    )
    ax.annotate(
        "",
        xy=(0.50, analysis_y + analysis_h),
        xytext=(0.8175, 0.715),
        xycoords=ax.transAxes,
        arrowprops=dict(
            arrowstyle="-|>",
            lw=1.4,
            color=base.PALETTE["outline"],
            mutation_scale=10,
            shrinkA=7,
            shrinkB=8,
        ),
    )

    rq_boxes = [
        (0.025, 0.145, 0.215, 0.205, "RQ1\nPerformance and\nprobability calibration", base.PALETTE["light_fill"]),
        (0.270, 0.145, 0.215, 0.205, "RQ2\nApplicability-domain\nrobustness", base.PALETTE["light_fill_2"]),
        (0.515, 0.145, 0.215, 0.205, "RQ3\nClass-conditional and\nsubgroup failure", base.PALETTE["light_fill_3"]),
        (0.760, 0.145, 0.215, 0.205, "RQ4\nCoverage–efficiency–\ninformativeness trade-offs", "#B3D6CC"),
    ]

    bus_y = 0.405
    ax.plot(
        [0.1325, 0.8675],
        [bus_y, bus_y],
        transform=ax.transAxes,
        color="#7A8B93",
        linewidth=1.2,
    )
    ax.plot(
        [0.50, 0.50],
        [analysis_y, bus_y],
        transform=ax.transAxes,
        color="#7A8B93",
        linewidth=1.2,
    )

    for x, y, width, height, text, fill in rq_boxes:
        patch = plt.Rectangle(
            (x, y),
            width,
            height,
            transform=ax.transAxes,
            facecolor=fill,
            edgecolor=base.PALETTE["outline"],
            linewidth=1.2,
        )
        ax.add_patch(patch)
        ax.text(
            x + width / 2,
            y + height / 2,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11.0,
            linespacing=1.22,
        )
        center_x = x + width / 2
        ax.annotate(
            "",
            xy=(center_x, y + height),
            xytext=(center_x, bus_y),
            xycoords=ax.transAxes,
            arrowprops=dict(
                arrowstyle="-|>",
                lw=1.05,
                color="#7A8B93",
                mutation_scale=9,
                shrinkA=1,
                shrinkB=7,
            ),
        )

    ax.set_title(
        "Confirmatory reliability-evaluation workflow",
        fontsize=15.5,
        fontweight="bold",
        pad=14,
    )
    fig.tight_layout()
    return base.save_figure(fig, "figure_1_confirmatory_design")
