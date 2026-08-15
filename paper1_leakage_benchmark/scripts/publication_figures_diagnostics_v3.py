from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import publication_artwork_common_v3 as u

m25 = u.load("25_finalize_submission_figures_v3.py", "paper1_diag_data")
m21 = u.load("21_build_manuscript_assets_v3_round3.py", "paper1_budget_data")


def figure4() -> None:
    u.configure()
    comp = pd.read_csv(m25.PARENT_CMP, keep_default_na=False)
    main_col = next(c for c in ["main_mean_effect", "primary_mean_effect", "source_faithful_mean_effect"] if c in comp.columns)
    frag_col = next(c for c in ["parent_mean_effect", "fragment_mean_effect", "dominant_fragment_mean_effect"] if c in comp.columns)
    comp["do"] = comp["dataset"].map(m25.DATASET_ORDER)
    comp["mo"] = comp["model"].map(m25.MODEL_ORDER)
    comp = comp.sort_values(["do", "mo"])

    fig, axs = plt.subplots(1, 2, figsize=(u.WIDTH_IN, 3.55),
                            gridspec_kw={"width_ratios": [0.82, 1.28]})
    fig.subplots_adjust(left=0.11, right=0.98, top=0.91, bottom=0.25, wspace=0.44)

    ax = axs[0]
    u.panel(ax, "A", "Structural consequences")
    ds = ["BBBP", "ClinTox", "HIV"]
    x = np.arange(3); width = 0.19
    series = [
        ("Multi-component", m25.MULTI, u.C["navy2"]),
        ("Scaffold changed", m25.SCAFF_CHANGED, u.C["navy"]),
        ("Similarity < 0.90", m25.SIM090, u.C["orange"]),
        ("Conflict groups", m25.CONFLICT, u.C["teal"]),
    ]
    for i, (label, values, color) in enumerate(series):
        ax.bar(x + (i - 1.5) * width, [values[d] for d in ds], width, color=color, label=label)
    ax.set_xticks(x, ds); ax.set_yscale("log"); ax.set_ylabel("Count"); u.clean(ax, "y")
    ax.legend(frameon=False, loc="upper left", fontsize=5.3, handlelength=1.2, labelspacing=0.22)

    ax = axs[1]
    u.panel(ax, "B", "Representation-sensitive effects")
    y = np.arange(len(comp)); a = comp[main_col].astype(float).to_numpy(); b = comp[frag_col].astype(float).to_numpy()
    labels = [f"{r.dataset} · {r.model}" for r in comp.itertuples(index=False)]
    for yy, x1, x2 in zip(y, a, b):
        ax.plot([x1, x2], [yy, yy], color=u.C["mid"], lw=1.0, zorder=1)
    p_source = ax.scatter(a, y, s=17, color=u.C["navy"], label="source-faithful", zorder=2)
    p_fragment = ax.scatter(b, y, s=18, marker="s", color=u.C["orange"], label="dominant fragment", zorder=2)
    ax.axvline(0, color=u.C["gray"], ls="--", lw=0.8)
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlabel("AUC effect")
    u.clean(ax, "x")
    fig.legend([p_source, p_fragment], ["source-faithful", "dominant fragment"],
               frameon=False, loc="lower center", ncol=2, fontsize=5.7,
               bbox_to_anchor=(0.72, 0.055), handletextpad=0.5, columnspacing=1.1)
    u.save(fig, "figure4_dominant_fragment_sensitivity_v3")


def figure5() -> None:
    u.configure()
    fig, axs = plt.subplots(2, 2, figsize=(u.WIDTH_IN, 4.15))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.93, bottom=0.12, wspace=0.44, hspace=0.58)

    for j, ds in enumerate(m21.REG):
        ax = axs[0, j]
        u.panel(ax, "A" if j == 0 else "B", f"{ds} - single-group")
        x = np.array(list(m21.BUDGET_SINGLE[ds])); y = np.array(list(m21.BUDGET_SINGLE[ds].values()))
        color = u.C["teal"] if ds == "ESOL" else u.C["navy"]
        ax.plot(x, y, marker="o", ms=3.0, lw=1.15, color=color)
        ax.axvline(20000, color=u.C["orange"], ls="--", lw=0.8)
        ax.annotate("20k cap", xy=(20000, y[-1]), xytext=(0.58, 0.63), textcoords="axes fraction",
                    fontsize=6.0, color=u.C["orange2"],
                    arrowprops=dict(arrowstyle="->", lw=0.7, color=u.C["orange2"]))
        ax.set_xlabel("Candidate budget"); ax.set_ylabel("Mean target-mean gap"); u.clean(ax, "both")

    ax = axs[1, 0]
    u.panel(ax, "C", "Singleton budget trajectory")
    for ds, color, marker in [("ESOL", u.C["teal"], "o"), ("FreeSolv", u.C["orange"], "s")]:
        x = np.array(list(m21.BUDGET_SINGLETON[ds])); y = np.array(list(m21.BUDGET_SINGLETON[ds].values()))
        ax.plot(x, y / y[0], marker=marker, ms=3.0, lw=1.05, color=color, label=ds)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Candidate budget")
    ax.set_ylabel("Gap / 100-candidate gap"); u.clean(ax, "both"); ax.legend(frameon=False)

    ax = axs[1, 1]
    u.panel(ax, "D", "Exact-size pairing")
    yy = np.arange(len(m21.DATASETS)); vals = np.array([m21.TEST_N[d] for d in m21.DATASETS], float)
    ax.scatter(vals, yy, s=25, color=u.C["navy"], label="size-matched", zorder=3)
    ax.scatter(vals, yy, s=18, marker="s", facecolor=u.C["white"], edgecolor=u.C["teal"],
               linewidth=1.0, label="target-balanced", zorder=4)
    ax.set_yticks(yy, m21.DATASETS); ax.invert_yaxis(); ax.set_xscale("log"); ax.set_xlabel("Test molecules")
    u.clean(ax, "x")
    ax.legend(frameon=False, loc="lower right", fontsize=5.6)
    u.save(fig, "figure5_candidate_budget_audit_v3")


def _assert_visible_yticklabels_inside(fig, ax, label: str) -> None:
    """Guard against Wiley/TIFF clipping of long visible row labels.

    The common artwork gate intentionally ignores tick labels because Matplotlib
    creates off-view tick artists on logarithmic axes. Figure 6A has long, visible
    categorical labels, so it receives a dedicated rendered-boundary check.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    left_edge = fig.bbox.x0 + 4.0
    outside = []
    for text in ax.get_yticklabels():
        if not text.get_visible() or not text.get_text().strip():
            continue
        bb = text.get_window_extent(renderer=renderer)
        if bb.x0 < left_edge:
            outside.append(text.get_text())
    if outside:
        raise AssertionError(f"{label} visible y tick labels exceed fixed canvas: {outside}")


def figure6() -> None:
    u.configure()
    primary = m25.primary_frame()
    mean_only = pd.read_csv(m25.MEAN_ONLY, keep_default_na=False)
    collateral = pd.read_csv(m25.COLLATERAL, keep_default_na=False)
    mean_only = mean_only[mean_only["freeze_label"].eq("main_regression")].copy()

    fig = plt.figure(figsize=(u.WIDTH_IN, 4.35))
    gs = fig.add_gridspec(2, 2)
    # Reserve a true left safety margin for the longest categorical label
    # ("FreeSolv · Mean-only") so both the PDF and 600-dpi TIFF remain intact.
    fig.subplots_adjust(left=0.245, right=0.98, top=0.93, bottom=0.14, wspace=0.58, hspace=0.58)

    ax = fig.add_subplot(gs[0, 0])
    u.panel(ax, "A", "Mean-only vs learned models")
    rows = []
    for ds in m25.REG:
        mo = mean_only[mean_only["dataset"].eq(ds)].iloc[0]
        rows.append((ds, "Mean-only", float(mo.mean_effect_size_minus_balanced_rmse),
                     float(mo.bootstrap_ci_low), float(mo.bootstrap_ci_high), u.C["orange"]))
        for model in ["Ridge", "RF", "XGB"]:
            r = primary[(primary["dataset"].eq(ds)) & (primary["model"].eq(model))].iloc[0]
            rows.append((ds, model, float(r.mean_effect), float(r.bootstrap_ci_low),
                         float(r.bootstrap_ci_high), u.C["teal2"]))
    labels = []
    for yy, (ds, model, effect, lo, hi, color) in enumerate(rows):
        ax.errorbar(effect, yy, xerr=[[effect - lo], [hi - effect]],
                    fmt="o" if model == "Mean-only" else "s", ms=3.0, color=color, lw=0.9, capsize=1.6)
        labels.append(f"{ds} · {model}")
    ax.axvline(0, color=u.C["gray"], ls="--", lw=0.8)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.tick_params(axis="y", labelsize=6.8, pad=2)
    ax.invert_yaxis(); ax.set_xlabel("RMSE improvement"); u.clean(ax, "x")
    _assert_visible_yticklabels_inside(fig, ax, "figure6 panel A")

    rng = np.random.default_rng(3)
    ax = fig.add_subplot(gs[0, 1])
    u.panel(ax, "B", "Target-mean gap ratio")
    for i, ds in enumerate(m25.DATASETS):
        g = collateral[collateral["dataset"].eq(ds)].copy()
        s = g["size_abs_target_mean_gap"].to_numpy(float); b = g["balanced_abs_target_mean_gap"].to_numpy(float)
        ratio = np.divide(b, s, out=np.full_like(b, np.nan), where=s > 0)
        finite = ratio[np.isfinite(ratio)]
        floor = max(np.min(finite[finite > 0]) * 0.35, 1e-6) if np.any(finite > 0) else 1e-6
        plotted = np.where(finite > 0, finite, floor)
        x = np.full(len(plotted), i) + rng.normal(0, 0.04, len(plotted))
        ax.scatter(x, plotted, s=9, alpha=0.55, color=u.C["teal"], edgecolors="none")
        ax.scatter([i], [np.median(plotted)], s=25, marker="D", color=u.C["navy2"], zorder=4)
    ax.axhline(1, color=u.C["gray"], ls="--", lw=0.8); ax.set_yscale("log")
    ax.set_xticks(range(6), m25.DATASETS, rotation=30, ha="right"); ax.set_ylabel("Balanced / size"); u.clean(ax, "y")

    ax = fig.add_subplot(gs[1, 0])
    u.panel(ax, "C", "Largest-scaffold fraction")
    for i, ds in enumerate(m25.DATASETS):
        g = collateral[collateral["dataset"].eq(ds)]
        delta = g["delta_balanced_minus_size_largest_test_scaffold_fraction"].to_numpy(float)
        x = np.full(len(delta), i) + rng.normal(0, 0.04, len(delta))
        ax.scatter(x, delta, s=9, alpha=0.55, color=u.C["orange"], edgecolors="none")
        ax.scatter([i], [np.mean(delta)], s=25, marker="D", color=u.C["navy2"], zorder=4)
    ax.axhline(0, color=u.C["gray"], ls="--", lw=0.8)
    ax.set_xticks(range(6), m25.DATASETS, rotation=30, ha="right"); ax.set_ylabel("Balanced - size"); u.clean(ax, "y")

    ax = fig.add_subplot(gs[1, 1])
    u.panel(ax, "D", "Effective scaffold number")
    for i, ds in enumerate(m25.DATASETS):
        g = collateral[collateral["dataset"].eq(ds)]
        s = g["size_effective_test_scaffolds"].to_numpy(float); b = g["balanced_effective_test_scaffolds"].to_numpy(float)
        valid = (s > 0) & (b > 0) & np.isfinite(s) & np.isfinite(b)
        values = np.log2(b[valid] / s[valid]); x = np.full(len(values), i) + rng.normal(0, 0.04, len(values))
        ax.scatter(x, values, s=9, alpha=0.55, color=u.C["navy"], edgecolors="none")
        ax.scatter([i], [np.mean(values)], s=25, marker="D", color=u.C["teal2"], zorder=4)
    ax.axhline(0, color=u.C["gray"], ls="--", lw=0.8)
    ax.set_xticks(range(6), m25.DATASETS, rotation=30, ha="right"); ax.set_ylabel("log2(balanced / size)"); u.clean(ax, "y")
    u.save(fig, "figure6_collateral_diagnostics_v3")
