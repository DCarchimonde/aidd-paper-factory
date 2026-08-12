from __future__ import annotations

"""Single authoritative final-artwork builder for Paper 1.

This script is deliberately presentation-only. It reads only already-frozen result
artifacts and does not run models, generate partitions, or recompute inferential
statistics. Historical round-3 plotting modules are used only as function libraries;
each final figure is rendered once in the authoritative order below.
"""

import importlib.util
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper1_leakage_benchmark"
SCRIPTS = PAPER / "scripts"
FIG = PAPER / "results" / "figures"

M21 = SCRIPTS / "21_build_manuscript_assets_v3_round3.py"
M25 = SCRIPTS / "25_finalize_submission_figures_v3.py"
M26 = SCRIPTS / "26_final_artwork_qc_v3.py"

TARGET_WIDTH_IN = 5.15  # 130.8 mm source canvas, safely inside 140-mm journal width.
MAX_WIDTH_MM = 140.0
MAX_HEIGHT_MM = 200.0

EXPECTED = [
    "figure1_audit_framework_v3",
    "figure2_primary_effects_v3",
    "figure3_acyclic_sensitivity_v3",
    "figure4_dominant_fragment_sensitivity_v3",
    "figure5_candidate_budget_audit_v3",
    "figure6_collateral_diagnostics_v3",
    "figureS1_dataset_construction_v3",
    "figureS2_budget_semantics_v3",
    "figureS3_multicomponent_audit_v3",
    "figureS4_supporting_metrics_v3",
    "figureS5_model_seed_sensitivity_v3",
]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import plotting module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_module(module) -> None:
    """Apply one typography/size policy and absorb old Matplotlib spacing syntax."""
    module.plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8.2,
        "axes.titlesize": 8.8,
        "axes.labelsize": 8.1,
        "xtick.labelsize": 7.1,
        "ytick.labelsize": 7.1,
        "legend.fontsize": 6.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    original_figure = module.plt.figure
    original_subplots = module.plt.subplots

    def scale_figsize(figsize):
        if figsize is None:
            return None
        w, h = float(figsize[0]), float(figsize[1])
        if w <= TARGET_WIDTH_IN:
            return (w, h)
        scale = TARGET_WIDTH_IN / w
        return (TARGET_WIDTH_IN, h * scale)

    def final_figure(*args, **kwargs):
        if "figsize" in kwargs:
            kwargs["figsize"] = scale_figsize(kwargs["figsize"])
        return original_figure(*args, **kwargs)

    def final_subplots(*args, **kwargs):
        # Historical figure code occasionally passed wspace/hspace directly.
        wspace = kwargs.pop("wspace", None)
        hspace = kwargs.pop("hspace", None)
        if wspace is not None or hspace is not None:
            gridspec_kw = dict(kwargs.pop("gridspec_kw", {}) or {})
            if wspace is not None:
                gridspec_kw["wspace"] = wspace
            if hspace is not None:
                gridspec_kw["hspace"] = hspace
            kwargs["gridspec_kw"] = gridspec_kw
        if "figsize" in kwargs:
            kwargs["figsize"] = scale_figsize(kwargs["figsize"])
        return original_subplots(*args, **kwargs)

    module.plt.figure = final_figure
    module.plt.subplots = final_subplots


def draw_card(module, ax, xy, w, h, lines, fc, ec, fs=6.3, bold=True):
    """Fixed-layout card: short prewrapped labels, no post-hoc auto-shrinking."""
    patch = module.FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor=fc,
        edgecolor=ec,
        linewidth=0.9,
        clip_on=False,
    )
    ax.add_patch(patch)
    text = ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        lines,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight="bold" if bold else "normal",
        linespacing=1.02,
        clip_on=False,
    )
    return patch, text


def validate_cards(fig, cards) -> None:
    """Fail only for a genuine rendered overflow, with a small tolerance."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for name, patch, text in cards:
        p = patch.get_window_extent(renderer=renderer)
        t = text.get_window_extent(renderer=renderer)
        # Two-pixel tolerance avoids false failures from anti-aliased glyph bounds.
        if t.x0 < p.x0 - 2 or t.x1 > p.x1 + 2 or t.y0 < p.y0 - 2 or t.y1 > p.y1 + 2:
            raise AssertionError(f"Final Figure 1 card overflow: {name}")


def figure1(module) -> None:
    """Dedicated, generous Figure 1 layout at intended journal width."""
    fig = module.plt.figure(figsize=(TARGET_WIDTH_IN, 4.20))
    gs = fig.add_gridspec(2, 2, wspace=0.43, hspace=0.52)
    cards = []

    ax = fig.add_subplot(gs[0, 0])
    module.panel(ax, "A", "Audited molecular universe")
    y = module.np.arange(6)
    vals = [module.N[d] for d in module.DATASETS]
    cols = [module.C["navy"] if d in module.CLS else module.C["teal"] for d in module.DATASETS]
    ax.barh(y, vals, color=cols, height=0.56, zorder=2)
    ax.set_yticks(y, module.DATASETS)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Clean molecules")
    module.clean(ax, "x")
    for yy, value in zip(y, vals):
        ax.text(value * 1.06, yy, f"{value:,}", va="center", fontsize=6.0)
    ax.legend(
        handles=[
            module.Rectangle((0, 0), 1, 1, color=module.C["navy"], label="Classification"),
            module.Rectangle((0, 0), 1, 1, color=module.C["teal"], label="Regression"),
        ],
        frameon=False,
        loc="lower right",
    )

    ax = fig.add_subplot(gs[0, 1])
    ax.set_axis_off()
    module.panel(ax, "B", "Exact-size target-mean perturbation")
    p, t = draw_card(module, ax, (0.12, 0.78), 0.76, 0.14, "Target-blind\ncandidate pool", module.C["pale_blue"], module.C["navy"], 6.2)
    cards.append(("B-candidate", p, t))
    module.arrow(ax, (0.50, 0.77), (0.50, 0.66))
    p, t = draw_card(module, ax, (0.12, 0.53), 0.76, 0.14, "Same realized\ntest-set size", module.C["white"], module.C["mid"], 6.2)
    cards.append(("B-size", p, t))
    module.arrow(ax, (0.50, 0.52), (0.27, 0.39), module.C["gray"])
    module.arrow(ax, (0.50, 0.52), (0.73, 0.39), module.C["teal2"])
    p, t = draw_card(module, ax, (0.02, 0.18), 0.43, 0.18, "Size-matched\nbaseline", module.C["pale_gray"], module.C["mid"], 6.1)
    cards.append(("B-baseline", p, t))
    p, t = draw_card(module, ax, (0.55, 0.18), 0.43, 0.18, "Target-mean-aware\nminimum gap", module.C["pale_teal"], module.C["teal"], 6.1)
    cards.append(("B-balanced", p, t))
    ax.text(0.50, 0.04, "Frozen pre-outcome: seed · scaffold rule · search budget", transform=ax.transAxes, ha="center", fontsize=5.4, color=module.C["gray"])

    ax = fig.add_subplot(gs[1, 0])
    ax.set_axis_off()
    module.panel(ax, "C", "Pre-outcome freeze and inference")
    steps = [
        ("Budget\nfrozen", module.C["pale_orange"], module.C["orange"]),
        ("Manifest\n+ hash", module.C["pale_blue"], module.C["navy"]),
        ("Model\nfit", module.C["pale_gray"], module.C["mid"]),
        ("Paired\neffect", module.C["pale_teal"], module.C["teal"]),
    ]
    for i, (label, fc, ec) in enumerate(steps):
        x = 0.01 + i * 0.247
        p, t = draw_card(module, ax, (x, 0.62), 0.20, 0.19, label, fc, ec, 5.7)
        cards.append((f"C-step-{i}", p, t))
        if i < 3:
            module.arrow(ax, (x + 0.205, 0.715), (x + 0.238, 0.715))
    for name, x, label in [
        ("pairs", 0.03, "20 unique\npairs"),
        ("boot", 0.36, "10,000 paired\nbootstraps"),
        ("holm", 0.69, "Wilcoxon +\nHolm"),
    ]:
        p, t = draw_card(module, ax, (x, 0.30), 0.28, 0.14, label, module.C["white"], module.C["mid"], 5.5, False)
        cards.append((f"C-{name}", p, t))
    ax.text(0.50, 0.08, "Inferential N = unique partition pairs, not model seeds", transform=ax.transAxes, ha="center", fontsize=5.4, color=module.C["gray"])

    ax = fig.add_subplot(gs[1, 1])
    ax.set_axis_off()
    module.panel(ax, "D", "Predeclared protocol sensitivities")
    p, t = draw_card(module, ax, (0.08, 0.67), 0.84, 0.15, "Acyclic semantics\nsingle-group ↔ singleton", module.C["pale_blue"], module.C["navy"], 6.0)
    cards.append(("D-acyclic", p, t))
    p, t = draw_card(module, ax, (0.08, 0.45), 0.84, 0.15, "Record representation\nfull record ↔ fragment", module.C["pale_orange"], module.C["orange"], 6.0)
    cards.append(("D-record", p, t))
    module.arrow(ax, (0.37, 0.64), (0.46, 0.34), module.C["navy"])
    module.arrow(ax, (0.63, 0.42), (0.54, 0.34), module.C["orange"])
    p, t = draw_card(module, ax, (0.17, 0.17), 0.66, 0.15, "Does the scientific\nclaim survive?", module.C["pale_teal"], module.C["teal"], 6.0)
    cards.append(("D-claim", p, t))
    ax.text(0.50, 0.04, "Report disagreement; do not resolve it post hoc.", transform=ax.transAxes, ha="center", fontsize=5.4, color=module.C["gray"])

    fig.suptitle("Auditable molecular benchmark construction", fontsize=9.4, fontweight="bold", y=0.995)
    validate_cards(fig, cards)
    module.save(fig, "figure1_audit_framework_v3")


def make_tiffs() -> None:
    for stem in EXPECTED:
        png = FIG / f"{stem}.png"
        if not png.exists():
            raise FileNotFoundError(png)
        with Image.open(png) as image:
            image.save(FIG / f"{stem}.tiff", format="TIFF", compression="tiff_lzw", dpi=(600, 600))


def pdf_dimensions_mm(path: Path) -> tuple[float, float]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        from PyPDF2 import PdfReader  # type: ignore
    page = PdfReader(str(path)).pages[0]
    return float(page.mediabox.width) * 25.4 / 72.0, float(page.mediabox.height) * 25.4 / 72.0


def preflight() -> None:
    print("\nFINAL ARTWORK DIMENSIONS")
    for stem in EXPECTED:
        pdf = FIG / f"{stem}.pdf"
        tiff = FIG / f"{stem}.tiff"
        if not pdf.exists() or not tiff.exists():
            raise FileNotFoundError(f"Missing final artwork for {stem}")
        w, h = pdf_dimensions_mm(pdf)
        print(f"  {stem}: {w:.1f} x {h:.1f} mm")
        if w > MAX_WIDTH_MM + 0.5 or h > MAX_HEIGHT_MM + 0.5:
            raise AssertionError(f"Artwork envelope exceeded: {stem} = {w:.1f} x {h:.1f} mm")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)

    m21 = load_module(M21, "paper1_final_base_figures")
    configure_module(m21)
    # Base module supplies Figure 3 and the first three SI visual summaries.
    m21.figure3()
    m21.supplementary_figures()

    m25 = load_module(M25, "paper1_final_effect_figures")
    configure_module(m25)
    # Use the polished primary-effect and SI robustness versions exactly once.
    m25.figure2()
    m25.figure_s4()
    m25.figure_s5()

    m26 = load_module(M26, "paper1_final_diagnostic_figures")
    configure_module(m26)
    # Use the final diagnostic implementations exactly once.
    m26.figure4()
    m26.figure5()
    m26.figure6()

    # Figure 1 is deliberately independent of historical patch layers.
    figure1(m21)

    make_tiffs()
    preflight()
    print("SUBMISSION-FINAL ARTWORK BUILDER: PASS")


if __name__ == "__main__":
    main()
