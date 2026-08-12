from __future__ import annotations

"""Journal of Chemometrics final artwork pass for Paper 1.

This pass is presentation-only.  It does not read model predictions to recompute
scientific results; it reruns the already-frozen plotting functions with journal-
compatible typography and intended-size canvases, replaces Figure 1 with a layout
that is robust at 140-mm reproduction width, emits 600-dpi TIFF companions, and
fails if any final PDF exceeds the journal's 140 mm x 200 mm artwork envelope.
"""

import importlib.util
import textwrap
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper1_leakage_benchmark"
SCRIPTS = PAPER / "scripts"
FIG = PAPER / "results" / "figures"

MODULE25 = SCRIPTS / "25_finalize_submission_figures_v3.py"
MODULE26 = SCRIPTS / "26_final_artwork_qc_v3.py"

CANVAS_WIDTH_IN = 5.15  # 130.8 mm before tight-bbox expansion
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


def apply_journal_matplotlib(module) -> None:
    """Use an Arial-family scientific figure font and cap source canvas width."""
    module.plt.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    original_figure = module.plt.figure
    original_subplots = module.plt.subplots

    def journal_figure(*args, **kwargs):
        figsize = kwargs.get("figsize")
        if figsize is not None and float(figsize[0]) > CANVAS_WIDTH_IN:
            kwargs["figsize"] = (CANVAS_WIDTH_IN, float(figsize[1]))
        return original_figure(*args, **kwargs)

    def journal_subplots(*args, **kwargs):
        figsize = kwargs.get("figsize")
        if figsize is not None and float(figsize[0]) > CANVAS_WIDTH_IN:
            kwargs["figsize"] = (CANVAS_WIDTH_IN, float(figsize[1]))
        return original_subplots(*args, **kwargs)

    module.plt.figure = journal_figure
    module.plt.subplots = journal_subplots


def fitted_box(module, ax, xy, w, h, text, fc, ec, fs=7.0, bold=False):
    """Draw a rounded card and guarantee that its label stays inside the card."""
    if text == "Does the scientific claim survive?":
        text = "Does the scientific\nclaim survive?"
    elif "\n" not in text and len(text) > 28:
        text = textwrap.fill(text, width=24)

    patch = module.FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor=fc,
        edgecolor=ec,
        linewidth=0.9,
    )
    ax.add_patch(patch)
    artist = ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight="bold" if bold else "normal",
        linespacing=1.03,
        clip_on=False,
    )

    # Shrink only when necessary; never allow text to cross the card boundary.
    for _ in range(14):
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        card = patch.get_window_extent(renderer=renderer)
        label = artist.get_window_extent(renderer=renderer)
        if label.width <= card.width - 6 and label.height <= card.height - 4:
            return artist
        new_fs = artist.get_fontsize() - 0.22
        if new_fs < 5.8:
            break
        artist.set_fontsize(new_fs)
    raise AssertionError(f"Figure card text does not fit: {text!r}")


def journal_figure1(module) -> None:
    """Redraw Figure 1 specifically for the journal's 140-mm reproduction width."""
    fig = module.plt.figure(figsize=(CANVAS_WIDTH_IN, 5.85))
    gs = fig.add_gridspec(2, 2, wspace=0.44, hspace=0.50)

    ax = fig.add_subplot(gs[0, 0])
    module.panel(ax, "A", "Audited molecular universe", letter_x=-0.10)
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
        ax.text(value * 1.06, yy, f"{value:,}", va="center", fontsize=6.4)
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
    module.panel(ax, "B", "Exact-size target-mean perturbation", letter_x=-0.08)
    fitted_box(module, ax, (0.18, 0.78), 0.64, 0.12, "Target-blind candidate pool", module.C["pale_blue"], module.C["navy"], 6.8, True)
    module.arrow(ax, (0.50, 0.77), (0.50, 0.67))
    fitted_box(module, ax, (0.18, 0.53), 0.64, 0.12, "Same realized test size", module.C["white"], module.C["mid"], 6.8, True)
    module.arrow(ax, (0.50, 0.52), (0.25, 0.40), module.C["gray"])
    module.arrow(ax, (0.50, 0.52), (0.75, 0.40), module.C["teal2"])
    fitted_box(module, ax, (0.02, 0.21), 0.42, 0.16, "Size-matched\nbaseline", module.C["pale_gray"], module.C["mid"], 6.8, True)
    fitted_box(module, ax, (0.56, 0.21), 0.42, 0.16, "Target-mean-aware\nminimum gap", module.C["pale_teal"], module.C["teal"], 6.8, True)
    ax.text(0.50, 0.06, "Fixed pre-outcome: seed · scaffold rule · search budget", transform=ax.transAxes, ha="center", fontsize=5.9, color=module.C["gray"])

    ax = fig.add_subplot(gs[1, 0])
    ax.set_axis_off()
    module.panel(ax, "C", "Pre-outcome freeze and inference", letter_x=-0.08)
    steps = [
        ("Budget\nfrozen", module.C["pale_orange"], module.C["orange"]),
        ("Manifest\n+ hash", module.C["pale_blue"], module.C["navy"]),
        ("Model\nfit", module.C["pale_gray"], module.C["mid"]),
        ("Paired\neffect", module.C["pale_teal"], module.C["teal"]),
    ]
    for i, (txt, fc, ec) in enumerate(steps):
        x = 0.005 + i * 0.249
        fitted_box(module, ax, (x, 0.64), 0.205, 0.17, txt, fc, ec, 6.0, True)
        if i < 3:
            module.arrow(ax, (x + 0.207, 0.725), (x + 0.240, 0.725))
    fitted_box(module, ax, (0.03, 0.33), 0.28, 0.12, "20 unique\npairs", module.C["white"], module.C["mid"], 5.9)
    fitted_box(module, ax, (0.36, 0.33), 0.28, 0.12, "10,000 paired\nbootstraps", module.C["white"], module.C["mid"], 5.9)
    fitted_box(module, ax, (0.69, 0.33), 0.28, 0.12, "Wilcoxon +\nHolm", module.C["white"], module.C["mid"], 5.9)
    ax.text(0.50, 0.11, "Inferential N = unique partition pairs, not model seeds", transform=ax.transAxes, ha="center", fontsize=5.8, color=module.C["gray"])

    ax = fig.add_subplot(gs[1, 1])
    ax.set_axis_off()
    module.panel(ax, "D", "Predeclared protocol sensitivities", letter_x=-0.08)
    left = fitted_box(module, ax, (0.08, 0.66), 0.84, 0.13, "Acyclic semantics\nsingle-group ↔ singleton", module.C["pale_blue"], module.C["navy"], 6.6, True)
    right = fitted_box(module, ax, (0.08, 0.45), 0.84, 0.13, "Record representation\nfull record ↔ fragment", module.C["pale_orange"], module.C["orange"], 6.6, True)
    module.arrow(ax, (0.36, 0.64), (0.45, 0.34), module.C["navy"])
    module.arrow(ax, (0.64, 0.43), (0.55, 0.34), module.C["orange"])
    fitted_box(module, ax, (0.18, 0.18), 0.64, 0.14, "Does the scientific claim survive?", module.C["pale_teal"], module.C["teal"], 6.6, True)
    ax.text(0.50, 0.06, "Report disagreement; do not resolve it post hoc.", transform=ax.transAxes, ha="center", fontsize=5.9, color=module.C["gray"])

    # Explicitly force a draw so all fitted-card checks execute before export.
    fig.canvas.draw()
    fig.suptitle("Benchmark construction as a controlled chemometric measurement process", fontsize=9.7, fontweight="bold", y=0.995)
    module.save(fig, "figure1_audit_framework_v3")


def make_tiffs() -> None:
    for stem in EXPECTED:
        png = FIG / f"{stem}.png"
        if not png.exists():
            raise FileNotFoundError(png)
        with Image.open(png) as image:
            image.save(
                FIG / f"{stem}.tiff",
                format="TIFF",
                compression="tiff_lzw",
                dpi=(600, 600),
            )


def pdf_dimensions_mm(path: Path) -> tuple[float, float]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        from PyPDF2 import PdfReader  # type: ignore
    page = PdfReader(str(path)).pages[0]
    width_pt = float(page.mediabox.width)
    height_pt = float(page.mediabox.height)
    return width_pt * 25.4 / 72.0, height_pt * 25.4 / 72.0


def preflight_dimensions() -> None:
    for stem in EXPECTED:
        pdf = FIG / f"{stem}.pdf"
        if not pdf.exists():
            raise FileNotFoundError(pdf)
        width, height = pdf_dimensions_mm(pdf)
        print(f"  {stem}: {width:.1f} x {height:.1f} mm")
        if width > MAX_WIDTH_MM + 0.5 or height > MAX_HEIGHT_MM + 0.5:
            raise AssertionError(
                f"Journal artwork envelope exceeded by {stem}: {width:.1f} x {height:.1f} mm"
            )


def main() -> None:
    # Finalize Figure 2 and SI diagnostic figures using the journal font/size.
    m25 = load_module(MODULE25, "paper1_finalize_figures_joc")
    apply_journal_matplotlib(m25)
    m25.main()

    # Finalize Figures 1, 4, 5, 6. Figure 1 receives a dedicated compact layout.
    m26 = load_module(MODULE26, "paper1_artwork_qc_joc")
    apply_journal_matplotlib(m26)
    m26.box = lambda ax, xy, w, h, text, fc, ec, fs=7.0, bold=False: fitted_box(
        m26, ax, xy, w, h, text, fc, ec, fs, bold
    )
    m26.figure1 = lambda: journal_figure1(m26)
    m26.main()

    make_tiffs()
    preflight_dimensions()
    print("JOURNAL OF CHEMOMETRICS ARTWORK PREFLIGHT: PASS")


if __name__ == "__main__":
    main()
