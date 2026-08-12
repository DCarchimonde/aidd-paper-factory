from __future__ import annotations

"""Journal of Chemometrics final artwork pass for Paper 1.

Presentation-only pass. Scientific results are not recomputed. The pass applies
journal-oriented typography, redraws Figure 1 at intended reproduction size,
emits 600-dpi TIFF companions, and verifies the final artwork envelope.
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

# Keep a small safety margin below the 140-mm maximum reproduction width. Tight
# bounding-box export can add a few millimetres around panel letters and labels.
CANVAS_WIDTH_IN = 5.15
MAX_WIDTH_MM = 140.0
MAX_HEIGHT_MM = 200.0
MIN_CARD_FONT = 5.8

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
    """Use a consistent Arial-family scientific figure font and cap canvas width."""
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


def _wrap_card_text(text: str, width_chars: int = 18) -> str:
    """Wrap every logical line so compact journal cards stay readable."""
    specials = {
        "Target-blind candidate pool": "Target-blind\ncandidate pool",
        "Same realized test size": "Same realized\ntest size",
        "Does the scientific claim survive?": "Does the scientific\nclaim survive?",
    }
    if text in specials:
        return specials[text]
    pieces: list[str] = []
    for logical_line in text.split("\n"):
        if len(logical_line) <= width_chars:
            pieces.append(logical_line)
        else:
            wrapped = textwrap.wrap(
                logical_line,
                width=width_chars,
                break_long_words=False,
                break_on_hyphens=False,
            )
            pieces.extend(wrapped or [logical_line])
    return "\n".join(pieces)


def fitted_box(module, ax, xy, w, h, text, fc, ec, fs=7.0, bold=False):
    """Draw a rounded card and prove that its label fits inside at final size."""
    # Width-aware wrapping is preferable to silently shrinking long labels to an
    # unreadable size. Wider cards get a slightly more generous character budget.
    char_budget = max(13, min(24, int(round(18 * w / 0.64))))
    text = _wrap_card_text(text, char_budget)

    patch = module.FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor=fc,
        edgecolor=ec,
        linewidth=0.9,
        clip_on=True,
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
        linespacing=1.01,
        clip_on=True,
    )

    # Measure real renderer extents. A 2-pixel internal margin is enough here
    # because the text is centred and the rounded-box pad is already included.
    for _ in range(20):
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        card = patch.get_window_extent(renderer=renderer)
        label = artist.get_window_extent(renderer=renderer)
        if label.width <= card.width - 4 and label.height <= card.height - 4:
            return artist
        new_fs = artist.get_fontsize() - 0.18
        if new_fs < MIN_CARD_FONT:
            break
        artist.set_fontsize(new_fs)
    raise AssertionError(
        f"Figure card text does not fit after wrapping: {text!r}; "
        f"card={w:.2f}x{h:.2f} axes fraction"
    )


def journal_figure1(module) -> None:
    """Redraw Figure 1 specifically for compact journal reproduction."""
    fig = module.plt.figure(figsize=(CANVAS_WIDTH_IN, 5.72))
    gs = fig.add_gridspec(2, 2, wspace=0.46, hspace=0.50)

    # A — molecular universe
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
        ax.text(value * 1.06, yy, f"{value:,}", va="center", fontsize=6.3)
    ax.legend(
        handles=[
            module.Rectangle((0, 0), 1, 1, color=module.C["navy"], label="Classification"),
            module.Rectangle((0, 0), 1, 1, color=module.C["teal"], label="Regression"),
        ],
        frameon=False,
        loc="lower right",
    )

    # B — exact-size pair. Explicit two-line labels avoid the failure mode seen
    # with the single-line candidate-pool label at compact width.
    ax = fig.add_subplot(gs[0, 1])
    ax.set_axis_off()
    module.panel(ax, "B", "Exact-size target-mean selection", letter_x=-0.08)
    fitted_box(module, ax, (0.15, 0.76), 0.70, 0.15, "Target-blind\ncandidate pool", module.C["pale_blue"], module.C["navy"], 6.7, True)
    module.arrow(ax, (0.50, 0.75), (0.50, 0.66))
    fitted_box(module, ax, (0.17, 0.51), 0.66, 0.15, "Same realized\ntest size", module.C["white"], module.C["mid"], 6.7, True)
    module.arrow(ax, (0.50, 0.50), (0.25, 0.39), module.C["gray"])
    module.arrow(ax, (0.50, 0.50), (0.75, 0.39), module.C["teal2"])
    fitted_box(module, ax, (0.01, 0.19), 0.45, 0.17, "Size-matched\nbaseline", module.C["pale_gray"], module.C["mid"], 6.6, True)
    fitted_box(module, ax, (0.54, 0.19), 0.45, 0.17, "Target-mean-aware\nminimum gap", module.C["pale_teal"], module.C["teal"], 6.6, True)
    ax.text(
        0.50, 0.055,
        "Frozen pre-outcome: seed · scaffold rule · search budget",
        transform=ax.transAxes, ha="center", fontsize=5.7, color=module.C["gray"],
    )

    # C — freeze and inference
    ax = fig.add_subplot(gs[1, 0])
    ax.set_axis_off()
    module.panel(ax, "C", "Freeze and partition-level inference", letter_x=-0.08)
    steps = [
        ("Budget\nfrozen", module.C["pale_orange"], module.C["orange"]),
        ("Manifest\n+ hash", module.C["pale_blue"], module.C["navy"]),
        ("Model\nfit", module.C["pale_gray"], module.C["mid"]),
        ("Paired\neffect", module.C["pale_teal"], module.C["teal"]),
    ]
    for i, (txt, fc, ec) in enumerate(steps):
        x = 0.005 + i * 0.249
        fitted_box(module, ax, (x, 0.64), 0.205, 0.18, txt, fc, ec, 5.9, True)
        if i < 3:
            module.arrow(ax, (x + 0.207, 0.73), (x + 0.240, 0.73))
    fitted_box(module, ax, (0.02, 0.32), 0.29, 0.14, "20 unique\npairs", module.C["white"], module.C["mid"], 5.8)
    fitted_box(module, ax, (0.355, 0.32), 0.29, 0.14, "10,000 paired\nbootstraps", module.C["white"], module.C["mid"], 5.8)
    fitted_box(module, ax, (0.69, 0.32), 0.29, 0.14, "Wilcoxon +\nHolm", module.C["white"], module.C["mid"], 5.8)
    ax.text(
        0.50, 0.10,
        "Inferential N = unique partition pairs, not model seeds",
        transform=ax.transAxes, ha="center", fontsize=5.65, color=module.C["gray"],
    )

    # D — stack the two sensitivity cards vertically. This removes the last
    # horizontal collision risk at narrow journal width.
    ax = fig.add_subplot(gs[1, 1])
    ax.set_axis_off()
    module.panel(ax, "D", "Predeclared protocol sensitivities", letter_x=-0.08)
    fitted_box(module, ax, (0.08, 0.66), 0.84, 0.14, "Acyclic semantics\nsingle-group ↔ singleton", module.C["pale_blue"], module.C["navy"], 6.3, True)
    fitted_box(module, ax, (0.08, 0.45), 0.84, 0.14, "Record representation\nfull record ↔ fragment", module.C["pale_orange"], module.C["orange"], 6.3, True)
    module.arrow(ax, (0.36, 0.64), (0.45, 0.35), module.C["navy"])
    module.arrow(ax, (0.64, 0.43), (0.55, 0.35), module.C["orange"])
    fitted_box(module, ax, (0.17, 0.17), 0.66, 0.16, "Does the scientific\nclaim survive?", module.C["pale_teal"], module.C["teal"], 6.3, True)
    ax.text(
        0.50, 0.055,
        "Report disagreement; do not resolve it post hoc.",
        transform=ax.transAxes, ha="center", fontsize=5.7, color=module.C["gray"],
    )

    fig.canvas.draw()  # execute all card-fit checks before export
    fig.suptitle(
        "Controlled chemometric audit of benchmark construction",
        fontsize=9.4, fontweight="bold", y=0.992,
    )
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
    # Figure 2 and supplementary diagnostics with journal font/size.
    m25 = load_module(MODULE25, "paper1_finalize_figures_joc")
    apply_journal_matplotlib(m25)
    m25.main()

    # Figures 1, 4, 5 and 6. Figure 1 uses the compact dedicated layout above.
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
