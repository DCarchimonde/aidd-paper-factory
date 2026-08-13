from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "paper1_leakage_benchmark" / "scripts"
FIG = ROOT / "paper1_leakage_benchmark" / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
WIDTH_IN = 5.25
MAX_WIDTH_MM = 140.0
MAX_HEIGHT_MM = 200.0

C = {
    "ink": "#20313A", "navy": "#315B73", "navy2": "#24485D",
    "teal": "#2B8C82", "teal2": "#176B64", "orange": "#D58A43",
    "orange2": "#A85F28", "pale_blue": "#EAF1F5", "pale_teal": "#E8F3EF",
    "pale_orange": "#FBF0E5", "pale_gray": "#F4F6F7", "gray": "#6D7A81",
    "mid": "#B8C2C7", "grid": "#E5EAEC", "white": "#FFFFFF", "purple": "#7A6F9B",
}
EXPECTED = [
    "figure1_audit_framework_v3", "figure2_primary_effects_v3",
    "figure3_acyclic_sensitivity_v3", "figure4_dominant_fragment_sensitivity_v3",
    "figure5_candidate_budget_audit_v3", "figure6_collateral_diagnostics_v3",
    "figureS1_dataset_construction_v3", "figureS2_budget_semantics_v3",
    "figureS3_multicomponent_audit_v3", "figureS4_supporting_metrics_v3",
    "figureS5_model_seed_sensitivity_v3",
]


def load(filename: str, name: str):
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8.0, "axes.titlesize": 8.8, "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "legend.fontsize": 6.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": C["mid"], "axes.labelcolor": C["ink"], "text.color": C["ink"],
        "xtick.color": C["ink"], "ytick.color": C["ink"], "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def panel(ax, letter: str, title: str) -> None:
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=8.8, fontweight="bold", pad=6)


def clean(ax, axis="x") -> None:
    ax.grid(axis=axis, color=C["grid"], lw=0.65, zorder=0)


def card(ax, xy, w, h, text, fc, ec, fs=6.7):
    patch = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.010,rounding_size=0.018",
                           transform=ax.transAxes, facecolor=fc, edgecolor=ec, linewidth=0.85)
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fs, fontweight="bold", linespacing=1.02)


def arrow(ax, start, end, color=None):
    ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes, arrowstyle="-|>",
                                 mutation_scale=9, linewidth=0.9, color=color or C["gray"]))


def save(fig, stem: str) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fb = fig.bbox
    bad = []
    for artist in fig.findobj(match=lambda x: isinstance(x, plt.Text)):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        bb = artist.get_window_extent(renderer=renderer)
        if bb.x0 < fb.x0 - 3 or bb.y0 < fb.y0 - 3 or bb.x1 > fb.x1 + 3 or bb.y1 > fb.y1 + 3:
            bad.append(artist.get_text().replace("\n", " / ")[:70])
    if bad:
        raise AssertionError(f"Rendered text outside fixed canvas in {stem}: {bad[:5]}")
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(FIG / f"{stem}.png", dpi=600)
    plt.close(fig)
    print(FIG / f"{stem}.pdf")


def finish() -> None:
    failures = []
    print("\nFINAL ARTWORK DIMENSIONS")
    for stem in EXPECTED:
        png = FIG / f"{stem}.png"
        pdf = FIG / f"{stem}.pdf"
        if not png.exists() or not pdf.exists() or pdf.stat().st_size == 0:
            failures.append(f"{stem}: missing output")
            continue
        with Image.open(png) as image:
            image.save(FIG / f"{stem}.tiff", format="TIFF", compression="tiff_lzw", dpi=(600, 600))
            w = image.width * 25.4 / 600.0
            h = image.height * 25.4 / 600.0
        print(f"  {stem}: {w:.1f} x {h:.1f} mm")
        if w > MAX_WIDTH_MM or h > MAX_HEIGHT_MM:
            failures.append(f"{stem}: {w:.1f} x {h:.1f} mm")
    if failures:
        raise AssertionError("Artwork preflight failed: " + "; ".join(failures))
