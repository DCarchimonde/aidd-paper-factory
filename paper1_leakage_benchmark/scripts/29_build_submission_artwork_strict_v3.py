from __future__ import annotations

"""Authoritative strict submission-artwork pass for Paper 1.

This is a thin orchestration layer over the consolidated artwork utilities in
script 28. It keeps the scientific state frozen and renders each final figure
exactly once. Statistical/SI figures use conservative source widths so labels,
legends, and colorbars included by ``bbox_inches='tight'`` remain comfortably
inside the Journal of Chemometrics 140-mm artwork envelope.
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "paper1_leakage_benchmark" / "scripts"
BASE = SCRIPTS / "28_build_submission_final_artwork_v3.py"


def load_base():
    spec = importlib.util.spec_from_file_location("paper1_submission_artwork_base", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import final artwork builder: {BASE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compact_si_save(module, original_save):
    """Use concise SI suptitles so title text cannot dominate tight-bbox width."""
    def save(fig, stem: str) -> None:
        title = getattr(fig, "_suptitle", None)
        if title is not None:
            if stem == "figureS4_supporting_metrics_v3":
                title.set_text("Supporting metrics: metric-dependent effects")
                title.set_fontsize(8.0)
            elif stem == "figureS5_model_seed_sensitivity_v3":
                title.set_text("RF/XGB model-seed sensitivity")
                title.set_fontsize(8.0)
        original_save(fig, stem)
    return save


def main() -> None:
    base = load_base()

    # Base figures already had comfortable margin below 140 mm at 5.00 inches.
    base.TARGET_WIDTH_IN = 5.00
    m21 = base.load_module(base.M21, "paper1_final_base_figures_strict")
    base.configure_module(m21)
    m21.figure3()
    m21.supplementary_figures()

    # S4's colorbar + long row labels make its tight bounding box much wider than
    # its nominal canvas. The previous 5.00-inch render measured 140.8 mm. Use
    # 4.65 inches for the complete m25 group, giving a robust margin rather than
    # living on the 140-mm boundary. Font point sizes are unchanged.
    base.TARGET_WIDTH_IN = 4.65
    m25 = base.load_module(base.M25, "paper1_final_effect_figures_strict")
    base.configure_module(m25)
    original_save = m25.save
    m25.save = compact_si_save(m25, original_save)
    m25.figure2()
    m25.figure_s4()
    m25.figure_s5()

    # Diagnostic figures remain at 5.00 inches; all previously measured <=129 mm.
    base.TARGET_WIDTH_IN = 5.00
    m26 = base.load_module(base.M26, "paper1_final_diagnostic_figures_strict")
    base.configure_module(m26)
    m26.figure4()
    m26.figure5()
    m26.figure6()

    # Figure 1 keeps its separately validated roomier layout; card overflow is
    # checked by the renderer before export.
    base.TARGET_WIDTH_IN = 5.15
    base.figure1(m21)

    base.make_tiffs()
    base.preflight()
    print("SUBMISSION-FINAL STRICT ARTWORK BUILDER: PASS")


if __name__ == "__main__":
    main()
