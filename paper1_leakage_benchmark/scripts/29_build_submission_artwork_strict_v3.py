from __future__ import annotations

"""Authoritative strict submission-artwork pass for Paper 1.

This is a thin orchestration layer over the consolidated artwork utilities in
script 28.  It keeps the scientific state frozen, renders each final figure once,
and adds two practical safeguards for the Journal of Chemometrics envelope:
(1) a 5.00-inch source width for statistical/SI figures, and
(2) compact final suptitles for the two SI figures whose development-stage titles
can otherwise widen a tight bounding box beyond 140 mm.
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
    """Shorten only SI suptitles that can dominate the tight-bbox width."""
    def save(fig, stem: str) -> None:
        title = getattr(fig, "_suptitle", None)
        if title is not None:
            if stem == "figureS4_supporting_metrics_v3":
                title.set_text("Supporting metrics: metric-dependent effect patterns")
                title.set_fontsize(8.5)
            elif stem == "figureS5_model_seed_sensitivity_v3":
                title.set_text("RF/XGB model-seed sensitivity")
                title.set_fontsize(8.5)
        original_save(fig, stem)
    return save


def main() -> None:
    base = load_base()

    # A slightly narrower source canvas gives a deterministic safety margin below
    # the 140-mm final-width limit after labels/colorbars are included by tight_bbox.
    base.TARGET_WIDTH_IN = 5.00

    m21 = base.load_module(base.M21, "paper1_final_base_figures_strict")
    base.configure_module(m21)
    m21.figure3()
    m21.supplementary_figures()

    m25 = base.load_module(base.M25, "paper1_final_effect_figures_strict")
    base.configure_module(m25)
    original_save = m25.save
    m25.save = compact_si_save(m25, original_save)
    m25.figure2()
    m25.figure_s4()
    m25.figure_s5()

    m26 = base.load_module(base.M26, "paper1_final_diagnostic_figures_strict")
    base.configure_module(m26)
    m26.figure4()
    m26.figure5()
    m26.figure6()

    # Keep the dedicated Figure 1 at its already validated, roomier 5.15-inch
    # layout; card overflow is checked by the renderer before export.
    base.TARGET_WIDTH_IN = 5.15
    base.figure1(m21)

    base.make_tiffs()
    base.preflight()
    print("SUBMISSION-FINAL STRICT ARTWORK BUILDER: PASS")


if __name__ == "__main__":
    main()
