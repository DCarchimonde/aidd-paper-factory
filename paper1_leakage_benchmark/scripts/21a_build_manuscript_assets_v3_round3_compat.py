from __future__ import annotations

"""Compatibility + Journal of Chemometrics artwork wrapper for Paper 1.

The frozen round-3 builder is retained as the scientific plotting source.  This
wrapper adapts Matplotlib spacing across versions, switches publication lettering
to an Arial-style sans serif, and caps the source canvas width so the exported
artwork is generated close to the journal's intended reproduction width rather
than being designed oversized and reduced later.
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "paper1_leakage_benchmark" / "scripts" / "21_build_manuscript_assets_v3_round3.py"
JOC_CANVAS_WIDTH_IN = 5.30  # leaves room for tight-bbox labels under the 140-mm journal limit


def load_target():
    spec = importlib.util.spec_from_file_location("paper1_round3_assets", TARGET)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import figure builder: {TARGET}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_target()
    module.plt.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    })

    original_figure = module.plt.figure
    original_subplots = module.plt.subplots

    def journal_figure(*args, **kwargs):
        figsize = kwargs.get("figsize")
        if figsize is not None and float(figsize[0]) > JOC_CANVAS_WIDTH_IN:
            kwargs["figsize"] = (JOC_CANVAS_WIDTH_IN, float(figsize[1]))
        return original_figure(*args, **kwargs)

    def compatible_subplots(*args, **kwargs):
        wspace = kwargs.pop("wspace", None)
        hspace = kwargs.pop("hspace", None)
        if wspace is not None or hspace is not None:
            gridspec_kw = dict(kwargs.pop("gridspec_kw", {}) or {})
            if wspace is not None:
                gridspec_kw["wspace"] = wspace
            if hspace is not None:
                gridspec_kw["hspace"] = hspace
            kwargs["gridspec_kw"] = gridspec_kw
        figsize = kwargs.get("figsize")
        if figsize is not None and float(figsize[0]) > JOC_CANVAS_WIDTH_IN:
            kwargs["figsize"] = (JOC_CANVAS_WIDTH_IN, float(figsize[1]))
        return original_subplots(*args, **kwargs)

    module.plt.figure = journal_figure
    module.plt.subplots = compatible_subplots
    module.main()
    print("ROUND-3 JOC ARTWORK COMPATIBILITY WRAPPER: PASS")


if __name__ == "__main__":
    main()
