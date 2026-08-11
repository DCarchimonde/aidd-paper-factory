from __future__ import annotations

"""Compatibility wrapper for the Paper 1 round-3 figure builder.

Some Matplotlib versions reject ``wspace``/``hspace`` when they are passed
straight to ``plt.subplots``.  The round-3 figure builder uses those kwargs in
supplementary figure construction.  This wrapper translates them into
``gridspec_kw`` before executing the frozen plotting module, preserving the
figure design while making the build portable across Matplotlib releases.
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "paper1_leakage_benchmark" / "scripts" / "21_build_manuscript_assets_v3_round3.py"


def load_target():
    spec = importlib.util.spec_from_file_location("paper1_round3_assets", TARGET)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import figure builder: {TARGET}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_target()
    original_subplots = module.plt.subplots

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
        return original_subplots(*args, **kwargs)

    module.plt.subplots = compatible_subplots
    module.main()
    print("ROUND-3 MATPLOTLIB COMPATIBILITY WRAPPER: PASS")


if __name__ == "__main__":
    main()
