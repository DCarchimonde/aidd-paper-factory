from __future__ import annotations

"""Paper 1 publication-final runner.

Consumes frozen scientific artifacts, regenerates manuscript tables/narratives,
runs one independent publication-artwork pass, applies journal gates, compiles
main/SI, and packages the submission bundle. No model fit or partition generation
is invoked.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "paper1_leakage_benchmark" / "scripts"
LEGACY = SCRIPTS / "22_build_paper1_q1_final_v3.py"
ARTWORK = SCRIPTS / "31_publication_artwork_final_v3.py"
GATE = SCRIPTS / "24_q1_submission_gate_v3.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("paper1_submission_library", LEGACY)
    if spec is None or spec.loader is None:
        raise RuntimeError(LEGACY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    r = load_runner()
    print("=" * 78)
    print("PAPER 1 PUBLICATION-FINAL BUILD — FROZEN SCIENCE / ONE ARTWORK PASS")
    print("=" * 78)
    r.verify_frozen_inputs()
    r.reset_outputs()
    r.refresh_manuscript_assets()
    r.run([sys.executable, str(ARTWORK)], cwd=ROOT)
    r.run([sys.executable, str(GATE)], cwd=ROOT)
    r.compile_latex()
    r.run([sys.executable, str(GATE), "--post-build"], cwd=ROOT)
    r.package_submission()
    print("\n" + "=" * 78)
    print("PAPER 1 PUBLICATION-FINAL BUILD: PASS")
    print("Main PDF:", r.BUNDLE / "main.pdf")
    print("SI PDF  :", r.BUNDLE / "supplementary.pdf")
    print("Manifest:", r.BUNDLE / "BUILD_MANIFEST.json")
    print("=" * 78)


if __name__ == "__main__":
    main()
