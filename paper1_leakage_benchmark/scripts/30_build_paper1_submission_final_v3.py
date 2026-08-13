from __future__ import annotations

"""Final Paper 1 submission runner.

Consumes only frozen scientific artifacts, uses one strict artwork builder, compiles
main/SI, runs the journal gates, and packages the submission bundle. No model fit,
partition generation, or inferential recomputation is invoked here.
"""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "paper1_leakage_benchmark" / "scripts"
LEGACY_RUNNER = SCRIPTS / "22_build_paper1_q1_final_v3.py"
ARTWORK = SCRIPTS / "29_build_submission_artwork_strict_v3.py"
GATE = SCRIPTS / "24_q1_submission_gate_v3.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("paper1_submission_runner_lib", LEGACY_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import runner library: {LEGACY_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    r = load_runner()
    print("=" * 78)
    print("PAPER 1 SUBMISSION-FINAL BUILD — FROZEN SCIENCE / ONE STRICT ARTWORK PASS")
    print("=" * 78)

    r.verify_frozen_inputs()
    r.reset_outputs()
    r.refresh_manuscript_assets()

    r.run([sys.executable, str(ARTWORK)], cwd=ROOT)
    print("AUTHORITATIVE STRICT FINAL ARTWORK: PASS")

    r.run([sys.executable, str(GATE)], cwd=ROOT)
    r.compile_latex()
    r.run([sys.executable, str(GATE), "--post-build"], cwd=ROOT)
    r.package_submission()

    manifest_path = r.BUNDLE / "BUILD_MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["final_artwork_builder"] = "paper1_leakage_benchmark/scripts/29_build_submission_artwork_strict_v3.py"
        manifest["pipeline"] = "frozen science -> strict artwork -> journal gate -> LaTeX -> post-build gate -> bundle"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("PAPER 1 SUBMISSION-FINAL BUILD: PASS")
    print("Main PDF:", r.BUNDLE / "main.pdf")
    print("SI PDF  :", r.BUNDLE / "supplementary.pdf")
    print("Manifest:", manifest_path)
    print("=" * 78)


if __name__ == "__main__":
    main()
