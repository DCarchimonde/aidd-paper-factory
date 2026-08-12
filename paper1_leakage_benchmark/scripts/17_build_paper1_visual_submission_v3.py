from __future__ import annotations

import ast
import py_compile
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "paper1_leakage_benchmark" / "scripts"
FINAL_RUNNER = SCRIPT_DIR / "22_build_paper1_q1_final_v3.py"

# Scripts directly executed by the submission-final pipeline.
ENTRY_SCRIPTS = [
    SCRIPT_DIR / "17_build_paper1_visual_submission_v3.py",
    SCRIPT_DIR / "20b_build_q1_tex_tables_v3.py",
    SCRIPT_DIR / "20c_write_q1_result_text_v3.py",
    SCRIPT_DIR / "22_build_paper1_q1_final_v3.py",
    SCRIPT_DIR / "24_q1_submission_gate_v3.py",
    SCRIPT_DIR / "28_build_submission_final_artwork_v3.py",
]

# Historical plotting modules are now function libraries only. The authoritative
# builder adapts their old Matplotlib spacing syntax internally, so they receive
# syntax compilation but not the direct-plt.subplots API rejection gate.
IMPORTED_FIGURE_LIBS = [
    SCRIPT_DIR / "21_build_manuscript_assets_v3_round3.py",
    SCRIPT_DIR / "25_finalize_submission_figures_v3.py",
    SCRIPT_DIR / "26_final_artwork_qc_v3.py",
]


def matplotlib_api_gate(script: Path) -> None:
    tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    bad: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_subplots = (
            isinstance(func, ast.Attribute)
            and func.attr == "subplots"
            and isinstance(func.value, ast.Name)
            and func.value.id == "plt"
        )
        if not is_subplots:
            continue
        direct = sorted(kw.arg for kw in node.keywords if kw.arg in {"wspace", "hspace"})
        if direct:
            bad.append(f"line {getattr(node, 'lineno', '?')}: {', '.join(direct)}")
    if bad:
        raise AssertionError(
            f"Matplotlib compatibility gate failed for {script.name}: direct spacing kwargs remain. "
            + "; ".join(bad)
        )


def main() -> None:
    print("PAPER 1 ENTRYPOINT -> SUBMISSION-FINAL FROZEN-SCIENCE PIPELINE")
    print("Running fast syntax/API preflight before any manuscript build work...")

    for script in ENTRY_SCRIPTS:
        py_compile.compile(str(script), doraise=True)
        matplotlib_api_gate(script)
        print("  OK", script.name)

    for script in IMPORTED_FIGURE_LIBS:
        py_compile.compile(str(script), doraise=True)
        print("  OK library", script.name)

    print("SUBMISSION-FINAL PYTHON PREFLIGHT: PASS")
    print("No model fitting or partition generation is invoked by this entrypoint.")

    completed = subprocess.run([sys.executable, str(FINAL_RUNNER)], cwd=str(ROOT))
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
