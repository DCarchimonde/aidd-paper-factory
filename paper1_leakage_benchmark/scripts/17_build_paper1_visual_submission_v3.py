from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "paper1_leakage_benchmark" / "scripts"
FINAL_RUNNER = SCRIPT_DIR / "22_build_paper1_q1_final_v3.py"
Q1_SCRIPTS = [
    SCRIPT_DIR / "20a_build_q1_scientific_controls_safe_v3.py",
    SCRIPT_DIR / "20b_build_q1_tex_tables_v3.py",
    SCRIPT_DIR / "20c_write_q1_result_text_v3.py",
    SCRIPT_DIR / "20d_capture_raw_source_provenance_v3.py",
    SCRIPT_DIR / "21_build_manuscript_assets_v3_round3.py",
    SCRIPT_DIR / "21a_build_manuscript_assets_v3_round3_compat.py",
    SCRIPT_DIR / "22_build_paper1_q1_final_v3.py",
    SCRIPT_DIR / "23_polish_q1_diagnostic_figures_v3.py",
    SCRIPT_DIR / "24_q1_submission_gate_v3.py",
]


def main() -> None:
    print("PAPER 1 BUILD ENTRYPOINT -> Q1 FINAL PIPELINE")
    print("Running Python syntax gate before any model jobs...")
    for script in Q1_SCRIPTS:
        py_compile.compile(str(script), doraise=True)
        print("  OK", script.name)
    print("Q1 PYTHON SYNTAX GATE: PASS")

    completed = subprocess.run([sys.executable, str(FINAL_RUNNER)], cwd=str(ROOT))
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
