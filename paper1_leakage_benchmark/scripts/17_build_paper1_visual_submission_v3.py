from __future__ import annotations

import ast
import py_compile
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "paper1_leakage_benchmark" / "scripts"
FINAL_RUNNER = SCRIPT_DIR / "22_build_paper1_q1_final_v3.py"
# Only scripts executed by the final pipeline are gated here. The legacy Round-3
# plotting module is intentionally excluded because it is executed only through
# the compatibility wrapper, which adapts its historical plt.subplots spacing kwargs.
Q1_SCRIPTS = [
    SCRIPT_DIR / "20a_build_q1_scientific_controls_safe_v3.py",
    SCRIPT_DIR / "20b_build_q1_tex_tables_v3.py",
    SCRIPT_DIR / "20c_write_q1_result_text_v3.py",
    SCRIPT_DIR / "20d_capture_raw_source_provenance_v3.py",
    SCRIPT_DIR / "21a_build_manuscript_assets_v3_round3_compat.py",
    SCRIPT_DIR / "22_build_paper1_q1_final_v3.py",
    SCRIPT_DIR / "23_polish_q1_diagnostic_figures_v3.py",
    SCRIPT_DIR / "24_q1_submission_gate_v3.py",
    SCRIPT_DIR / "25_finalize_submission_figures_v3.py",
]


def matplotlib_api_gate(script: Path) -> None:
    """Reject direct wspace/hspace kwargs to plt.subplots before long-running jobs start."""
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
        direct = sorted(
            kw.arg for kw in node.keywords
            if kw.arg in {"wspace", "hspace"}
        )
        if direct:
            bad.append(f"line {getattr(node, 'lineno', '?')}: {', '.join(direct)}")
    if bad:
        raise AssertionError(
            f"Matplotlib compatibility gate failed for {script.name}: direct spacing kwargs to plt.subplots are unsupported in the target environment; use gridspec_kw. "
            + "; ".join(bad)
        )


def main() -> None:
    print("PAPER 1 BUILD ENTRYPOINT -> Q1 FINAL PIPELINE")
    print("Running Python syntax + Matplotlib API compatibility gates before any model jobs...")
    for script in Q1_SCRIPTS:
        py_compile.compile(str(script), doraise=True)
        matplotlib_api_gate(script)
        print("  OK", script.name)
    print("Q1 PYTHON + MATPLOTLIB API GATES: PASS")

    completed = subprocess.run([sys.executable, str(FINAL_RUNNER)], cwd=str(ROOT))
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
