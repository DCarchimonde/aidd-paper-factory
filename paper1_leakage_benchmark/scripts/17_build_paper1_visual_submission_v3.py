from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FINAL_RUNNER = ROOT / "paper1_leakage_benchmark" / "scripts" / "22_build_paper1_q1_final_v3.py"


def main() -> None:
    print("PAPER 1 BUILD ENTRYPOINT -> Q1 FINAL PIPELINE")
    completed = subprocess.run([sys.executable, str(FINAL_RUNNER)], cwd=str(ROOT))
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
