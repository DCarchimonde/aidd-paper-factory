from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "paper1_leakage_benchmark" / "scripts"
MATERIALIZER = SCRIPTS / "33_materialize_metric_coupling_null_v1.py"
GENERATED = SCRIPTS / "_generated" / "34_run_metric_coupling_null_v1_materialized.py"


def main() -> None:
    if not MATERIALIZER.exists() or MATERIALIZER.stat().st_size == 0:
        raise FileNotFoundError(MATERIALIZER)
    subprocess.run([sys.executable, "-u", str(MATERIALIZER)], cwd=str(ROOT), check=True)
    if not GENERATED.exists() or GENERATED.stat().st_size == 0:
        raise FileNotFoundError(GENERATED)
    command = [sys.executable, "-u", str(GENERATED), *sys.argv[1:]]
    raise SystemExit(subprocess.call(command, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
