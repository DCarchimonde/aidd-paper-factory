from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "paper1_leakage_benchmark" / "scripts" / "42_run_paper1_sarqsar_overnight_hardened_v1.py"

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040


def set_awake(enabled: bool) -> None:
    if os.name != "nt":
        return
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED if enabled else ES_CONTINUOUS
    result = ctypes.windll.kernel32.SetThreadExecutionState(flags)
    if result == 0:
        print("NOTICE: Windows sleep-prevention request was not accepted; verify power settings manually.", flush=True)
    else:
        print("WINDOWS KEEP-AWAKE:", "ENABLED" if enabled else "RELEASED", flush=True)


def main() -> None:
    if not RUNNER.exists() or RUNNER.stat().st_size == 0:
        raise FileNotFoundError(RUNNER)
    set_awake(True)
    try:
        command = [sys.executable, "-u", str(RUNNER), *sys.argv[1:]]
        raise SystemExit(subprocess.call(command, cwd=str(ROOT)))
    finally:
        set_awake(False)


if __name__ == "__main__":
    main()
