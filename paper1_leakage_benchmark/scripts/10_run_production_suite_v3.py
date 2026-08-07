from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper1_leakage_benchmark"
SCRIPT_DIR = PAPER_DIR / "scripts"
OUT_DIR = PAPER_DIR / "results" / "model_rerun_v3"
JOB_DIR = OUT_DIR / "jobs" / "production"
STALE_ROOT = OUT_DIR / "stale_jobs_v3"
MODEL_PROTOCOL = PAPER_DIR / "MODEL_PROTOCOL_V3.md"
RUNNER = SCRIPT_DIR / "09_run_models_v3.py"
COMPLETENESS = SCRIPT_DIR / "11_audit_model_completeness_v3.py"
ANALYSIS = SCRIPT_DIR / "12_analyze_partition_effects_v3.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_stale_protocol_jobs() -> int:
    if not JOB_DIR.exists():
        return 0
    current_sha = sha256_file(MODEL_PROTOCOL)
    stale: list[Path] = []
    for path in JOB_DIR.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            stale.append(path)
            continue
        if str(payload.get("model_protocol_sha256", "")) != current_sha:
            stale.append(path)
    if not stale:
        return 0
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = STALE_ROOT / stamp
    for source in stale:
        relative = source.relative_to(JOB_DIR)
        destination = archive_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
    print(
        f"Archived {len(stale)} stale production jobs with an older model protocol to {archive_root}"
    )
    return len(stale)


def run_command(args: list[str], label: str) -> None:
    print("\n" + "=" * 88)
    print(label)
    print("COMMAND:", " ".join(args))
    print("=" * 88)
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    required = [MODEL_PROTOCOL, RUNNER, COMPLETENESS, ANALYSIS]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    archive_stale_protocol_jobs()

    python = sys.executable
    production_runs = [
        (
            "MAIN CLASSIFICATION PRODUCTION",
            [
                python,
                str(RUNNER),
                "--freeze-label", "main_classification",
                "--datasets", "all",
                "--protocols", "default",
                "--partition-seeds", "all",
                "--models", "all",
                "--run-type", "production",
            ],
        ),
        (
            "MAIN REGRESSION PRODUCTION",
            [
                python,
                str(RUNNER),
                "--freeze-label", "main_regression",
                "--datasets", "all",
                "--protocols", "default",
                "--partition-seeds", "all",
                "--models", "all",
                "--run-type", "production",
            ],
        ),
        (
            "ACYCLIC SINGLETON SENSITIVITY PRODUCTION",
            [
                python,
                str(RUNNER),
                "--freeze-label", "acyclic_singleton_sensitivity",
                "--datasets", "all",
                "--protocols", "default",
                "--partition-seeds", "all",
                "--models", "all",
                "--run-type", "production",
            ],
        ),
    ]

    for label, command in production_runs:
        run_command(command, label)

    run_command([python, str(COMPLETENESS)], "PRODUCTION COMPLETENESS GATE")
    run_command([python, str(ANALYSIS)], "PARTITION-LEVEL PRIMARY ANALYSIS")

    print("\n" + "=" * 88)
    print("PAPER 1 MODEL PRODUCTION SUITE V3 COMPLETED")
    print("All production jobs, completeness checks, and primary partition-level analyses finished.")
    print("=" * 88)


if __name__ == "__main__":
    main()
