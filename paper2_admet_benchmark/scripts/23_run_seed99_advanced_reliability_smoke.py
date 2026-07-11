from __future__ import annotations

"""Run the remaining frozen advanced reliability methods on technical seed 99.

This script must be executed before any confirmatory seed 101-110 results are inspected.
It validates:
- density-ratio-weighted conformal for classification and regression;
- locally adaptive normalized conformal for regression;
- selective prediction AURC with repeated random-rejection baselines.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper2_admet_benchmark"
SCRIPT_DIR = PAPER_DIR / "scripts"
APPLICABILITY_DIR = PAPER_DIR / "results" / "applicability"
CONFORMAL_DIR = PAPER_DIR / "results" / "conformal"
SELECTIVE_DIR = PAPER_DIR / "results" / "selective"

MODE = "confirmatory_smoke_seed99"
PATTERN = "*confirm*seed99*_predictions.csv"
ENDPOINTS = "bbbp,clintox,esol,lipophilicity"
SPLITS = ",".join(
    [
        "split_confirm_random_seed99",
        "split_confirm_scaffold_seed99",
        "split_confirm_cluster_seed99",
    ]
)


def run(args: list[str]) -> None:
    print("\nRUN:", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def require_nonempty(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise RuntimeError(f"Expected non-empty output: {path}")
    print(f"validated {path.name}: {len(frame)} rows")


def main() -> None:
    applicability = APPLICABILITY_DIR / f"applicability_details_{MODE}.csv"
    if not applicability.exists():
        raise FileNotFoundError(
            f"Missing {applicability}. Run 19_run_seed99_reliability_smoke.py first."
        )

    common = [
        "--mode",
        MODE,
        "--pattern",
        PATTERN,
        "--endpoints",
        ENDPOINTS,
        "--splits",
        SPLITS,
    ]

    run([sys.executable, str(SCRIPT_DIR / "20_shift_weighted_conformal.py"), *common])
    run([sys.executable, str(SCRIPT_DIR / "21_adaptive_regression_conformal.py"), *common])
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "22_selective_prediction_v2.py"),
            *common,
            "--applicability-details",
            str(applicability),
            "--random-repeats",
            "500",
        ]
    )

    require_nonempty(CONFORMAL_DIR / f"shift_weighted_conformal_summary_{MODE}.csv")
    require_nonempty(CONFORMAL_DIR / f"adaptive_regression_conformal_summary_{MODE}.csv")
    require_nonempty(SELECTIVE_DIR / f"selective_prediction_v2_curves_{MODE}.csv")
    require_nonempty(SELECTIVE_DIR / f"selective_prediction_v2_aurc_{MODE}.csv")

    print("\nSeed-99 advanced reliability smoke complete.")
    print("Do not inspect or generate confirmatory seeds 101-110 until these outputs are reviewed.")


if __name__ == "__main__":
    main()
