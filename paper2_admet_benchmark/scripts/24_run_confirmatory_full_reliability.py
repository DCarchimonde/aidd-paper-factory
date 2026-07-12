from __future__ import annotations

"""Run the frozen full confirmatory reliability pipeline.

This script consumes only the previously generated confirmatory predictions for
seeds 101-110 (random/scaffold) and 101-105 (similarity-cluster). It does not
train models or alter split definitions. The script runs all locked reliability
analyses, validates expected row counts, and then creates cross-seed aggregates.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper2_admet_benchmark"
SCRIPT_DIR = PAPER_DIR / "scripts"
PREDICTION_DIR = PAPER_DIR / "results" / "predictions"
CALIBRATION_DIR = PAPER_DIR / "results" / "calibration"
CONFORMAL_DIR = PAPER_DIR / "results" / "conformal"
APPLICABILITY_DIR = PAPER_DIR / "results" / "applicability"
SELECTIVE_DIR = PAPER_DIR / "results" / "selective"
TABLE_DIR = PAPER_DIR / "results" / "tables"

MODE = "confirmatory_full"
MONDRIAN_MODE = "confirmatory"
PATTERN = "*confirm_*seed1??*_predictions.csv"
ENDPOINTS = ["bbbp", "clintox", "esol", "lipophilicity"]
RANDOM_SEEDS = list(range(101, 111))
CLUSTER_SEEDS = list(range(101, 106))
SPLITS = (
    [f"split_confirm_random_seed{s}" for s in RANDOM_SEEDS]
    + [f"split_confirm_scaffold_seed{s}" for s in RANDOM_SEEDS]
    + [f"split_confirm_cluster_seed{s}" for s in CLUSTER_SEEDS]
)


def load_script_module(name: str, filename: str):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_subprocess(args: list[str]) -> None:
    print("\nRUN:", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def preflight_predictions() -> None:
    files = sorted(PREDICTION_DIR.glob(PATTERN))
    if len(files) != 1200:
        raise RuntimeError(
            f"Expected exactly 1200 confirmatory prediction files, found {len(files)} "
            f"with pattern {PATTERN}."
        )

    frames = []
    for path in files:
        frame = pd.read_csv(
            path,
            usecols=["endpoint", "task_type", "split_col", "role", "model"],
        )
        frames.append(frame.drop_duplicates())
    index = pd.concat(frames, ignore_index=True).drop_duplicates()

    actual_splits = set(index["split_col"].astype(str))
    missing_splits = sorted(set(SPLITS) - actual_splits)
    extra_splits = sorted(actual_splits - set(SPLITS))
    if missing_splits or extra_splits:
        raise RuntimeError(
            f"Prediction split mismatch. Missing={missing_splits}; extra={extra_splits}"
        )
    if set(index["endpoint"].str.lower()) != set(ENDPOINTS):
        raise RuntimeError(
            f"Endpoint mismatch: {sorted(index['endpoint'].str.lower().unique())}"
        )
    if set(index["role"]) != {"calibration", "test"}:
        raise RuntimeError(f"Role mismatch: {sorted(index['role'].unique())}")

    print(
        f"Preflight passed: files={len(files)}, endpoints={index['endpoint'].nunique()}, "
        f"splits={index['split_col'].nunique()}, roles={index['role'].nunique()}."
    )


def run_applicability() -> tuple[Path, Path]:
    ad = load_script_module(
        "paper2_ad_confirmatory_full", "10_applicability_domain_analysis.py"
    )
    APPLICABILITY_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []

    for endpoint in ENDPOINTS:
        print(f"\n========== applicability {endpoint} ==========")
        frames.extend(
            ad.analyze_endpoint(
                endpoint=endpoint,
                split_cols=SPLITS,
                chunk_size=256,
                top_k=5,
                high_threshold=0.70,
                medium_threshold=0.40,
            )
        )

    if not frames:
        raise RuntimeError("Applicability analysis produced zero frames.")
    details = pd.concat(frames, ignore_index=True)
    summary = ad.summarize_details(details, top_k=5)

    details_path = APPLICABILITY_DIR / f"applicability_details_{MODE}.csv"
    summary_path = APPLICABILITY_DIR / f"applicability_summary_{MODE}.csv"
    details.to_csv(details_path, index=False)
    summary.to_csv(summary_path, index=False)
    print("saved", details_path)
    print("saved", summary_path)
    return details_path, summary_path


def run_domain_conditioned(details_path: Path, conformal_path: Path) -> tuple[Path, Path]:
    domain = load_script_module(
        "paper2_domain_confirmatory_full", "12_domain_conditioned_reliability.py"
    )

    predictions = domain.load_test_predictions()
    predictions = domain.filter_mode(predictions, endpoints=ENDPOINTS, splits=SPLITS)

    applicability = pd.read_csv(details_path, low_memory=False)
    applicability = applicability[applicability["role"] == "test"].copy()
    applicability = domain.filter_mode(
        applicability, endpoints=ENDPOINTS, splits=SPLITS
    )

    conformal = pd.read_csv(conformal_path, low_memory=False)
    conformal = domain.filter_mode(conformal, endpoints=ENDPOINTS, splits=SPLITS)

    performance = domain.compute_domain_performance(predictions, applicability)
    conformal_summary = domain.compute_domain_conformal(conformal, applicability)
    if performance.empty or conformal_summary.empty:
        raise RuntimeError("Domain-conditioned analysis produced an empty output.")

    performance_path = (
        APPLICABILITY_DIR / f"domain_conditioned_performance_{MODE}.csv"
    )
    conformal_out_path = (
        APPLICABILITY_DIR / f"domain_conditioned_conformal_{MODE}.csv"
    )
    performance.to_csv(performance_path, index=False)
    conformal_summary.to_csv(conformal_out_path, index=False)
    print("saved", performance_path)
    print("saved", conformal_out_path)
    return performance_path, conformal_out_path


def validate_csv(path: Path, expected_rows: int | None = None) -> int:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = len(pd.read_csv(path, low_memory=False))
    if expected_rows is not None and rows != expected_rows:
        raise RuntimeError(f"{path.name}: expected {expected_rows} rows, found {rows}")
    if rows == 0:
        raise RuntimeError(f"{path.name}: output is empty")
    print(f"validated {path.name}: {rows} rows")
    return rows


def main() -> None:
    preflight_predictions()
    for directory in [
        CALIBRATION_DIR,
        CONFORMAL_DIR,
        APPLICABILITY_DIR,
        SELECTIVE_DIR,
        TABLE_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    split_arg = ",".join(SPLITS)
    endpoint_arg = ",".join(ENDPOINTS)

    run_subprocess(
        [
            sys.executable,
            str(SCRIPT_DIR / "18_calibration_analysis_v2.py"),
            "--mode",
            MODE,
            "--pattern",
            PATTERN,
        ]
    )

    run_subprocess(
        [
            sys.executable,
            str(SCRIPT_DIR / "07_conformal_analysis.py"),
            "--mode",
            MODE,
            "--pattern",
            PATTERN,
            "--alphas",
            "0.1,0.2",
        ]
    )

    details_path, _ = run_applicability()
    marginal_details = CONFORMAL_DIR / f"conformal_predictions_{MODE}.csv"
    run_domain_conditioned(details_path, marginal_details)

    run_subprocess(
        [
            sys.executable,
            str(SCRIPT_DIR / "13_mondrian_conformal_analysis.py"),
            "--mode",
            MONDRIAN_MODE,
            "--pattern",
            PATTERN,
            "--endpoints",
            "bbbp,clintox",
            "--splits",
            split_arg,
            "--alphas",
            "0.1,0.2",
        ]
    )

    run_subprocess(
        [
            sys.executable,
            str(SCRIPT_DIR / "20_shift_weighted_conformal.py"),
            "--mode",
            MODE,
            "--pattern",
            PATTERN,
            "--endpoints",
            endpoint_arg,
            "--splits",
            split_arg,
            "--alphas",
            "0.1,0.2",
        ]
    )

    run_subprocess(
        [
            sys.executable,
            str(SCRIPT_DIR / "21_adaptive_regression_conformal.py"),
            "--mode",
            MODE,
            "--pattern",
            PATTERN,
            "--endpoints",
            endpoint_arg,
            "--splits",
            split_arg,
            "--alphas",
            "0.1,0.2",
        ]
    )

    run_subprocess(
        [
            sys.executable,
            str(SCRIPT_DIR / "22_selective_prediction_v2.py"),
            "--mode",
            MODE,
            "--pattern",
            PATTERN,
            "--endpoints",
            endpoint_arg,
            "--splits",
            split_arg,
            "--applicability-details",
            str(details_path),
            "--random-repeats",
            "500",
        ]
    )

    run_subprocess(
        [
            sys.executable,
            str(SCRIPT_DIR / "25_domain_conditioned_all_conformal.py"),
            "--mode",
            MODE,
        ]
    )

    run_subprocess(
        [
            sys.executable,
            str(SCRIPT_DIR / "26_aggregate_confirmatory_results.py"),
            "--mode",
            MODE,
        ]
    )

    paths = {
        CALIBRATION_DIR / "calibration_summary_v2_confirmatory_full.csv": 1200,
        CONFORMAL_DIR / "conformal_summary_confirmatory_full.csv": 1200,
        CONFORMAL_DIR / "mondrian_conformal_summary_confirmatory.csv": 800,
        CONFORMAL_DIR
        / "shift_weighted_conformal_summary_confirmatory_full.csv": 1200,
        CONFORMAL_DIR
        / "adaptive_regression_conformal_summary_confirmatory_full.csv": 400,
        SELECTIVE_DIR
        / "selective_prediction_v2_curves_confirmatory_full.csv": 19000,
        SELECTIVE_DIR
        / "selective_prediction_v2_aurc_confirmatory_full.csv": 1000,
        APPLICABILITY_DIR
        / "domain_conditioned_all_conformal_confirmatory_full.csv": None,
        TABLE_DIR / "confirmatory_aggregate_long.csv": None,
    }
    for path, expected in paths.items():
        validate_csv(path, expected)

    print("\nFULL CONFIRMATORY RELIABILITY PIPELINE COMPLETE.")
    print("All frozen methods and cross-seed aggregate outputs were generated.")
    print(
        "Do not alter methods based on individual seed results; inspect aggregate tables first."
    )


if __name__ == "__main__":
    main()
