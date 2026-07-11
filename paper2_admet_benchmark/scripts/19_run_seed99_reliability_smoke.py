from __future__ import annotations

"""Run the complete seed-99 reliability smoke pipeline without touching MVP outputs.

This technical-only workflow consumes prediction files from confirmatory seed 99 and
writes outputs with suffix `confirmatory_smoke_seed99`. It covers:
- dual-view calibration;
- marginal split conformal;
- Mondrian class-conditional conformal;
- chemical applicability-domain diagnostics;
- selective-prediction curves;
- domain-conditioned performance and conformal coverage.

Seed 99 is excluded from confirmatory claims and exists only to verify code paths.
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

MODE = "confirmatory_smoke_seed99"
PATTERN = "*confirm*seed99*_predictions.csv"
ENDPOINTS = ["bbbp", "clintox", "esol", "lipophilicity"]
SPLITS = [
    "split_confirm_random_seed99",
    "split_confirm_scaffold_seed99",
    "split_confirm_cluster_seed99",
]
ALPHAS = [0.10, 0.20]


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


def prediction_files_exist() -> None:
    files = sorted(PREDICTION_DIR.glob(PATTERN))
    if not files:
        raise FileNotFoundError(
            f"No seed-99 confirmatory prediction files matched {PATTERN} in {PREDICTION_DIR}."
        )
    expected_split_tokens = [split.replace("split_", "") for split in SPLITS]
    missing = [token for token in expected_split_tokens if not any(token in path.name for path in files)]
    if missing:
        raise FileNotFoundError(f"Missing prediction files for split tokens: {missing}")
    print(f"Found {len(files)} seed-99 prediction files across all required splits.")


def run_applicability() -> tuple[Path, Path]:
    ad = load_script_module("paper2_ad_seed99", "10_applicability_domain_analysis.py")
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


def run_selective(details_path: Path) -> Path:
    selective = load_script_module("paper2_selective_seed99", "11_selective_prediction_analysis.py")
    SELECTIVE_DIR.mkdir(parents=True, exist_ok=True)
    predictions = selective.load_test_predictions()
    applicability = pd.read_csv(details_path)
    applicability = applicability[applicability["role"] == "test"].copy()
    predictions = predictions[
        predictions["endpoint"].isin(ENDPOINTS) & predictions["split_col"].isin(SPLITS)
    ].copy()
    applicability = applicability[
        applicability["endpoint"].isin(ENDPOINTS) & applicability["split_col"].isin(SPLITS)
    ].copy()
    merge_cols = ["endpoint", "task_type", "split_col", "row_index"]
    ad_cols = merge_cols + ["max_tanimoto_to_train", "domain_bin", "unseen_scaffold"]
    merged = predictions.merge(
        applicability[ad_cols],
        on=merge_cols,
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        raise RuntimeError("Selective-prediction merge produced zero rows.")
    rows: list[dict] = []
    for _, group in merged.groupby(["endpoint", "task_type", "split_col", "model"], sort=True):
        enriched = selective.add_uncertainty_columns(group)
        task_type = str(group["task_type"].iloc[0])
        for uncertainty_col in selective.uncertainty_columns(task_type):
            rows.extend(selective.evaluate_curve(enriched, uncertainty_col))
    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("Selective-prediction analysis produced zero rows.")
    output_path = SELECTIVE_DIR / f"selective_prediction_summary_{MODE}.csv"
    summary.to_csv(output_path, index=False)
    print("saved", output_path)
    return output_path


def run_domain_conditioned(details_path: Path, conformal_path: Path) -> tuple[Path, Path]:
    domain = load_script_module("paper2_domain_seed99", "12_domain_conditioned_reliability.py")
    predictions = domain.load_test_predictions()
    predictions = domain.filter_mode(predictions, endpoints=ENDPOINTS, splits=SPLITS)
    applicability = pd.read_csv(details_path)
    applicability = applicability[applicability["role"] == "test"].copy()
    applicability = domain.filter_mode(applicability, endpoints=ENDPOINTS, splits=SPLITS)
    conformal = pd.read_csv(conformal_path)
    conformal = domain.filter_mode(conformal, endpoints=ENDPOINTS, splits=SPLITS)
    performance = domain.compute_domain_performance(predictions, applicability)
    conformal_summary = domain.compute_domain_conformal(conformal, applicability)
    if performance.empty or conformal_summary.empty:
        raise RuntimeError("Domain-conditioned analysis produced an empty output.")
    performance_path = APPLICABILITY_DIR / f"domain_conditioned_performance_{MODE}.csv"
    conformal_out_path = APPLICABILITY_DIR / f"domain_conditioned_conformal_{MODE}.csv"
    performance.to_csv(performance_path, index=False)
    conformal_summary.to_csv(conformal_out_path, index=False)
    print("saved", performance_path)
    print("saved", conformal_out_path)
    return performance_path, conformal_out_path


def run_mondrian() -> tuple[Path, Path]:
    mondrian = load_script_module("paper2_mondrian_seed99", "13_mondrian_conformal_analysis.py")
    predictions = mondrian.load_predictions(PATTERN)
    predictions = predictions[
        (predictions["task_type"] == "classification")
        & predictions["endpoint"].isin(["bbbp", "clintox"])
        & predictions["split_col"].isin(SPLITS)
    ].copy()
    if predictions.empty:
        raise RuntimeError("No classification predictions remain for seed-99 Mondrian analysis.")
    summary_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    skipped = 0
    for keys, group in predictions.groupby(["endpoint", "split_col", "model"], sort=True):
        endpoint, split_col, model = keys
        calibration = group[group["role"] == "calibration"].copy()
        test = group[group["role"] == "test"].copy()
        if calibration.empty or test.empty:
            skipped += 1
            continue
        score0 = mondrian.class_scores(calibration, class_label=0)
        score1 = mondrian.class_scores(calibration, class_label=1)
        if len(score0) == 0 or len(score1) == 0:
            skipped += 1
            continue
        print(
            f"mondrian-smoke endpoint={endpoint} split={split_col} model={model} "
            f"cal_n0={len(score0)} cal_n1={len(score1)}"
        )
        for alpha in ALPHAS:
            qhat0 = mondrian.conformal_quantile(score0, alpha)
            qhat1 = mondrian.conformal_quantile(score1, alpha)
            test_out = mondrian.make_prediction_sets(test, qhat_0=qhat0, qhat_1=qhat1)
            test_out["alpha"] = alpha
            test_out["nominal_coverage"] = 1 - alpha
            summary_rows.append(
                mondrian.summarize(
                    endpoint=endpoint,
                    split_col=split_col,
                    model=model,
                    alpha=alpha,
                    qhat_0=qhat0,
                    qhat_1=qhat1,
                    calibration=calibration,
                    test_out=test_out,
                )
            )
            prediction_rows.append(test_out)
    summary = pd.DataFrame(summary_rows)
    details = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    if summary.empty or details.empty:
        raise RuntimeError("Mondrian analysis produced empty outputs.")
    summary_path = CONFORMAL_DIR / f"mondrian_conformal_summary_{MODE}.csv"
    details_path = CONFORMAL_DIR / f"mondrian_conformal_predictions_{MODE}.csv"
    summary.to_csv(summary_path, index=False)
    details.to_csv(details_path, index=False)
    print("saved", summary_path)
    print("saved", details_path)
    print(f"Mondrian groups: {summary.groupby(['endpoint', 'split_col', 'model']).ngroups}; skipped={skipped}")
    return summary_path, details_path


def main() -> None:
    prediction_files_exist()
    for directory in [CALIBRATION_DIR, CONFORMAL_DIR, APPLICABILITY_DIR, SELECTIVE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

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
        ]
    )

    conformal_path = CONFORMAL_DIR / f"conformal_predictions_{MODE}.csv"
    if not conformal_path.exists():
        raise FileNotFoundError(f"Expected conformal detail file was not created: {conformal_path}")

    details_path, _ = run_applicability()
    run_selective(details_path)
    run_domain_conditioned(details_path, conformal_path)
    run_mondrian()

    print("\nSeed-99 reliability smoke pipeline complete.")
    print("This output is technical validation only and must not enter confirmatory claims.")


if __name__ == "__main__":
    main()
