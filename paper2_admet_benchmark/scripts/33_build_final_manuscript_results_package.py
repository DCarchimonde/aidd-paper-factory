from __future__ import annotations

"""Package frozen confirmatory results into manuscript-ready evidence tables.

This script performs no model fitting and no statistical re-analysis beyond descriptive
averaging already used for the RQ1 headline summaries. It validates required final
outputs, creates compact manuscript tables, copies frozen headline tables into one
folder, and writes a SHA256 integrity manifest and an RQ-to-evidence map.
"""

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper2_admet_benchmark"
TABLE_DIR = PAPER_DIR / "results" / "tables"
OUT_DIR = PAPER_DIR / "results" / "manuscript_assets"
OUT_TABLE_DIR = OUT_DIR / "tables"


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise RuntimeError(f"Required table is empty: {path}")
    return df


def finite_mean(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(x.mean()) if len(x) else np.nan


def wide_rq1(
    aggregate: pd.DataFrame,
    task_type: str,
    metrics: list[str],
    source_table: str,
) -> pd.DataFrame:
    subset = aggregate[
        (aggregate["source_table"] == source_table)
        & (aggregate["task_type"] == task_type)
        & (aggregate["metric"].isin(metrics))
    ].copy()
    if subset.empty:
        raise RuntimeError(
            f"No RQ1 rows for source={source_table}, task_type={task_type}, metrics={metrics}"
        )

    rows: list[dict[str, object]] = []
    for (endpoint, split_type), group in subset.groupby(
        ["endpoint", "split_type"], sort=True
    ):
        row: dict[str, object] = {
            "endpoint": endpoint,
            "split_type": split_type,
            "aggregation_note": "descriptive mean across frozen model/regime combinations",
        }
        for metric in metrics:
            metric_group = group[group["metric"] == metric]
            row[metric] = finite_mean(metric_group["mean"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["endpoint", "split_type"]).reset_index(drop=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_table(source_name: str, destination_name: str | None = None) -> Path:
    source = TABLE_DIR / source_name
    if not source.exists():
        raise FileNotFoundError(source)
    destination = OUT_TABLE_DIR / (destination_name or source_name)
    shutil.copy2(source, destination)
    return destination


def main() -> None:
    OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    aggregate = read_required(TABLE_DIR / "confirmatory_aggregate_long.csv")

    rq1_classification = wide_rq1(
        aggregate,
        task_type="classification",
        metrics=["roc_auc", "pr_auc", "balanced_accuracy"],
        source_table="baseline_test",
    )
    rq1_regression = wide_rq1(
        aggregate,
        task_type="regression",
        metrics=["rmse", "mae", "r2"],
        source_table="baseline_test",
    )
    rq1_calibration = wide_rq1(
        aggregate,
        task_type="classification",
        metrics=[
            "brier_score",
            "negative_log_likelihood",
            "ece_probability",
            "ece_confidence",
        ],
        source_table="calibration_test",
    )

    generated: list[Path] = []
    rq1_paths = [
        ("table_rq1_classification_performance.csv", rq1_classification),
        ("table_rq1_regression_performance.csv", rq1_regression),
        ("table_rq1_classification_calibration.csv", rq1_calibration),
    ]
    for name, frame in rq1_paths:
        path = OUT_TABLE_DIR / name
        frame.to_csv(path, index=False)
        generated.append(path)

    frozen_tables = [
        (
            "headline_classification_conformal_alpha01_complete.csv",
            "table_rq3_rq4_classification_conformal.csv",
        ),
        (
            "paired_classification_effects_headline_alpha01.csv",
            "table_rq3_rq4_classification_paired_effects.csv",
        ),
        (
            "headline_regression_conformal_alpha01_complete.csv",
            "table_rq4_regression_conformal.csv",
        ),
        (
            "paired_regression_effects_headline_alpha01.csv",
            "table_rq4_regression_paired_effects.csv",
        ),
        (
            "selective_prediction_matched_paurc_headline.csv",
            "table_rq2_rq3_selective_prediction.csv",
        ),
        (
            "ad_continuous_headline_alpha01.csv",
            "table_rq2_ad_continuous.csv",
        ),
        (
            "ad_threshold_sensitivity_headline_alpha01.csv",
            "table_rq2_ad_threshold_sensitivity.csv",
        ),
        (
            "ad_similarity_quartiles_headline_alpha01.csv",
            "table_rq2_ad_similarity_quartiles.csv",
        ),
    ]
    for source_name, destination_name in frozen_tables:
        generated.append(copy_table(source_name, destination_name))

    expected_rows = {
        "table_rq1_classification_performance.csv": 6,
        "table_rq1_regression_performance.csv": 6,
        "table_rq1_classification_calibration.csv": 6,
        "table_rq3_rq4_classification_conformal.csv": 18,
        "table_rq3_rq4_classification_paired_effects.csv": 12,
        "table_rq4_regression_conformal.csv": 18,
        "table_rq4_regression_paired_effects.csv": 12,
        "table_rq2_rq3_selective_prediction.csv": 18,
        "table_rq2_ad_continuous.csv": 12,
        "table_rq2_ad_threshold_sensitivity.csv": 72,
        "table_rq2_ad_similarity_quartiles.csv": 48,
    }

    manifest_rows: list[dict[str, object]] = []
    for path in sorted(generated):
        rows = len(read_required(path))
        expected = expected_rows[path.name]
        if rows != expected:
            raise RuntimeError(f"{path.name}: expected {expected} rows, found {rows}")
        manifest_rows.append(
            {
                "file": path.relative_to(PAPER_DIR).as_posix(),
                "rows": rows,
                "expected_rows": expected,
                "row_count_valid": rows == expected,
                "sha256": sha256(path),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = OUT_DIR / "final_results_integrity_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    evidence = pd.DataFrame(
        [
            {
                "research_question": "RQ1",
                "claim_scope": "Predictive performance and calibration across random, scaffold, and cluster splits",
                "primary_files": "table_rq1_classification_performance.csv; table_rq1_regression_performance.csv; table_rq1_classification_calibration.csv",
            },
            {
                "research_question": "RQ2",
                "claim_scope": "Applicability-domain association, threshold sensitivity, and selective prediction",
                "primary_files": "table_rq2_ad_continuous.csv; table_rq2_ad_threshold_sensitivity.csv; table_rq2_ad_similarity_quartiles.csv; table_rq2_rq3_selective_prediction.csv",
            },
            {
                "research_question": "RQ3",
                "claim_scope": "Marginal summaries concealing class-conditional and subgroup failure",
                "primary_files": "table_rq3_rq4_classification_conformal.csv; table_rq3_rq4_classification_paired_effects.csv; table_rq2_rq3_selective_prediction.csv",
            },
            {
                "research_question": "RQ4",
                "claim_scope": "Mondrian, shift-weighted, and adaptive conformal coverage-efficiency trade-offs",
                "primary_files": "table_rq3_rq4_classification_conformal.csv; table_rq3_rq4_classification_paired_effects.csv; table_rq4_regression_conformal.csv; table_rq4_regression_paired_effects.csv",
            },
        ]
    )
    evidence_path = OUT_DIR / "research_question_evidence_map.csv"
    evidence.to_csv(evidence_path, index=False)

    readme = OUT_DIR / "README.md"
    readme.write_text(
        "# Frozen manuscript results package\n\n"
        "This directory contains manuscript-ready confirmatory headline tables. "
        "No model fitting is performed by the packaging script. The integrity manifest "
        "records row counts and SHA256 hashes. Descriptive tables averaging across model "
        "families must not treat models as independent inferential replicates. Model-specific "
        "cross-seed confidence intervals remain in the original results/tables directory.\n",
        encoding="utf-8",
    )

    print("saved", manifest_path)
    print("saved", evidence_path)
    print("saved", readme)
    print(
        "Final manuscript results package complete. "
        f"tables={len(generated)}, manifest_rows={len(manifest)}, rq_rows={len(evidence)}."
    )


if __name__ == "__main__":
    main()
