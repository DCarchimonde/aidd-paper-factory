from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared_utils.dataset_registry import DATASETS
from shared_utils.raw_audit_v2 import (
    canonicalize_smiles,
    duplicate_group_rows,
    processed_set_audit,
    sha256_file,
)

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")

PAPER_DIR = ROOT / "paper1_leakage_benchmark"
RAW_DIR = PAPER_DIR / "data" / "raw"
PROCESSED_DIR = PAPER_DIR / "data" / "processed"
OUT_DIR = PAPER_DIR / "results" / "split_rebuild_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit raw molecular datasets and processed-set traceability.")
    parser.add_argument("--datasets", default="all")
    return parser.parse_args()


def audit_one(dataset: str) -> tuple[dict, list[dict], list[dict]]:
    spec = DATASETS[dataset]
    raw_path = RAW_DIR / f"{dataset.lower()}_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw dataset: {raw_path}")
    raw = pd.read_csv(raw_path, keep_default_na=False, low_memory=False)
    required = {spec.smiles_col, spec.target_col}
    missing = required.difference(raw.columns)
    if missing:
        raise KeyError(f"{raw_path} missing columns: {sorted(missing)}")

    work = raw[[spec.smiles_col, spec.target_col]].copy()
    work.columns = ["source_smiles", "source_target"]
    work["target_numeric"] = pd.to_numeric(work["source_target"], errors="coerce")
    canonical = work["source_smiles"].map(canonicalize_smiles)
    work["canonical_smiles_v2"] = canonical.map(lambda item: item[0])
    work["smiles_status"] = canonical.map(lambda item: item[1])

    valid = work.loc[
        work["canonical_smiles_v2"].notna() & work["target_numeric"].notna()
    ].copy()
    duplicates = duplicate_group_rows(
        valid,
        dataset=dataset,
        canonical_col="canonical_smiles_v2",
        target_col="target_numeric",
    )
    duplicate_df = pd.DataFrame(duplicates)

    issues: list[dict] = []
    issue_mask = work["smiles_status"].ne("valid") | work["target_numeric"].isna()
    for idx, row in work.loc[issue_mask].iterrows():
        issues.append(
            {
                "dataset": dataset,
                "raw_row_index": int(idx),
                "source_smiles": str(row["source_smiles"]),
                "source_target": str(row["source_target"]),
                "smiles_status": str(row["smiles_status"]),
                "target_is_numeric": bool(pd.notna(row["target_numeric"])),
            }
        )

    nonempty = work["source_smiles"].astype(str).str.strip().ne("")
    exact_sizes = work.loc[nonempty].groupby("source_smiles").size()
    classification_bad = 0
    if spec.task_type == "classification":
        values = work["target_numeric"].dropna()
        classification_bad = int((~values.isin([0, 1])).sum())

    summary = {
        "dataset": dataset,
        "task_type": spec.task_type,
        "raw_file": str(raw_path.relative_to(ROOT)),
        "raw_file_sha256": sha256_file(raw_path),
        "raw_n_rows": int(len(work)),
        "raw_missing_smiles": int(work["smiles_status"].eq("missing").sum()),
        "raw_invalid_smiles": int(work["smiles_status"].eq("invalid").sum()),
        "raw_missing_or_nonnumeric_target": int(work["target_numeric"].isna().sum()),
        "classification_targets_outside_0_1": classification_bad,
        "raw_valid_rows": int(len(valid)),
        "raw_unique_exact_smiles": int(work.loc[nonempty, "source_smiles"].nunique()),
        "raw_duplicate_exact_smiles_groups": int((exact_sizes > 1).sum()),
        "raw_unique_canonical_smiles": int(valid["canonical_smiles_v2"].nunique()),
        "raw_duplicate_canonical_groups": int(len(duplicate_df)),
        "raw_duplicate_canonical_rows_beyond_first": int(
            (duplicate_df["n_raw_rows"] - 1).sum() if not duplicate_df.empty else 0
        ),
        "raw_conflicting_target_groups": int(
            duplicate_df["has_target_conflict"].sum() if not duplicate_df.empty else 0
        ),
        "max_duplicate_target_spread": float(
            duplicate_df["target_spread"].max() if not duplicate_df.empty else 0.0
        ),
    }
    summary.update(
        processed_set_audit(
            PROCESSED_DIR / f"{dataset.lower()}_clean.csv",
            valid,
        )
    )
    summary["requires_policy_decision"] = bool(
        summary["raw_conflicting_target_groups"] > 0
        or summary["classification_targets_outside_0_1"] > 0
        or not summary["processed_matches_raw_valid_unique_set"]
        or summary["processed_duplicate_canonical_groups"] > 0
        or summary["processed_conflicting_target_groups"] > 0
    )
    return summary, duplicates, issues


def main() -> None:
    args = parse_args()
    datasets = list(DATASETS) if args.datasets == "all" else [x.strip() for x in args.datasets.split(",")]
    summaries: list[dict] = []
    duplicates: list[dict] = []
    issues: list[dict] = []
    for dataset in datasets:
        if dataset not in DATASETS:
            raise KeyError(f"Unknown dataset: {dataset}")
        print(f"auditing raw {dataset}")
        summary, duplicate_rows, issue_rows = audit_one(dataset)
        summaries.append(summary)
        duplicates.extend(duplicate_rows)
        issues.extend(issue_rows)

    summary_df = pd.DataFrame(summaries)
    duplicate_df = pd.DataFrame(duplicates)
    issue_df = pd.DataFrame(issues)
    summary_path = OUT_DIR / "raw_data_audit_v2.csv"
    duplicate_path = OUT_DIR / "raw_duplicate_groups_v2.csv"
    issue_path = OUT_DIR / "raw_invalid_or_missing_rows_v2.csv"
    summary_df.to_csv(summary_path, index=False)
    duplicate_df.to_csv(duplicate_path, index=False)
    issue_df.to_csv(issue_path, index=False)

    print("\nRaw-data audit summary:")
    print(summary_df.to_string(index=False))
    print("\nSaved:")
    print(summary_path)
    print(duplicate_path)
    print(issue_path)
    print(
        "\nRAW DATA AUDIT REQUIRES POLICY REVIEW"
        if bool(summary_df["requires_policy_decision"].any())
        else "\nRAW DATA AUDIT PASSED WITHOUT POLICY EXCEPTIONS"
    )


if __name__ == "__main__":
    main()
