from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared_utils.cleaning_policy_v2 import build_clean_dataset_v2
from shared_utils.dataset_registry import DATASETS

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")

PAPER_DIR = ROOT / "paper1_leakage_benchmark"
RAW_DIR = PAPER_DIR / "data" / "raw"
OUT_DATA_DIR = PAPER_DIR / "data" / "processed_v2"
OUT_RESULT_DIR = PAPER_DIR / "results" / "split_rebuild_v2"
OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_RESULT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build audited canonical clean_v2 datasets.")
    parser.add_argument("--datasets", default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = list(DATASETS) if args.datasets == "all" else [x.strip() for x in args.datasets.split(",")]

    summaries: list[dict] = []
    decision_frames: list[pd.DataFrame] = []
    lineage_frames: list[pd.DataFrame] = []

    for dataset in datasets:
        if dataset not in DATASETS:
            raise KeyError(f"Unknown dataset: {dataset}")
        spec = DATASETS[dataset]
        raw_path = RAW_DIR / f"{dataset.lower()}_raw.csv"
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)

        print(f"building clean_v2 {dataset}")
        raw = pd.read_csv(raw_path, keep_default_na=False, low_memory=False)
        result = build_clean_dataset_v2(
            raw,
            dataset=dataset,
            smiles_col=spec.smiles_col,
            target_col=spec.target_col,
            task_type=spec.task_type,
        )
        clean_path = OUT_DATA_DIR / f"{dataset.lower()}_clean_v2.csv"
        result.clean.to_csv(clean_path, index=False)
        summaries.append({**result.summary, "clean_v2_file": str(clean_path.relative_to(ROOT))})
        decision_frames.append(result.group_decisions)
        lineage_frames.append(result.row_lineage)

    summary = pd.DataFrame(summaries)
    decisions = pd.concat(decision_frames, ignore_index=True) if decision_frames else pd.DataFrame()
    lineage = pd.concat(lineage_frames, ignore_index=True) if lineage_frames else pd.DataFrame()

    summary_path = OUT_RESULT_DIR / "cleaning_summary_v2.csv"
    decision_path = OUT_RESULT_DIR / "cleaning_group_decisions_v2.csv"
    lineage_path = OUT_RESULT_DIR / "cleaning_row_lineage_v2.csv"
    summary.to_csv(summary_path, index=False)
    decisions.to_csv(decision_path, index=False)
    lineage.to_csv(lineage_path, index=False)

    print("\nCleaning summary:")
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(summary_path)
    print(decision_path)
    print(lineage_path)
    print("\nCLEANING V2 PASSED")


if __name__ == "__main__":
    main()
