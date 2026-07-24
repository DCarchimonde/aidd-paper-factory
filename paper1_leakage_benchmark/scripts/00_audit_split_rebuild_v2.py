from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared_utils.dataset_registry import DATASETS
from shared_utils.scaffold_identity import prepare_scaffold_frame
from shared_utils.split_manifest_v2 import (
    duplicate_target_audit,
    split_manifest_rows,
    summarize_partition,
)
from shared_utils.split_search_v2 import (
    add_legacy_scaffold_split_v2,
    add_random_split_v2,
    add_searched_scaffold_split_v2,
)

PAPER_DIR = ROOT / "paper1_leakage_benchmark"
PROCESSED_DIR = PAPER_DIR / "data" / "processed"
AUDIT_DIR = PAPER_DIR / "results" / "split_rebuild_v2"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [
    42, 123, 2024, 2026, 3407,
    7, 19, 71, 101, 211,
    307, 401, 503, 601, 701,
    809, 907, 1009, 1201, 1429,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit rebuilt molecular partitions before retraining."
    )
    parser.add_argument("--datasets", default="all")
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=20,
        choices=range(1, len(SEEDS) + 1),
    )
    parser.add_argument("--n-trials", type=int, default=300)
    parser.add_argument(
        "--acyclic-mode",
        choices=["single_group", "singleton"],
        default="single_group",
    )
    return parser.parse_args()


def load_dataset(name: str) -> pd.DataFrame:
    candidates = [
        PROCESSED_DIR / f"{name.lower()}_clean.csv",
        PROCESSED_DIR / f"{name.lower()}_splits.csv",
    ]
    for path in candidates:
        if path.exists():
            frame = pd.read_csv(path, keep_default_na=False)
            required = {"canonical_smiles", "target"}
            missing = required.difference(frame.columns)
            if missing:
                raise KeyError(f"{path} missing columns: {sorted(missing)}")
            return frame[["canonical_smiles", "target"]].copy()
    raise FileNotFoundError(f"No processed input found for {name}")


def append_partition(
    *,
    split_df: pd.DataFrame,
    dataset: str,
    task_type: str,
    protocol: str,
    seed: int | None,
    split_col: str,
    meta: dict,
    manifest_rows: list[dict],
    audit_rows: list[dict],
) -> None:
    manifest_rows.extend(
        split_manifest_rows(
            split_df,
            dataset=dataset,
            protocol=protocol,
            partition_seed=seed,
            split_col=split_col,
            partition_hash_value=meta["partition_hash"],
        )
    )
    audit_rows.append(
        summarize_partition(
            split_df,
            dataset=dataset,
            task_type=task_type,
            protocol=protocol,
            partition_seed=seed,
            split_col=split_col,
            meta=meta,
        )
    )


def main() -> None:
    args = parse_args()
    datasets = (
        list(DATASETS)
        if args.datasets == "all"
        else [item.strip() for item in args.datasets.split(",")]
    )
    seeds = SEEDS[: args.n_seeds]
    manifest_rows: list[dict] = []
    audit_rows: list[dict] = []
    duplicate_rows: list[dict] = []

    for dataset in datasets:
        if dataset not in DATASETS:
            raise KeyError(f"Unknown dataset: {dataset}")
        spec = DATASETS[dataset]
        print(f"\n========== {dataset} ==========")
        loaded = load_dataset(dataset)
        duplicate_rows.append(duplicate_target_audit(loaded, dataset))
        base = prepare_scaffold_frame(
            loaded,
            acyclic_mode=args.acyclic_mode,
        )

        legacy_df, legacy_meta = add_legacy_scaffold_split_v2(base)
        append_partition(
            split_df=legacy_df,
            dataset=dataset,
            task_type=spec.task_type,
            protocol="legacy_scaffold",
            seed=None,
            split_col="split_legacy_scaffold_v2",
            meta=legacy_meta,
            manifest_rows=manifest_rows,
            audit_rows=audit_rows,
        )

        for seed in seeds:
            random_df, random_meta = add_random_split_v2(
                base,
                target_col="target",
                task_type=spec.task_type,
                seed=seed,
            )
            append_partition(
                split_df=random_df,
                dataset=dataset,
                task_type=spec.task_type,
                protocol="random_observation",
                seed=seed,
                split_col="split_random_v2",
                meta=random_meta,
                manifest_rows=manifest_rows,
                audit_rows=audit_rows,
            )

            size_df, size_meta = add_searched_scaffold_split_v2(
                base,
                seed=seed,
                objective="size_only",
                n_trials=args.n_trials,
            )
            append_partition(
                split_df=size_df,
                dataset=dataset,
                task_type=spec.task_type,
                protocol="size_matched_scaffold",
                seed=seed,
                split_col="split_size_matched_scaffold_v2",
                meta=size_meta,
                manifest_rows=manifest_rows,
                audit_rows=audit_rows,
            )

            balanced_df, balanced_meta = add_searched_scaffold_split_v2(
                base,
                seed=seed,
                objective="target_balanced",
                n_trials=args.n_trials,
            )
            append_partition(
                split_df=balanced_df,
                dataset=dataset,
                task_type=spec.task_type,
                protocol="target_balanced_scaffold",
                seed=seed,
                split_col="split_target_balanced_scaffold_v2",
                meta=balanced_meta,
                manifest_rows=manifest_rows,
                audit_rows=audit_rows,
            )

    manifest = pd.DataFrame(manifest_rows)
    audit = pd.DataFrame(audit_rows)
    duplicates = pd.DataFrame(duplicate_rows)

    uniqueness = (
        audit.groupby(["dataset", "protocol"], as_index=False)
        .agg(
            n_requested_partitions=("partition_hash", "size"),
            n_unique_partitions=("partition_hash", "nunique"),
            min_n_test=("n_test", "min"),
            max_n_test=("n_test", "max"),
            mean_size_deviation=("size_deviation_from_target", "mean"),
            min_abs_target_mean_gap=("abs_target_mean_gap", "min"),
            mean_abs_target_mean_gap=("abs_target_mean_gap", "mean"),
            max_abs_target_mean_gap=("abs_target_mean_gap", "max"),
        )
    )
    frequencies = (
        audit.groupby(
            ["dataset", "protocol", "partition_hash"],
            as_index=False,
        )
        .agg(
            repeat_count=("partition_hash", "size"),
            seeds=(
                "partition_seed",
                lambda values: ",".join(
                    str(int(value))
                    for value in values.dropna().astype(int)
                ),
            ),
        )
        .sort_values(
            ["dataset", "protocol", "repeat_count"],
            ascending=[True, True, False],
        )
    )

    suffix = args.acyclic_mode
    paths = {
        "manifest": AUDIT_DIR / f"split_manifest_v2_{suffix}.csv",
        "audit": AUDIT_DIR / f"split_audit_v2_{suffix}.csv",
        "uniqueness": AUDIT_DIR / f"split_uniqueness_v2_{suffix}.csv",
        "frequencies": AUDIT_DIR / f"split_hash_frequencies_v2_{suffix}.csv",
        "duplicates": AUDIT_DIR / "duplicate_target_audit_v2.csv",
    }
    manifest.to_csv(paths["manifest"], index=False)
    audit.to_csv(paths["audit"], index=False)
    uniqueness.to_csv(paths["uniqueness"], index=False)
    frequencies.to_csv(paths["frequencies"], index=False)
    duplicates.to_csv(paths["duplicates"], index=False)

    print("\nSaved:")
    for path in paths.values():
        print(path)
    print("\nUniqueness summary:")
    print(uniqueness.to_string(index=False))


if __name__ == "__main__":
    main()
