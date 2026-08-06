from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared_utils.dataset_registry import DATASETS
from shared_utils.scaffold_identity import prepare_scaffold_frame
from shared_utils.split_candidate_pool_v3 import (
    generate_candidate_pool,
    materialize_candidate_split,
    select_paired_candidates,
)
from shared_utils.split_manifest_v2 import split_manifest_rows, summarize_partition
from shared_utils.split_search_v2 import add_legacy_scaffold_split_v2, add_random_split_v2

RDLogger.DisableLog("rdApp.warning")

PAPER_DIR = ROOT / "paper1_leakage_benchmark"
DATA_DIR = PAPER_DIR / "data" / "processed_v2"
OUT_DIR = PAPER_DIR / "results" / "split_rebuild_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [
    42, 123, 2024, 2026, 3407,
    7, 19, 71, 101, 211,
    307, 401, 503, 601, 701,
    809, 907, 1009, 1201, 1429,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit matched-budget random scaffold candidate pools on clean_v2 data."
    )
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--n-seeds", type=int, default=20, choices=range(1, 21))
    parser.add_argument("--n-candidates", type=int, default=5000)
    parser.add_argument(
        "--acyclic-mode",
        choices=["single_group", "singleton"],
        default="single_group",
    )
    return parser.parse_args()


def append_partition(
    frame: pd.DataFrame,
    *,
    dataset: str,
    task_type: str,
    protocol: str,
    seed: int | None,
    split_col: str,
    meta: dict,
    manifests: list[dict],
    audits: list[dict],
) -> None:
    manifests.extend(
        split_manifest_rows(
            frame,
            dataset=dataset,
            protocol=protocol,
            partition_seed=seed,
            split_col=split_col,
            partition_hash_value=meta["partition_hash"],
        )
    )
    row = summarize_partition(
        frame,
        dataset=dataset,
        task_type=task_type,
        protocol=protocol,
        partition_seed=seed,
        split_col=split_col,
        meta=meta,
    )
    row.update({key: value for key, value in meta.items() if key not in row})
    audits.append(row)


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
    pool_frames: list[pd.DataFrame] = []
    pair_rows: list[dict] = []

    for dataset in datasets:
        if dataset not in DATASETS:
            raise KeyError(f"Unknown dataset: {dataset}")
        spec = DATASETS[dataset]
        clean_path = DATA_DIR / f"{dataset.lower()}_clean_v2.csv"
        if not clean_path.exists():
            raise FileNotFoundError(
                f"Missing {clean_path}. Run 04_build_clean_data_v2.py first."
            )
        print(f"\n========== {dataset} ==========")
        clean = pd.read_csv(clean_path, keep_default_na=False)
        base = prepare_scaffold_frame(clean, acyclic_mode=args.acyclic_mode)

        legacy_df, legacy_meta = add_legacy_scaffold_split_v2(base)
        append_partition(
            legacy_df,
            dataset=dataset,
            task_type=spec.task_type,
            protocol="legacy_scaffold",
            seed=None,
            split_col="split_legacy_scaffold_v2",
            meta=legacy_meta,
            manifests=manifest_rows,
            audits=audit_rows,
        )

        for seed in seeds:
            random_df, random_meta = add_random_split_v2(
                base,
                target_col="target",
                task_type=spec.task_type,
                seed=seed,
            )
            append_partition(
                random_df,
                dataset=dataset,
                task_type=spec.task_type,
                protocol="random_observation",
                seed=seed,
                split_col="split_random_v2",
                meta=random_meta,
                manifests=manifest_rows,
                audits=audit_rows,
            )

            candidates, groups, generation_meta = generate_candidate_pool(
                base,
                seed=seed,
                n_candidates=args.n_candidates,
            )
            size_candidate, balanced_candidate, pool_table, pair_meta = select_paired_candidates(
                candidates,
                groups,
                seed=seed,
                total_n=len(base),
                total_target_sum=float(base["target"].sum()),
            )
            pool_table.insert(0, "dataset", dataset)
            pool_table.insert(1, "task_type", spec.task_type)
            pool_table.insert(2, "partition_seed", seed)
            pool_frames.append(pool_table)

            size_df, size_meta = materialize_candidate_split(
                base,
                groups,
                size_candidate,
                split_col="split_size_matched_scaffold_v3",
            )
            size_meta.update(
                {
                    "protocol": "size_matched_scaffold",
                    **generation_meta,
                    **pair_meta,
                    "selection_role": "target_blind_size_baseline",
                }
            )
            append_partition(
                size_df,
                dataset=dataset,
                task_type=spec.task_type,
                protocol="size_matched_scaffold",
                seed=seed,
                split_col="split_size_matched_scaffold_v3",
                meta=size_meta,
                manifests=manifest_rows,
                audits=audit_rows,
            )

            balanced_df, balanced_meta = materialize_candidate_split(
                base,
                groups,
                balanced_candidate,
                split_col="split_target_balanced_scaffold_v3",
            )
            balanced_meta.update(
                {
                    "protocol": "target_balanced_scaffold",
                    **generation_meta,
                    **pair_meta,
                    "selection_role": "same_pool_same_size_min_target_gap",
                }
            )
            append_partition(
                balanced_df,
                dataset=dataset,
                task_type=spec.task_type,
                protocol="target_balanced_scaffold",
                seed=seed,
                split_col="split_target_balanced_scaffold_v3",
                meta=balanced_meta,
                manifests=manifest_rows,
                audits=audit_rows,
            )

            pair_rows.append(
                {
                    "dataset": dataset,
                    "task_type": spec.task_type,
                    "partition_seed": seed,
                    **generation_meta,
                    **pair_meta,
                    "size_partition_hash": size_meta["partition_hash"],
                    "balanced_partition_hash": balanced_meta["partition_hash"],
                }
            )

    manifests = pd.DataFrame(manifest_rows)
    audits = pd.DataFrame(audit_rows)
    pools = pd.concat(pool_frames, ignore_index=True) if pool_frames else pd.DataFrame()
    pairs = pd.DataFrame(pair_rows)

    if not bool(pairs["exact_size_match"].all()):
        raise AssertionError("At least one candidate-pool pair is not exactly size matched")
    if not bool((pairs["unique_candidates"] >= 1).all()):
        raise AssertionError("At least one candidate pool is empty")

    uniqueness = (
        audits.groupby(["dataset", "protocol"], as_index=False)
        .agg(
            n_requested_partitions=("partition_hash", "size"),
            n_unique_partitions=("partition_hash", "nunique"),
            min_n_test=("n_test", "min"),
            max_n_test=("n_test", "max"),
            mean_size_deviation=("size_deviation_from_target", "mean"),
            mean_abs_target_mean_gap=("abs_target_mean_gap", "mean"),
        )
    )
    pair_summary = (
        pairs.groupby(["dataset", "task_type"], as_index=False)
        .agg(
            n_pairs=("partition_seed", "size"),
            n_exact_size_matches=("exact_size_match", "sum"),
            n_same_partitions=("same_partition", "sum"),
            min_unique_candidates=("unique_candidates", "min"),
            mean_unique_candidates=("unique_candidates", "mean"),
            min_same_size_candidates=("n_same_size_candidates", "min"),
            mean_same_size_candidates=("n_same_size_candidates", "mean"),
            mean_size_matched_target_gap=("size_matched_target_gap", "mean"),
            mean_target_balanced_target_gap=("target_balanced_target_gap", "mean"),
            mean_target_gap_absolute_reduction=("target_gap_absolute_reduction", "mean"),
            mean_target_gap_relative_reduction=("target_gap_relative_reduction", "mean"),
            mean_same_size_gap_median=("same_size_gap_median", "mean"),
            mean_size_baseline_empirical_cdf=("size_baseline_empirical_cdf", "mean"),
            mean_balanced_min_tie_fraction=("balanced_min_tie_fraction", "mean"),
        )
    )

    dataset_tag = "all" if args.datasets == "all" else "-".join(datasets)
    suffix = (
        f"{dataset_tag}_{args.acyclic_mode}_"
        f"{args.n_seeds}s_{args.n_candidates}c"
    )
    paths = {
        "manifest": OUT_DIR / f"split_manifest_v3_{suffix}.csv",
        "audit": OUT_DIR / f"split_audit_v3_{suffix}.csv",
        "candidate_pool": OUT_DIR / f"candidate_pool_v3_{suffix}.csv",
        "pairs": OUT_DIR / f"split_pairs_v3_{suffix}.csv",
        "pair_summary": OUT_DIR / f"split_pair_summary_v3_{suffix}.csv",
        "uniqueness": OUT_DIR / f"split_uniqueness_v3_{suffix}.csv",
    }
    manifests.to_csv(paths["manifest"], index=False)
    audits.to_csv(paths["audit"], index=False)
    pools.to_csv(paths["candidate_pool"], index=False)
    pairs.to_csv(paths["pairs"], index=False)
    pair_summary.to_csv(paths["pair_summary"], index=False)
    uniqueness.to_csv(paths["uniqueness"], index=False)

    print("\nPaired candidate-pool summary:")
    print(pair_summary.to_string(index=False))
    print("\nUniqueness summary:")
    print(uniqueness.to_string(index=False))
    print("\nSaved:")
    for path in paths.values():
        print(path)
    print("\nCANDIDATE POOL V3 AUDIT PASSED")


if __name__ == "__main__":
    main()
