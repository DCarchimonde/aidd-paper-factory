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
from shared_utils.scaffold_identity import prepare_scaffold_frame
from shared_utils.split_candidate_pool_v3 import (
    generate_candidate_pool,
    materialize_candidate_split,
    select_paired_candidates,
)

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
DEFAULT_BUDGETS = [100, 300, 500, 1000, 3000, 5000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit convergence of matched-budget scaffold candidate pools."
    )
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--n-seeds", type=int, default=3, choices=range(1, 21))
    parser.add_argument(
        "--budgets",
        default=",".join(str(value) for value in DEFAULT_BUDGETS),
    )
    parser.add_argument(
        "--acyclic-mode",
        choices=["single_group", "singleton"],
        default="single_group",
    )
    parser.add_argument("--absolute-threshold", type=float, default=0.02)
    parser.add_argument("--relative-threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = (
        list(DATASETS)
        if args.datasets == "all"
        else [item.strip() for item in args.datasets.split(",")]
    )
    budgets = sorted({int(item.strip()) for item in args.budgets.split(",")})
    if not budgets or budgets[0] < 2:
        raise ValueError("All candidate budgets must be at least 2")
    seeds = SEEDS[: args.n_seeds]

    rows: list[dict] = []
    for dataset in datasets:
        if dataset not in DATASETS:
            raise KeyError(f"Unknown dataset: {dataset}")
        clean_path = DATA_DIR / f"{dataset.lower()}_clean_v2.csv"
        if not clean_path.exists():
            raise FileNotFoundError(
                f"Missing {clean_path}. Run 04_build_clean_data_v2.py first."
            )
        spec = DATASETS[dataset]
        clean = pd.read_csv(clean_path, keep_default_na=False)
        base = prepare_scaffold_frame(clean, acyclic_mode=args.acyclic_mode)
        print(f"\n========== {dataset} ==========")

        for seed in seeds:
            print(f"seed={seed}")
            for budget in budgets:
                candidates, groups, generation_meta = generate_candidate_pool(
                    base,
                    seed=seed,
                    n_candidates=budget,
                )
                size_candidate, balanced_candidate, _, pair_meta = select_paired_candidates(
                    candidates,
                    groups,
                    seed=seed,
                    total_n=len(base),
                    total_target_sum=float(base["target"].sum()),
                )
                size_df, size_meta = materialize_candidate_split(
                    base,
                    groups,
                    size_candidate,
                    split_col="split_size_matched_scaffold_v3",
                )
                balanced_df, balanced_meta = materialize_candidate_split(
                    base,
                    groups,
                    balanced_candidate,
                    split_col="split_target_balanced_scaffold_v3",
                )
                size_n = int(size_df["split_size_matched_scaffold_v3"].eq("test").sum())
                balanced_n = int(
                    balanced_df["split_target_balanced_scaffold_v3"].eq("test").sum()
                )
                if size_n != balanced_n:
                    raise AssertionError(
                        f"Budget audit lost exact size matching for "
                        f"{dataset}, seed={seed}, budget={budget}"
                    )
                rows.append(
                    {
                        "dataset": dataset,
                        "task_type": spec.task_type,
                        "acyclic_mode": args.acyclic_mode,
                        "partition_seed": seed,
                        "requested_candidates": budget,
                        **generation_meta,
                        **pair_meta,
                        "size_partition_hash": size_meta["partition_hash"],
                        "balanced_partition_hash": balanced_meta["partition_hash"],
                    }
                )

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(
            ["dataset", "task_type", "acyclic_mode", "requested_candidates"],
            as_index=False,
        )
        .agg(
            n_seeds=("partition_seed", "size"),
            min_unique_candidates=("unique_candidates", "min"),
            mean_unique_candidates=("unique_candidates", "mean"),
            max_duplicate_candidate_fraction=("duplicate_candidate_fraction", "max"),
            min_same_size_candidates=("n_same_size_candidates", "min"),
            mean_same_size_candidates=("n_same_size_candidates", "mean"),
            mean_size_matched_target_gap=("size_matched_target_gap", "mean"),
            mean_target_balanced_target_gap=("target_balanced_target_gap", "mean"),
            mean_target_gap_absolute_reduction=("target_gap_absolute_reduction", "mean"),
            mean_target_gap_relative_reduction=("target_gap_relative_reduction", "mean"),
            mean_same_size_gap_median=("same_size_gap_median", "mean"),
            mean_size_baseline_empirical_cdf=("size_baseline_empirical_cdf", "mean"),
            mean_balanced_min_tie_fraction=("balanced_min_tie_fraction", "mean"),
            n_unique_balanced_partitions=("balanced_partition_hash", "nunique"),
        )
    )

    convergence_rows: list[dict] = []
    group_cols = ["dataset", "task_type", "acyclic_mode"]
    for keys, group in summary.groupby(group_cols, sort=True):
        ordered = group.sort_values("requested_candidates").reset_index(drop=True)
        for index in range(len(ordered) - 1):
            lower = ordered.iloc[index]
            higher = ordered.iloc[index + 1]
            lower_gap = float(lower["mean_target_balanced_target_gap"])
            higher_gap = float(higher["mean_target_balanced_target_gap"])
            absolute_change = abs(lower_gap - higher_gap)
            relative_change = absolute_change / max(abs(lower_gap), 1e-12)
            convergence_rows.append(
                {
                    "dataset": keys[0],
                    "task_type": keys[1],
                    "acyclic_mode": keys[2],
                    "lower_budget": int(lower["requested_candidates"]),
                    "higher_budget": int(higher["requested_candidates"]),
                    "lower_mean_balanced_gap": lower_gap,
                    "higher_mean_balanced_gap": higher_gap,
                    "absolute_change": absolute_change,
                    "relative_change": relative_change,
                    "absolute_threshold": float(args.absolute_threshold),
                    "relative_threshold": float(args.relative_threshold),
                    "meets_absolute_rule": bool(
                        absolute_change <= args.absolute_threshold + 1e-15
                    ),
                    "meets_relative_rule": bool(
                        relative_change <= args.relative_threshold + 1e-15
                    ),
                    "meets_stopping_rule": bool(
                        absolute_change <= args.absolute_threshold + 1e-15
                        and relative_change <= args.relative_threshold + 1e-15
                    ),
                }
            )
    convergence = pd.DataFrame(convergence_rows)

    dataset_tag = "all" if args.datasets == "all" else "-".join(datasets)
    budget_tag = f"{min(budgets)}-{max(budgets)}"
    suffix = (
        f"{dataset_tag}_{args.acyclic_mode}_{args.n_seeds}s_{budget_tag}c"
    )
    detail_path = OUT_DIR / f"candidate_budget_detail_v3_{suffix}.csv"
    summary_path = OUT_DIR / f"candidate_budget_summary_v3_{suffix}.csv"
    convergence_path = OUT_DIR / f"candidate_budget_convergence_v3_{suffix}.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    convergence.to_csv(convergence_path, index=False)

    print("\nCandidate-budget summary:")
    print(summary.to_string(index=False))
    print("\nAdjacent-budget convergence decisions:")
    print(convergence.to_string(index=False))
    print("\nSaved:")
    print(detail_path)
    print(summary_path)
    print(convergence_path)
    print("\nCANDIDATE BUDGET V3 AUDIT COMPLETED")


if __name__ == "__main__":
    main()
