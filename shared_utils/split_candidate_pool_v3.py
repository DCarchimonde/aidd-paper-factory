"""Matched-budget random scaffold candidate pools for Paper 1."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from shared_utils.scaffold_identity import assert_valid_split, partition_hash
from shared_utils.split_search_v2 import ScaffoldGroup, build_scaffold_groups


@dataclass(frozen=True)
class PoolCandidate:
    group_indices: tuple[int, ...]
    scaffold_key: tuple[str, ...]
    candidate_hash: str
    n_test: int
    size_deviation: float


def _hash_scaffolds(scaffolds: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(scaffolds).encode("utf-8")).hexdigest()


def _seeded_rank(seed: int, candidate_hash: str) -> str:
    return hashlib.sha256(f"{seed}|{candidate_hash}".encode("utf-8")).hexdigest()


def _random_prefix_candidate(
    groups: list[ScaffoldGroup],
    *,
    target_n: int,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    order = [int(value) for value in rng.permutation(len(groups))]
    selected: list[int] = []
    selected_n = 0

    for idx in order:
        if selected_n >= target_n:
            break
        selected.append(idx)
        selected_n += groups[idx].n

    if len(selected) > 1:
        last = selected[-1]
        without_last = selected_n - groups[last].n
        if without_last > 0 and abs(without_last - target_n) <= abs(selected_n - target_n):
            selected.pop()
            selected_n = without_last

    if not selected:
        selected = [min(range(len(groups)), key=lambda idx: abs(groups[idx].n - target_n))]
    if sum(groups[idx].n for idx in selected) >= sum(group.n for group in groups):
        selected = selected[:-1]
    if not selected:
        raise RuntimeError("Could not generate a non-empty proper scaffold candidate")
    return tuple(sorted(selected))


def generate_candidate_pool(
    df: pd.DataFrame,
    *,
    seed: int,
    n_candidates: int,
    test_size: float = 0.2,
    target_col: str = "target",
) -> tuple[list[PoolCandidate], list[ScaffoldGroup], dict]:
    """Generate target-blind random scaffold candidates with a fixed budget."""
    if n_candidates < 2:
        raise ValueError("n_candidates must be at least 2")
    groups = build_scaffold_groups(df, target_col=target_col)
    target_n = int(round(len(df) * test_size))
    rng = np.random.default_rng(int(seed))
    unique: dict[str, PoolCandidate] = {}

    for _ in range(n_candidates):
        selected = _random_prefix_candidate(groups, target_n=target_n, rng=rng)
        scaffold_key = tuple(sorted(groups[idx].scaffold for idx in selected))
        candidate_hash = _hash_scaffolds(scaffold_key)
        if candidate_hash in unique:
            continue
        n_test = int(sum(groups[idx].n for idx in selected))
        unique[candidate_hash] = PoolCandidate(
            group_indices=selected,
            scaffold_key=scaffold_key,
            candidate_hash=candidate_hash,
            n_test=n_test,
            size_deviation=float(abs(n_test - target_n) / max(target_n, 1)),
        )

    candidates = list(unique.values())
    if not candidates:
        raise RuntimeError("Candidate pool is empty")
    return candidates, groups, {
        "partition_seed": int(seed),
        "requested_candidates": int(n_candidates),
        "unique_candidates": int(len(candidates)),
        "duplicate_candidate_fraction": float(1.0 - len(candidates) / n_candidates),
        "target_test_n": int(target_n),
        "candidate_generation": "random_scaffold_prefix_closest_crossing",
        "target_blind_generation": True,
    }


def candidate_target_gap(
    candidate: PoolCandidate,
    groups: list[ScaffoldGroup],
    *,
    total_n: int,
    total_target_sum: float,
) -> float:
    test_sum = float(sum(groups[idx].target_sum for idx in candidate.group_indices))
    test_n = int(candidate.n_test)
    train_n = int(total_n - test_n)
    if test_n <= 0 or train_n <= 0:
        return float("inf")
    test_mean = test_sum / test_n
    train_mean = (total_target_sum - test_sum) / train_n
    return float(abs(test_mean - train_mean))


def select_paired_candidates(
    candidates: list[PoolCandidate],
    groups: list[ScaffoldGroup],
    *,
    seed: int,
    total_n: int,
    total_target_sum: float,
) -> tuple[PoolCandidate, PoolCandidate, pd.DataFrame, dict]:
    """Select a target-blind size baseline and a same-size target-balanced pair."""
    min_size_deviation = min(candidate.size_deviation for candidate in candidates)
    size_ties = [
        candidate
        for candidate in candidates
        if np.isclose(candidate.size_deviation, min_size_deviation, rtol=0.0, atol=1e-15)
    ]
    size_candidate = min(size_ties, key=lambda item: _seeded_rank(seed, item.candidate_hash))
    matched = [candidate for candidate in candidates if candidate.n_test == size_candidate.n_test]
    if not matched:
        raise AssertionError("The selected size baseline is missing from its own candidate pool")

    rows: list[dict] = []
    for candidate in candidates:
        gap = candidate_target_gap(
            candidate,
            groups,
            total_n=total_n,
            total_target_sum=total_target_sum,
        )
        rows.append(
            {
                "candidate_hash": candidate.candidate_hash,
                "n_test": candidate.n_test,
                "size_deviation": candidate.size_deviation,
                "abs_target_mean_gap": gap,
                "eligible_same_size": candidate.n_test == size_candidate.n_test,
                "selected_size_matched": candidate.candidate_hash == size_candidate.candidate_hash,
            }
        )
    pool_table = pd.DataFrame(rows)
    matched_table = pool_table.loc[pool_table["eligible_same_size"]].copy()
    matched_gaps = matched_table["abs_target_mean_gap"].to_numpy(dtype=float)
    best_gap = float(np.min(matched_gaps))
    baseline_gap = float(
        pool_table.loc[pool_table["selected_size_matched"], "abs_target_mean_gap"].iloc[0]
    )
    best_mask = np.isclose(matched_gaps, best_gap, rtol=0.0, atol=1e-15)
    balanced_hashes = set(
        matched_table.loc[
            np.isclose(
                matched_table["abs_target_mean_gap"],
                best_gap,
                rtol=0.0,
                atol=1e-15,
            ),
            "candidate_hash",
        ]
    )
    balanced_candidate = min(
        [candidate for candidate in matched if candidate.candidate_hash in balanced_hashes],
        key=lambda item: _seeded_rank(seed + 10_000_019, item.candidate_hash),
    )
    pool_table["selected_target_balanced"] = pool_table["candidate_hash"].eq(
        balanced_candidate.candidate_hash
    )

    absolute_reduction = float(baseline_gap - best_gap)
    relative_reduction = (
        float(absolute_reduction / baseline_gap)
        if baseline_gap > 0
        else 0.0
    )
    meta = {
        "size_matched_n_test": int(size_candidate.n_test),
        "target_balanced_n_test": int(balanced_candidate.n_test),
        "exact_size_match": bool(size_candidate.n_test == balanced_candidate.n_test),
        "n_min_size_ties": int(len(size_ties)),
        "n_same_size_candidates": int(len(matched)),
        "size_matched_target_gap": baseline_gap,
        "target_balanced_target_gap": best_gap,
        "target_gap_absolute_reduction": absolute_reduction,
        "target_gap_relative_reduction": relative_reduction,
        "same_size_gap_min": best_gap,
        "same_size_gap_q05": float(np.quantile(matched_gaps, 0.05)),
        "same_size_gap_q25": float(np.quantile(matched_gaps, 0.25)),
        "same_size_gap_median": float(np.quantile(matched_gaps, 0.50)),
        "same_size_gap_mean": float(np.mean(matched_gaps)),
        "same_size_gap_q75": float(np.quantile(matched_gaps, 0.75)),
        "same_size_gap_q95": float(np.quantile(matched_gaps, 0.95)),
        "same_size_gap_max": float(np.max(matched_gaps)),
        "size_baseline_empirical_cdf": float(
            np.mean(matched_gaps <= baseline_gap + 1e-15)
        ),
        "n_balanced_min_ties": int(np.sum(best_mask)),
        "balanced_min_tie_fraction": float(np.mean(best_mask)),
        "same_partition": bool(size_candidate.candidate_hash == balanced_candidate.candidate_hash),
    }
    if not meta["exact_size_match"]:
        raise AssertionError("Candidate-pool paired split lost exact size matching")
    if meta["target_balanced_target_gap"] > meta["size_matched_target_gap"] + 1e-15:
        raise AssertionError("Target-balanced selection is worse than its paired baseline")
    return size_candidate, balanced_candidate, pool_table, meta


def materialize_candidate_split(
    df: pd.DataFrame,
    groups: list[ScaffoldGroup],
    candidate: PoolCandidate,
    *,
    split_col: str,
) -> tuple[pd.DataFrame, dict]:
    out = df.copy().reset_index(drop=True)
    test_idx = np.concatenate([groups[idx].indices for idx in candidate.group_indices])
    out[split_col] = "train"
    out.loc[test_idx, split_col] = "test"
    assert_valid_split(out, split_col, require_scaffold_disjoint=True)
    return out, {
        "candidate_hash": candidate.candidate_hash,
        "actual_test_n": int(len(test_idx)),
        "size_deviation": candidate.size_deviation,
        "partition_hash": partition_hash(out, split_col),
    }
