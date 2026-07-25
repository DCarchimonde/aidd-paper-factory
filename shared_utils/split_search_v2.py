"""One audited engine for size-matched and target-balanced scaffold splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from shared_utils.scaffold_identity import assert_valid_split, partition_hash

SplitObjective = Literal["size_only", "target_balanced"]


@dataclass(frozen=True)
class ScaffoldGroup:
    scaffold: str
    indices: np.ndarray
    n: int
    target_sum: float


def build_scaffold_groups(
    df: pd.DataFrame,
    target_col: str = "target",
) -> list[ScaffoldGroup]:
    if "scaffold" not in df.columns:
        raise KeyError("Scaffolds must be prepared before grouping.")
    if target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")
    groups: list[ScaffoldGroup] = []
    covered: list[int] = []
    for scaffold, sub in df.groupby("scaffold", sort=True, dropna=False):
        indices = sub.index.to_numpy(dtype=int)
        groups.append(
            ScaffoldGroup(
                scaffold=str(scaffold),
                indices=indices,
                n=int(len(indices)),
                target_sum=float(sub[target_col].sum()),
            )
        )
        covered.extend(indices.tolist())
    if sorted(covered) != list(range(len(df))):
        raise AssertionError(
            "Scaffold groups do not cover every molecule exactly once."
        )
    return groups


def add_random_split_v2(
    df: pd.DataFrame,
    *,
    target_col: str,
    task_type: str,
    seed: int,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, dict]:
    out = df.copy().reset_index(drop=True)
    idx = np.arange(len(out))
    stratify = None
    if task_type == "classification":
        counts = out[target_col].value_counts()
        if len(counts) > 1 and int(counts.min()) >= 2:
            stratify = out[target_col]
    _, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    out["split_random_v2"] = "train"
    out.loc[test_idx, "split_random_v2"] = "test"
    assert_valid_split(
        out,
        "split_random_v2",
        require_scaffold_disjoint=False,
    )
    return out, {
        "protocol": "random_observation",
        "partition_seed": int(seed),
        "nominal_target_test_n": int(round(len(out) * test_size)),
        "target_test_n": int(round(len(out) * test_size)),
        "actual_test_n": int(len(test_idx)),
        "partition_hash": partition_hash(out, "split_random_v2"),
    }


def add_legacy_scaffold_split_v2(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, dict]:
    """Reproduce the old largest-groups-first split for sensitivity only."""
    out = df.copy().reset_index(drop=True)
    groups = sorted(
        build_scaffold_groups(out),
        key=lambda group: group.n,
        reverse=True,
    )
    target_n = int(round(len(out) * test_size))
    selected: list[ScaffoldGroup] = []
    selected_n = 0
    for group in groups:
        if selected_n < target_n:
            selected.append(group)
            selected_n += group.n
    test_idx = np.concatenate([group.indices for group in selected])
    out["split_legacy_scaffold_v2"] = "train"
    out.loc[test_idx, "split_legacy_scaffold_v2"] = "test"
    assert_valid_split(
        out,
        "split_legacy_scaffold_v2",
        require_scaffold_disjoint=True,
    )
    return out, {
        "protocol": "legacy_scaffold",
        "partition_seed": None,
        "nominal_target_test_n": target_n,
        "target_test_n": target_n,
        "actual_test_n": int(len(test_idx)),
        "size_deviation": float(
            abs(len(test_idx) - target_n) / max(target_n, 1)
        ),
        "partition_hash": partition_hash(
            out,
            "split_legacy_scaffold_v2",
        ),
    }


def _components(
    selected_n: int,
    selected_sum: float,
    target_n: int,
    global_mean: float,
    target_std: float,
) -> tuple[float, float]:
    if selected_n <= 0:
        return float("inf"), float("inf")
    size_component = abs(selected_n - target_n) / max(target_n, 1)
    mean_component = abs(
        (selected_sum / selected_n) - global_mean
    ) / max(target_std, 1e-12)
    return float(size_component), float(mean_component)


def _candidate_key(
    *,
    objective: SplitObjective,
    size_component: float,
    mean_component: float,
    tie_breaker: float,
) -> tuple[float, ...]:
    if objective == "size_only":
        return (size_component, tie_breaker)
    return (size_component, mean_component, tie_breaker)


def _final_key(
    *,
    objective: SplitObjective,
    within_constraints: bool,
    size_component: float,
    mean_component: float,
    scaffold_key: tuple[str, ...],
) -> tuple:
    penalty = 0 if within_constraints else 1
    if objective == "size_only":
        return (penalty, size_component, scaffold_key)
    return (penalty, size_component, mean_component, scaffold_key)


def _search(
    groups: list[ScaffoldGroup],
    *,
    n_rows: int,
    global_mean: float,
    target_std: float,
    seed: int,
    objective: SplitObjective,
    test_size: float,
    n_trials: int,
    candidate_pool: int,
    min_target_ratio: float,
    max_target_ratio: float,
    target_n_override: int | None,
) -> tuple[list[int], dict]:
    nominal_target_n = int(round(n_rows * test_size))
    target_n = (
        int(target_n_override)
        if target_n_override is not None
        else nominal_target_n
    )
    if target_n <= 0 or target_n >= n_rows:
        raise ValueError(f"Invalid target test size: {target_n}")
    min_n = max(1, int(np.floor(target_n * min_target_ratio)))
    max_n = max(min_n, int(np.ceil(target_n * max_target_ratio)))
    best_selected: list[int] | None = None
    best_key: tuple | None = None
    best_meta: dict | None = None

    for trial in range(n_trials):
        rng = np.random.default_rng(int(seed) * 1_000_003 + trial)
        remaining = list(range(len(groups)))
        selected: list[int] = []
        selected_n = 0
        selected_sum = 0.0

        while remaining and selected_n < min_n:
            candidate_ids = rng.choice(
                remaining,
                size=min(candidate_pool, len(remaining)),
                replace=False,
            )
            candidates: list[tuple[tuple[float, ...], int]] = []
            for raw_idx in candidate_ids:
                idx = int(raw_idx)
                group = groups[idx]
                new_n = selected_n + group.n
                if selected_n > 0 and new_n > max_n:
                    continue
                size_component, mean_component = _components(
                    new_n,
                    selected_sum + group.target_sum,
                    target_n,
                    global_mean,
                    target_std,
                )
                candidates.append(
                    (
                        _candidate_key(
                            objective=objective,
                            size_component=size_component,
                            mean_component=mean_component,
                            tie_breaker=float(rng.random()) * 1e-12,
                        ),
                        idx,
                    )
                )
            if not candidates:
                break
            _, chosen = min(candidates, key=lambda item: item[0])
            group = groups[chosen]
            selected.append(chosen)
            selected_n += group.n
            selected_sum += group.target_sum
            remaining.remove(chosen)

        if not selected:
            continue
        size_component, mean_component = _components(
            selected_n,
            selected_sum,
            target_n,
            global_mean,
            target_std,
        )
        scaffold_key = tuple(sorted(groups[idx].scaffold for idx in selected))
        within_constraints = min_n <= selected_n <= max_n
        key = _final_key(
            objective=objective,
            within_constraints=within_constraints,
            size_component=size_component,
            mean_component=mean_component,
            scaffold_key=scaffold_key,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_selected = list(selected)
            best_meta = {
                "objective": objective,
                "objective_value": (
                    size_component
                    if objective == "size_only"
                    else mean_component
                ),
                "size_component": size_component,
                "mean_component": mean_component,
                "nominal_target_test_n": nominal_target_n,
                "target_test_n": target_n,
                "reference_test_n": (
                    int(target_n_override)
                    if target_n_override is not None
                    else None
                ),
                "actual_test_n": selected_n,
                "min_test_n": min_n,
                "max_test_n": max_n,
                "within_size_constraints": bool(within_constraints),
                "n_trials": int(n_trials),
                "candidate_pool": int(candidate_pool),
                "partition_seed": int(seed),
                "selection_priority": (
                    "size_only"
                    if objective == "size_only"
                    else "size_then_target_mean"
                ),
            }

    if best_selected is None or best_meta is None:
        raise RuntimeError(f"Could not create split for objective={objective}")
    return best_selected, best_meta


def add_searched_scaffold_split_v2(
    df: pd.DataFrame,
    *,
    seed: int,
    objective: SplitObjective,
    target_col: str = "target",
    test_size: float = 0.2,
    n_trials: int = 300,
    candidate_pool: int = 80,
    min_target_ratio: float = 0.95,
    max_target_ratio: float = 1.25,
    target_n_override: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Create audited scaffold splits with size-first selection."""
    if objective not in {"size_only", "target_balanced"}:
        raise ValueError(f"Unsupported objective: {objective}")
    out = df.copy().reset_index(drop=True)
    groups = build_scaffold_groups(out, target_col=target_col)
    target = out[target_col].to_numpy(dtype=float)
    target_std = float(np.std(target, ddof=0))
    if not np.isfinite(target_std) or target_std <= 0:
        target_std = 1.0
    selected, meta = _search(
        groups,
        n_rows=len(out),
        global_mean=float(np.mean(target)),
        target_std=target_std,
        seed=seed,
        objective=objective,
        test_size=test_size,
        n_trials=n_trials,
        candidate_pool=candidate_pool,
        min_target_ratio=min_target_ratio,
        max_target_ratio=max_target_ratio,
        target_n_override=target_n_override,
    )
    test_idx = np.concatenate([groups[idx].indices for idx in selected])
    split_col = (
        "split_size_matched_scaffold_v2"
        if objective == "size_only"
        else "split_target_balanced_scaffold_v2"
    )
    out[split_col] = "train"
    out.loc[test_idx, split_col] = "test"
    assert_valid_split(out, split_col, require_scaffold_disjoint=True)
    meta = dict(meta)
    meta.update(
        {
            "protocol": (
                "size_matched_scaffold"
                if objective == "size_only"
                else "target_balanced_scaffold"
            ),
            "partition_hash": partition_hash(out, split_col),
        }
    )
    return out, meta
