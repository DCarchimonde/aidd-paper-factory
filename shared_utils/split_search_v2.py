"""One audited engine for size-matched and target-balanced scaffold splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

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
    target_n = int(round(len(out) * test_size))
    return out, {
        "protocol": "random_observation",
        "partition_seed": int(seed),
        "nominal_target_test_n": target_n,
        "target_test_n": target_n,
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


def _make_meta(
    *,
    objective: SplitObjective,
    nominal_target_n: int,
    target_n: int,
    actual_n: int,
    size_component: float,
    mean_component: float,
    min_n: int,
    max_n: int,
    n_trials: int,
    candidate_pool: int,
    seed: int,
    target_n_override: int | None,
    strict_size_lock: bool,
    incumbent_source: str,
) -> dict:
    return {
        "objective": objective,
        "objective_value": (
            size_component if objective == "size_only" else mean_component
        ),
        "size_component": size_component,
        "mean_component": mean_component,
        "nominal_target_test_n": nominal_target_n,
        "target_test_n": target_n,
        "reference_test_n": (
            int(target_n_override) if target_n_override is not None else None
        ),
        "actual_test_n": actual_n,
        "min_test_n": min_n,
        "max_test_n": max_n,
        "within_size_constraints": bool(min_n <= actual_n <= max_n),
        "n_trials": int(n_trials),
        "candidate_pool": int(candidate_pool),
        "partition_seed": int(seed),
        "selection_priority": (
            "size_only"
            if objective == "size_only"
            else "exact_size_then_target_mean"
            if strict_size_lock
            else "size_then_target_mean"
        ),
        "strict_size_lock": bool(strict_size_lock),
        "incumbent_source": incumbent_source,
    }


def _reference_incumbent(
    groups: list[ScaffoldGroup],
    reference_test_scaffolds: Sequence[str],
    *,
    target_n: int,
    global_mean: float,
    target_std: float,
) -> tuple[list[int], float, float, tuple[str, ...]]:
    scaffold_to_index = {group.scaffold: idx for idx, group in enumerate(groups)}
    reference = tuple(sorted(str(value) for value in reference_test_scaffolds))
    missing = [scaffold for scaffold in reference if scaffold not in scaffold_to_index]
    if missing:
        raise KeyError(f"Reference test scaffolds not found: {missing[:5]}")

    selected = [scaffold_to_index[scaffold] for scaffold in reference]
    selected_n = int(sum(groups[idx].n for idx in selected))
    if selected_n != target_n:
        raise AssertionError(
            "Reference partition does not match the requested locked test size: "
            f"{selected_n} != {target_n}"
        )
    selected_sum = float(sum(groups[idx].target_sum for idx in selected))
    size_component, mean_component = _components(
        selected_n,
        selected_sum,
        target_n,
        global_mean,
        target_std,
    )
    return selected, size_component, mean_component, reference


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
    reference_test_scaffolds: Sequence[str] | None,
) -> tuple[list[int], dict]:
    nominal_target_n = int(round(n_rows * test_size))
    target_n = (
        int(target_n_override)
        if target_n_override is not None
        else nominal_target_n
    )
    if target_n <= 0 or target_n >= n_rows:
        raise ValueError(f"Invalid target test size: {target_n}")

    strict_size_lock = (
        objective == "target_balanced"
        and target_n_override is not None
        and reference_test_scaffolds is not None
    )
    if strict_size_lock:
        min_n = target_n
        max_n = target_n
    else:
        min_n = max(1, int(np.floor(target_n * min_target_ratio)))
        max_n = max(min_n, int(np.ceil(target_n * max_target_ratio)))

    best_selected: list[int] | None = None
    best_key: tuple | None = None
    best_meta: dict | None = None

    if strict_size_lock:
        (
            reference_selected,
            reference_size_component,
            reference_mean_component,
            reference_key,
        ) = _reference_incumbent(
            groups,
            reference_test_scaffolds,
            target_n=target_n,
            global_mean=global_mean,
            target_std=target_std,
        )
        best_selected = reference_selected
        best_key = (reference_mean_component, reference_key)
        best_meta = _make_meta(
            objective=objective,
            nominal_target_n=nominal_target_n,
            target_n=target_n,
            actual_n=target_n,
            size_component=reference_size_component,
            mean_component=reference_mean_component,
            min_n=min_n,
            max_n=max_n,
            n_trials=n_trials,
            candidate_pool=candidate_pool,
            seed=seed,
            target_n_override=target_n_override,
            strict_size_lock=True,
            incumbent_source="size_matched_reference",
        )

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
                if new_n > max_n:
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

        if not selected or not (min_n <= selected_n <= max_n):
            continue
        if strict_size_lock and selected_n != target_n:
            continue

        size_component, mean_component = _components(
            selected_n,
            selected_sum,
            target_n,
            global_mean,
            target_std,
        )
        scaffold_key = tuple(sorted(groups[idx].scaffold for idx in selected))
        if objective == "size_only":
            key = (size_component, scaffold_key)
        elif strict_size_lock:
            key = (mean_component, scaffold_key)
        else:
            key = (size_component, mean_component, scaffold_key)

        if best_key is None or key < best_key:
            best_key = key
            best_selected = list(selected)
            best_meta = _make_meta(
                objective=objective,
                nominal_target_n=nominal_target_n,
                target_n=target_n,
                actual_n=selected_n,
                size_component=size_component,
                mean_component=mean_component,
                min_n=min_n,
                max_n=max_n,
                n_trials=n_trials,
                candidate_pool=candidate_pool,
                seed=seed,
                target_n_override=target_n_override,
                strict_size_lock=strict_size_lock,
                incumbent_source="stochastic_search",
            )

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
    reference_test_scaffolds: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Create audited scaffold splits with optional exact paired-size locking."""
    if objective not in {"size_only", "target_balanced"}:
        raise ValueError(f"Unsupported objective: {objective}")
    if reference_test_scaffolds is not None and objective != "target_balanced":
        raise ValueError("Reference scaffolds are only valid for target balancing.")
    if reference_test_scaffolds is not None and target_n_override is None:
        raise ValueError("Reference scaffolds require target_n_override.")

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
        reference_test_scaffolds=reference_test_scaffolds,
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

    if reference_test_scaffolds is not None:
        expected = int(target_n_override)
        actual = int(out[split_col].eq("test").sum())
        if actual != expected:
            raise AssertionError(
                f"Strict paired-size lock failed: {actual} != {expected}"
            )

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
