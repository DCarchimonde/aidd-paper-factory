from __future__ import annotations

"""Fail-closed RACER-C gate selection with simultaneous exact bounds."""

import csv
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from scipy.stats import beta


THRESHOLD_GRID = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)


@dataclass(frozen=True)
class PolicyConstraints:
    critical_class: int = 1
    retention_floor: float = 0.50
    critical_error_ceiling: float = 0.10
    familywise_alpha: float = 0.05
    minimum_class_n: int = 25
    minimum_selected_critical_n: int = 25


def one_sided_exact_lower(successes: int, n: int, alpha: float) -> float:
    if n <= 0:
        return 0.0
    if successes <= 0:
        return 0.0
    return float(beta.ppf(alpha, successes, n - successes + 1))


def one_sided_exact_upper(successes: int, n: int, alpha: float) -> float:
    if n <= 0:
        return 1.0
    if successes >= n:
        return 1.0
    return float(beta.ppf(1.0 - alpha, successes + 1, n - successes))


def validate_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("policy rows are empty")
    required = {"structure_id", "true_class", "predicted_class", "risk_percentile"}
    missing = required - set(materialized[0])
    if missing:
        raise ValueError(f"policy rows missing columns: {sorted(missing)}")
    ids = [str(row["structure_id"]) for row in materialized]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("policy structure IDs must be unique and nonblank")
    for row in materialized:
        row["true_class"] = int(row["true_class"])
        row["predicted_class"] = int(row["predicted_class"])
        row["risk_percentile"] = float(row["risk_percentile"])
        if row["true_class"] not in {0, 1} or row["predicted_class"] not in {0, 1}:
            raise ValueError("policy classes must be binary")
        if not 0.0 <= row["risk_percentile"] <= 1.0:
            raise ValueError("risk percentiles must lie in [0,1]")
    return materialized


def evaluate_pair(
    rows: list[dict[str, object]],
    thresholds: Mapping[int, float],
    constraints: PolicyConstraints,
    simultaneous_alpha: float,
) -> dict[str, object]:
    selected = [
        row
        for row in rows
        if float(row["risk_percentile"]) <= thresholds[int(row["predicted_class"])]
    ]
    totals = {label: sum(int(row["true_class"]) == label for row in rows) for label in (0, 1)}
    selected_counts = {
        label: sum(int(row["true_class"]) == label for row in selected)
        for label in (0, 1)
    }
    critical = constraints.critical_class
    critical_errors = sum(
        int(row["true_class"]) == critical
        and int(row["predicted_class"]) != critical
        for row in selected
    )
    retention_lower = {
        label: one_sided_exact_lower(
            selected_counts[label], totals[label], simultaneous_alpha
        )
        for label in (0, 1)
    }
    error_upper = one_sided_exact_upper(
        critical_errors, selected_counts[critical], simultaneous_alpha
    )
    count_ready = all(totals[label] >= constraints.minimum_class_n for label in (0, 1))
    count_ready = count_ready and (
        selected_counts[critical] >= constraints.minimum_selected_critical_n
    )
    feasible = (
        count_ready
        and all(
            retention_lower[label] >= constraints.retention_floor
            for label in (0, 1)
        )
        and error_upper <= constraints.critical_error_ceiling
    )
    return {
        "threshold_0": thresholds[0],
        "threshold_1": thresholds[1],
        "n": len(rows),
        "selected_n": len(selected),
        "class_0_n": totals[0],
        "class_1_n": totals[1],
        "selected_class_0_n": selected_counts[0],
        "selected_class_1_n": selected_counts[1],
        "class_0_retention_lower": retention_lower[0],
        "class_1_retention_lower": retention_lower[1],
        "critical_errors": critical_errors,
        "critical_error_upper": error_upper,
        "count_ready": count_ready,
        "feasible": feasible,
    }


def select_policy(
    rows: Iterable[Mapping[str, object]],
    constraints: PolicyConstraints = PolicyConstraints(),
    threshold_grid: tuple[float, ...] = THRESHOLD_GRID,
) -> tuple[str, dict[str, object] | None, list[dict[str, object]]]:
    materialized = validate_rows(rows)
    pairs = list(itertools.product(threshold_grid, repeat=2))
    # Two class-retention bounds plus one critical-class error bound are tested
    # for each candidate pair.  This conservative Bonferroni contract remains
    # valid despite deterministic selection among the 36 pairs.
    simultaneous_alpha = constraints.familywise_alpha / (len(pairs) * 3)
    evaluations = [
        evaluate_pair(
            materialized,
            {0: threshold_0, 1: threshold_1},
            constraints,
            simultaneous_alpha,
        )
        for threshold_0, threshold_1 in pairs
    ]
    feasible = [row for row in evaluations if bool(row["feasible"])]
    if not feasible:
        return "policy-infeasible", None, evaluations
    critical = constraints.critical_class
    other = 1 - critical

    def rank(row: Mapping[str, object]) -> tuple[float, float, float, float, float]:
        total_retention = float(row["selected_n"]) / float(row["n"])
        smaller_class_retention = min(
            float(row["selected_class_0_n"]) / float(row["class_0_n"]),
            float(row["selected_class_1_n"]) / float(row["class_1_n"]),
        )
        return (
            -total_retention,
            -smaller_class_retention,
            float(row["threshold_0"]) + float(row["threshold_1"]),
            float(row[f"threshold_{critical}"]),
            float(row[f"threshold_{other}"]),
        )

    chosen = min(feasible, key=rank)
    return "selected", chosen, evaluations


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
