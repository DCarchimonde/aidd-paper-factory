from __future__ import annotations

"""Pre-model grouped role and conformal-resolution audit for RACER-C.

Input rows contain standardized structure IDs and labels only.  The script does
not fit a model, select a gate, or inspect predictions.  Candidate allocations are
all retained; this stage does not choose the winning allocation.
"""

import argparse
import csv
import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[3]
P2 = ROOT / "paper2_admet_benchmark"
DEFAULT_CONFIG = P2 / "configs" / "racer_c" / "study_design.yaml"
DEFAULT_INPUT_DIR = P2 / "data" / "processed" / "racer_c" / "role_inputs"
DEFAULT_OUTPUT_DIR = P2 / "results" / "racer_c_preflight"
ROLES = ("dev", "policy", "conformal", "test")
TRACK_GROUP_COLUMN = {
    "random_grouped": "structure_id",
    "strict_scaffold": "murcko_scaffold_id",
    "similarity_cluster": "similarity_cluster_id",
}


@dataclass(frozen=True)
class AllocationResult:
    assignment: dict[str, str]
    role_class_counts: dict[str, Counter[int]]
    role_totals: Counter[str]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stable_tie(seed: int, group_id: str) -> str:
    return hashlib.sha256(f"{seed}|{group_id}".encode("utf-8")).hexdigest()


def validate_role_input(rows: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("role input is empty")
    required = {"endpoint", "structure_id", "target", "murcko_scaffold_id", "similarity_cluster_id"}
    missing = required - set(materialized[0])
    if missing:
        raise ValueError(f"role input missing columns: {sorted(missing)}")
    endpoints = {row["endpoint"] for row in materialized}
    if len(endpoints) != 1:
        raise ValueError(f"one endpoint per role-input file required, got {sorted(endpoints)}")
    ids = [row["structure_id"] for row in materialized]
    if any(not value for value in ids):
        raise ValueError("blank structure_id")
    if len(ids) != len(set(ids)):
        raise ValueError("role input must have exactly one row per standardized structure")
    for row in materialized:
        if row["target"] not in {"0", "1", 0, 1}:
            raise ValueError(f"non-binary target for {row['structure_id']}: {row['target']}")
        row["target"] = str(row["target"])
    return materialized


def allocate_groups(
    rows: Iterable[Mapping[str, str]],
    group_column: str,
    fractions: Mapping[str, float],
    seed: int,
    use_labels_for_assignment: bool = True,
) -> AllocationResult:
    materialized = validate_role_input(rows)
    if group_column not in materialized[0]:
        raise ValueError(f"missing group column: {group_column}")
    if any(not row[group_column] for row in materialized):
        raise ValueError(f"blank {group_column}; track must fail closed")
    if set(fractions) != set(ROLES) or not math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-9):
        raise ValueError("fractions must define dev/policy/conformal/test and sum to one")

    grouped: dict[str, Counter[int]] = defaultdict(Counter)
    for row in materialized:
        grouped[row[group_column]][int(row["target"])] += 1
    totals = Counter(int(row["target"]) for row in materialized)
    targets = {
        role: {label: fractions[role] * totals[label] for label in (0, 1)}
        for role in ROLES
    }
    target_total = {role: fractions[role] * len(materialized) for role in ROLES}
    role_class_counts = {role: Counter() for role in ROLES}
    role_totals: Counter[str] = Counter()
    assignment: dict[str, str] = {}

    def ordering_key(group_id: str) -> tuple[object, ...]:
        group_size = sum(grouped[group_id].values())
        if use_labels_for_assignment:
            return (
                -max(grouped[group_id][0], grouped[group_id][1]),
                -group_size,
                stable_tie(seed, group_id),
                group_id,
            )
        # Scaffold and similarity-cluster tracks are covariate-only shifts.
        # Their group order, just like their role objective below, must remain
        # invariant under *any* reassignment of labels at fixed group IDs.
        return (-group_size, stable_tie(seed, group_id), group_id)

    ordered_groups = sorted(grouped, key=ordering_key)

    def objective(candidate_role: str, group_id: str) -> tuple[float, int]:
        score = 0.0
        for role in ROLES:
            added_total = sum(grouped[group_id].values()) if role == candidate_role else 0
            after_total = role_totals[role] + added_total
            score += ((after_total - target_total[role]) / max(target_total[role], 1.0)) ** 2
            if use_labels_for_assignment:
                for label in (0, 1):
                    added = grouped[group_id][label] if role == candidate_role else 0
                    after = role_class_counts[role][label] + added
                    score += ((after - targets[role][label]) / max(targets[role][label], 1.0)) ** 2
        return score, ROLES.index(candidate_role)

    for group_id in ordered_groups:
        role = min(ROLES, key=lambda candidate: objective(candidate, group_id))
        assignment[group_id] = role
        role_class_counts[role].update(grouped[group_id])
        role_totals[role] += sum(grouped[group_id].values())

    if len(assignment) != len(grouped):
        raise AssertionError("group assignment is incomplete")
    return AllocationResult(assignment, role_class_counts, role_totals)


def allocation_id(fractions: Mapping[str, float]) -> str:
    return "_".join(str(round(100 * fractions[role])) for role in ROLES)


def conformal_k(n: int, alpha: float) -> int:
    return math.ceil((n + 1) * (1.0 - alpha))


def audit_one(
    rows: list[dict[str, str]],
    track: str,
    fractions: Mapping[str, float],
    seed: int,
    minimum_retention: float,
    alpha: float,
) -> tuple[dict[str, str | int], list[dict[str, str | int | float]]]:
    group_column = TRACK_GROUP_COLUMN[track]
    assignment_uses_labels = track == "random_grouped"
    result = allocate_groups(
        rows,
        group_column,
        fractions,
        seed,
        use_labels_for_assignment=assignment_uses_labels,
    )
    endpoint = rows[0]["endpoint"]
    overall = Counter(int(row["target"]) for row in rows)
    scaffold_count = len({row["murcko_scaffold_id"] for row in rows if row["murcko_scaffold_id"]})
    selected_conf = {
        label: math.floor(minimum_retention * result.role_class_counts["conformal"][label])
        for label in (0, 1)
    }
    reasons: list[str] = []
    for label in (0, 1):
        if overall[label] < 350:
            reasons.append(f"class_{label}_total_lt_350")
        if result.role_class_counts["policy"][label] < 25:
            reasons.append(f"class_{label}_policy_lt_25")
        if result.role_class_counts["conformal"][label] < 70:
            reasons.append(f"class_{label}_conf_lt_70")
        if selected_conf[label] < 35:
            reasons.append(f"class_{label}_selected_conf_lt_35")
        if result.role_class_counts["test"][label] < 70:
            reasons.append(f"class_{label}_test_lt_70")
    if scaffold_count < 100:
        reasons.append("murcko_scaffolds_lt_100")

    summary: dict[str, str | int] = {
        "endpoint": endpoint,
        "track": track,
        "seed": seed,
        "allocation": allocation_id(fractions),
        "group_column": group_column,
        "assignment_uses_labels": str(assignment_uses_labels).lower(),
        "n_structures": len(rows),
        "n_groups": len(result.assignment),
        "n_murcko_scaffolds": scaffold_count,
        "class_0_n": overall[0],
        "class_1_n": overall[1],
    }
    for role in ROLES:
        summary[f"{role}_n"] = result.role_totals[role]
        summary[f"{role}_class_0_n"] = result.role_class_counts[role][0]
        summary[f"{role}_class_1_n"] = result.role_class_counts[role][1]
    summary["selected_conf_class_0_floor"] = selected_conf[0]
    summary["selected_conf_class_1_floor"] = selected_conf[1]
    summary["primary_count_gate"] = "pass" if not reasons else "fail"
    summary["failure_reasons"] = "|".join(reasons)

    resolution: list[dict[str, str | int | float]] = []
    for population, counts in (
        ("full", {label: result.role_class_counts["conformal"][label] for label in (0, 1)}),
        ("selected_floor", selected_conf),
    ):
        for label in (0, 1):
            n = counts[label]
            k = conformal_k(n, alpha)
            resolution.append(
                {
                    "endpoint": endpoint,
                    "track": track,
                    "seed": seed,
                    "allocation": allocation_id(fractions),
                    "population": population,
                    "true_class": label,
                    "n": n,
                    "alpha": alpha,
                    "order_statistic_k": k,
                    "finite_threshold": str(k <= n).lower(),
                    "primary_precision_minimum_35": str(n >= 35).lower(),
                }
            )
    return summary, resolution


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows and not fieldnames:
        raise ValueError(f"empty output requires explicit fieldnames: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames or list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--endpoints",
        default="",
        help="optional comma-separated endpoint allowlist for staged preflight audits",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    input_paths = sorted(args.input_dir.glob("*_role_input.csv"))
    requested = {value.strip() for value in args.endpoints.split(",") if value.strip()}
    if requested:
        by_endpoint = {
            path.name.removesuffix("_role_input.csv"): path for path in input_paths
        }
        missing = requested - set(by_endpoint)
        if missing:
            raise FileNotFoundError(f"missing requested role inputs: {sorted(missing)}")
        input_paths = [by_endpoint[endpoint] for endpoint in sorted(requested)]
    if not input_paths:
        raise FileNotFoundError(
            f"no role inputs under {args.input_dir}; acquire, clean, and hash data first"
        )
    allocations = config["outer_role_candidates"]
    tracks = config["tracks"]
    seeds = config["main_split_seeds"]
    minimum_retention = float(config["gate"]["minimum_planned_retention"])
    alpha = float(config["conformal"]["alpha_primary"])

    summaries: list[dict] = []
    resolutions: list[dict] = []
    failures: list[dict[str, str]] = []
    for path in input_paths:
        rows = validate_role_input(read_csv(path))
        for track in tracks:
            for fractions in allocations:
                for seed in seeds:
                    try:
                        summary, resolution = audit_one(
                            rows, track, fractions, int(seed), minimum_retention, alpha
                        )
                    except ValueError as exc:
                        failures.append(
                            {
                                "endpoint": rows[0]["endpoint"],
                                "track": track,
                                "allocation": allocation_id(fractions),
                                "seed": str(seed),
                                "reason": str(exc),
                            }
                        )
                        continue
                    summaries.append(summary)
                    resolutions.extend(resolution)

    if summaries:
        write_csv(args.output_dir / "role_count_feasibility.csv", summaries)
        write_csv(args.output_dir / "conformal_resolution.csv", resolutions)
    write_csv(
        args.output_dir / "role_feasibility_failures.csv",
        failures,
        fieldnames=["endpoint", "track", "allocation", "seed", "reason"],
    )
    print(f"feasibility rows: {len(summaries)}")
    print(f"resolution rows: {len(resolutions)}")
    print(f"fail-closed rows: {len(failures)}")
    return 0 if summaries else 2


if __name__ == "__main__":
    raise SystemExit(main())
