from __future__ import annotations

"""Prepare a label-blind, development-only seed-99 GPU benchmark plan."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from role_feasibility import allocate_groups, read_csv, validate_role_input


ROOT = Path(__file__).resolve().parents[3]
P2 = ROOT / "paper2_admet_benchmark"
DEFAULT_ROLE_INPUT = (
    P2 / "data" / "processed" / "racer_c" / "role_inputs" / "Tox21_NR_ER_role_input.csv"
)
DEFAULT_CLEAN = P2 / "data" / "processed" / "racer_c" / "Tox21_NR_ER_clean.csv"
DEFAULT_DECISIONS = (
    P2 / "results" / "racer_c_phase2_preflight" / "endpoint_eligibility_decision.csv"
)
DEFAULT_OUTPUT = (
    P2 / "results" / "racer_c_phase3_preflight" / "seed99_gpu_benchmark_plan.json"
)
FRACTIONS = {"dev": 0.50, "policy": 0.20, "conformal": 0.15, "test": 0.15}
SEED = 99
TRACK = "strict_scaffold"
GROUP_COLUMN = "murcko_scaffold_id"
META_FOLDS = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_tie(seed: int, group_id: str) -> str:
    return hashlib.sha256(f"{seed}|{group_id}".encode("utf-8")).hexdigest()


def label_blind_group_folds(
    rows: Iterable[Mapping[str, str]], group_column: str, n_folds: int, seed: int
) -> dict[str, int]:
    materialized = [dict(row) for row in rows]
    if n_folds < 2:
        raise ValueError("at least two folds are required")
    groups: dict[str, int] = Counter(str(row[group_column]) for row in materialized)
    if "" in groups:
        raise ValueError(f"blank {group_column}")
    totals = [0] * n_folds
    assignment: dict[str, int] = {}
    ordered = sorted(
        groups,
        key=lambda group_id: (
            -groups[group_id],
            stable_tie(seed, group_id),
            group_id,
        ),
    )
    for group_id in ordered:
        fold = min(range(n_folds), key=lambda value: (totals[value], value))
        assignment[group_id] = fold
        totals[fold] += groups[group_id]
    if sum(totals) != len(materialized):
        raise AssertionError("fold allocation lost rows")
    return assignment


def read_clean(path: Path, allowed_ids: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["structure_id"] in allowed_ids
        ]
    by_id = {row["structure_id"]: row for row in rows}
    if set(by_id) != allowed_ids or len(rows) != len(by_id):
        raise ValueError("clean table does not map one-to-one to development IDs")
    return [by_id[value] for value in sorted(by_id)]


def class_counts(rows: Iterable[Mapping[str, str]]) -> dict[str, int]:
    counts = Counter(int(row["target"]) for row in rows)
    return {"class_0_n": counts[0], "class_1_n": counts[1]}


def assert_primary(endpoint: str, path: Path) -> int:
    rows = read_csv(path)
    matches = [row for row in rows if row["endpoint"] == endpoint]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one endpoint decision for {endpoint}")
    if matches[0]["eligibility_status"] != "primary_candidate":
        raise ValueError(f"benchmark endpoint is not primary_candidate: {endpoint}")
    return sum(row["eligibility_status"] == "primary_candidate" for row in rows)


def build_plan(
    role_rows: list[dict[str, str]],
    clean_rows: list[dict[str, str]],
    primary_endpoint_count: int,
) -> dict[str, object]:
    endpoint = role_rows[0]["endpoint"]
    allocation = allocate_groups(
        role_rows,
        GROUP_COLUMN,
        FRACTIONS,
        SEED,
        use_labels_for_assignment=False,
    )
    dev_ids = {
        row["structure_id"]
        for row in role_rows
        if allocation.assignment[row[GROUP_COLUMN]] == "dev"
    }
    dev_rows = [row for row in clean_rows if row["structure_id"] in dev_ids]
    if len(dev_rows) != len(dev_ids):
        raise ValueError("development clean rows are incomplete")
    outer_assignment = label_blind_group_folds(
        dev_rows, GROUP_COLUMN, META_FOLDS, SEED
    )
    outer_by_row = {
        row["structure_id"]: outer_assignment[row[GROUP_COLUMN]] for row in dev_rows
    }
    jobs: list[dict[str, object]] = []
    for outer_fold in range(META_FOLDS):
        outer_train = [
            row for row in dev_rows if outer_by_row[row["structure_id"]] != outer_fold
        ]
        outer_valid = [
            row for row in dev_rows if outer_by_row[row["structure_id"]] == outer_fold
        ]
        inner_assignment = label_blind_group_folds(
            outer_train, GROUP_COLUMN, 2, SEED * 100 + outer_fold
        )
        for inner_fold in range(2):
            fit_rows = [
                row
                for row in outer_train
                if inner_assignment[row[GROUP_COLUMN]] != inner_fold
            ]
            predict_rows = [
                row
                for row in outer_train
                if inner_assignment[row[GROUP_COLUMN]] == inner_fold
            ]
            jobs.append(
                {
                    "job_id": f"outer_{outer_fold}_inner_{inner_fold}",
                    "stage": "inner_oof",
                    "fit_n": len(fit_rows),
                    "predict_n": len(predict_rows),
                    **{f"fit_{key}": value for key, value in class_counts(fit_rows).items()},
                    **{
                        f"predict_{key}": value
                        for key, value in class_counts(predict_rows).items()
                    },
                }
            )
        jobs.append(
            {
                "job_id": f"outer_{outer_fold}_final",
                "stage": "outer_final",
                "fit_n": len(outer_train),
                "predict_n": len(outer_valid),
                **{f"fit_{key}": value for key, value in class_counts(outer_train).items()},
                **{
                    f"predict_{key}": value
                    for key, value in class_counts(outer_valid).items()
                },
            }
        )
    if len(jobs) != 9:
        raise AssertionError("three outer folds require six inner and three final fits")
    return {
        "status": "ready_for_target_gpu_component_benchmark",
        "scientific_interpretation": "technical timing and lineage only; not a model result",
        "endpoint": endpoint,
        "seed": SEED,
        "track": TRACK,
        "allocation": "50_20_15_15",
        "trainer_label_roles": ["dev"],
        "policy_conformal_test_predictions_generated": False,
        "performance_metrics_permitted": False,
        "dev_n": len(dev_rows),
        **class_counts(dev_rows),
        "meta_folds": META_FOLDS,
        "dmpnn_fit_jobs_per_endpoint_track_seed": len(jobs),
        "dmpnn_final_fit_equivalents_per_endpoint_track_seed": 6.0,
        "primary_endpoint_count": primary_endpoint_count,
        "primary_tracks": 3,
        "primary_seeds": 5,
        "primary_endpoint_track_seed_cells": primary_endpoint_count * 3 * 5,
        "jobs": jobs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-input", type=Path, default=DEFAULT_ROLE_INPUT)
    parser.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    role_rows = validate_role_input(read_csv(args.role_input))
    endpoint = role_rows[0]["endpoint"]
    primary_endpoint_count = assert_primary(endpoint, args.decisions)
    clean_rows = read_clean(
        args.clean, {row["structure_id"] for row in role_rows}
    )
    plan = build_plan(role_rows, clean_rows, primary_endpoint_count)
    plan.update(
        {
            "role_input_sha256": sha256_file(args.role_input),
            "clean_input_sha256": sha256_file(args.clean),
            "endpoint_decisions_sha256": sha256_file(args.decisions),
            "script_sha256": sha256_file(Path(__file__)),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(plan, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
