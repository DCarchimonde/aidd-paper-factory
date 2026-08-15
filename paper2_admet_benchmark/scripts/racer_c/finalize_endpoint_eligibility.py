from __future__ import annotations

"""Finalize count-only Tox21 endpoint eligibility before any model fitting."""

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
P2 = ROOT / "paper2_admet_benchmark"
DEFAULT_CANDIDATES = P2 / "protocols" / "endpoint_candidate_manifest.csv"
DEFAULT_MANIFESTS = P2 / "data" / "manifests" / "racer_c"
DEFAULT_ROLES = P2 / "results" / "racer_c_phase2_preflight" / "role_count_feasibility.csv"
DEFAULT_OUTPUT = P2 / "results" / "racer_c_phase2_preflight" / "endpoint_eligibility_decision.csv"
SELECTED_ALLOCATION = "50_20_15_15"
PRIMARY_TOTAL_CLASS_MINIMUM = 350
PRIMARY_REQUIRED_CELLS = 15


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def decide(
    cleaning: dict[str, object], role_rows: list[dict[str, str]]
) -> tuple[str, str, str, int, int]:
    endpoint = str(cleaning["endpoint"])
    n0 = int(cleaning["clean_class_0_n"])
    n1 = int(cleaning["clean_class_1_n"])
    selected = [
        row
        for row in role_rows
        if row["endpoint"] == endpoint and row["allocation"] == SELECTED_ALLOCATION
    ]
    passing = sum(row["primary_count_gate"] == "pass" for row in selected)
    total = len(selected)
    if min(n0, n1) < PRIMARY_TOTAL_CLASS_MINIMUM:
        return (
            "calibration-limited",
            "freeze1_calibration_limited",
            f"clean class minimum {min(n0, n1)} < {PRIMARY_TOTAL_CLASS_MINIMUM}; no primary claim",
            passing,
            total,
        )
    if total == PRIMARY_REQUIRED_CELLS and passing == total:
        return (
            "primary_candidate",
            "freeze1_primary_candidate",
            f"all {total}/{total} track-seed count cells pass under {SELECTED_ALLOCATION}",
            passing,
            total,
        )
    if total:
        failed_tracks = sorted(
            {
                row["track"]
                for row in selected
                if row["primary_count_gate"] != "pass"
            }
        )
        return (
            "track_limited_secondary",
            "freeze1_secondary_candidate",
            f"{passing}/{total} cells pass; failed tracks={'+'.join(failed_tracks)}",
            passing,
            total,
        )
    return (
        "calibration-limited",
        "freeze1_calibration_limited",
        "total clean class count failed before similarity clustering",
        passing,
        total,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFESTS)
    parser.add_argument("--roles", type=Path, default=DEFAULT_ROLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write-candidate-manifest", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = read_csv(args.candidates)
    role_rows = read_csv(args.roles)
    decisions: list[dict[str, object]] = []
    for row in candidates:
        endpoint = row["endpoint"]
        if not endpoint.startswith("Tox21_"):
            continue
        path = args.manifest_dir / f"{endpoint}_cleaning.json"
        if not path.exists():
            raise FileNotFoundError(path)
        cleaning = json.loads(path.read_text(encoding="utf-8"))
        eligibility, candidate_status, reason, passing, total = decide(
            cleaning, role_rows
        )
        n0 = int(cleaning["clean_class_0_n"])
        n1 = int(cleaning["clean_class_1_n"])
        clean_n = int(cleaning["clean_unique_structures"])
        row.update(
            {
                "clean_n": str(clean_n),
                "class_0_n": str(n0),
                "class_1_n": str(n1),
                "positive_prevalence": f"{n1 / clean_n:.10f}",
                "unique_standardized_structures": str(clean_n),
                "duplicate_count": str(cleaning["duplicate_rows_aggregated"]),
                "conflict_count": str(
                    cleaning["conflicting_structure_groups_excluded"]
                ),
                "murcko_scaffold_count": str(cleaning["murcko_scaffold_count"]),
                "candidate_status": candidate_status,
                "eligibility_status": eligibility,
                "eligibility_reason": reason,
            }
        )
        decisions.append(
            {
                "endpoint": endpoint,
                "source_family": "NCATS_Tox21_2014",
                "clean_n": clean_n,
                "class_0_n": n0,
                "class_1_n": n1,
                "selected_allocation": SELECTED_ALLOCATION,
                "passing_track_seed_cells": passing,
                "audited_track_seed_cells": total,
                "eligibility_status": eligibility,
                "eligibility_reason": reason,
                "selection_used_model_outputs": "false",
            }
        )
    write_csv(args.output, decisions)
    if args.write_candidate_manifest:
        write_csv(args.candidates, candidates)
    counts: dict[str, int] = {}
    for decision in decisions:
        key = str(decision["eligibility_status"])
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
