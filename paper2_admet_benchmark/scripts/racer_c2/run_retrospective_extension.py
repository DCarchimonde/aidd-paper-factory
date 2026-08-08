from __future__ import annotations

"""Evaluate the fixed RACER-C2 family on completed RACER-C v1 artifacts.

This is an additive, CPU-only post-processing run.  It never retrains a base
model and never writes inside the v1 source directory.  Because the v1 test
outcomes were already known during RACER-C2 development, every output is
explicitly retrospective and cannot be relabelled as prospective evidence.
"""

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import yaml

from core import (
    ActionCertificateConstraints,
    ScoreConfiguration,
    certify_final_sets,
    compose_candidate_scores,
    mondrian_thresholds,
    prediction_sets,
    set_metric_record,
    state_labels,
)


P2_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCK = (
    P2_ROOT / "configs" / "racer_c2" / "retrospective_extension_lock_v0.yaml"
)
DEFAULT_SOURCE = P2_ROOT / "results" / "racer_c_confirmatory_v1"
DEFAULT_OUTPUT = P2_ROOT / "results" / "racer_c2_retrospective_extension_v0"

RAW_REQUIRED_COLUMNS = {
    "structure_id",
    "role",
    "target",
    "meta_fold",
    "stack_p",
    "risk_percentile",
}
TEST_REQUIRED_COLUMNS = {"structure_id", "target", "method", "state"}
VALID_ROLES = ("dev", "policy", "conformal", "test")

METRIC_FIELDS = [
    "endpoint",
    "track",
    "seed",
    "cell",
    "method",
    "method_role",
    "evaluation_role",
    "n",
    "class_0_n",
    "class_0_correct_singleton_n",
    "class_0_wrong_singleton_n",
    "class_0_covered_n",
    "class_0_empty_n",
    "class_0_coverage",
    "class_0_csy",
    "class_0_wrong_singleton_exposure",
    "class_0_empty_exposure",
    "class_1_n",
    "class_1_correct_singleton_n",
    "class_1_wrong_singleton_n",
    "class_1_covered_n",
    "class_1_empty_n",
    "class_1_coverage",
    "class_1_csy",
    "class_1_wrong_singleton_exposure",
    "class_1_empty_exposure",
    "macro_csy",
    "worst_csy",
    "ambiguous_n",
    "empty_n",
    "singleton_n",
    "ambiguity_rate",
    "empty_rate",
    "singleton_rate",
    "qhat_0",
    "qhat_1",
]

PREDICTION_FIELDS = [
    "endpoint",
    "track",
    "seed",
    "cell",
    "structure_id",
    "target",
    "method",
    "state",
    "include_0",
    "include_1",
    "score_0",
    "score_1",
    "qhat_0",
    "qhat_1",
    "risk_percentile",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary_path = Path(temporary)
        if temporary_path.exists():
            temporary_path.unlink()


def write_csv(
    path: Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(fields), lineterminator="\n", extrasaction="raise"
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary_path = Path(temporary)
        if temporary_path.exists():
            temporary_path.unlink()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def parse_cell_name(name: str) -> tuple[str, str, int]:
    parts = name.split("__")
    if len(parts) != 3 or not parts[2].startswith("seed"):
        raise ValueError(f"unrecognized cell name: {name}")
    endpoint, track = parts[:2]
    seed = int(parts[2][4:])
    if not endpoint or not track:
        raise ValueError(f"unrecognized cell name: {name}")
    return endpoint, track, seed


def load_lock(path: Path) -> dict[str, object]:
    lock = yaml.safe_load(path.read_text(encoding="utf-8"))
    if lock.get("lock_status") != "retrospective_extension_fixed":
        raise RuntimeError("RACER-C2 retrospective lock is not fixed")
    if lock.get("source_panel_already_known") is not True:
        raise RuntimeError("the retrospective source-panel disclosure is missing")
    if lock.get("confirmatory_claim_authorized") is not False:
        raise RuntimeError("retrospective output cannot authorize a confirmatory claim")
    methods = list(lock.get("fixed_method_family", []))
    names = [str(row["method"]) for row in methods]
    if not methods or len(names) != len(set(names)):
        raise ValueError("the fixed RACER-C2 method family is empty or duplicated")
    if str(lock.get("primary_method")) not in names:
        raise ValueError("the primary RACER-C2 method is absent from the fixed family")
    for row in methods:
        ScoreConfiguration(
            float(row["t_max"]),
            float(row["gamma_0"]),
            float(row["gamma_1"]),
            float(row["counterfactual_blend"]),
        ).validate()
    return lock


def assert_separate_output(source: Path, output: Path) -> None:
    source_resolved = source.resolve()
    output_resolved = output.resolve()
    if output_resolved == source_resolved or source_resolved in output_resolved.parents:
        raise ValueError("RACER-C2 output must not be the v1 source or its child")


def source_cell_directories(
    source: Path, lock: Mapping[str, object]
) -> list[Path]:
    run_summary_path = source / "run_summary.json"
    if not run_summary_path.is_file():
        raise FileNotFoundError(run_summary_path)
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    expected = lock["expected_source"]
    if run_summary.get("status") != "complete_confirmatory_primary_study":
        raise RuntimeError("v1 source run is not complete")
    if int(run_summary.get("primary_cell_count", -1)) != int(expected["cell_count"]):
        raise RuntimeError("v1 source cell count does not match the lock")
    if int(run_summary.get("method_cell_count", -1)) != int(
        expected["source_method_cell_count"]
    ):
        raise RuntimeError("v1 source method-cell count does not match the lock")
    cell_root = source / "cells"
    directories = sorted(
        path
        for path in cell_root.iterdir()
        if path.is_dir() and (path / "raw_predictions.csv").is_file()
    )
    expected_names = {
        f"{endpoint}__{track}__seed{int(seed)}"
        for endpoint in expected["endpoints"]
        for track in expected["tracks"]
        for seed in expected["seeds"]
    }
    observed_names = {path.name for path in directories}
    if observed_names != expected_names:
        missing = sorted(expected_names - observed_names)
        extra = sorted(observed_names - expected_names)
        raise RuntimeError(f"v1 cell matrix mismatch; missing={missing}, extra={extra}")
    return directories


def recover_test_labels(
    path: Path, expected_method_count: int
) -> tuple[dict[str, int], tuple[str, ...]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = TEST_REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        labels: dict[str, int] = {}
        methods_by_id: dict[str, set[str]] = defaultdict(set)
        all_methods: set[str] = set()
        for row in reader:
            structure_id = row["structure_id"]
            target = int(row["target"])
            if target not in {0, 1}:
                raise ValueError(f"non-binary test target in {path}")
            if structure_id in labels and labels[structure_id] != target:
                raise RuntimeError(f"conflicting test target for {structure_id}")
            labels[structure_id] = target
            method = row["method"]
            methods_by_id[structure_id].add(method)
            all_methods.add(method)
    if len(all_methods) != expected_method_count:
        raise RuntimeError(
            f"source test method count is {len(all_methods)}/{expected_method_count}"
        )
    incomplete = [
        structure_id
        for structure_id, methods in methods_by_id.items()
        if methods != all_methods
    ]
    if incomplete:
        raise RuntimeError("source test predictions are incomplete by structure ID")
    return labels, tuple(sorted(all_methods))


def load_and_verify_cell(
    directory: Path, lock: Mapping[str, object]
) -> tuple[dict[str, list[dict[str, str]]], dict[str, object]]:
    endpoint, track, seed = parse_cell_name(directory.name)
    raw_path = directory / "raw_predictions.csv"
    test_path = directory / "test_predictions.csv"
    metrics_path = directory / "metrics.csv"
    raw_manifest_path = directory / "raw_manifest.json"
    final_manifest_path = directory / "final_manifest.json"
    for path in (
        raw_path,
        test_path,
        metrics_path,
        raw_manifest_path,
        final_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    if raw_manifest.get("status") != "complete_raw_predictions":
        raise RuntimeError(f"incomplete v1 raw cell: {directory.name}")
    if final_manifest.get("status") != "complete_final_evaluation":
        raise RuntimeError(f"incomplete v1 final cell: {directory.name}")
    expected_method_count = int(lock["expected_source"]["source_method_count"])
    checks = {
        "endpoint": endpoint,
        "track": track,
        "seed": seed,
        "raw_predictions_sha256": sha256_file(raw_path),
        "test_predictions_sha256": sha256_file(test_path),
        "metrics_sha256": sha256_file(metrics_path),
    }
    for key in ("endpoint", "track", "seed"):
        if raw_manifest.get(key) != checks[key] or final_manifest.get(key) != checks[key]:
            raise RuntimeError(f"v1 manifest identity mismatch in {directory.name}: {key}")
    if raw_manifest.get("raw_predictions_sha256") != checks["raw_predictions_sha256"]:
        raise RuntimeError(f"v1 raw hash mismatch in {directory.name}")
    if final_manifest.get("test_predictions_sha256") != checks["test_predictions_sha256"]:
        raise RuntimeError(f"v1 test hash mismatch in {directory.name}")
    if final_manifest.get("metrics_sha256") != checks["metrics_sha256"]:
        raise RuntimeError(f"v1 metrics hash mismatch in {directory.name}")
    if int(final_manifest.get("method_count", -1)) != expected_method_count:
        raise RuntimeError(f"v1 method count mismatch in {directory.name}")

    with raw_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = RAW_REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{raw_path} is missing columns: {sorted(missing)}")
        rows = list(reader)
    identifiers = [row["structure_id"] for row in rows]
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise RuntimeError(f"blank or duplicated raw structure IDs in {directory.name}")
    by_role: dict[str, list[dict[str, str]]] = {role: [] for role in VALID_ROLES}
    for row in rows:
        role = row["role"]
        if role not in by_role:
            raise ValueError(f"unexpected role {role!r} in {directory.name}")
        if role == "test":
            if row["target"]:
                raise RuntimeError(f"v1 raw test target is not sealed in {directory.name}")
        elif row["target"] not in {"0", "1"}:
            raise ValueError(f"non-test target is missing in {directory.name}")
        by_role[role].append(row)
    role_counts = {role: len(by_role[role]) for role in VALID_ROLES}
    if role_counts != {key: int(value) for key, value in raw_manifest["role_counts"].items()}:
        raise RuntimeError(f"v1 role counts mismatch in {directory.name}")
    if any(not by_role[role] for role in VALID_ROLES):
        raise RuntimeError(f"v1 role is empty in {directory.name}")

    test_labels, source_methods = recover_test_labels(test_path, expected_method_count)
    raw_test_ids = {row["structure_id"] for row in by_role["test"]}
    if set(test_labels) != raw_test_ids:
        raise RuntimeError(f"v1 test IDs do not join exactly in {directory.name}")
    for row in by_role["test"]:
        row["target"] = str(test_labels[row["structure_id"]])
    if len(by_role["test"]) != int(final_manifest.get("test_n", -1)):
        raise RuntimeError(f"v1 test count mismatch in {directory.name}")
    source_record = {
        **checks,
        "cell": directory.name,
        "role_counts": role_counts,
        "source_methods": list(source_methods),
        "test_labels_recovered_by_structure_id": True,
        "source_artifacts_modified": False,
    }
    return by_role, source_record


def arrays(
    rows: Sequence[Mapping[str, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    targets = np.asarray([int(row["target"]) for row in rows], dtype=np.int8)
    probability = np.asarray([float(row["stack_p"]) for row in rows], dtype=float)
    risk = np.asarray([float(row["risk_percentile"]) for row in rows], dtype=float)
    identifiers = [str(row["structure_id"]) for row in rows]
    if not len(rows) or not np.isfinite(probability).all() or not np.isfinite(risk).all():
        raise ValueError("RACER-C2 role arrays are empty or non-finite")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("stack probabilities lie outside [0,1]")
    if np.any((risk < 0.0) | (risk > 1.0)):
        raise ValueError("risk percentiles lie outside [0,1]")
    if set(np.unique(targets)) != {0, 1}:
        raise ValueError("each RACER-C2 role requires both classes")
    return targets, probability, risk, identifiers


def configuration(row: Mapping[str, object]) -> ScoreConfiguration:
    return ScoreConfiguration(
        t_max=float(row["t_max"]),
        gamma_0=float(row["gamma_0"]),
        gamma_1=float(row["gamma_1"]),
        counterfactual_blend=float(row["counterfactual_blend"]),
    )


def metric_row(
    endpoint: str,
    track: str,
    seed: int,
    cell: str,
    method: str,
    method_role: str,
    evaluation_role: str,
    targets: np.ndarray,
    sets: np.ndarray,
    thresholds: Mapping[int, float],
) -> dict[str, object]:
    record = set_metric_record(targets, sets)
    n = int(record["n"])
    singleton_n = int(
        record["class_0_correct_singleton_n"]
        + record["class_0_wrong_singleton_n"]
        + record["class_1_correct_singleton_n"]
        + record["class_1_wrong_singleton_n"]
    )
    # Each singleton belongs to exactly one true-class denominator, so the four
    # class-specific counts above partition singleton rows.
    return {
        "endpoint": endpoint,
        "track": track,
        "seed": seed,
        "cell": cell,
        "method": method,
        "method_role": method_role,
        "evaluation_role": evaluation_role,
        **record,
        "singleton_n": singleton_n,
        "ambiguity_rate": int(record["ambiguous_n"]) / n,
        "empty_rate": int(record["empty_n"]) / n,
        "singleton_rate": singleton_n / n,
        "qhat_0": float(thresholds[0]),
        "qhat_1": float(thresholds[1]),
    }


def certificate_constraints(lock: Mapping[str, object]) -> ActionCertificateConstraints:
    row = lock["policy_certificate_diagnostic"]
    return ActionCertificateConstraints(
        familywise_alpha=float(row["familywise_alpha"]),
        coverage_floor=(
            None if row["coverage_floor"] is None else float(row["coverage_floor"])
        ),
        wrong_singleton_ceiling=(
            None
            if row["wrong_singleton_ceiling"] is None
            else float(row["wrong_singleton_ceiling"])
        ),
        empty_exposure_ceiling=(
            None
            if row["empty_exposure_ceiling"] is None
            else float(row["empty_exposure_ceiling"])
        ),
        critical_class=int(row["critical_class"]),
        critical_csy_floor=(
            None
            if row["critical_csy_floor"] is None
            else float(row["critical_csy_floor"])
        ),
        minimum_true_class_n=int(row["minimum_true_class_n"]),
    )


def flatten_certificate(
    endpoint: str,
    track: str,
    seed: int,
    cell: str,
    method: str,
    certificate: Mapping[str, object],
) -> dict[str, object]:
    classes = certificate["classes"]
    row: dict[str, object] = {
        "endpoint": endpoint,
        "track": track,
        "seed": seed,
        "cell": cell,
        "method": method,
        "certificate_status": certificate["status"],
        "simultaneous_alpha": certificate["simultaneous_alpha"],
        "tested_constraint_count": certificate["tested_constraint_count"],
        "selection_grid_test_count": certificate["selection_grid_test_count"],
    }
    for label in (0, 1):
        values = classes[str(label)]
        row.update(
            {
                f"class_{label}_n": values["n"],
                f"class_{label}_count_ready": values["count_ready"],
                f"class_{label}_coverage_lower": values["coverage_lower"],
                f"class_{label}_csy_lower": values["csy_lower"],
                f"class_{label}_wrong_singleton_upper": values[
                    "wrong_singleton_upper"
                ],
                f"class_{label}_empty_exposure_upper": values[
                    "empty_exposure_upper"
                ],
                f"class_{label}_passed": values["passed"],
            }
        )
    return row


def evaluate_cell(
    directory: Path, lock: Mapping[str, object]
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    endpoint, track, seed = parse_cell_name(directory.name)
    by_role, source_record = load_and_verify_cell(directory, lock)
    role_arrays = {role: arrays(by_role[role]) for role in ("conformal", "policy", "test")}
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    certificate_rows: list[dict[str, object]] = []
    constraints = certificate_constraints(lock)
    for method_row in lock["fixed_method_family"]:
        method = str(method_row["method"])
        method_role = str(method_row["role"])
        selected = configuration(method_row)
        scores: dict[str, np.ndarray] = {}
        targets: dict[str, np.ndarray] = {}
        for role, (y, probability, risk, _) in role_arrays.items():
            targets[role] = y
            scores[role] = compose_candidate_scores(
                probability, risk, None, selected
            )
        thresholds = mondrian_thresholds(
            scores["conformal"], targets["conformal"], float(lock["conformal"]["alpha"])
        )
        sets = {
            role: prediction_sets(scores[role], thresholds)
            for role in ("policy", "test")
        }
        for role in ("policy", "test"):
            metric_rows.append(
                metric_row(
                    endpoint,
                    track,
                    seed,
                    directory.name,
                    method,
                    method_role,
                    role,
                    targets[role],
                    sets[role],
                    thresholds,
                )
            )
        certificate = certify_final_sets(targets["policy"], sets["policy"], constraints)
        certificate_rows.append(
            flatten_certificate(
                endpoint, track, seed, directory.name, method, certificate
            )
        )
        test_y, _, test_risk, test_ids = role_arrays["test"]
        states = state_labels(sets["test"])
        for index, structure_id in enumerate(test_ids):
            prediction_rows.append(
                {
                    "endpoint": endpoint,
                    "track": track,
                    "seed": seed,
                    "cell": directory.name,
                    "structure_id": structure_id,
                    "target": int(test_y[index]),
                    "method": method,
                    "state": states[index],
                    "include_0": bool(sets["test"][index, 0]),
                    "include_1": bool(sets["test"][index, 1]),
                    "score_0": float(scores["test"][index, 0]),
                    "score_1": float(scores["test"][index, 1]),
                    "qhat_0": float(thresholds[0]),
                    "qhat_1": float(thresholds[1]),
                    "risk_percentile": float(test_risk[index]),
                }
            )
    return metric_rows, prediction_rows, certificate_rows, source_record


def group_mean_rows(
    rows: Sequence[Mapping[str, object]], grouping: Sequence[str]
) -> list[dict[str, object]]:
    metrics = (
        "macro_csy",
        "worst_csy",
        "class_0_coverage",
        "class_1_coverage",
        "class_0_csy",
        "class_1_csy",
        "class_0_wrong_singleton_exposure",
        "class_1_wrong_singleton_exposure",
        "ambiguity_rate",
        "empty_rate",
        "singleton_rate",
    )
    buckets: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[key] for key in grouping)].append(row)
    output = []
    for key, values in sorted(buckets.items(), key=lambda item: tuple(map(str, item[0]))):
        record = {name: value for name, value in zip(grouping, key)}
        record["cell_count"] = len(values)
        for metric in metrics:
            record[f"mean_{metric}"] = float(
                np.mean([float(row[metric]) for row in values])
            )
        output.append(record)
    return output


def bootstrap_cluster_mean_ci(
    differences: Sequence[Mapping[str, object]],
    iterations: int,
    seed: int,
    confidence_level: float,
) -> tuple[float, float]:
    if iterations < 1 or not 0.0 < confidence_level < 1.0:
        raise ValueError("invalid paired cluster-bootstrap settings")
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in differences:
        clusters[str(row["cluster"])].append(float(row["difference"]))
    names = sorted(clusters)
    if not names:
        raise ValueError("paired comparison has no clusters")
    cluster_values = [np.asarray(clusters[name], dtype=float) for name in names]
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled = rng.integers(0, len(cluster_values), size=len(cluster_values))
        values = np.concatenate([cluster_values[position] for position in sampled])
        estimates[index] = float(np.mean(values))
    tail = (1.0 - confidence_level) / 2.0
    return (
        float(np.quantile(estimates, tail)),
        float(np.quantile(estimates, 1.0 - tail)),
    )


def comparison_scopes(lock: Mapping[str, object]) -> list[tuple[str, set[str]]]:
    all_tracks = set(str(value) for value in lock["expected_source"]["tracks"])
    shift_tracks = set(str(value) for value in lock["scope"]["primary_development_tracks"])
    negative_tracks = set(str(value) for value in lock["scope"]["negative_control_tracks"])
    output = [
        ("all_tracks", all_tracks),
        ("chemical_shift_primary_scope", shift_tracks),
        ("negative_control", negative_tracks),
    ]
    output.extend((f"track:{track}", {track}) for track in sorted(all_tracks))
    return output


def paired_comparisons(
    test_metrics: Sequence[Mapping[str, object]], lock: Mapping[str, object]
) -> list[dict[str, object]]:
    primary = str(lock["primary_method"])
    methods = [str(row["method"]) for row in lock["fixed_method_family"]]
    comparators = [method for method in methods if method != primary]
    by_key = {
        (str(row["cell"]), str(row["method"])): row for row in test_metrics
    }
    bootstrap = lock["paired_cluster_bootstrap"]
    output: list[dict[str, object]] = []
    for scope_position, (scope, tracks) in enumerate(comparison_scopes(lock)):
        cells = sorted(
            {
                str(row["cell"])
                for row in test_metrics
                if str(row["track"]) in tracks
            }
        )
        for comparator_position, comparator in enumerate(comparators):
            difference_rows: list[dict[str, object]] = []
            primary_values: list[float] = []
            comparator_values: list[float] = []
            for cell in cells:
                candidate = by_key[(cell, primary)]
                baseline = by_key[(cell, comparator)]
                candidate_value = float(candidate["macro_csy"])
                baseline_value = float(baseline["macro_csy"])
                primary_values.append(candidate_value)
                comparator_values.append(baseline_value)
                difference_rows.append(
                    {
                        "cluster": f"{candidate['endpoint']}__{candidate['track']}",
                        "difference": candidate_value - baseline_value,
                    }
                )
            differences = np.asarray(
                [float(row["difference"]) for row in difference_rows], dtype=float
            )
            ci_low, ci_high = bootstrap_cluster_mean_ci(
                difference_rows,
                int(bootstrap["iterations"]),
                int(bootstrap["seed"]) + scope_position * 100 + comparator_position,
                float(bootstrap["confidence_level"]),
            )
            tolerance = 1.0e-15
            output.append(
                {
                    "scope": scope,
                    "tracks": "|".join(sorted(tracks)),
                    "candidate": primary,
                    "comparator": comparator,
                    "cell_count": len(cells),
                    "cluster_count": len({row["cluster"] for row in difference_rows}),
                    "candidate_mean_macro_csy": float(np.mean(primary_values)),
                    "comparator_mean_macro_csy": float(np.mean(comparator_values)),
                    "mean_difference": float(np.mean(differences)),
                    "median_difference": float(np.median(differences)),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "wins": int(np.sum(differences > tolerance)),
                    "ties": int(np.sum(np.abs(differences) <= tolerance)),
                    "losses": int(np.sum(differences < -tolerance)),
                    "confirmatory_interpretation_allowed": False,
                }
            )
    return output


def validate_complete_output(
    output: Path, lock: Mapping[str, object], source_manifest_sha256: str
) -> dict[str, object] | None:
    summary_path = output / "run_summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_cells = int(lock["expected_source"]["cell_count"])
    expected_methods = len(lock["fixed_method_family"])
    expected_metric_rows = expected_cells * expected_methods * 2
    required = {
        "all_role_metrics.csv": int(summary.get("metric_row_count", -1)),
        "all_test_predictions.csv": int(summary.get("prediction_row_count", -1)),
        "policy_certificates.csv": int(summary.get("certificate_row_count", -1)),
        "aggregate_metrics.csv": int(summary.get("aggregate_row_count", -1)),
        "paired_comparisons.csv": int(summary.get("comparison_row_count", -1)),
    }
    if (
        summary.get("status") != "complete_retrospective_racer_c2_extension"
        or int(summary.get("source_cell_count", -1)) != expected_cells
        or int(summary.get("method_count", -1)) != expected_methods
        or int(summary.get("metric_row_count", -1)) != expected_metric_rows
        or summary.get("source_manifest_sha256") != source_manifest_sha256
        or summary.get("confirmatory_claim_authorized") is not False
    ):
        return None
    hashes = summary.get("output_sha256", {})
    for name, expected_rows in required.items():
        path = output / name
        if not path.is_file() or hashes.get(name) != sha256_file(path):
            return None
        if expected_rows < 1 or len(read_csv(path)) != expected_rows:
            return None
    return summary


def run(lock_path: Path, source: Path, output: Path, resume: bool) -> dict[str, object]:
    lock = load_lock(lock_path)
    assert_separate_output(source, output)
    directories = source_cell_directories(source, lock)
    # Hash the immutable source inventory before any possible resume decision.
    source_inventory = [
        {
            "cell": directory.name,
            "raw_predictions_sha256": sha256_file(directory / "raw_predictions.csv"),
            "test_predictions_sha256": sha256_file(directory / "test_predictions.csv"),
            "metrics_sha256": sha256_file(directory / "metrics.csv"),
            "raw_manifest_sha256": sha256_file(directory / "raw_manifest.json"),
            "final_manifest_sha256": sha256_file(directory / "final_manifest.json"),
        }
        for directory in directories
    ]
    source_manifest_sha256 = stable_json_sha256(source_inventory)
    if output.exists():
        if not resume:
            raise FileExistsError(f"output exists; use --resume to validate it: {output}")
        complete = validate_complete_output(output, lock, source_manifest_sha256)
        if complete is None:
            raise RuntimeError("existing RACER-C2 output is incomplete or mismatched")
        print(f"RESUME VERIFIED: {output}")
        return complete

    output.mkdir(parents=True, exist_ok=False)
    all_metrics: list[dict[str, object]] = []
    all_predictions: list[dict[str, object]] = []
    all_certificates: list[dict[str, object]] = []
    verified_sources: list[dict[str, object]] = []
    for index, directory in enumerate(directories, start=1):
        metrics, predictions, certificates, source_record = evaluate_cell(directory, lock)
        all_metrics.extend(metrics)
        all_predictions.extend(predictions)
        all_certificates.extend(certificates)
        verified_sources.append(source_record)
        print(f"RACER-C2 cell {index}/{len(directories)}: {directory.name}", flush=True)

    expected_cells = int(lock["expected_source"]["cell_count"])
    expected_methods = len(lock["fixed_method_family"])
    if len(all_metrics) != expected_cells * expected_methods * 2:
        raise RuntimeError("RACER-C2 metric matrix is incomplete")
    if len(all_certificates) != expected_cells * expected_methods:
        raise RuntimeError("RACER-C2 certificate matrix is incomplete")
    test_metrics = [row for row in all_metrics if row["evaluation_role"] == "test"]
    aggregate = group_mean_rows(
        test_metrics, ("method", "method_role", "endpoint", "track")
    )
    comparisons = paired_comparisons(test_metrics, lock)

    write_csv(output / "all_role_metrics.csv", all_metrics, METRIC_FIELDS)
    write_csv(output / "all_test_predictions.csv", all_predictions, PREDICTION_FIELDS)
    certificate_fields = list(all_certificates[0])
    write_csv(output / "policy_certificates.csv", all_certificates, certificate_fields)
    aggregate_fields = list(aggregate[0])
    write_csv(output / "aggregate_metrics.csv", aggregate, aggregate_fields)
    comparison_fields = list(comparisons[0])
    write_csv(output / "paired_comparisons.csv", comparisons, comparison_fields)
    source_inventory_after = [
        {
            "cell": directory.name,
            "raw_predictions_sha256": sha256_file(directory / "raw_predictions.csv"),
            "test_predictions_sha256": sha256_file(directory / "test_predictions.csv"),
            "metrics_sha256": sha256_file(directory / "metrics.csv"),
            "raw_manifest_sha256": sha256_file(directory / "raw_manifest.json"),
            "final_manifest_sha256": sha256_file(directory / "final_manifest.json"),
        }
        for directory in directories
    ]
    if source_inventory_after != source_inventory:
        raise RuntimeError("immutable v1 source changed during RACER-C2 evaluation")
    atomic_json(
        output / "source_manifest.json",
        {
            "status": "verified_immutable_v1_source",
            "source_root": str(source.resolve()),
            "source_manifest_sha256": source_manifest_sha256,
            "inventory": source_inventory,
            "verified_cells": verified_sources,
        },
    )
    output_files = (
        "all_role_metrics.csv",
        "all_test_predictions.csv",
        "policy_certificates.csv",
        "aggregate_metrics.csv",
        "paired_comparisons.csv",
        "source_manifest.json",
    )
    certificate_counts = Counter(
        str(row["certificate_status"]) for row in all_certificates
    )
    summary = {
        "status": "complete_retrospective_racer_c2_extension",
        "algorithm": lock["algorithm"],
        "algorithm_version": lock["algorithm_version"],
        "study_role": lock["study_role"],
        "source_panel_already_known": True,
        "confirmatory_claim_authorized": False,
        "base_models_retrained": False,
        "old_methods_rerun": False,
        "v1_source_modified": False,
        "source_cell_count": expected_cells,
        "source_method_cell_count": int(
            lock["expected_source"]["source_method_cell_count"]
        ),
        "method_count": expected_methods,
        "metric_row_count": len(all_metrics),
        "policy_method_cell_count": len(all_certificates),
        "test_method_cell_count": len(test_metrics),
        "prediction_row_count": len(all_predictions),
        "certificate_row_count": len(all_certificates),
        "aggregate_row_count": len(aggregate),
        "comparison_row_count": len(comparisons),
        "policy_certificate_status_counts": dict(certificate_counts),
        "source_manifest_sha256": source_manifest_sha256,
        "lock_sha256": sha256_file(lock_path),
        "output_sha256": {
            name: sha256_file(output / name) for name in output_files
        },
        "primary_method": lock["primary_method"],
        "selected_configuration": next(
            row
            for row in lock["fixed_method_family"]
            if row["method"] == lock["primary_method"]
        ),
    }
    atomic_json(output / "run_summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(args.lock, args.source, args.output_dir, args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
