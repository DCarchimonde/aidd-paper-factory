from __future__ import annotations

"""Select a RACER-C2 score configuration using development labels only.

This command is intentionally unable to evaluate v1 test labels.  The completed
RACER-C v1 panel is a retrospective development resource for v2, not a fresh
confirmatory panel.  Prospective endpoint evaluation requires a later frozen
runner and protocol tag.
"""

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml

from core import (
    ScoreConfiguration,
    compose_candidate_scores,
    crossfit_candidate_error_scores,
    mondrian_thresholds,
    prediction_sets,
    reference_midrank_percentiles,
    select_development_configuration,
    set_metric_record,
)


P2_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = P2_ROOT / "configs" / "racer_c2" / "development_lock_v0.yaml"
DEFAULT_CELL_ROOT = P2_ROOT / "results" / "racer_c_confirmatory_v1" / "cells"
DEFAULT_OUTPUT = (
    P2_ROOT
    / "results"
    / "racer_c2_development"
    / "development_only_score_selection.json"
)

PROBABILITY_COLUMNS = (
    "ecfp_p",
    "dmpnn_p",
    "molformer_p",
    "stack_p",
    "unrestricted_p",
)
RELIABILITY_COLUMNS = (
    "disagreement",
    "ecfp_distance",
    "local_oof_brier_loss",
    "bri",
)
REQUIRED_COLUMNS = {
    "structure_id",
    "role",
    "target",
    "meta_fold",
    "risk_percentile",
    *PROBABILITY_COLUMNS,
    *RELIABILITY_COLUMNS,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
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


def read_development_rows(path: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Materialize only dev rows and count all roles without parsing outer labels."""

    development: list[dict[str, str]] = []
    role_counts: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or ()))
            raise ValueError(f"{path} is missing columns: {missing}")
        for row in reader:
            role = str(row["role"])
            role_counts[role] = role_counts.get(role, 0) + 1
            if role != "dev":
                continue
            if not row["target"] or not row["meta_fold"]:
                raise ValueError(f"development label/fold is missing in {path}")
            development.append(row)
    if set(role_counts) != {"dev", "policy", "conformal", "test"}:
        raise ValueError(f"{path} does not contain all four outer roles")
    identifiers = [row["structure_id"] for row in development]
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError(f"development structure IDs are blank or duplicated in {path}")
    return development, role_counts


def arrays(rows: Sequence[Mapping[str, str]]) -> dict[str, np.ndarray]:
    targets = np.asarray([int(row["target"]) for row in rows], dtype=np.int8)
    folds = np.asarray([int(row["meta_fold"]) for row in rows], dtype=np.int8)
    probabilities = np.column_stack(
        [np.asarray([float(row[column]) for row in rows]) for column in PROBABILITY_COLUMNS]
    )
    reliability = np.column_stack(
        [np.asarray([float(row[column]) for row in rows]) for column in RELIABILITY_COLUMNS]
    )
    risk = np.asarray([float(row["risk_percentile"]) for row in rows], dtype=float)
    if set(np.unique(targets)) != {0, 1} or len(set(folds.tolist())) < 2:
        raise ValueError("development rows require both classes and at least two folds")
    if not np.isfinite(probabilities).all() or not np.isfinite(reliability).all():
        raise ValueError("development numerical features are incomplete")
    return {
        "targets": targets,
        "folds": folds,
        "probabilities": probabilities,
        "reliability": reliability,
        "risk": risk,
    }


def leave_fold_sets(
    stack_probability: np.ndarray,
    risk_percentile: np.ndarray,
    counterfactual_scores: np.ndarray | None,
    targets: np.ndarray,
    folds: np.ndarray,
    configuration: ScoreConfiguration,
    alpha: float,
) -> np.ndarray:
    output = np.zeros((len(targets), 2), dtype=bool)
    for fold in sorted(set(folds.tolist())):
        fit = folds != fold
        query = folds == fold
        if counterfactual_scores is None:
            fit_percentiles = None
            query_percentiles = None
        else:
            fit_percentiles = reference_midrank_percentiles(
                counterfactual_scores[fit], counterfactual_scores[fit]
            )
            query_percentiles = reference_midrank_percentiles(
                counterfactual_scores[fit], counterfactual_scores[query]
            )
        fit_scores = compose_candidate_scores(
            stack_probability[fit], risk_percentile[fit], fit_percentiles, configuration
        )
        query_scores = compose_candidate_scores(
            stack_probability[query],
            risk_percentile[query],
            query_percentiles,
            configuration,
        )
        thresholds = mondrian_thresholds(fit_scores, targets[fit], alpha)
        output[query] = prediction_sets(query_scores, thresholds)
    return output


def evaluate_cell(
    directory: Path,
    configurations: Sequence[ScoreConfiguration],
    alpha: float,
    training_seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    table_path = directory / "raw_predictions.csv"
    development, role_counts = read_development_rows(table_path)
    values = arrays(development)
    counterfactual = None
    if any(configuration.counterfactual_blend > 0.0 for configuration in configurations):
        counterfactual, external = crossfit_candidate_error_scores(
            values["probabilities"],
            values["reliability"],
            values["targets"],
            values["folds"],
            training_seed,
        )
        if external is not None:
            raise RuntimeError(
                "development-only audit unexpectedly produced external scores"
            )
    baseline_configuration = ScoreConfiguration(1.0, 0.0, 0.0)
    baseline_sets = leave_fold_sets(
        values["probabilities"][:, 3],
        values["risk"],
        counterfactual,
        values["targets"],
        values["folds"],
        baseline_configuration,
        alpha,
    )
    baseline = set_metric_record(values["targets"], baseline_sets)
    evaluations: list[dict[str, object]] = []
    for configuration in configurations:
        candidate_sets = leave_fold_sets(
            values["probabilities"][:, 3],
            values["risk"],
            counterfactual,
            values["targets"],
            values["folds"],
            configuration,
            alpha,
        )
        candidate = set_metric_record(values["targets"], candidate_sets)
        evaluations.append(
            {
                "cell": directory.name,
                "t_max": configuration.t_max,
                "gamma_0": configuration.gamma_0,
                "gamma_1": configuration.gamma_1,
                "counterfactual_blend": configuration.counterfactual_blend,
                "baseline_macro_csy": baseline["macro_csy"],
                "baseline_class_0_coverage": baseline["class_0_coverage"],
                "baseline_class_1_coverage": baseline["class_1_coverage"],
                "candidate_macro_csy": candidate["macro_csy"],
                "candidate_class_0_coverage": candidate["class_0_coverage"],
                "candidate_class_1_coverage": candidate["class_1_coverage"],
                "candidate_class_0_csy": candidate["class_0_csy"],
                "candidate_class_1_csy": candidate["class_1_csy"],
            }
        )
    source = {
        "cell": directory.name,
        "raw_predictions_sha256": sha256_file(table_path),
        "development_n": len(development),
        "role_counts": role_counts,
        "labels_materialized": ["dev"],
    }
    return evaluations, source


def cell_track(directory: Path) -> str:
    parts = directory.name.split("__")
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"unrecognized RACER-C cell name: {directory.name}")
    return parts[1]


def load_config(path: Path) -> dict[str, object]:
    row = yaml.safe_load(path.read_text(encoding="utf-8"))
    if row.get("lock_status") != "development_only_not_frozen":
        raise RuntimeError("RACER-C2 development audit requires an unfrozen development lock")
    if row.get("known_v1_test_reuse") != "prohibited_as_confirmatory":
        raise RuntimeError("RACER-C2 config does not preserve the v1 test firewall")
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cell-root", type=Path, default=DEFAULT_CELL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-cells", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if not args.cell_root.is_dir():
        raise FileNotFoundError(args.cell_root)
    directories = sorted(
        path
        for path in args.cell_root.iterdir()
        if path.is_dir() and (path / "raw_predictions.csv").is_file()
    )
    if len(directories) != args.expected_cells:
        raise RuntimeError(
            f"development cell count mismatch: {len(directories)}/{args.expected_cells}"
        )
    score = config["score_family"]
    selection_tracks = tuple(str(track) for track in score["selection_tracks"])
    if not selection_tracks or len(selection_tracks) != len(set(selection_tracks)):
        raise ValueError("selection_tracks must be nonempty and unique")
    discovered_track_counts: dict[str, int] = {}
    for directory in directories:
        track = cell_track(directory)
        discovered_track_counts[track] = discovered_track_counts.get(track, 0) + 1
    selection_directories = [
        directory for directory in directories if cell_track(directory) in selection_tracks
    ]
    expected_selection_cells = int(score["expected_selection_cells"])
    if len(selection_directories) != expected_selection_cells:
        raise RuntimeError(
            "development selection cell count mismatch: "
            f"{len(selection_directories)}/{expected_selection_cells}"
        )
    configurations = [
        ScoreConfiguration(
            t_max=float(t_max),
            gamma_0=float(gamma_0),
            gamma_1=float(gamma_1),
            counterfactual_blend=float(blend),
        )
        for t_max in score["t_max_candidates"]
        for gamma_0 in score["gamma_0_candidates"]
        for gamma_1 in score["gamma_1_candidates"]
        for blend in score["counterfactual_blend_candidates"]
    ]
    evaluations: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    for index, directory in enumerate(selection_directories, start=1):
        rows, source = evaluate_cell(
            directory,
            configurations,
            float(config["conformal"]["alpha"]),
            int(config["training_seed"]),
        )
        evaluations.extend(rows)
        sources.append(source)
        print(
            "RACER-C2 development selection cell "
            f"{index}/{len(selection_directories)}: {directory.name}"
        )
    selected, summary = select_development_configuration(
        evaluations,
        float(score["mean_coverage_shortfall_margin"]),
        float(score["minimum_cell_class_coverage"]),
    )
    record: dict[str, object] = {
        "status": "complete_development_only_score_selection",
        "algorithm": "RACER-C2",
        "algorithm_version": str(config["algorithm_version"]),
        "selected_configuration": {
            "t_max": selected.t_max,
            "gamma_0": selected.gamma_0,
            "gamma_1": selected.gamma_1,
            "counterfactual_blend": selected.counterfactual_blend,
        },
        "configuration_summary": summary,
        "discovered_cell_count": len(directories),
        "selection_cell_count": len(selection_directories),
        "selection_tracks": list(selection_tracks),
        "negative_control_tracks": list(score["negative_control_tracks"]),
        "discovered_track_counts": discovered_track_counts,
        "evaluation_row_count": len(evaluations),
        "allowed_label_roles": ["dev"],
        "policy_labels_used": False,
        "conformal_labels_used": False,
        "test_labels_used": False,
        "scientific_test_predictions_generated": False,
        "known_v1_test_reuse": "prohibited_as_confirmatory",
        "source_manifest_sha256": stable_json_sha256(sources),
        "source_manifest": sources,
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "freeze_authorized": False,
        "next_gate": "prospective endpoint, precision, comparator, and protocol review",
    }
    atomic_json(args.output, record)
    print(json.dumps(record["selected_configuration"], sort_keys=True))
    print(f"DEVELOPMENT ONLY: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
