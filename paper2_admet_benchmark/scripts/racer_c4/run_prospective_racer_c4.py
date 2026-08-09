from __future__ import annotations

"""One-pass, fail-closed independent validation for RACER-C4/TAME.

The public leaderboard batch is architecture development only.  Fresh seeds
211--215 are used for the final EPA batch.  Final labels are neither downloaded
nor parsed until the development gate passes and every final prediction has
been written and hashed in a promotion record.
"""

import argparse
import csv
import json
import math
import subprocess
import sys
import traceback
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml
from scipy.special import expit, logit
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB


ROOT = Path(__file__).resolve().parents[3]
P2 = ROOT / "paper2_admet_benchmark"
RACER_C = P2 / "scripts" / "racer_c"
sys.path.insert(0, str(RACER_C))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from racer_c4_core import (  # noqa: E402
    audit_density_ratio,
    binary_nonconformity,
    clipped,
    cross_fitted_density_ratio,
    ordinary_mondrian_sets,
    set_metrics,
    transport_envelope,
    weighted_mondrian_sets,
)
from racer_c4_io import (  # noqa: E402
    ENDPOINT_PROPERTIES,
    acquire_final_label_bytes,
    acquire_unlabeled_sources,
    atomic_csv,
    atomic_json,
    fingerprints,
    open_final_labels_after_promotion,
    physchem_features,
    read_clean_endpoint,
    read_development_sdf,
    read_structure_table,
    sha256_file,
    stable_sha256,
)


DEFAULT_LOCK = P2 / "configs" / "racer_c4" / "prospective_lock_v1.yaml"
METHOD_BASELINE = "ordinary_mondrian_global_stack"
METHOD_WEIGHT_ECFP = "density_weighted_mondrian_ecfp"
METHOD_WEIGHT_PHYSCHEM = "density_weighted_mondrian_physchem"
METHOD_WEIGHT_SCORE = "density_weighted_mondrian_score_view"
METHOD_EQUAL = "ordinary_mondrian_equal_logit"
METHOD_TAME = "RACER-C4_TAME"
METHODS = (
    METHOD_BASELINE,
    METHOD_WEIGHT_ECFP,
    METHOD_WEIGHT_PHYSCHEM,
    METHOD_WEIGHT_SCORE,
    METHOD_EQUAL,
    METHOD_TAME,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def git_state() -> dict[str, object]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"head": head, "tracked_dirty_paths": dirty}


def validate_lock(lock: Mapping[str, object]) -> None:
    if lock.get("lock_status") != "candidate_frozen_before_final_epa_label_open":
        raise RuntimeError("unexpected RACER-C4 lock status")
    endpoints = list(lock["endpoint_order"])
    if endpoints != list(ENDPOINT_PROPERTIES):
        raise RuntimeError("endpoint order differs from the official Tox21 property map")
    primary = list(lock["primary_endpoints"])
    if len(primary) != 6 or not set(primary).issubset(endpoints):
        raise RuntimeError("the frozen six-endpoint primary panel is invalid")
    if list(lock["roles"]["prospective_seeds"]) != [211, 212, 213, 214, 215]:
        raise RuntimeError("fresh prospective seed contract changed")
    if list(lock["envelope"]["protected_labels"]) != [0]:
        raise RuntimeError("protected-label development decision changed")


def ensure_training_processed(
    sources: Mapping[str, Path], processed_dir: Path, manifest_dir: Path, endpoints: Sequence[str]
) -> None:
    ready = all((processed_dir / f"{endpoint}_clean.csv").is_file() for endpoint in endpoints)
    if ready:
        return
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(RACER_C / "prepare_tox21_challenge.py"),
        "--archive",
        str(sources["training_archive"]),
        "--sdf",
        str(sources["training_sdf"]),
        "--processed-dir",
        str(processed_dir),
        "--manifest-dir",
        str(manifest_dir),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    missing = [
        endpoint
        for endpoint in endpoints
        if not (processed_dir / f"{endpoint}_clean.csv").is_file()
    ]
    if missing:
        raise RuntimeError(f"deterministic training preparation missed {missing}")


def make_models(lock: Mapping[str, object], seed: int) -> tuple[object, ...]:
    config = {row["name"]: row for row in lock["base_predictor"]["components"]}
    logistic = config["logistic"]
    random_forest = config["random_forest"]
    extra_trees = config["extra_trees"]
    naive_bayes = config["bernoulli_nb"]
    return (
        LogisticRegression(
            C=float(logistic["C"]),
            solver=str(logistic["solver"]),
            max_iter=int(logistic["max_iter"]),
            random_state=int(seed),
        ),
        RandomForestClassifier(
            n_estimators=int(random_forest["n_estimators"]),
            max_features=str(random_forest["max_features"]),
            min_samples_leaf=int(random_forest["min_samples_leaf"]),
            n_jobs=int(random_forest["n_jobs"]),
            random_state=int(seed) + 1,
        ),
        ExtraTreesClassifier(
            n_estimators=int(extra_trees["n_estimators"]),
            max_features=str(extra_trees["max_features"]),
            min_samples_leaf=int(extra_trees["min_samples_leaf"]),
            n_jobs=int(extra_trees["n_jobs"]),
            random_state=int(seed) + 2,
        ),
        BernoulliNB(alpha=float(naive_bayes["alpha"])),
    )


def component_probabilities(models: Sequence[object], features: np.ndarray) -> np.ndarray:
    return clipped(np.column_stack([model.predict_proba(features)[:, 1] for model in models]))


def equal_logit_probability(probabilities: np.ndarray) -> np.ndarray:
    return clipped(expit(np.mean(logit(clipped(probabilities)), axis=1)))


def split_internal_roles(target: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(target))
    fit, held = train_test_split(
        indices, test_size=0.40, stratify=target, random_state=int(seed)
    )
    router, conformal = train_test_split(
        held, test_size=0.50, stratify=target[held], random_state=int(seed) + 100
    )
    for name, values in (("fit", fit), ("router", router), ("conformal", conformal)):
        if set(np.unique(target[values])) != {0, 1}:
            raise RuntimeError(f"internal {name} role is not binary for seed {seed}")
    return fit, router, conformal


def _full_sets(n: int) -> np.ndarray:
    return np.ones((int(n), 2), dtype=bool)


def _expand_eligible_sets(values: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    output = _full_sets(len(eligible))
    output[eligible] = np.asarray(values, dtype=bool)
    return output


def run_cell(
    endpoint: str,
    phase: str,
    seed: int,
    training_rows: Sequence[Mapping[str, str]],
    training_target: np.ndarray,
    training_ecfp: np.ndarray,
    training_physchem: np.ndarray,
    target_rows: Sequence[Mapping[str, str]],
    target_ecfp: np.ndarray,
    target_physchem: np.ndarray,
    lock: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    fit, router, conformal = split_internal_roles(training_target, seed)
    models = make_models(lock, seed)
    for model in models:
        model.fit(training_ecfp[fit], training_target[fit])
    router_probability = component_probabilities(models, training_ecfp[router])
    conformal_probability = component_probabilities(models, training_ecfp[conformal])
    target_probability = component_probabilities(models, target_ecfp)
    router_lock = lock["base_predictor"]["router"]
    stacker = LogisticRegression(
        C=float(router_lock["C"]),
        solver=str(router_lock["solver"]),
        max_iter=int(router_lock["max_iter"]),
        random_state=int(seed),
    )
    stacker.fit(logit(router_probability), training_target[router])
    conformal_stack = clipped(stacker.predict_proba(logit(conformal_probability))[:, 1])
    target_stack = clipped(stacker.predict_proba(logit(target_probability))[:, 1])
    conformal_scores = binary_nonconformity(conformal_stack)
    target_scores = binary_nonconformity(target_stack)
    alpha = {int(key): float(value) for key, value in lock["conformal"]["alpha_by_class"].items()}

    training_ids = {str(row["structure_id"]) for row in training_rows}
    overlap = np.asarray(
        [str(row["structure_id"]) in training_ids for row in target_rows], dtype=bool
    )
    eligible = ~overlap
    if int(eligible.sum()) < int(lock["transport"]["audit"]["minimum_target_n"]):
        raise RuntimeError(f"too few non-overlapping target structures for {endpoint}")

    ordinary_eligible = ordinary_mondrian_sets(
        conformal_scores, training_target[conformal], target_scores[eligible], alpha
    )
    equal_conformal_scores = binary_nonconformity(equal_logit_probability(conformal_probability))
    equal_target_scores = binary_nonconformity(equal_logit_probability(target_probability))
    equal_eligible = ordinary_mondrian_sets(
        equal_conformal_scores,
        training_target[conformal],
        equal_target_scores[eligible],
        alpha,
    )

    density = lock["transport"]["domain_estimator"]
    audit_lock = lock["transport"]["audit"]
    ratio_arguments = {
        "folds": int(density["folds"]),
        "c_value": float(density["C"]),
        "weight_min": float(density["weight_min"]),
        "weight_max": float(density["weight_max"]),
    }
    audit_arguments = {
        "minimum_source_n": int(audit_lock["minimum_source_n"]),
        "minimum_target_n": int(audit_lock["minimum_target_n"]),
        "minimum_class_n": int(audit_lock["minimum_class_n"]),
        "minimum_ess_fraction": float(audit_lock["minimum_ess_fraction"]),
        "maximum_clip_fraction": float(audit_lock["maximum_clip_fraction"]),
        "maximum_domain_auc": float(audit_lock["maximum_domain_auc"]),
        "maximum_gap_ratio": float(audit_lock["maximum_gap_ratio"]),
    }
    score_source_view = np.column_stack(
        [logit(conformal_probability), logit(conformal_stack)]
    )
    score_target_view = np.column_stack(
        [logit(target_probability[eligible]), logit(target_stack[eligible])]
    )
    view_inputs = (
        ("ecfp_bits", training_ecfp[conformal], target_ecfp[eligible], seed + 20000),
        (
            "physchem_descriptors",
            training_physchem[conformal],
            target_physchem[eligible],
            seed + 25000,
        ),
        (
            "component_and_stack_logits",
            score_source_view,
            score_target_view,
            seed + 30000,
        ),
    )
    weighted_by_view: dict[str, np.ndarray] = {}
    audit_by_view = {}
    audit_rows: list[dict[str, object]] = []
    for view, source_features, target_features, view_seed in view_inputs:
        ratio = cross_fitted_density_ratio(
            source_features,
            target_features,
            training_target[conformal],
            seed=view_seed,
            **ratio_arguments,
        )
        audit = audit_density_ratio(
            view,
            ratio,
            training_target[conformal],
            int(eligible.sum()),
            **audit_arguments,
        )
        weighted_by_view[view] = weighted_mondrian_sets(
                conformal_scores,
                training_target[conformal],
                target_scores[eligible],
                ratio.source_weights,
                ratio.target_weights,
                alpha,
            )
        audit_by_view[view] = audit
        audit_rows.append(
            {
                "phase": phase,
                "endpoint": endpoint,
                "seed": seed,
                **asdict(audit),
            }
        )
    envelope_lock = lock["envelope"]
    decision = transport_envelope(
        ordinary_eligible,
        [weighted_by_view[view] for view in lock["transport"]["views"]],
        [audit_by_view[view] for view in lock["transport"]["views"]],
        protected_labels=list(envelope_lock["protected_labels"]),
        quorum=str(envelope_lock["quorum"]),
        minimum_active_views=int(envelope_lock["minimum_active_views"]),
    )
    method_sets = {
        METHOD_BASELINE: _expand_eligible_sets(ordinary_eligible, eligible),
        METHOD_WEIGHT_ECFP: _expand_eligible_sets(weighted_by_view["ecfp_bits"], eligible),
        METHOD_WEIGHT_PHYSCHEM: _expand_eligible_sets(weighted_by_view["physchem_descriptors"], eligible),
        METHOD_WEIGHT_SCORE: _expand_eligible_sets(weighted_by_view["component_and_stack_logits"], eligible),
        METHOD_EQUAL: _expand_eligible_sets(equal_eligible, eligible),
        METHOD_TAME: _expand_eligible_sets(decision.membership, eligible),
    }
    rows: list[dict[str, object]] = []
    active_views = ";".join(decision.active_views)
    for method, membership in method_sets.items():
        for position, target_row in enumerate(target_rows):
            rows.append(
                {
                    "phase": phase,
                    "endpoint": endpoint,
                    "seed": seed,
                    "method": method,
                    "sample_id": str(target_row["sample_id"]),
                    "structure_id": str(target_row["structure_id"]),
                    "excluded_structure_overlap": int(overlap[position]),
                    "excluded_structure_invalid": 0,
                    "probability_1": float(target_stack[position]),
                    "include_0": int(membership[position, 0]),
                    "include_1": int(membership[position, 1]),
                    "active_transport_views": active_views if method == METHOD_TAME else "",
                    "failed_closed": int(decision.failed_closed) if method == METHOD_TAME else 0,
                }
            )
    return rows, audit_rows


def labels_from_development(
    rows: Sequence[Mapping[str, str]], labels: Mapping[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for position, row in enumerate(rows):
        output[str(row["sample_id"])] = {
            endpoint: float(labels[endpoint][position]) for endpoint in labels
        }
    return output


def evaluate_predictions(
    prediction_rows: Sequence[Mapping[str, object]],
    labels: Mapping[str, Mapping[str, float]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in prediction_rows:
        if int(row["excluded_structure_overlap"]) or int(row["excluded_structure_invalid"]):
            continue
        grouped[(str(row["endpoint"]), int(row["seed"]), str(row["method"]))].append(row)
    output: list[dict[str, object]] = []
    for (endpoint, seed, method), rows in sorted(grouped.items()):
        observed_rows = []
        targets = []
        for row in rows:
            value = float(labels[str(row["sample_id"])][endpoint])
            if math.isfinite(value):
                observed_rows.append(row)
                targets.append(int(value))
        if not observed_rows or set(targets) != {0, 1}:
            continue
        membership = np.asarray(
            [[int(row["include_0"]), int(row["include_1"])] for row in observed_rows],
            dtype=bool,
        )
        output.append(
            {
                "endpoint": endpoint,
                "seed": seed,
                "method": method,
                **set_metrics(np.asarray(targets, dtype=np.int8), membership),
            }
        )
    return output


def paired_rows(metrics: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    baseline = {
        (str(row["endpoint"]), int(row["seed"])): row
        for row in metrics
        if row["method"] == METHOD_BASELINE
    }
    output: list[dict[str, object]] = []
    delta_fields = (
        "class_0_coverage",
        "class_1_coverage",
        "minimum_class_coverage",
        "macro_csy",
        "class_0_wrong_singleton_exposure",
        "class_1_wrong_singleton_exposure",
        "ambiguous_rate",
        "empty_rate",
    )
    for row in metrics:
        key = (str(row["endpoint"]), int(row["seed"]))
        if key not in baseline:
            raise RuntimeError(f"missing baseline metric for {key}")
        paired = dict(row)
        for field in delta_fields:
            paired[f"{field}_delta"] = float(row[field]) - float(baseline[key][field])
        output.append(paired)
    return output


def summarize_metrics(metrics: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in metrics:
        grouped[str(row["method"])].append(row)
    fields = (
        "macro_csy",
        "minimum_class_coverage",
        "class_0_coverage",
        "class_1_coverage",
        "class_0_wrong_singleton_exposure",
        "class_1_wrong_singleton_exposure",
        "ambiguous_rate",
        "empty_rate",
    )
    output: list[dict[str, object]] = []
    for method, rows in sorted(grouped.items()):
        record: dict[str, object] = {"method": method, "cell_count": len(rows)}
        for field in fields:
            record[f"mean_{field}"] = float(np.nanmean([float(row[field]) for row in rows]))
        output.append(record)
    return output


def structural_invariants(predictions: Sequence[Mapping[str, object]]) -> dict[str, int]:
    by_key = {
        (
            str(row["endpoint"]),
            int(row["seed"]),
            str(row["sample_id"]),
            str(row["method"]),
        ): row
        for row in predictions
        if not int(row["excluded_structure_overlap"])
        and not int(row["excluded_structure_invalid"])
    }
    cells = sorted({(key[0], key[1], key[2]) for key in by_key})
    inclusion = 0
    empty = 0
    new_singleton = 0
    for endpoint, seed, sample_id in cells:
        baseline = by_key[(endpoint, seed, sample_id, METHOD_BASELINE)]
        tame = by_key[(endpoint, seed, sample_id, METHOD_TAME)]
        base = np.asarray([int(baseline["include_0"]), int(baseline["include_1"])], dtype=bool)
        candidate = np.asarray([int(tame["include_0"]), int(tame["include_1"])], dtype=bool)
        inclusion += int(np.any(base & ~candidate))
        empty += int(not candidate.any())
        new_singleton += int(candidate.sum() == 1 and base.sum() != 1)
    return {
        "baseline_inclusion_violations": inclusion,
        "tame_empty_sets": empty,
        "new_singletons": new_singleton,
    }


def development_gate(
    predictions: Sequence[Mapping[str, object]],
    audits: Sequence[Mapping[str, object]],
    paired: Sequence[Mapping[str, object]],
    lock: Mapping[str, object],
) -> dict[str, object]:
    primary = set(lock["primary_endpoints"])
    candidate = [
        row for row in paired if row["method"] == METHOD_TAME and row["endpoint"] in primary
    ]
    structural = structural_invariants(
        [row for row in predictions if row["endpoint"] in primary]
    )
    active_by_cell: dict[tuple[str, int], int] = defaultdict(int)
    required_views = set(lock["transport"]["views"])
    for row in audits:
        if (
            row["endpoint"] in primary
            and row["view"] in required_views
            and bool(row["active"])
        ):
            active_by_cell[(str(row["endpoint"]), int(row["seed"]))] += 1
    two_view_fraction = float(
        np.mean([active_by_cell[(endpoint, seed)] >= 2 for endpoint in primary for seed in lock["roles"]["development_seeds"]])
    )
    means = {
        field: float(np.mean([float(row[field]) for row in candidate]))
        for field in (
            "class_0_coverage_delta",
            "class_1_coverage_delta",
            "macro_csy_delta",
            "class_0_wrong_singleton_exposure_delta",
            "class_1_wrong_singleton_exposure_delta",
        )
    }
    gate_lock = lock["promotion_gate"]
    checks = {
        "primary_cell_count": len(candidate) == int(gate_lock["primary_cell_count"]),
        "baseline_inclusion": structural["baseline_inclusion_violations"] == 0,
        "no_empty_sets": structural["tame_empty_sets"] == 0,
        "no_new_singletons": structural["new_singletons"] == 0,
        "coverage_nonnegative": min(means["class_0_coverage_delta"], means["class_1_coverage_delta"]) >= -1.0e-12,
        "wrong_singleton_nonpositive": max(
            means["class_0_wrong_singleton_exposure_delta"],
            means["class_1_wrong_singleton_exposure_delta"],
        ) <= 1.0e-12,
        "macro_csy_noninferiority": means["macro_csy_delta"] >= float(gate_lock["macro_csy_noninferiority_margin"]),
        "two_view_activity": two_view_fraction >= float(gate_lock["minimum_two_view_active_cell_fraction"]),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "passed": all(checks.values()),
        "checks": checks,
        "structural_invariants": structural,
        "two_view_active_cell_fraction": two_view_fraction,
        "mean_deltas": means,
        "final_labels_downloaded_or_parsed": False,
    }


def hierarchical_bootstrap_primary(
    prediction_rows: Sequence[Mapping[str, object]],
    labels: Mapping[str, Mapping[str, float]],
    primary_endpoints: Sequence[str],
    seeds: Sequence[int],
    *,
    repetitions: int = 2000,
    seed: int = 44021,
) -> dict[str, object]:
    table: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray]] = {}
    for endpoint in primary_endpoints:
        for run_seed in seeds:
            for method in (METHOD_BASELINE, METHOD_TAME):
                selected = [
                    row
                    for row in prediction_rows
                    if row["endpoint"] == endpoint
                    and int(row["seed"]) == int(run_seed)
                    and row["method"] == method
                    and not int(row["excluded_structure_overlap"])
                    and not int(row["excluded_structure_invalid"])
                    and math.isfinite(float(labels[str(row["sample_id"])][endpoint]))
                ]
                target = np.asarray(
                    [int(labels[str(row["sample_id"])][endpoint]) for row in selected],
                    dtype=np.int8,
                )
                member = np.asarray(
                    [[int(row["include_0"]), int(row["include_1"])] for row in selected],
                    dtype=bool,
                )
                table[(endpoint, int(run_seed), method)] = (target, member)
    def estimate(indices_by_endpoint: Mapping[str, np.ndarray] | None = None, sampled_endpoints: Sequence[str] | None = None) -> float:
        endpoint_values = []
        use_endpoints = list(primary_endpoints) if sampled_endpoints is None else list(sampled_endpoints)
        for endpoint in use_endpoints:
            seed_values = []
            for run_seed in seeds:
                baseline_target, baseline_set = table[(endpoint, int(run_seed), METHOD_BASELINE)]
                tame_target, tame_set = table[(endpoint, int(run_seed), METHOD_TAME)]
                if not np.array_equal(baseline_target, tame_target):
                    raise RuntimeError("bootstrap method target alignment failed")
                indices = (
                    np.arange(len(baseline_target))
                    if indices_by_endpoint is None
                    else indices_by_endpoint[endpoint]
                )
                base = set_metrics(baseline_target[indices], baseline_set[indices])
                tame = set_metrics(tame_target[indices], tame_set[indices])
                seed_values.append(
                    float(tame["minimum_class_coverage"])
                    - float(base["minimum_class_coverage"])
                )
            endpoint_values.append(float(np.mean(seed_values)))
        return float(np.mean(endpoint_values))
    point = estimate()
    rng = np.random.default_rng(seed)
    draws = np.empty(int(repetitions), dtype=float)
    for repetition in range(int(repetitions)):
        sampled_endpoints = rng.choice(primary_endpoints, size=len(primary_endpoints), replace=True).tolist()
        indices = {}
        # Preserve the bootstrap draw's first-occurrence order while avoiding
        # duplicate within-endpoint resamples.  A set is not valid here:
        # PYTHONHASHSEED can change its traversal order and therefore assign
        # different portions of the fixed RNG stream to different endpoints.
        for endpoint in dict.fromkeys(sampled_endpoints):
            target = table[(endpoint, int(seeds[0]), METHOD_BASELINE)][0]
            # Preserve each endpoint's class counts so a rare-class bootstrap
            # draw cannot silently turn the class-balanced estimand marginal.
            indices[endpoint] = np.concatenate(
                [
                    rng.choice(
                        np.flatnonzero(target == label),
                        size=int(np.sum(target == label)),
                        replace=True,
                    )
                    for label in (0, 1)
                ]
            )
        draws[repetition] = estimate(indices, sampled_endpoints)
    return {
        "estimand": "endpoint_seed_equal_mean_minimum_class_coverage_delta",
        "estimate": point,
        "bootstrap_repetitions": int(repetitions),
        "bootstrap_seed": int(seed),
        "percentile_95_interval": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
    }


def run_panel(
    phase: str,
    endpoints: Sequence[str],
    seeds: Sequence[int],
    processed_dir: Path,
    target_rows: Sequence[Mapping[str, str]],
    lock: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    model_target_rows = [
        row for row in target_rows if str(row.get("structure_status", "pass")) == "pass"
    ]
    invalid_target_rows = [
        row for row in target_rows if str(row.get("structure_status", "pass")) != "pass"
    ]
    if not model_target_rows:
        raise RuntimeError("every target structure failed standardization")
    target_ecfp = fingerprints(
        model_target_rows, int(lock["base_predictor"]["fingerprint"]["n_bits"])
    )
    target_physchem = physchem_features(model_target_rows)
    prediction_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    total = len(endpoints) * len(seeds)
    position = 0
    for endpoint in endpoints:
        training_rows, training_target = read_clean_endpoint(
            processed_dir / f"{endpoint}_clean.csv"
        )
        training_ecfp = fingerprints(
            training_rows, int(lock["base_predictor"]["fingerprint"]["n_bits"])
        )
        training_physchem = physchem_features(training_rows)
        for run_seed in seeds:
            position += 1
            print(
                f"[{phase}] {position}/{total} endpoint={endpoint} seed={run_seed}",
                flush=True,
            )
            cell_rows, cell_audits = run_cell(
                endpoint,
                phase,
                int(run_seed),
                training_rows,
                training_target,
                training_ecfp,
                training_physchem,
                model_target_rows,
                target_ecfp,
                target_physchem,
                lock,
            )
            for invalid in invalid_target_rows:
                for method in METHODS:
                    cell_rows.append(
                        {
                            "phase": phase,
                            "endpoint": endpoint,
                            "seed": int(run_seed),
                            "method": method,
                            "sample_id": str(invalid["sample_id"]),
                            "structure_id": str(invalid["structure_id"]),
                            "excluded_structure_overlap": 0,
                            "excluded_structure_invalid": 1,
                            "probability_1": math.nan,
                            "include_0": 1,
                            "include_1": 1,
                            "active_transport_views": "",
                            "failed_closed": int(method == METHOD_TAME),
                        }
                    )
            prediction_rows.extend(cell_rows)
            audit_rows.extend(cell_audits)
    return prediction_rows, audit_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RACER-C4 independent validation")
    parser.add_argument("--mode", choices=("validate", "development", "full"), default="full")
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--source-root", type=Path, default=ROOT / ".local" / "racer_c4_sources")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / ".local" / "racer_c4_processed")
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / ".local" / "racer_c4_manifests")
    parser.add_argument("--output", type=Path, default=ROOT / ".local" / "racer_c4_results")
    parser.add_argument("--scope", choices=("primary", "all"), default="all")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        lock = yaml.safe_load(args.lock.read_text(encoding="utf-8"))
        validate_lock(lock)
        state = git_state()
        if state["tracked_dirty_paths"] and not args.allow_dirty:
            raise RuntimeError(
                "tracked worktree changes are prohibited for the sealed run: "
                + ", ".join(state["tracked_dirty_paths"])
            )
        validation = {
            "status": "pass",
            "lock_sha256": sha256_file(args.lock),
            "git": state,
            "final_labels_opened": False,
        }
        atomic_json(args.output / "validation.json", validation)
        if args.mode == "validate":
            print(json.dumps(validation, indent=2, sort_keys=True))
            return 0

        endpoints = (
            list(lock["primary_endpoints"])
            if args.scope == "primary"
            else list(lock["endpoint_order"])
        )
        sources = acquire_unlabeled_sources(lock, args.source_root)
        ensure_training_processed(
            sources, args.processed_dir, args.manifest_dir, endpoints
        )
        development_rows, development_labels_array = read_development_sdf(
            sources["development_sdf"]
        )
        expected_development = int(
            lock["data_sources"]["leaderboard_development_labels"]["expected_structure_count"]
        )
        if len(development_rows) != expected_development:
            raise RuntimeError(
                f"development structure count mismatch: {len(development_rows)}"
            )
        development_predictions, development_audits = run_panel(
            "development",
            endpoints,
            list(lock["roles"]["development_seeds"]),
            args.processed_dir,
            development_rows,
            lock,
        )
        development_labels = labels_from_development(
            development_rows, development_labels_array
        )
        development_metrics = evaluate_predictions(
            development_predictions, development_labels
        )
        development_paired = paired_rows(development_metrics)
        gate = development_gate(
            development_predictions, development_audits, development_paired, lock
        )
        atomic_csv(args.output / "development_metrics.csv", development_metrics)
        atomic_csv(args.output / "development_paired.csv", development_paired)
        atomic_csv(args.output / "development_summary.csv", summarize_metrics(development_metrics))
        atomic_csv(args.output / "development_transport_audit.csv", development_audits)
        atomic_json(args.output / "development_gate.json", gate)
        if args.mode == "development":
            print(json.dumps(gate, indent=2, sort_keys=True))
            return 0 if gate["passed"] else 3
        if not gate["passed"]:
            raise RuntimeError(
                "development promotion gate failed; final label download and parse remain blocked"
            )

        final_rows = read_structure_table(sources["final_structures"])
        expected_final = int(
            lock["data_sources"]["final_epa_structures"]["expected_structure_count"]
        )
        if len(final_rows) != expected_final:
            raise RuntimeError(f"final structure count mismatch: {len(final_rows)}")
        final_predictions, final_audits = run_panel(
            "final",
            endpoints,
            list(lock["roles"]["prospective_seeds"]),
            args.processed_dir,
            final_rows,
            lock,
        )
        prediction_path = args.output / "sealed_final_predictions.csv"
        audit_path = args.output / "sealed_final_transport_audit.csv"
        atomic_csv(prediction_path, final_predictions)
        atomic_csv(audit_path, final_audits)
        expected_label_sha = str(lock["data_sources"]["final_epa_labels"]["sha256"])
        promotion = {
            "status": "predictions_sealed_before_final_labels",
            "development_gate_passed": True,
            "final_labels_opened": False,
            "expected_final_label_sha256": expected_label_sha,
            "sealed_predictions_sha256": sha256_file(prediction_path),
            "sealed_transport_audit_sha256": sha256_file(audit_path),
            "lock_sha256": sha256_file(args.lock),
            "git_head": state["head"],
            "prospective_seeds": list(lock["roles"]["prospective_seeds"]),
            "endpoint_scope": endpoints,
        }
        promotion_path = args.output / "promotion_record.json"
        atomic_json(promotion_path, promotion)

        # The label file is first acquired and interpreted after the promotion
        # record above exists and binds every prediction byte.
        final_label_path = acquire_final_label_bytes(lock, args.source_root)
        final_labels = open_final_labels_after_promotion(
            final_label_path,
            promotion_path,
            expected_label_sha,
            list(lock["endpoint_order"]),
        )
        structure_ids = {str(row["sample_id"]) for row in final_rows}
        if set(final_labels) != structure_ids:
            raise RuntimeError(
                "final label/sample identity mismatch: "
                f"missing={sorted(structure_ids - set(final_labels))[:10]} "
                f"extra={sorted(set(final_labels) - structure_ids)[:10]}"
            )
        final_metrics = evaluate_predictions(final_predictions, final_labels)
        final_paired = paired_rows(final_metrics)
        primary_paired = [
            row
            for row in final_paired
            if row["endpoint"] in set(lock["primary_endpoints"])
            and row["method"] == METHOD_TAME
        ]
        primary_bootstrap = hierarchical_bootstrap_primary(
            final_predictions,
            final_labels,
            list(lock["primary_endpoints"]),
            list(lock["roles"]["prospective_seeds"]),
        )
        mean_macro_delta = float(
            np.mean([float(row["macro_csy_delta"]) for row in primary_paired])
        )
        report = {
            "status": "complete_independent_final_epa_evaluation",
            "algorithm": lock["algorithm"],
            "development_gate_passed": True,
            "predictions_sealed_before_final_labels": True,
            "final_labels_opened_after_promotion": True,
            "final_label_sha256": sha256_file(final_label_path),
            "primary_cell_count": len(primary_paired),
            "primary_inference": primary_bootstrap,
            "primary_mean_macro_csy_delta": mean_macro_delta,
            "macro_csy_noninferiority_margin": float(
                lock["promotion_gate"]["macro_csy_noninferiority_margin"]
            ),
            "result_interpretation": (
                "coverage_gain_with_efficiency_noninferiority"
                if primary_bootstrap["estimate"] > 0.0
                and mean_macro_delta
                >= float(lock["promotion_gate"]["macro_csy_noninferiority_margin"])
                else "negative_or_mixed_retain_without_retuning"
            ),
            "scientific_superiority_claim_authorized": False,
            "arbitrary_shift_coverage_guarantee_claim_authorized": False,
        }
        atomic_csv(args.output / "final_metrics.csv", final_metrics)
        atomic_csv(args.output / "final_paired.csv", final_paired)
        atomic_csv(args.output / "final_summary.csv", summarize_metrics(final_metrics))
        atomic_json(args.output / "final_report.json", report)
        manifest = {
            "status": "complete",
            "lock_sha256": sha256_file(args.lock),
            "promotion_record_sha256": sha256_file(promotion_path),
            "sealed_predictions_sha256": sha256_file(prediction_path),
            "final_metrics_sha256": sha256_file(args.output / "final_metrics.csv"),
            "final_paired_sha256": sha256_file(args.output / "final_paired.csv"),
            "final_summary_sha256": sha256_file(args.output / "final_summary.csv"),
            "final_report_sha256": sha256_file(args.output / "final_report.json"),
            "git_head": state["head"],
            "target_label_permutation_would_not_change_predictions": True,
            "prediction_identity_sha256": stable_sha256(
                [
                    [row["endpoint"], row["seed"], row["method"], row["sample_id"], row["include_0"], row["include_1"]]
                    for row in final_predictions
                ]
            ),
        }
        atomic_json(args.output / "manifest.json", manifest)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "status": "failed_closed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_json(args.output / "failure.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
