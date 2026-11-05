"""CPU experiment runner for the CHEMOLAB rejection revision.

The EPA outcomes were already inspected in the submitted work. All new EPA
comparisons are explicitly retrospective. No code here creates a new seal or
claims a coverage theorem. Legacy files and results are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
P2 = ROOT / "paper2_admet_benchmark"
LEGACY = P2 / "scripts" / "racer_c4"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LEGACY))
sys.path.insert(0, str(P2 / "scripts" / "racer_c"))
sys.path.insert(0, str(HERE))
import core  # noqa: E402
from racer_c4_core import (  # noqa: E402
    audit_density_ratio, binary_nonconformity, clipped,
    cross_fitted_density_ratio, transport_envelope, weighted_mondrian_sets,
)
from racer_c4_io import (  # noqa: E402
    ENDPOINT_PROPERTIES, acquire_final_label_bytes, acquire_unlabeled_sources,
    atomic_csv, atomic_json, fingerprints, parse_final_label_text,
    physchem_features, read_clean_endpoint, read_development_sdf,
    read_structure_table, sha256_file, stable_sha256,
)
from run_prospective_racer_c4 import (  # noqa: E402
    component_probabilities, make_models, split_internal_roles,
)

DEFAULT_PROTOCOL = HERE / "protocol.json"
LEGACY_LOCK = P2 / "configs" / "racer_c4" / "prospective_lock_v1.yaml"
DEFAULT_CACHE = ROOT / ".local" / "chemolab_revision_20261104" / "cache"
VIEWS = ("physchem_descriptors", "component_and_stack_logits")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def package_versions() -> dict[str, str]:
    names = ["numpy", "scipy", "scikit-learn", "rdkit", "PyYAML", "threadpoolctl", "pandas"]
    return {name: importlib.metadata.version(name) for name in names}


def code_hashes() -> dict[str, str]:
    paths = [p for p in HERE.rglob("*.py") if "tests" not in p.parts and not p.name.startswith("test_")]
    paths += list(LEGACY.glob("*.py"))
    paths += [P2 / "scripts/racer_c/prepare_tox21_challenge.py",
              P2 / "scripts/racer_c/prepare_classification_endpoint.py",
              P2 / "protocols/chemical_standardization_contract.yaml"]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(paths)}


def protect_output(path: Path) -> None:
    """Reject legacy trees, repository root and any nonempty unowned output."""
    resolved = path.resolve()
    forbidden = [ROOT, P2, LEGACY, P2 / "results", P2 / "data", ROOT / "paper2_latex"]
    if resolved in forbidden or any(parent in resolved.parents for parent in forbidden[2:]):
        raise ValueError("output must be a new revision directory, not a legacy source/results tree")


def verify_hashes(base: Path, expected: Mapping[str, str]) -> None:
    for relative, digest in expected.items():
        path = base / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"missing or changed cached input: {relative}")


def check_processing_environment() -> None:
    """Enforce the inherited chemical identity contract before downloading."""
    from rdkit import rdBase
    contract = yaml.safe_load((P2 / "protocols/chemical_standardization_contract.yaml").read_text())
    expected = str(contract["standardization"]["rdkit_version"])
    if rdBase.rdkitVersion != expected:
        raise RuntimeError(f"historical structure processing requires RDKit {expected}; "
                           f"found {rdBase.rdkitVersion}. Install revision_2026/requirements.txt "
                           "in a new Python 3.12 environment before prepare/run.")


def prepare(cache: Path, lock: dict) -> tuple[dict[str, Path], Path]:
    """Acquire hash-locked raw bytes and own a separate processed cache."""
    check_processing_environment()
    cache = cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    sources = acquire_unlabeled_sources(lock, cache / "sources")
    processed = cache / "processed"
    receipt = cache / "preparation.json"
    preparation_identity = {
        "legacy_lock_sha256": sha256_file(LEGACY_LOCK),
        "rdkit_version": package_versions()["rdkit"],
        "standardization_source_sha256": sha256_file(P2 / "scripts/racer_c/prepare_classification_endpoint.py"),
        "preparation_source_sha256": sha256_file(P2 / "scripts/racer_c/prepare_tox21_challenge.py"),
        "contract_sha256": sha256_file(P2 / "protocols/chemical_standardization_contract.yaml"),
    }
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["identity"] != preparation_identity:
            raise RuntimeError("prepared cache uses different code or RDKit; use a new --cache")
        verify_hashes(cache, saved["input_hashes"])
        return sources, processed
    if processed.exists() and any(processed.iterdir()):
        raise RuntimeError("unverified processed cache found; use a new --cache")
    command = [sys.executable, str(P2 / "scripts/racer_c/prepare_tox21_challenge.py"),
               "--archive", str(sources["training_archive"]), "--sdf", str(sources["training_sdf"]),
               "--processed-dir", str(processed), "--manifest-dir", str(cache / "manifests")]
    subprocess.run(command, cwd=ROOT, check=True)
    paths = list(sources.values()) + [processed / f"{endpoint}_clean.csv" for endpoint in ENDPOINT_PROPERTIES]
    atomic_json(receipt, {"identity": preparation_identity,
                         "input_hashes": {str(p.relative_to(cache)): sha256_file(p) for p in paths}})
    return sources, processed


def load_target(phase: str, sources: dict[str, Path], lock: dict, cache: Path):
    if phase == "development":
        rows, columns = read_development_sdf(sources["development_sdf"])
        expected = lock["data_sources"]["leaderboard_development_labels"]["expected_structure_count"]
        labels = {row["sample_id"]: {ep: float(y[i]) for ep, y in columns.items()}
                  for i, row in enumerate(rows)}
        input_label_path = sources["development_sdf"]
    else:
        rows = read_structure_table(sources["final_structures"])
        expected = lock["data_sources"]["final_epa_structures"]["expected_structure_count"]
        input_label_path = acquire_final_label_bytes(lock, cache / "sources")
        labels = parse_final_label_text(input_label_path.read_text(encoding="utf-8-sig"), list(ENDPOINT_PROPERTIES))
    if len(rows) != expected or len({r["sample_id"] for r in rows}) != len(rows):
        raise RuntimeError("unexpected target count or duplicate sample IDs")
    if set(labels) != {row["sample_id"] for row in rows}:
        raise RuntimeError("target structure and outcome IDs differ")
    return rows, labels, input_label_path


def max_tanimoto(train: np.ndarray, target: np.ndarray, block: int = 128) -> np.ndarray:
    """Exact maximum binary Tanimoto to fit rows; no query outcomes used."""
    reference = np.asarray(train, dtype=np.float32)
    query = np.asarray(target, dtype=np.float32)
    output = np.empty(len(query), dtype=float)
    reference_size = reference.sum(axis=1)
    for start in range(0, len(query), block):
        batch = query[start:start + block]
        intersection = batch @ reference.T
        union = batch.sum(axis=1)[:, None] + reference_size[None, :] - intersection
        similarity = np.divide(intersection, union, out=np.ones_like(intersection), where=union > 0)
        output[start:start + len(batch)] = similarity.max(axis=1)
    return output


def fit_predictors(x: np.ndarray, y: np.ndarray, qx: np.ndarray, lock: dict, seed: int,
                   jobs: int, model_names: Sequence[str]):
    fit, router, cal = split_internal_roles(y, seed)
    components = make_models(lock, seed)
    for estimator in components:
        if hasattr(estimator, "n_estimators") and hasattr(estimator, "n_jobs"):
            estimator.set_params(n_jobs=jobs)
    use_ensemble = any(model in {"stack", "equal_logit"} for model in model_names)
    selected = components if use_ensemble else components[:1]
    for estimator in selected:
        estimator.fit(x[fit], y[fit])
    cp = component_probabilities(selected, x[cal])
    qp = component_probabilities(selected, qx)
    probabilities = {"logistic": (cp[:, 0], qp[:, 0], logit(cp[:, :1]), logit(qp[:, :1]))}
    if use_ensemble:
        ec = clipped(expit(logit(cp).mean(axis=1)))
        eq = clipped(expit(logit(qp).mean(axis=1)))
        probabilities["equal_logit"] = (ec, eq, np.column_stack([logit(cp), logit(ec)]),
                                        np.column_stack([logit(qp), logit(eq)]))
    if "stack" in model_names:
        rp = component_probabilities(selected, x[router])
        router_lock = lock["base_predictor"]["router"]
        stacker = LogisticRegression(C=float(router_lock["C"]), solver=router_lock["solver"],
                                     max_iter=int(router_lock["max_iter"]), random_state=seed)
        stacker.fit(logit(rp), y[router])
        sc = clipped(stacker.predict_proba(logit(cp))[:, 1])
        sq = clipped(stacker.predict_proba(logit(qp))[:, 1])
        probabilities["stack"] = (sc, sq, np.column_stack([logit(cp), logit(sc)]),
                                  np.column_stack([logit(qp), logit(sq)]))
    return (fit, router, cal), {name: probabilities[name] for name in model_names}


def estimate_views(cal_x, target_x, cal_phys, target_phys, cal_logits, target_logits,
                   cal_y, lock, seed):
    density = lock["transport"]["domain_estimator"]
    inputs = [("ecfp_bits", cal_x, target_x, 20000),
              ("physchem_descriptors", cal_phys, target_phys, 25000),
              ("component_and_stack_logits", cal_logits, target_logits, 30000)]
    result, audits = {}, {}
    for view, sx, tx, offset in inputs:
        result[view] = cross_fitted_density_ratio(sx, tx, cal_y, seed=seed + offset,
            folds=int(density["folds"]), c_value=float(density["C"]),
            weight_min=float(density["weight_min"]), weight_max=float(density["weight_max"]))
        audits[view] = audit_density_ratio(view, result[view], cal_y, len(tx), **lock["transport"]["audit"])
    return result, audits


def generate_methods(cal_p, cal_y, target_p, ids, similarity, ratios, audits, lock,
                     protocol, seed, alpha, sensitivities=False, diagnostics=None):
    """Pure target-label-free decisions; target outcomes are not arguments."""
    sets = {name: core.conformal_sets(cal_p, cal_y, target_p, alpha=alpha, method=name)
            for name in ("marginal_lac", "mondrian_lac", "marginal_aps", "mondrian_aps")}
    baseline = core.repair_empty(sets["mondrian_lac"])
    sets["mondrian_lac_empty_to_full"] = baseline
    weighted = {}
    for view, ratio in ratios.items():
        weighted[view] = weighted_mondrian_sets(binary_nonconformity(cal_p), cal_y,
            binary_nonconformity(target_p), ratio.source_weights, ratio.target_weights, alpha)
        sets[f"lr_weighted_mondrian_{view}"] = weighted[view]
    for label_name, protected in (("label0_historical", [0]), ("label1_active", [1]), ("both", [0, 1])):
        sets[f"tame_{label_name}"] = transport_envelope(baseline, [weighted[v] for v in VIEWS],
            [audits[v] for v in VIEWS], protected_labels=protected, quorum="all", minimum_active_views=2).membership
    sets["tame_label1_no_gates"] = transport_envelope(baseline, [weighted[v] for v in VIEWS],
        [replace(audits[v], active=True, reason="gate_disabled_ablation") for v in VIEWS],
        protected_labels=[1], quorum="all", minimum_active_views=2).membership
    sets["tame_label1_union"] = transport_envelope(baseline, [weighted[v] for v in VIEWS],
        [audits[v] for v in VIEWS], protected_labels=[1], quorum="any", minimum_active_views=2).membership
    for view in VIEWS:
        sets[f"tame_label1_single_{view}"] = transport_envelope(baseline, [weighted[view]], [audits[view]],
            protected_labels=[1], quorum="any", minimum_active_views=1).membership
    priorities = {kind: core.deferral_priority(target_p, ids, kind=kind, similarity=similarity, seed=seed + 44000)
                  for kind in ("uncertainty", "ad", "random")}
    # A Mondrian singleton can disagree with probability argmax. This stronger
    # comparator ranks the probability of its omitted label, rather than just
    # distance to 0.5, within each matched singleton class.
    matched_priorities = {**priorities, "omitted_label_probability": np.where(baseline[:, 0], target_p, 1 - target_p)}
    for reference in ("label0_historical", "label1_active", "both"):
        for kind, priority in matched_priorities.items():
            sets[f"matched_{reference}_{kind}"] = core.matched_budget_deferral(
                baseline, sets[f"tame_{reference}"], priority, ids, match_classes=True)
    single_n = int((baseline.sum(axis=1) == 1).sum())
    for fraction in protocol["fixed_deferral_fractions"]:
        budget = int(math.floor(float(fraction) * single_n))
        for kind, priority in priorities.items():
            sets[f"mondrian_defer_{kind}_{float(fraction):g}"] = core.fixed_budget_deferral(baseline, budget, priority, ids)
            # Point-prediction rejectors use the same accepted-count budget as
            # the CP expansion curve, including baseline ambiguous/empty rows.
            total_defer = len(ids) - single_n + budget
            sets[f"point_abstain_{kind}_{float(fraction):g}"] = core.point_abstention(target_p, total_defer, priority, ids)
    sets["full_set"] = core.full_sets(len(ids))
    if sensitivities:
        for parameter, values in protocol["audit_sensitivity"].items():
            for value in values:
                params = {**lock["transport"]["audit"], parameter: value}
                changed = [audit_density_ratio(v, ratios[v], cal_y, len(ids), **params) for v in VIEWS]
                sets[f"tame_label1_sensitivity_{parameter}_{value:g}"] = transport_envelope(
                    baseline, [weighted[v] for v in VIEWS], changed,
                    protected_labels=[1], quorum="all", minimum_active_views=2).membership
    ablation_references = ["tame_label1_no_gates", "tame_label1_union"] + [f"tame_label1_single_{v}" for v in VIEWS]
    if sensitivities:
        ablation_references += [f"tame_label1_sensitivity_{parameter}_{value:g}"
                                for parameter, values in protocol["audit_sensitivity"].items() for value in values]
    for reference in ablation_references:
        sets[f"matched_{reference}_omitted_label_probability"] = core.matched_budget_deferral(
            baseline, sets[reference], matched_priorities["omitted_label_probability"], ids, match_classes=True)
    control_config = protocol.get("transport_null", {})
    if control_config.get("enabled", False):
        from paper2_admet_benchmark.revision_2026.target_null import generate_controls
        controls, control_diagnostics = generate_controls(cal_p, cal_y, target_p, ids, baseline,
            ratios, audits, VIEWS, alpha=alpha, seed=seed,
            replicates=int(control_config["replicates"]), edges=control_config["probability_bin_edges"])
        sets.update(controls)
        if diagnostics is not None:
            diagnostics["query_weight_alignment_controls"] = control_diagnostics
    return sets


def ensure_run(output: Path, identity: dict, resume: bool) -> None:
    protect_output(output)
    receipt = output / "provenance.json"
    if receipt.exists():
        if not resume:
            raise RuntimeError("output already contains a run; use --resume for identical inputs or a new --output")
        if json.loads(receipt.read_text())["identity"] != identity:
            raise RuntimeError("resume provenance differs (code/config/data/environment); use a new --output")
        return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("refusing a nonempty output directory without matching provenance")
    output.mkdir(parents=True, exist_ok=True)
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        head = "unavailable"
    atomic_json(receipt, {"identity": identity, "git_head_at_start": head,
                         "epa_is_new_blinded_validation": False,
                         "reference_methods_required_before_superiority_claim": protocol_reference_status()})


def protocol_reference_status():
    return ["full_CoDrug_and_KMM_CP_are_run_by_execute_revision_not_this_shared_predictor_subcommand"]


def validate_cell_receipt(cell: Path) -> bool:
    receipt = cell / "complete.json"
    if not receipt.exists():
        return False
    saved = json.loads(receipt.read_text())
    verify_hashes(cell, saved["files"])
    return True


def write_cell(cell, phase, endpoint, seed, model, target_rows, labels, eligible,
               p_all, cal_rows, cal_y, cal_p, all_roles, methods_by_alpha, audits,
               component_diagnostics=None, selection_by_method=None, raw_methods_by_alpha=None,
               max_train_tanimoto=None):
    cell.mkdir(parents=True, exist_ok=True)
    rows, metrics, target_manifest = [], [], []
    eval_y_all = np.asarray([labels[r["sample_id"]][endpoint] for r in target_rows], dtype=float)
    observed = eligible & np.isfinite(eval_y_all)
    for i, row in enumerate(target_rows):
        target_manifest.append({"phase": phase, "endpoint": endpoint, "seed": seed, "model": model,
            "sample_id": row["sample_id"], "structure_id": row["structure_id"],
            "standardized_smiles": row["standardized_smiles"],
            "probability_1": float(p_all[i]) if np.isfinite(p_all[i]) else "",
            "max_train_tanimoto": float(max_train_tanimoto[i]) if max_train_tanimoto is not None and np.isfinite(max_train_tanimoto[i]) else "",
            "eligible": int(eligible[i]),
            "excluded_structure_overlap": int(row.get("excluded_structure_overlap", 0)),
            "excluded_structure_invalid": int(row.get("structure_status", "pass") != "pass")})
    for alpha, methods in methods_by_alpha.items():
        for method, small_set in methods.items():
            membership = core.full_sets(len(target_rows))
            membership[eligible] = small_set
            raw_membership = membership.copy()
            if raw_methods_by_alpha is not None:
                raw_membership[eligible] = raw_methods_by_alpha[alpha][method]
            selected = eligible.copy()
            if selection_by_method is not None:
                selected[eligible] = selection_by_method[method]
            for i, manifest_row in enumerate(target_manifest):
                rows.append({**manifest_row, "alpha": alpha, "method": method,
                    "y": int(eval_y_all[i]) if np.isfinite(eval_y_all[i]) else "",
                    "include_0": int(membership[i, 0]), "include_1": int(membership[i, 1]),
                    "method_selected": int(selected[i]),
                    "original_include_0": int(raw_membership[i, 0]), "original_include_1": int(raw_membership[i, 1])})
            metrics.append({"phase": phase, "endpoint": endpoint, "seed": seed, "model": model,
                "alpha": alpha, "method": method, "n_target": len(target_rows),
                "n_eligible": int(eligible.sum()), "n_missing_labels": int((eligible & ~observed).sum()),
                "target_eligible_singletons": int((small_set.sum(axis=1) == 1).sum()),
                "target_eligible_deferred": int((small_set.sum(axis=1) != 1).sum()),
                "target_eligible_clear": int(((small_set.sum(axis=1) == 1) & small_set[:, 0]).sum()),
                **core.decision_metrics(eval_y_all[observed].astype(int), membership[observed])})
    calibration = [{"sample_id": r.get("sample_id", r["structure_id"]), "structure_id": r["structure_id"],
                    "standardized_smiles": r["standardized_smiles"], "y": int(y), "probability_1": float(p)}
                   for r, y, p in zip(cal_rows, cal_y, cal_p)]
    atomic_csv(cell / "predictions.csv", rows)
    atomic_csv(cell / "metrics.csv", metrics)
    atomic_csv(cell / "target_manifest.csv", target_manifest)
    atomic_csv(cell / "calibration.csv", calibration)
    atomic_csv(cell / "roles.csv", all_roles)
    atomic_json(cell / "audits.json", {v: asdict(a) for v, a in audits.items()})
    atomic_json(cell / "component_audits.json", component_diagnostics or {})
    names = ("predictions.csv", "metrics.csv", "target_manifest.csv", "calibration.csv", "roles.csv", "audits.json", "component_audits.json")
    atomic_json(cell / "complete.json", {"status": "complete", "files": {n: sha256_file(cell / n) for n in names}})


def density_components(cal_phys, target_phys, cal_p, cal_y, target_p, alphas, selected, protocol):
    """Fit optional diagnostic ratio components once, then vary CP alpha."""
    from paper2_admet_benchmark.revision_2026.density_baselines import (
        kde_density_ratio, kernel_mean_matching, weighted_empirical_mondrian,
    )
    outputs, diagnostics = {a: {} for a in alphas}, {}
    config = protocol["density_components"]
    if "kde" in selected:
        sw, tw, diag = kde_density_ratio(cal_phys, target_phys, **config["kde"])
        diagnostics["kde"] = diag
        for alpha in alphas:
            outputs[alpha]["kde_ratio_mondrian_physchem_component"] = weighted_mondrian_sets(
                core.candidate_scores(cal_p), cal_y, core.candidate_scores(target_p), sw, tw, alpha)
    if "kmm" in selected:
        weights, diag = kernel_mean_matching(cal_phys, target_phys, **config["kmm"])
        diagnostics["kmm"] = diag
        for alpha in alphas:
            outputs[alpha]["kmm_mondrian_physchem_component"] = weighted_empirical_mondrian(
                cal_p, cal_y, target_p, weights, alpha)
    return outputs, diagnostics


def expected_method_names(protocol, sensitivities=True, selected_density=()):
    """The scheduled method set is defined before target outcomes are read."""
    methods = ["marginal_lac", "mondrian_lac", "marginal_aps", "mondrian_aps",
               "mondrian_lac_empty_to_full"]
    methods += [f"lr_weighted_mondrian_{v}" for v in ("ecfp_bits", *VIEWS)]
    methods += [f"tame_{r}" for r in ("label0_historical", "label1_active", "both")]
    methods += ["tame_label1_no_gates", "tame_label1_union"]
    methods += [f"tame_label1_single_{v}" for v in VIEWS]
    methods += [f"matched_{r}_{k}" for r in ("label0_historical", "label1_active", "both")
                for k in ("uncertainty", "ad", "random", "omitted_label_probability")]
    methods += [f"{family}_{kind}_{float(f):g}" for f in protocol["fixed_deferral_fractions"]
                for kind in ("uncertainty", "ad", "random") for family in ("mondrian_defer", "point_abstain")]
    methods += ["full_set"]
    if sensitivities:
        methods += [f"tame_label1_sensitivity_{parameter}_{value:g}"
                    for parameter, values in protocol["audit_sensitivity"].items() for value in values]
    ablation_references = ["tame_label1_no_gates", "tame_label1_union"] + [f"tame_label1_single_{v}" for v in VIEWS]
    if sensitivities:
        ablation_references += [f"tame_label1_sensitivity_{parameter}_{value:g}"
                               for parameter, values in protocol["audit_sensitivity"].items() for value in values]
    methods += [f"matched_{name}_omitted_label_probability" for name in ablation_references]
    if "kde" in selected_density:
        methods.append("kde_ratio_mondrian_physchem_component")
    if "kmm" in selected_density:
        methods.append("kmm_mondrian_physchem_component")
    if protocol.get("transport_null", {}).get("enabled", False):
        from paper2_admet_benchmark.revision_2026.target_null import method_names
        methods += method_names(protocol["transport_null"]["replicates"])
    return methods


def execute_shared_case(source_rows, y, target_rows, labels, *, phase, endpoint, seed,
                        models, alphas, protocol, lock, output, jobs=2, sensitivities=True,
                        selected_density=(), prepared_features=None):
    """Execute one arbitrary binary endpoint on a common target universe.

    Used for the historical challenge and the separately sourced validation
    panels. Selection and set construction never receive target outcomes.
    """
    valid = np.asarray([r.get("structure_status", "pass") == "pass" for r in target_rows])
    if prepared_features is None:
        x, source_phys = fingerprints(source_rows), physchem_features(source_rows)
        valid_rows = [r for r, ok in zip(target_rows, valid) if ok]
        vx, vp = fingerprints(valid_rows), physchem_features(valid_rows)
        tx = np.zeros((len(target_rows), vx.shape[1]), dtype=vx.dtype)
        tp = np.zeros((len(target_rows), vp.shape[1]), dtype=float)
        tx[valid], tp[valid] = vx, vp
    else:
        x, source_phys, tx, tp = prepared_features
    source_ids = {r["structure_id"] for r in source_rows}
    overlap = np.asarray([r["structure_id"] in source_ids for r in target_rows])
    eligible = valid & ~overlap
    rows_for_endpoint = [{**r, "excluded_structure_overlap": int(overlap[i])} for i, r in enumerate(target_rows)]
    if eligible.sum() < 2:
        raise RuntimeError(f"{endpoint}: less than two usable target structures")
    pending = [model for model in models if not validate_cell_receipt(
        output / "cells" / f"{phase}__{endpoint}__{seed}__{model}")]
    if not pending:
        print(f"resume verified {endpoint} seed={seed}", flush=True)
        return
    roles, predictions = fit_predictors(x, y, tx[eligible], lock, seed, jobs, pending)
    fit, router, cal = roles
    source_roles = [{"structure_id": source_rows[i]["structure_id"],
                     "standardized_smiles": source_rows[i]["standardized_smiles"], "y": int(y[i]), "role": role}
                    for role, indices in zip(("fit", "router", "conformal"), roles) for i in indices]
    similarity = max_tanimoto(x[fit], tx[eligible])
    ids = [r["sample_id"] for r, ok in zip(target_rows, eligible) if ok]
    expected = set(expected_method_names(protocol, sensitivities, selected_density))
    for model in pending:
        cp, qp, cl, ql = predictions[model]
        ratios, audits = estimate_views(x[cal], tx[eligible], source_phys[cal], tp[eligible], cl, ql, y[cal], lock, seed)
        generation_audits = {}
        methods = {alpha: generate_methods(cp, y[cal], qp, ids, similarity, ratios, audits, lock,
                   protocol, seed, alpha, sensitivities, generation_audits) for alpha in alphas}
        component_sets, component_audits = density_components(source_phys[cal], tp[eligible],
            cp, y[cal], qp, alphas, selected_density, protocol)
        component_audits.update(generation_audits)
        for alpha in alphas:
            methods[alpha].update(component_sets[alpha])
            if set(methods[alpha]) != expected:
                raise RuntimeError("generated methods differ from the prespecified matrix")
        p_all = np.full(len(target_rows), np.nan)
        p_all[eligible] = qp
        sim_all = np.full(len(target_rows), np.nan)
        sim_all[eligible] = similarity
        cell = output / "cells" / f"{phase}__{endpoint}__{seed}__{model}"
        write_cell(cell, phase, endpoint, seed, model, rows_for_endpoint, labels, eligible, p_all,
                   [source_rows[i] for i in cal], y[cal], cp, source_roles, methods, audits, component_audits,
                   max_train_tanimoto=sim_all)
        print(f"complete {phase} {endpoint} seed={seed} model={model}", flush=True)


def run(args, protocol, lock):
    sources, processed = prepare(args.cache, lock)
    target_rows, labels, label_path = load_target(args.phase, sources, lock, args.cache)
    endpoints = args.endpoints or list(ENDPOINT_PROPERTIES)
    seeds = args.seeds or protocol[f"{args.phase}_seeds"]
    models = args.models or protocol["models"]
    alphas = args.alphas or protocol["alpha_grid"]
    selected_density = getattr(args, "density_components", [])
    if not set(endpoints).issubset(ENDPOINT_PROPERTIES):
        raise ValueError("unknown Tox21 endpoint")
    if not set(models).issubset({"logistic", "stack", "equal_logit"}):
        raise ValueError("unknown model")
    for sequence in (endpoints, seeds, models, alphas):
        if not sequence or len(sequence) != len(set(sequence)):
            raise ValueError("selection lists must be nonempty and duplicate-free")
    if any(not 0 < a < 1 for a in alphas):
        raise ValueError("alpha must lie in (0,1)")
    identity = {"protocol_sha256": sha256_file(args.protocol), "legacy_lock_sha256": sha256_file(LEGACY_LOCK),
        "code": code_hashes(), "environment": {"python": platform.python_version(), "packages": package_versions()},
        "phase": args.phase, "endpoints": endpoints, "seeds": seeds, "models": models, "alphas": alphas,
        "jobs": args.jobs, "sensitivities": args.sensitivities,
        "density_components": selected_density,
        "preparation_sha256": sha256_file(args.cache / "preparation.json"),
        "outcome_file_sha256": sha256_file(label_path)}
    ensure_run(args.output, identity, args.resume)
    atomic_json(args.output / "protocol_snapshot.json", protocol)
    valid = np.asarray([r.get("structure_status", "pass") == "pass" for r in target_rows])
    valid_rows = [r for r, ok in zip(target_rows, valid) if ok]
    valid_x, valid_phys = fingerprints(valid_rows), physchem_features(valid_rows)
    target_x = np.zeros((len(target_rows), valid_x.shape[1]), dtype=valid_x.dtype)
    target_phys = np.zeros((len(target_rows), valid_phys.shape[1]), dtype=float)
    target_x[valid], target_phys[valid] = valid_x, valid_phys
    start = time.monotonic()
    for endpoint in endpoints:
        source_rows, y = read_clean_endpoint(processed / f"{endpoint}_clean.csv")
        x, source_phys = fingerprints(source_rows), physchem_features(source_rows)
        for seed in seeds:
            execute_shared_case(source_rows, y, target_rows, labels, phase=args.phase,
                endpoint=endpoint, seed=seed, models=models, alphas=alphas, protocol=protocol,
                lock=lock, output=args.output, jobs=args.jobs, sensitivities=args.sensitivities,
                selected_density=selected_density,
                prepared_features=(x, source_phys, target_x, target_phys))
    summaries = []
    for endpoint in endpoints:
        for seed in seeds:
            for model in models:
                cell = args.output / "cells" / f"{args.phase}__{endpoint}__{seed}__{model}"
                if not validate_cell_receipt(cell):
                    raise RuntimeError(f"incomplete cell: {cell}")
                summaries.extend(read_csv(cell / "metrics.csv"))
    atomic_csv(args.output / "all_metrics.csv", summaries)
    atomic_json(args.output / "run_complete.json", {"status": "completed_requested_matrix",
        "phase": args.phase, "cells": len(endpoints) * len(seeds) * len(models),
        "epa_is_new_blinded_validation": False, "full_reference_baselines_complete": False,
        "all_metrics_sha256": sha256_file(args.output / "all_metrics.csv"),
        "elapsed_seconds": time.monotonic() - start})


def smoke(args, protocol, lock):
    """Synthetic numerical/integration check only; never a molecular result."""
    rng = np.random.default_rng(739)
    y = np.tile([0, 1], 160)
    x = rng.integers(0, 2, (320, 32), dtype=np.uint8)
    qx = rng.integers(0, 2, (48, 32), dtype=np.uint8)
    phys = rng.normal(size=(320, 9))
    qphys = rng.normal(size=(48, 9))
    small_lock = json.loads(json.dumps(lock))
    for entry in small_lock["base_predictor"]["components"]:
        if "n_estimators" in entry:
            entry["n_estimators"] = 8
    roles, probabilities = fit_predictors(x, y, qx, small_lock, 7, args.jobs, protocol["models"])
    cal = roles[2]
    result = {"status": "synthetic_smoke_only", "molecular_experiments_run": False, "models": {}}
    for model, (cp, qp, cl, ql) in probabilities.items():
        ratios, audits = estimate_views(x[cal], qx, phys[cal], qphys, cl, ql, y[cal], small_lock, 7)
        sets = generate_methods(cp, y[cal], qp, [f"synthetic-{i}" for i in range(48)], max_tanimoto(x[roles[0]], qx),
                                ratios, audits, small_lock, protocol, 7, 0.1, True)
        assert all(v.shape == (48, 2) for v in sets.values())
        result["models"][model] = {"methods": len(sets), "all_shapes_pass": True}
    print(json.dumps(result, indent=2))
    if args.output:
        protect_output(args.output)
        if args.output.exists() and any(args.output.iterdir()):
            raise RuntimeError("smoke output must be new/empty")
        atomic_json(args.output / "synthetic_smoke.json", result)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "smoke"])
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase", choices=["development", "retrospective_epa"], default="development")
    parser.add_argument("--endpoints", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--models", nargs="+", choices=["logistic", "stack", "equal_logit"])
    parser.add_argument("--alphas", nargs="+", type=float)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--sensitivities", action="store_true")
    parser.add_argument("--density-components", nargs="+", choices=["kde", "kmm"], default=[],
                        help="optional diagnostic components; neither is a full published reference reproduction")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be a positive integer")
    if args.command == "run" and args.output is None:
        parser.error("run requires a new --output directory")
    args.cache = args.cache.resolve()
    if args.output:
        args.output = args.output.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    protocol = json.loads(args.protocol.read_text())
    lock = yaml.safe_load(LEGACY_LOCK.read_text())
    # This also controls BLAS/OpenMP threads used by linear algebra and LR.
    with threadpool_limits(limits=args.jobs):
        if args.command == "prepare":
            sources, processed = prepare(args.cache, lock)
            print(json.dumps({"status": "prepared", "cache": str(args.cache), "processed": str(processed)}))
        elif args.command == "smoke":
            smoke(args, protocol, lock)
        else:
            run(args, protocol, lock)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
