"""Build the complete revision tables, figures and manuscript from real runs.

The declared matrix is checked before results are written.  No analysis
discovers a favorable comparator from the outcomes.  Molecules, not repeated
training seeds, are the resampling units.  Existing output is never replaced.
Run with ``python -m paper2_admet_benchmark.revision_2026.reporting --help``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd

from .core import decision_metrics
from .summarize import METRICS, bootstrap_comparison, paired_cells

HERE = Path(__file__).resolve().parent
MANUSCRIPT = HERE.parents[1] / "paper2_latex/resubmission_20261104/manuscript.tex"
IDENTITY = ["phase", "endpoint", "seed", "model", "alpha", "method"]
FLAGS = ["include_0", "include_1"]
PRIMARY_METHOD = "tame_label1_active"
PRIMARY_REFERENCE = "matched_label1_active_omitted_label_probability"
ALIASES = {"false_clearance": "active_false_clearance", "false_discard": "inactive_false_discard",
           "clearance_contamination": "active_fraction_among_cleared", "deferred_rate": "defer_rate"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value):
    path.write_text(json.dumps(clean_json(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def confined(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("manifest paths must be nonempty strings")
    path = (root / relative).resolve()
    if path == root.resolve() or root.resolve() not in path.parents:
        raise ValueError("manifest path escapes run root: " + relative)
    return path


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    if "predictor" in frame and "model" not in frame:
        frame = frame.rename(columns={"predictor": "model"})
    required = set(IDENTITY + ["sample_id", "structure_id", "y", "eligible"] + FLAGS)
    if not required.issubset(frame):
        raise ValueError("prediction columns missing: " + str(sorted(required - set(frame))))
    frame = frame.copy()
    for col in ["sample_id", "structure_id"]:
        if frame[col].isna().any():
            # Invalid structures may have no structure identity but still
            # require a unique target identity; eligible rows may not.
            if col == "sample_id" or frame.loc[frame[col].isna(), "eligible"].astype(float).eq(1).any():
                raise ValueError("missing " + col)
        frame[col] = frame[col].fillna("").astype(str)
    if frame.sample_id.eq("").any():
        raise ValueError("empty sample_id")
    for col in FLAGS + ["eligible"]:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        if not frame[col].isin([0, 1]).all():
            raise ValueError(col + " must be 0/1")
        frame[col] = frame[col].astype(np.int8)
    frame["seed"] = pd.to_numeric(frame.seed, errors="raise")
    if not (np.isfinite(frame.seed) & (frame.seed == np.floor(frame.seed))).all():
        raise ValueError("seed must be a finite integer")
    frame["seed"] = frame.seed.astype(int)
    frame["alpha"] = pd.to_numeric(frame.alpha, errors="raise")
    if not (np.isfinite(frame.alpha) & (frame.alpha > 0) & (frame.alpha < 1)).all():
        raise ValueError("alpha must lie in (0,1)")
    frame["y"] = pd.to_numeric(frame.y, errors="raise")
    if not (frame.y.isna() | frame.y.isin([-1, 0, 1])).all():
        raise ValueError("outcomes must be 0/1 or explicitly missing")
    if frame.duplicated(IDENTITY + ["sample_id"]).any():
        raise ValueError("duplicate prediction identity")
    selection_col = "method_selected" if "method_selected" in frame else "external_selected"
    frame["method_selected"] = frame[selection_col] if selection_col in frame else frame.eligible
    frame["method_selected"] = pd.to_numeric(frame.method_selected, errors="raise")
    if not frame.method_selected.isin([0, 1]).all():
        raise ValueError("method_selected must be 0/1")
    for c in (0, 1):
        raw = f"original_include_{c}"
        old = f"external_original_include_{c}"
        frame[raw] = frame[raw] if raw in frame else frame[old] if old in frame else frame[f"include_{c}"]
        frame[raw] = pd.to_numeric(frame[raw], errors="raise")
        if not frame[raw].isin([0, 1]).all():
            raise ValueError(raw + " must be 0/1")
    unselected = (frame.eligible == 1) & (frame.method_selected == 0)
    if not frame.loc[unselected, FLAGS].eq(1).all().all():
        raise ValueError("unselected target rows must defer as full sets in full-cohort analysis")
    return frame


def _facts(group: pd.DataFrame) -> pd.DataFrame:
    return group.sort_values("sample_id")[["sample_id", "structure_id", "y", "eligible"]].reset_index(drop=True)


def verify_job(run_root: Path, job: dict) -> tuple[pd.DataFrame, dict]:
    """Verify completion hash, the full declared method matrix, and row identity."""
    prediction = confined(run_root, job["predictions"])
    receipt_path = confined(run_root, job["completion"])
    if not prediction.is_file() or not receipt_path.is_file():
        raise ValueError("incomplete declared job: " + job["job_id"])
    receipt = json.loads(receipt_path.read_text())
    files = receipt.get("files", receipt.get("output_sha256", {}))
    if not files:
        raise ValueError("completion receipt has no file hashes: " + job["job_id"])
    if job.get("family") in {"shared", "full_reference"} and receipt.get("status") != "complete":
        raise ValueError("declared execution job lacks a complete status")
    if job.get("family") == "shared" and "audits.json" not in files:
        raise ValueError("shared completion receipt does not bind original gate diagnostics")
    pred_verified = False
    for relative, expected in files.items():
        path = (receipt_path.parent / relative).resolve()
        if receipt_path.parent.resolve() not in path.parents:
            raise ValueError("receipt file escapes its cell")
        if not path.is_file() or sha256(path) != expected:
            raise ValueError("changed or missing completed output: " + str(path))
        pred_verified |= path == prediction
    if not pred_verified:
        raise ValueError("predictions are not covered by the completion receipt")
    frame = _normalize(pd.read_csv(prediction, dtype={"sample_id": str, "structure_id": str}))
    if frame.empty:
        raise ValueError("empty prediction file")
    for column in ["phase", "endpoint", "seed", "model"]:
        if set(frame[column]) != {job[column]}:
            raise ValueError("job identity mismatch: " + column)
    expected = {(float(a), str(m)) for a in job["alphas"] for m in job["methods"]}
    actual = set(zip(frame.alpha, frame.method))
    if actual != expected:
        raise ValueError(f"method/alpha matrix differs for {job['job_id']}: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    facts = None
    for _, group in frame.groupby(["alpha", "method"], sort=True):
        observed = _facts(group)
        if facts is None:
            facts = observed
        elif not facts.equals(observed):
            raise ValueError("method/alpha rows disagree on outcomes, structure or eligibility")
    verify_matched_budgets(frame)
    return frame, {"job_id": job["job_id"], "predictions": job["predictions"],
                   "sha256": sha256(prediction), "completion_sha256": sha256(receipt_path),
                   "rows": len(frame), "methods": len(job["methods"]), "alphas": job["alphas"]}


def verify_matched_budgets(frame: pd.DataFrame):
    """Check all eligible rows, including missing outcomes; budgets are label-free."""
    for alpha, group in frame.groupby("alpha"):
        methods = set(group.method)
        if "mondrian_lac_empty_to_full" not in methods:
            return
        base = group[(group.method == "mondrian_lac_empty_to_full") & (group.eligible == 1)].sort_values("sample_id")
        b = base[FLAGS].to_numpy(bool)
        singleton = b.sum(1) == 1
        references = [("tame_" + stem, "matched_" + stem + "_") for stem in ["label0_historical", "label1_active", "both"]]
        references += [(name, "matched_" + name + "_") for name in methods if name.startswith("null_") and "query_permutation_" in name]
        references += [(name, "matched_" + name + "_") for name in methods if name.startswith("tame_") and
                       "matched_" + name + "_omitted_label_probability" in methods]
        for ref_name, prefix in references:
            if ref_name not in methods:
                continue
            reference = group[(group.method == ref_name) & (group.eligible == 1)].sort_values("sample_id")[FLAGS].to_numpy(bool)
            if np.any(b & ~reference):
                raise ValueError("TAME violates baseline containment")
            expected = [int(np.sum(singleton & b[:, c] & (reference.sum(1) == 2))) for c in (0, 1)]
            for name in sorted(m for m in methods if m.startswith(prefix)):
                matched = group[(group.method == name) & (group.eligible == 1)].sort_values("sample_id")[FLAGS].to_numpy(bool)
                if np.any(b & ~matched):
                    raise ValueError("matched rule violates baseline containment")
                actual = [int(np.sum(singleton & b[:, c] & (matched.sum(1) == 2))) for c in (0, 1)]
                if actual != expected:
                    raise ValueError(f"class-specific deferral budgets differ: {alpha}, {name}")


def recompute_metrics(frame: pd.DataFrame) -> list[dict]:
    output = []
    for keys, group in frame.groupby(IDENTITY, sort=True):
        identity = dict(zip(IDENTITY, keys))
        valid = group[(group.eligible == 1) & group.y.isin([0, 1])]
        selected = valid[valid.method_selected == 1]
        eligible_sets = group.loc[group.eligible == 1, FLAGS].to_numpy(bool)
        sizes = eligible_sets.sum(axis=1)
        operational_n = len(sizes)
        common = {**identity, "n_target": len(group), "n_eligible": int(group.eligible.sum()),
                  "n_eligible_labeled": len(valid), "n_selected_eligible_labeled": len(selected),
                  "target_selection_rate": float(group.loc[group.eligible == 1, "method_selected"].mean()),
                  "n_missing_eligible_outcomes": int(((group.eligible == 1) & ~group.y.isin([0, 1])).sum()),
                  "operational_defer_n": int(np.sum(sizes != 1)),
                  "operational_inactive_n": int(np.sum((sizes == 1) & eligible_sets[:, 0])),
                  "operational_active_n": int(np.sum((sizes == 1) & eligible_sets[:, 1])),
                  "operational_empty_n": int(np.sum(sizes == 0)),
                  "operational_defer_rate": float(np.mean(sizes != 1)) if operational_n else math.nan}
        for population, rows, columns in [("full_eligible_labeled", valid, FLAGS),
                                         ("selected_eligible_labeled", selected, ["original_include_0", "original_include_1"])]:
            metrics = decision_metrics(rows.y.to_numpy(int), rows[columns].to_numpy(bool))
            metrics.update({new: metrics[old] for old, new in ALIASES.items()})
            metrics["labeled_defer_rate"] = metrics["defer_rate"]
            output.append({**common, "population": population, **metrics})
    return output


def aggregate_metrics(cells: pd.DataFrame, panels: dict[str, list[str]]) -> pd.DataFrame:
    """Equal assay/seed means with no silent dropping of undefined cells."""
    numeric = [c for c in cells.select_dtypes(include=[np.number]) if c not in ["seed", "alpha"]]
    rows = []
    group_keys = ["phase", "model", "alpha", "method", "population"]
    for panel_name, endpoints in panels.items():
        for keys, group in cells[cells.endpoint.isin(endpoints)].groupby(group_keys, sort=True):
            row = {**dict(zip(group_keys, keys)), "panel": panel_name, "cells": len(group),
                   "endpoints": len(set(group.endpoint)), "seeds": len(set(group.seed))}
            for metric in numeric:
                values = group[metric].to_numpy(float)
                row[metric] = float(np.mean(values)) if np.isfinite(values).all() else math.nan
                row[metric + "_undefined_cells"] = int((~np.isfinite(values)).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def gate_stratified_targeting(cells: pd.DataFrame, gates: list[dict]) -> pd.DataFrame:
    """Own-budget excess error conditional on original gate status; descriptive only."""
    gate_frame = pd.DataFrame(gates)
    keys = ["phase", "endpoint", "seed", "model"]
    data = cells[(cells.population == "full_eligible_labeled") & (cells.alpha == .1)]
    rows = []
    for identity, group in data.groupby(keys, sort=True):
        index = group.set_index("method")
        for method in index.index:
            if not method.startswith("tame_"):
                continue
            own = PRIMARY_REFERENCE if method == PRIMARY_METHOD else "matched_" + method + "_omitted_label_probability"
            if own not in index.index:
                continue
            left, right = index.loc[method], index.loc[own]
            rows.append({**dict(zip(keys, identity)), "method": method, "reference": own,
                "excess_false_inactive": left.active_false_clearance - right.active_false_clearance,
                "labeled_deferral_difference": left.defer_rate - right.defer_rate})
    values = pd.DataFrame(rows).merge(gate_frame, on=keys, how="left", validate="many_to_one")
    if values.original_gate_status.isna().any():
        raise ValueError("missing original gate status for own-budget diagnostics")
    summary = []
    for identity, group in values.groupby(["phase", "model", "method", "original_gate_status"], sort=True):
        summary.append({**dict(zip(["phase", "model", "method", "original_gate_status"], identity)),
            "cells": len(group), "endpoints": group.endpoint.nunique(), "training_seeds": group.seed.nunique(),
            "mean_excess_false_inactive": float(np.mean(group.excess_false_inactive.to_numpy(float))),
            "excess_false_inactive_undefined_cells": int((~np.isfinite(group.excess_false_inactive.to_numpy(float))).sum()),
            "mean_labeled_deferral_difference": float(np.mean(group.labeled_deferral_difference.to_numpy(float))),
            "labeled_deferral_difference_undefined_cells": int((~np.isfinite(group.labeled_deferral_difference.to_numpy(float))).sum()),
            "interpretation": "Descriptive conditional cell mean; not a causal gate effect or independent-seed inference"})
    return pd.DataFrame(summary)


@dataclass(frozen=True)
class Contrast:
    comparison_id: str
    phase: str
    panel: str
    model: str
    method: str
    reference_model: str
    reference: str
    alpha: float
    role: str = "secondary"


def declared_contrasts(protocol: dict, jobs: list[dict]) -> list[Contrast]:
    """Choose every contrast from declared methods before reading outcomes."""
    primary = protocol["primary_comparison"]
    models = protocol["models"]
    phases = sorted({job["phase"] for job in jobs})
    inventory = {}
    for job in jobs:
        inventory.setdefault((job["phase"], job["model"]), set()).update(job["methods"])
    alpha = float(primary["alpha"])
    contrasts = []
    for phase in phases:
        historical = phase in {"development", "retrospective_epa"}
        endpoints = sorted({j["endpoint"] for j in jobs if j["phase"] == phase})
        main_panels = ["primary_six"] if historical else ["endpoint__" + e for e in endpoints]
        sensitivity_panels = ["all_assays"] if historical else []
        for model in models:
            for panel in main_panels + sensitivity_panels:
                contrasts.append(Contrast(f"{phase}__{model}__{panel}__matched_omitted_label",
                    phase, panel, model, primary["method"], model, primary["reference"], alpha,
                    "primary" if phase == "retrospective_epa" and model == primary["predictor"] and panel == "primary_six" else "sensitivity"))
        # Endpoint-specific effects are reported even when their directions
        # differ. New assay families are never pooled into the Tox21 estimand.
        if historical:
            for endpoint in endpoints:
                contrasts.append(Contrast(f"{phase}__stack__endpoint__{endpoint}", phase,
                    "endpoint__" + endpoint, "stack", primary["method"], "stack", primary["reference"], alpha, "endpoint_sensitivity"))
        simple = ["matched_label1_active_uncertainty", "matched_label1_active_ad", "matched_label1_active_random"]
        ablations = ["tame_label1_no_gates", "tame_label1_union", "tame_label1_single_physchem_descriptors",
                     "tame_label1_single_component_and_stack_logits"]
        standards = ["mondrian_lac_empty_to_full", "marginal_lac", "mondrian_aps",
                     "lr_weighted_mondrian_physchem_descriptors", "lr_weighted_mondrian_component_and_stack_logits",
                     "lr_weighted_mondrian_ecfp_bits"]
        full_models = sorted({j["model"] for j in jobs if j["phase"] == phase and j.get("family") == "full_reference"})
        for panel in main_panels:
            for alternative_alpha in protocol["alpha_grid"]:
                if float(alternative_alpha) != alpha:
                    contrasts.append(Contrast(f"{phase}__stack__{panel}__alpha_{alternative_alpha:g}", phase,
                        panel, "stack", primary["method"], "stack", primary["reference"], float(alternative_alpha), "alpha_sensitivity"))
            for reference in simple + ablations + standards:
                contrasts.append(Contrast(f"{phase}__stack__{panel}__{reference}", phase, panel, "stack",
                    primary["method"], "stack", reference, alpha, "secondary"))
            for parameter, values in protocol["audit_sensitivity"].items():
                for value in values:
                    reference = f"tame_label1_sensitivity_{parameter}_{value:g}"
                    contrasts.append(Contrast(f"{phase}__stack__{panel}__{reference}", phase,
                        panel, "stack", primary["method"], "stack", reference, alpha, "gate_sensitivity"))
            for variant in ["label0_historical", "both"]:
                contrasts.append(Contrast(f"{phase}__stack__{panel}__matched_{variant}", phase,
                    panel, "stack", "tame_" + variant, "stack", "matched_" + variant + "_omitted_label_probability", alpha))
            for model in ["equal_logit", "logistic"]:
                contrasts.append(Contrast(f"{phase}__{model}__{panel}__ordinary_vs_stack", phase, panel, model,
                    "mondrian_lac_empty_to_full", "stack", "mondrian_lac_empty_to_full", alpha, "predictor_sensitivity"))
            for model in full_models:
                for method in sorted(inventory[(phase, model)]):
                    contrasts.append(Contrast(f"{phase}__{panel}__stack_vs_{model}__{method}", phase, panel, "stack",
                        primary["method"], model, method, alpha, "end_to_end_reference"))
            for method in sorted(m for m in inventory[(phase, "stack")] if m.startswith("null_") and "query_permutation_" in m):
                # Each null is compared at its own output budget. Permuting
                # query weights may change the number of expanded compounds.
                contrasts.append(Contrast(f"{phase}__{panel}__{method}__own_budget", phase, panel, "stack",
                    method, "stack", "matched_" + method + "_omitted_label_probability", alpha, "query_weight_alignment_null"))
            for method in sorted(m for m in inventory[(phase, "stack")] if m.startswith("tame_label1_") and
                                 "matched_" + m + "_omitted_label_probability" in inventory[(phase, "stack")]):
                contrasts.append(Contrast(f"{phase}__{panel}__{method}__own_budget", phase, panel, "stack",
                    method, "stack", "matched_" + method + "_omitted_label_probability", alpha, "ablation_own_budget"))
    for c in contrasts:
        if c.method not in inventory.get((c.phase, c.model), set()) or c.reference not in inventory.get((c.phase, c.reference_model), set()):
            raise ValueError("declared comparison method absent from job manifest: " + c.comparison_id)
    if len({c.comparison_id for c in contrasts}) != len(contrasts):
        raise ValueError("duplicate declared comparison IDs")
    return contrasts


def interpret_primary(results: list[dict], *, matrix_complete: bool, reference_complete: bool) -> dict:
    """A fixed decision rule; secondary results cannot rescue a failed primary."""
    matches = [r for r in results if r["role"] == "primary" and r["metric"] == "active_false_clearance"]
    if len(matches) != 1:
        raise ValueError("exactly one declared primary comparison is required")
    r = matches[0]
    lo, hi, delta = r.get("ci_95_low"), r.get("ci_95_high"), r.get("delta")
    finite = all(v is not None and math.isfinite(float(v)) for v in [lo, hi, delta])
    if not matrix_complete or not reference_complete:
        decision, evidence = "INCOMPLETE_DO_NOT_SUBMIT", "required_matrix_or_reference_missing"
    elif not finite:
        decision, evidence = "NO_GO_ALGORITHM_ADVANTAGE", "primary_interval_undefined"
    elif hi < 0:
        decision, evidence = "GO_TO_AUTHOR_REVIEW", "primary_retrospective_targeting_advantage_supported"
    elif lo > 0:
        decision, evidence = "NO_GO_ALGORITHM_ADVANTAGE", "primary_simple_comparator_is_better"
    else:
        decision, evidence = "NO_GO_ALGORITHM_ADVANTAGE", "primary_advantage_not_established"
    return {"decision": decision, "evidence": evidence, "primary_comparison_id": r["comparison_id"],
            "primary_delta": delta, "primary_ci_95_low": lo, "primary_ci_95_high": hi,
            "decision_meaning": "Author-review readiness and primary claim support only; not a journal acceptance prediction. EPA is retrospective. Secondary pointwise contrasts cannot override the primary outcome."}


def _comparison_frames(frames, contrasts):
    needed = {(c.phase, model, method, c.alpha) for c in contrasts
              for model, method in [(c.model, c.method), (c.reference_model, c.reference)]}
    columns = IDENTITY + ["sample_id", "structure_id", "y", "eligible"] + FLAGS
    for frame in frames:
        mask = np.fromiter(((p, model, m, float(a)) in needed for p, model, m, a in
                            frame[["phase", "model", "method", "alpha"]].itertuples(index=False, name=None)), bool, count=len(frame))
        yield frame.loc[mask, columns].copy()


def compute_contrasts(frame: pd.DataFrame, contrasts: list[Contrast], panels: dict, replicates: int, seed: int):
    results = []
    for contrast in contrasts:
        metadata = asdict(contrast)
        try:
            selected = frame[(frame.phase == contrast.phase) & (frame.alpha.astype(float) == contrast.alpha) &
                             frame.endpoint.isin(panels[contrast.panel]) &
                             (((frame.model == contrast.model) & (frame.method == contrast.method)) |
                              ((frame.model == contrast.reference_model) & (frame.method == contrast.reference)))]
            cells, structures, seeds = paired_cells(selected, contrast.method, contrast.reference, contrast.model,
                contrast.reference_model, contrast.alpha, panels[contrast.panel], contrast.phase)
            rows = bootstrap_comparison(cells, len(structures), replicates, seed)
            results.extend({**metadata, "molecular_clusters": len(structures), "training_seeds": len(seeds),
                            "cells": len(cells), **row} for row in rows)
        except ValueError as exc:
            if "lacks a class" not in str(exc):
                raise
            if contrast.role == "primary":
                raise ValueError("primary panel has an absent class; primary inference cannot be silently changed") from exc
            results.extend({**metadata, "metric": metric, "method_value": None, "reference_value": None,
                            "delta": None, "ci_95_low": None, "ci_95_high": None,
                            "valid_bootstrap_replicates": 0, "total_bootstrap_replicates": replicates,
                            "interval_status": "undefined_absent_class_in_fixed_panel"} for metric in METRICS)
    return results


def _method_label(method: str) -> str:
    return {PRIMARY_METHOD: "TAME: active protection", PRIMARY_REFERENCE: "Omitted-label probability",
            "matched_label1_active_uncertainty": "Probability uncertainty", "matched_label1_active_ad": "Chemical similarity",
            "matched_label1_active_random": "Random", "mondrian_lac_empty_to_full": "Mondrian LAC",
            "marginal_lac": "Marginal LAC", "mondrian_aps": "Mondrian APS"}.get(method, method.replace("_", " "))


def _model_label(model: str) -> str:
    return {"logistic": "Logistic", "stack": "Stack", "equal_logit": "Equal-logit",
            "codrug": "CoDrug (printed margins)", "codrug_margin_swap": "CoDrug (swapped margins)",
            "kmm_cp": "KMM-CP"}.get(model, model)


def _reference_label(model: str, method: str) -> str:
    if model in {"codrug", "codrug_margin_swap"}:
        return _model_label(model) + ": " + {"codrug_global": "global", "codrug_mondrian": "candidate class"}.get(method, _method_label(method))
    if model == "kmm_cp":
        prefix = "Selective KMM-CP" if method.startswith("skmm_cp_") else "KMM-CP"
        suffix = method.removeprefix("skmm_cp_").removeprefix("kmm_cp_")
        return prefix + ": " + {"global": "global", "mondrian_candidate": "candidate class",
                              "mondrian_paper_predicted": "predicted group"}.get(suffix, suffix)
    return _method_label(method) if model == "stack" else _model_label(model) + ": " + _method_label(method)


def plot_results(aggregate: pd.DataFrame, paired: pd.DataFrame, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    figures = output / "figures"
    figures.mkdir()
    phase = "retrospective_epa"
    common = aggregate[(aggregate.phase == phase) & (aggregate.panel == "primary_six") &
                       (aggregate.model == "stack") & (aggregate.alpha == .1) &
                       (aggregate.population == "full_eligible_labeled")]
    names = [PRIMARY_METHOD, PRIMARY_REFERENCE, "matched_label1_active_uncertainty", "matched_label1_active_ad", "matched_label1_active_random"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    for name in names:
        row = common[common.method == name]
        if len(row) != 1:
            raise ValueError("figure requires every declared matched-budget method")
        axes[0].scatter(row.operational_defer_rate * 100, row.active_false_clearance * 100, s=42, label=_method_label(name))
    axes[0].set(xlabel="Deferred eligible compounds (%)", ylabel="False-inactive exposure (%)", title="Matched additional-deferral budget")
    axes[0].legend(fontsize=7)
    for prefix, label in [("mondrian_defer_uncertainty_", "Mondrian + uncertainty"),
                          ("mondrian_defer_ad_", "Mondrian + similarity"),
                          ("mondrian_defer_random_", "Mondrian + random"),
                          ("point_abstain_uncertainty_", "Point classifier + uncertainty")]:
        part = common[common.method.str.startswith(prefix)].sort_values("operational_defer_rate")
        axes[1].plot(part.operational_defer_rate * 100, part.active_false_clearance * 100, marker="o", markersize=3, label=label)
    axes[1].set(xlabel="Deferred eligible compounds (%)", ylabel="False-inactive exposure (%)", title="Predeclared workload curves")
    axes[1].legend(fontsize=7)
    for suffix in ["pdf", "png"]:
        fig.savefig(figures / f"error_deferral.{suffix}", dpi=200)
    plt.close(fig)

    forest = paired[(paired.phase == phase) & (paired.metric == "active_false_clearance") &
                    (paired.panel == "primary_six") & (paired.reference == PRIMARY_REFERENCE) &
                    (paired.method == PRIMARY_METHOD) & (paired.alpha == .1)].copy()
    if forest.model.duplicated().any() or set(forest.model) != {"logistic", "stack", "equal_logit"}:
        raise ValueError("predictor figure requires one primary-alpha matched contrast per predictor")
    forest = forest.set_index("model").reindex(["logistic", "stack", "equal_logit"]).reset_index()
    fig, ax = plt.subplots(figsize=(7.0, 3.0), constrained_layout=True)
    y = np.arange(len(forest))
    d = pd.to_numeric(forest.delta) * 100
    low = pd.to_numeric(forest.ci_95_low) * 100
    high = pd.to_numeric(forest.ci_95_high) * 100
    ax.scatter(d, y, color="#245b80", s=36)
    for i, (lo, hi) in enumerate(zip(low, high)):
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [i, i], color="#245b80", linewidth=2)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set(yticks=y, yticklabels=[_model_label(m) for m in forest.model], xlabel="TAME minus omitted-label comparator (percentage points)",
           title="False-inactive exposure: shared-molecule bootstrap")
    for suffix in ["pdf", "png"]:
        fig.savefig(figures / f"predictor_paired_effects.{suffix}", dpi=200)
    plt.close(fig)
    transfer = paired[(paired.phase.isin(["independent_random", "independent_scaffold"])) &
                      (paired.metric == "active_false_clearance") & (paired.reference == PRIMARY_REFERENCE) &
                      (paired.method == PRIMARY_METHOD) & (paired.alpha == .1) &
                      paired.model.isin(["logistic", "stack", "equal_logit"])].copy()
    if transfer.duplicated(["phase", "panel", "model"]).any():
        raise ValueError("additional-family figure has duplicate primary-alpha contrasts")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), constrained_layout=True)
    for phase, ax in zip(["independent_random", "independent_scaffold"], axes):
        part = transfer[transfer.phase == phase].sort_values(["panel", "model"])
        y = np.arange(len(part))
        values = pd.to_numeric(part.delta, errors="coerce") * 100
        lo = pd.to_numeric(part.ci_95_low, errors="coerce") * 100
        hi = pd.to_numeric(part.ci_95_high, errors="coerce") * 100
        ax.scatter(values, y, s=28, color="#245b80")
        for i, (left, right) in enumerate(zip(lo, hi)):
            if np.isfinite(left) and np.isfinite(right):
                ax.plot([left, right], [i, i], color="#245b80", linewidth=1.5)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        labels = [p.replace("endpoint__", "").replace("CYP2D6_Veith", "CYP2D6") + " / " + _model_label(m) for p, m in zip(part.panel, part.model)]
        ax.set(yticks=y, yticklabels=labels, xlabel="Difference in false-negative exposure (pp)",
               title="Fixed random target" if phase.endswith("random") else "Fixed scaffold target")
    for suffix in ["pdf", "png"]:
        fig.savefig(figures / f"additional_assay_families.{suffix}", dpi=200)
    plt.close(fig)
    reference = aggregate[(aggregate.phase == "retrospective_epa") & (aggregate.panel == "primary_six") &
        (aggregate.alpha == .1) & (aggregate.population == "full_eligible_labeled") &
        (~aggregate.model.isin(["logistic", "stack", "equal_logit"]) |
         ((aggregate.model == "stack") & aggregate.method.isin([PRIMARY_METHOD, PRIMARY_REFERENCE])))].copy()
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    for i, row in enumerate(reference.sort_values(["model", "method"]).itertuples()):
        label = _reference_label(row.model, row.method)
        ax.scatter(row.operational_defer_rate * 100, row.active_false_clearance * 100,
                   s=36, marker=["o", "s", "^", "D"][i % 4], label=label)
    ax.set(xlabel="Deferred eligible compounds (%)", ylabel="False-inactive exposure (%)",
           title="Full-reference systems at their native workloads")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, .5), fontsize=6.7)
    for suffix in ["pdf", "png"]:
        fig.savefig(figures / f"full_reference_workload.{suffix}", dpi=200)
    plt.close(fig)
    return [str(p.relative_to(output)) for p in sorted(figures.iterdir())]


def latex_escape(text):
    mapping = {"\\": r"\textbackslash{}", "_": r"\_", "&": r"\&", "%": r"\%", "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}"}
    return "".join(mapping.get(c, c) for c in str(text))


def percent(value, digits=2):
    return "undefined" if value is None or not np.isfinite(float(value)) else f"{100*float(value):.{digits}f}"


def result_text(aggregate: pd.DataFrame, paired: pd.DataFrame, decision: dict, synthetic: dict | None = None) -> tuple[str, str, str]:
    """Render only computed outputs; no invented tables or numeric placeholders."""
    primary = paired[(paired.role == "primary") & (paired.metric == "active_false_clearance")]
    if len(primary) != 1:
        raise ValueError("cannot render a manuscript without the primary result")
    p = primary.iloc[0]
    numeric_sentence = (f"At the same class-specific additional-deferral budget, active-protection TAME had false-inactive exposure of {percent(p.method_value)}\\% "
        f"versus {percent(p.reference_value)}\\% for the omitted-label-probability rule. The TAME-minus-comparator difference was {percent(p.delta)} percentage points "
        f"(shared-molecule bootstrap 95\\% interval, {percent(p.ci_95_low)} to {percent(p.ci_95_high)} points).")
    if decision["evidence"] == "primary_retrospective_targeting_advantage_supported":
        conclusion = "The primary retrospective comparison supports better error targeting than the declared omitted-label-probability rule at the matched budget in this fixed assay panel. It does not establish general superiority across molecular settings or a new coverage guarantee."
    elif decision["evidence"] == "primary_simple_comparator_is_better":
        conclusion = "The primary comparison favors the simpler omitted-label-probability rule. These results do not support an algorithmic advantage for the evaluated TAME expansion; its coverage increase relative to ordinary prediction must be interpreted together with its deferral cost."
    else:
        conclusion = "The primary comparison does not establish improved error targeting beyond the omitted-label-probability rule. A coverage increase relative to ordinary prediction alone is therefore insufficient to support an algorithmic-advantage claim."
    sections = [r"\section{Results}", r"\subsection{Primary comparison at matched deferral}", numeric_sentence,
                "This comparison used the original stack, LAC scores, alpha 0.10, the six previously declared assays, and equal weighting of assay--seed metrics. All new EPA comparisons are retrospective.",
                r"\subsection{Coverage, decisions, and deferral}",
                "Table~\\ref{tab:revision_results} reports the principal shared-predictor comparators and every end-to-end reference configuration. Coverage and error use eligible targets with observed outcomes; operational deferral uses all eligible targets. Unselected compounds in support-selective methods defer as full sets. Selected-population metrics and all secondary methods are reported separately in the machine-readable tables."]
    table = aggregate[(aggregate.phase == "retrospective_epa") & (aggregate.panel == "primary_six") &
                      (aggregate.alpha == .1) & (aggregate.population == "full_eligible_labeled") &
                      ((aggregate.model == "stack") | ~aggregate.model.isin(["logistic", "stack", "equal_logit"]))].copy()
    allowed = {PRIMARY_METHOD, PRIMARY_REFERENCE, "matched_label1_active_uncertainty", "matched_label1_active_ad", "matched_label1_active_random",
               "mondrian_lac_empty_to_full", "marginal_lac", "mondrian_aps", "lr_weighted_mondrian_physchem_descriptors", "lr_weighted_mondrian_component_and_stack_logits"}
    table = table[table.method.isin(allowed) | (table.model != "stack")].sort_values(["model", "method"])
    sections += [r"\begin{table}[p]\centering\small\singlespacing", r"\caption{Retrospective six-assay means at alpha 0.10. All values are percentages; the deferral column uses all eligible targets, including missing outcomes. End-to-end references use their own graph predictors and native budgets, so these rows are not equal-budget wrapper effects.}",
                 r"\label{tab:revision_results}", r"\begin{tabular}{p{0.44\textwidth}rrrr}\toprule",
                 r"Method & \shortstack{False-\\inactive} & \shortstack{Minimum\\coverage} & MacroCSY & Deferral\\\midrule"]
    for row in table.itertuples():
        name = _reference_label(row.model, row.method)
        sections.append(f"{latex_escape(name)} & {percent(row.active_false_clearance)} & {percent(row.minimum_class_coverage)} & {percent(row.macro_csy)} & {percent(getattr(row, 'operational_defer_rate', row.defer_rate))}\\\\")
    sections += [r"\bottomrule\end{tabular}\end{table}",
                 r"\begin{figure}[p]\centering\includegraphics[width=\textwidth]{figures/error_deferral.pdf}",
                 r"\caption{Class-specific decision error and deferral. Left: matched numbers of expanded class-0 and class-1 baseline singletons. Right: predeclared fixed-workload curves. Rates are computed per assay and seed before equal-weight aggregation; The workload axis uses all eligible targets, including missing outcomes; error denominators use observed outcomes only. Thus matched rules share the operational budget even when labeled-subset deferral differs.}\end{figure}",
                 r"\subsection{Predictor sensitivity and full-method comparisons}",
                 "All predeclared predictor, simple-rule, transport-ablation, and full-reference contrasts are included in the paired-effects table, with pointwise exploratory intervals. The three-predictor matched comparison is displayed in Figure~\\ref{fig:revision_predictors}. End-to-end references may change the predictor as well as the uncertainty procedure, so their contrasts are not attributed solely to a post-prediction wrapper.",
                 r"\begin{figure}[htbp]\centering\includegraphics[width=0.95\textwidth]{figures/predictor_paired_effects.pdf}",
                 r"\caption{TAME minus omitted-label-probability differences in false-inactive exposure across the three predeclared predictors. Lower values favor TAME. Intervals resample molecular identities jointly across assays and retain every seed fit.}\label{fig:revision_predictors}\end{figure}"]
    sections += [r"\begin{figure}[p]\centering\includegraphics[width=\textwidth]{figures/full_reference_workload.pdf}",
                 r"\caption{Full-reference systems use their native workloads. Deferral includes all eligible targets, whereas error rates use observed outcomes. Both CoDrug margin settings and every declared KMM interpretation are retained; these points do not isolate equal-budget wrapper effects.}\end{figure}"]
    transfers = paired[(paired.phase.isin(["independent_random", "independent_scaffold"])) &
        (paired.model == "stack") & (paired.method == PRIMARY_METHOD) & (paired.reference == PRIMARY_REFERENCE) &
        (paired.metric == "active_false_clearance") & (paired.alpha == .1)].sort_values(["phase", "panel"])
    sections.append(r"\subsection{Additional assay families}")
    for row in transfers.itertuples():
        endpoint = row.panel.replace("endpoint__", "")
        split = "random" if row.phase.endswith("random") else "scaffold"
        sections.append(f"For {latex_escape(endpoint)} under the fixed {split} split, the stacked-predictor TAME-minus-comparator difference in false-negative exposure was {percent(row.delta)} percentage points "
                        f"(95\\% interval, {percent(row.ci_95_low)} to {percent(row.ci_95_high)}).")
    sections += ["Each endpoint retains its original positive-label meaning. These within-dataset holdout effects are separate sensitivity results and are not pooled into the Tox21 primary estimate.",
                 r"\begin{figure}[p]\centering\includegraphics[width=\textwidth]{figures/additional_assay_families.pdf}",
                 r"\caption{Additional endpoint-family evaluation. Differences compare TAME with its own class-budget-matched omitted-label-probability rule. Endpoint meanings and fixed random/scaffold targets remain distinct; training-seed repeats do not enlarge the biological sample count.}\end{figure}",
                 r"\subsection{Query-weight alignment and discordant decisions}"]
    null = paired[(paired.phase == "retrospective_epa") & (paired.panel == "primary_six") &
                  (paired.metric == "active_false_clearance") & (paired.role == "query_weight_alignment_null")]
    for ungated in [False, True]:
        values = pd.to_numeric(null.loc[null.method.str.startswith("null_ungated_") == ungated, "delta"], errors="coerce")
        if len(values) and values.notna().all():
            sections.append(f"Across the {len(values)} fixed {'ungated' if ungated else 'gated'} query-weight permutations, differences from each control's own matched probability rule ranged from "
                            f"{percent(values.min())} to {percent(values.max())} percentage points. These repeated perturbations diagnose alignment sensitivity and are not independent molecular replications.")
    sections.append("The accompanying discordant-decision tables identify which compounds were deferred only by each rule, separate true positives from true negatives, and retain the observed-budget correction when labels are missing. Source calibration weights and the fitted target-density distribution remain intact in the permutation controls; their results therefore do not identify the effect of removing all target information.")
    sections.append("Every ungated, union, single-view, and threshold-sensitivity expansion is also compared with its own class-budget-matched probability rule. The gate-stratified table reports their excess false-inactive exposure conditional on the original two-view pass/fallback status. These are descriptive conditional cell means, not causal gate effects; direct differences between expansions at different workloads are not interpreted as improved targeting.")
    if synthetic is not None:
        sections += [r"\subsection{Controlled mechanism checks}",
            f"Across {synthetic['replicates']} prespecified simulations, the no-shift and same-covariate concept-shift scenarios had identical prediction-and-audit fingerprints in {synthetic['identical_fingerprint_pairs']} paired replicates. "
            f"For active-protection TAME, their mean class-1 coverages were {percent(synthetic['no_shift_class_1_coverage'])}\\% and {percent(synthetic['concept_shift_class_1_coverage'])}\\%, respectively. "
            "These controlled simulations demonstrate the distinction between unchanged observable covariates and changed outcome mechanisms. They are not additional molecular samples or an estimate of clinical utility.",
            r"\begin{figure}[p]\centering\includegraphics[width=\textwidth]{figures/synthetic_mechanisms.pdf}",
            r"\caption{Prespecified synthetic mechanism checks with a frozen known source posterior. Oracle marginal density ratios are available for overlapping Gaussian covariate shift; no such ratio repairs support loss, and knowledge of the X-density ratio does not restore a changed conditional outcome mechanism. These simulations are reported separately from molecular experiments.}\end{figure}"]
    abstract = ("Prediction-set expansion can improve molecular assay coverage by deferring more compounds. We compared transport-based expansion with simpler rules at identical additional-deferral budgets and original singleton-class composition. "
                "The analysis used common molecular predictions, twelve Tox21 assays, three classical predictors, predeclared alpha and workload grids, complete paper-based shift-aware reference implementations, and separate AMES, hERG, DILI, and CYP2D6 holdouts. " + numeric_sentence + " " + conclusion)
    return "\n\n".join(sections), abstract, conclusion


def write_manuscript(output, aggregate, paired, decision, compile_pdf=True, synthetic=None):
    sections, abstract, conclusion = result_text(aggregate, paired, decision, synthetic)
    latex = output / "latex"
    latex.mkdir()
    shutil.copytree(output / "figures", latex / "figures")
    (latex / "results_generated.tex").write_text(sections + "\n", encoding="utf-8")
    template = MANUSCRIPT.read_text(encoding="utf-8")
    start, end = template.index(r"\section{Results}"), template.index(r"\section{Discussion}")
    manuscript = template[:start] + "\\input{results_generated}\n\n" + template[end:]
    import re
    manuscript = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", lambda _: "\\begin{abstract}\n" + abstract + "\n\\end{abstract}", manuscript, count=1, flags=re.S)
    manuscript = re.sub(r"\\begin\{center\}\s*\\fbox\{\\parbox.*?\\end\{center\}", "", manuscript, count=1, flags=re.S)
    manuscript = re.sub(r"\\pending\{Replace this paragraph.*?\}", lambda _: conclusion, manuscript, count=1, flags=re.S)
    manuscript = manuscript.replace("Revision working draft, 4 November 2026", "Results generated from the completed revision matrix")
    manuscript = manuscript.replace(r"\usepackage{amsmath,amssymb,booktabs,array}", r"\usepackage{amsmath,amssymb,booktabs,array,graphicx}")
    manuscript = manuscript.replace("The revised experimental results remain pending", "The revised experimental results are reported below")
    if r"\pending{" in manuscript:
        # The macro definition is \newcommand{\pending}[1], not a call.
        raise ValueError("unresolved pending result placeholder in generated manuscript")
    (latex / "manuscript_completed.tex").write_text(manuscript, encoding="utf-8")
    # The response is generated from the same computed primary result and
    # named methods, so execution does not leave manuscript filling to a user.
    response = """# Response to review of CHEMOLAB-D-26-01118

The revision changes the main question to error targeting at an identical deferral budget. The marginal/class-conditional coverage distinction is acknowledged as established theory, and no new validity theorem is claimed. Historical Stage I numerical examples are excluded from the revised evidence because their row-level outputs could not be recovered for a finite-sample quantile correction. All new predictions use the specified correct implementation.

The primary comparator ranks the omitted label's probability within each original singleton class. TAME and this comparator have exactly the same class-specific expansion counts on all eligible target structures. Uncertainty, chemical similarity, random deferral, full conformal reference systems, model and score sensitivity, and gate/weight-alignment controls are included in the declared matrix. The complete method tables retain unfavorable results and both published and swapped CoDrug margin orientations.

PRIMARY_RESULT

The label-0 historical method and new label-1 objective are distinguished. Tox21 labels represent assay activity; AMES, standard hERG, DILI and CYP2D6 retain their own positive-class meanings. Full-cohort decision metrics and selected-population coverage are reported separately. Repeated training seeds are not treated as independent compounds; intervals resample molecular identities jointly across assays and methods.

The full CoDrug and KMM-CP code consists of independent paper-based implementations, not the unavailable author repositories. The CoDrug energy loss, auxiliary molecules, AttentiveFP training and KDE are included. KMM-CP includes bandwidth selection, ordinary and joint selective optimization, post-selection refitting, and both candidate-label and literal predicted-label interpretations of the paper's Mondrian ambiguity. Model configurations, numerical choices and outputs are recorded.

The primary result and every secondary comparison are available in all_paired_contrasts.csv; the complete assay/seed/method tables are in all_cell_metrics.csv and all_panel_metrics.csv. The generated manuscript includes the real results and figures. The EPA comparisons are retrospective because its labels had already been accessed. Additional endpoint-family holdouts are not presented as a newly collected or blinded cohort. Authors should review the generated interpretation and declarations before submitting this as a new manuscript.
""".replace("PRIMARY_RESULT", abstract.replace(r"\%", "%"))
    (output / "response_generated.md").write_text(response, encoding="utf-8")
    cover = """Dear Editor,

Please consider the substantially revised manuscript, “Class-Specific Errors and Deferral under Molecular Distribution Shift: A Budget-Controlled Comparison of Conformal and Abstention Rules,” as a new submission following CHEMOLAB-D-26-01118.

The scientific question now concerns class-specific error targeting at an identical deferral budget. The established marginal-versus-class-conditional coverage distinction is background, and no new coverage-validity theorem is claimed. The study includes a decision-specific probability comparator, shared-predictor conformal and abstention methods, complete paper-based CoDrug and KMM-CP implementations, own-budget transport ablations, query-weight alignment controls, controlled mechanism simulations, and separately reported AMES, standard hERG, DILI and CYP2D6 endpoint-family holdouts. EPA comparisons are explicitly retrospective. The revised methods distinguish assay outcomes from clinical toxicity and preserve molecular dependence in paired inference.

RESULT

The accompanying response identifies the changes and their evidence. The method implementations and complete result tables are provided with the manuscript's repository.

Sincerely,
Siyuan Tong, on behalf of the authors

[Generated draft for author review. Confirm author approval, journal declarations and final files before submission. This letter has not been sent.]
""".replace("RESULT", abstract.replace(r"\%", "%"))
    (output / "cover_letter_generated.md").write_text(cover, encoding="utf-8")
    if compile_pdf:
        if shutil.which("pdflatex") is None:
            return {"source": "latex/manuscript_completed.tex", "pdf_status": "pdflatex_unavailable_source_complete"}
        for _ in range(2):
            proc = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "manuscript_completed.tex"],
                                  cwd=latex, text=True, capture_output=True)
            (latex / "build.log").write_text(proc.stdout + proc.stderr)
            if proc.returncode:
                raise RuntimeError("generated manuscript compilation failed; inspect latex/build.log")
        if "undefined references" in proc.stdout.lower() or "undefined citations" in proc.stdout.lower():
            raise RuntimeError("generated manuscript has unresolved references")
        return {"source": "latex/manuscript_completed.tex", "pdf": "latex/manuscript_completed.pdf", "pdf_status": "compiled"}
    return {"source": "latex/manuscript_completed.tex", "pdf_status": "not_requested"}


def read_synthetic(run_root: Path, protocol: dict):
    """Validate the separate simulation matrix; never call it molecular evidence."""
    directory = run_root / "synthetic_mechanisms"
    file_names = ["report.json", "replicate_metrics.csv", "transport_audits.csv", "descriptive_summary.csv"]
    if not all((directory / name).is_file() for name in file_names):
        raise ValueError("complete synthetic_mechanisms outputs are required before final reporting")
    report = json.loads((directory / "report.json").read_text())
    expected = protocol.get("synthetic_mechanisms", {"replicates": 30, "seed": 73000,
        "n_calibration": 400, "n_target": 800, "alpha": .1})
    for key in ["replicates", "n_calibration", "n_target", "alpha"]:
        if report[key] != expected[key]:
            raise ValueError("synthetic configuration differs from protocol: " + key)
    if report["first_seed"] != expected["seed"]:
        raise ValueError("synthetic seed sequence differs from protocol")
    frame = pd.read_csv(directory / "replicate_metrics.csv")
    from .synthetic_mechanisms import SCENARIOS
    if set(frame.scenario) != set(SCENARIOS) or frame.duplicated(["scenario", "seed", "method"]).any():
        raise ValueError("synthetic scenario matrix is invalid")
    seeds = set(range(expected["seed"], expected["seed"] + expected["replicates"]))
    expected_methods = {"mondrian_lac_empty_to_full", "tame_label1_active",
        "matched_tame_label1_active_omitted_label_probability", "tame_label1_ungated",
        "matched_tame_label1_ungated_omitted_label_probability", "estimated_ratio_linear_mondrian",
        "estimated_ratio_quadratic_mondrian"}
    for scenario in SCENARIOS:
        expected_method_names = expected_methods | ({"oracle_x_ratio_mondrian"} if scenario != "support_loss" else set())
        if set(frame.loc[frame.scenario == scenario, "method"]) != expected_method_names:
            raise ValueError("synthetic method matrix incomplete: " + scenario)
    for _, group in frame.groupby(["scenario", "method"]):
        if set(group.seed) != seeds:
            raise ValueError("synthetic method has missing or extra seeds")
    audits = pd.read_csv(directory / "transport_audits.csv")
    expected_audits = {(scenario, seed, view) for scenario in SCENARIOS for seed in seeds for view in ["linear", "quadratic"]}
    if (audits.duplicated(["scenario", "seed", "view"]).any() or
            set(audits[["scenario", "seed", "view"]].itertuples(index=False, name=None)) != expected_audits):
        raise ValueError("synthetic audit matrix incomplete")
    # No-shift and changed-conditional data were deliberately paired in X.
    a = frame[(frame.scenario == "no_shift") & (frame.method == PRIMARY_METHOD)].sort_values("seed")
    b = frame[(frame.scenario == "same_x_concept_shift") & (frame.method == PRIMARY_METHOD)].sort_values("seed")
    if len(a) != expected["replicates"] or len(b) != len(a):
        raise ValueError("synthetic paired mechanism comparison is incomplete")
    same = int(np.sum(a.label_free_fingerprint.to_numpy() == b.label_free_fingerprint.to_numpy()))
    if same != expected["replicates"]:
        raise ValueError("synthetic same-X mechanisms no longer share predictions/audits")
    summary = {"replicates": expected["replicates"], "identical_fingerprint_pairs": same,
        "no_shift_class_1_coverage": float(a.class_1_coverage.mean()),
        "concept_shift_class_1_coverage": float(b.class_1_coverage.mean()),
        "input_sha256": {name: sha256(directory / name) for name in file_names},
        "interpretation": "Controlled simulations only; separate from all molecular estimands."}
    return frame, summary


def plot_synthetic(frame: pd.DataFrame, output: Path):
    import matplotlib.pyplot as plt
    from .synthetic_mechanisms import SCENARIOS
    metrics = ["class_1_coverage", "deferred_rate"]
    grouped = frame.groupby(["scenario", "method"])[metrics].mean().reset_index()
    grouped.to_csv(output / "synthetic_mechanism_means.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1), constrained_layout=True)
    labels = ["No shift", "Gaussian shift", "Support loss", "Same X; changed Y|X"]
    for method, label in [("mondrian_lac_empty_to_full", "Mondrian"), (PRIMARY_METHOD, "TAME"),
                          ("oracle_x_ratio_mondrian", "Known X-density ratio")]:
        part = grouped[grouped.method == method].set_index("scenario").reindex(SCENARIOS)
        for ax, metric in zip(axes, metrics):
            ax.plot(np.arange(4), part[metric].to_numpy() * 100, marker="o", label=label)
    for ax, label in zip(axes, ["Class-1 coverage (%)", "Deferral (%)"]):
        ax.set(xticks=np.arange(4), xticklabels=labels, ylabel=label)
        ax.tick_params(axis="x", rotation=15)
        ax.legend(fontsize=8)
    for suffix in ["pdf", "png"]:
        fig.savefig(output / "figures" / f"synthetic_mechanisms.{suffix}", dpi=200)
    plt.close(fig)


def build_report(run_root: Path, output: Path, *, protocol: dict | None = None,
                 replicates: int | None = None, compile_pdf: bool = True) -> dict:
    run_root, output = Path(run_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("report output must be new; previous results are never overwritten")
    manifest_path = run_root / "required_jobs.json"
    if not manifest_path.is_file():
        raise ValueError("required_jobs.json is missing; cannot infer a matrix from successful outputs")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("execution_kind") == "diagnostic_pilot":
        raise ValueError("a diagnostic pilot cannot generate a completed-study manuscript")
    protocol_path = run_root / "protocol_snapshot.json"
    if not protocol_path.is_file():
        protocol_path = HERE / "protocol.json"
    snapshot = json.loads(protocol_path.read_text())
    if protocol is not None and protocol != snapshot:
        raise ValueError("supplied protocol differs from the run snapshot")
    protocol = snapshot
    if manifest.get("protocol_sha256") and manifest["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("protocol snapshot hash does not match declared jobs")
    jobs = manifest["jobs"]
    if not jobs or len({j["job_id"] for j in jobs}) != len(jobs):
        raise ValueError("empty or duplicate declared jobs")
    if len({j["predictions"] for j in jobs}) != len(jobs):
        raise ValueError("each declared job must have a distinct prediction file")
    families = {j.get("implementation", j.get("model")) for j in jobs if j.get("family") == "full_reference"}
    if not {"codrug", "kmm_cp"}.issubset(families):
        raise ValueError("complete CoDrug and KMM-CP jobs are required; component or external import results are insufficient")
    contrasts = declared_contrasts(protocol, jobs)
    phases = sorted({j["phase"] for j in jobs})
    all_endpoints = sorted({j["endpoint"] for j in jobs})
    # Ensure that a subset run cannot masquerade as the formal matrix. Every
    # phase uses its own assay family and fixed target cohort.
    from paper2_admet_benchmark.scripts.racer_c4.racer_c4_io import ENDPOINT_PROPERTIES
    independent_lock = json.loads((HERE / "independent_sources.json").read_text())
    phase_specs = {p: {"endpoints": list(ENDPOINT_PROPERTIES), "seeds": protocol[f"{p}_seeds"]}
                   for p in ["development", "retrospective_epa"]}
    phase_specs.update({p: {"endpoints": list(independent_lock["sources"]), "seeds": independent_lock["split_contract"]["fit_seeds"]}
                       for p in ["independent_random", "independent_scaffold"]})
    if set(phases) != set(phase_specs):
        raise ValueError("formal report requires both historical phases and both additional-family split phases")
    for phase, spec in phase_specs.items():
        actual = {j["endpoint"] for j in jobs if j["phase"] == phase}
        if actual != set(spec["endpoints"]):
            raise ValueError("phase-specific endpoint matrix incomplete: " + phase)
        saved = manifest.get("phase_specs", {}).get(phase)
        if saved is not None and (set(saved["endpoints"]) != set(spec["endpoints"]) or set(saved["seeds"]) != set(spec["seeds"])):
            raise ValueError("manifest phase differs from locked scientific matrix: " + phase)
    panels = {"primary_six": protocol["primary_comparison"]["endpoint_panel"], "all_assays": list(ENDPOINT_PROPERTIES)}
    panels.update({"endpoint__" + e: [e] for e in all_endpoints})
    expected_cells = {(phase, endpoint, int(seed), model) for phase, spec in phase_specs.items() for endpoint in spec["endpoints"]
                      for seed in spec["seeds"] for model in protocol["models"]}
    actual_cells = {(j["phase"], j["endpoint"], int(j["seed"]), j["model"]) for j in jobs if j.get("family", "shared") == "shared"}
    if actual_cells != expected_cells:
        raise ValueError("shared predictor endpoint/seed matrix is incomplete or contains extra jobs")
    if not {"codrug", "codrug_margin_swap", "kmm_cp"}.issubset(families):
        raise ValueError("both original-margin and swapped-margin CoDrug runs plus KMM-CP are required")
    for family in sorted(families):
        subset = [j for j in jobs if j.get("family") == "full_reference" and j.get("implementation", j["model"]) == family]
        expected = {(p, e, int(s)) for p, spec in phase_specs.items() for e in spec["endpoints"] for s in spec["seeds"]}
        if {(j["phase"], j["endpoint"], int(j["seed"])) for j in subset} != expected:
            raise ValueError("full-reference endpoint/seed matrix is incomplete: " + family)
    for job in jobs:
        if set(map(float, job["alphas"])) != set(map(float, protocol["alpha_grid"])):
            raise ValueError("formal report requires the complete declared alpha grid")
    # Re-enumerate from the actual execution API, not merely from successful
    # manifest columns. Dropping a whole competitor must not appear complete.
    from . import execute_revision, independent_data
    from .full_baselines import codrug, kmm_cp
    locked_jobs = execute_revision.build_required_jobs(protocol,
        execute_revision.case_specs(protocol, independent_data.case_specs()),
        codrug.METHOD_NAMES, kmm_cp.METHOD_NAMES)
    if {j["job_id"] for j in jobs} != {j["job_id"] for j in locked_jobs}:
        raise ValueError("job identities differ from the execution API's complete matrix")
    locked = {j["job_id"]: j for j in locked_jobs}
    for job in jobs:
        if set(job["methods"]) != set(locked[job["job_id"]]["methods"]):
            raise ValueError("method inventory differs from the execution API: " + job["job_id"])
    needed = {(c.phase, model, method, c.alpha) for c in contrasts for model, method in [(c.model, c.method), (c.reference_model, c.reference)]}
    all_metrics, inference, receipts, cohorts, failures, gates = [], [], [], {}, [], []
    for job in jobs:
        try:
            frame, receipt = verify_job(run_root, job)
            if job["family"] == "shared":
                audit_path = confined(run_root, job["completion"]).parent / "audits.json"
                audit = json.loads(audit_path.read_text())
                from .runner import VIEWS
                if not set(VIEWS).issubset(audit):
                    raise ValueError("original two-view gate diagnostics incomplete")
                if any(not isinstance(audit[v].get("active"), bool) for v in VIEWS):
                    raise ValueError("invalid original gate activation flag")
                gates.append({k: job[k] for k in ["phase", "endpoint", "seed", "model"]} |
                    {"original_gate_status": "both_pass" if all(audit[v]["active"] for v in VIEWS) else "fallback"})
        except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
            failures.extend({"job_id": job["job_id"], "phase": job["phase"], "endpoint": job["endpoint"],
                "seed": job["seed"], "model": job["model"], "alpha": alpha, "method": method,
                "error": str(exc)} for alpha in job["alphas"] for method in job["methods"])
            continue
        receipts.append(receipt)
        facts = _facts(frame[(frame.alpha == frame.alpha.iloc[0]) & (frame.method == frame.method.iloc[0])])
        key = (job["phase"], job["endpoint"])
        if key in cohorts and not facts.equals(cohorts[key]):
            raise ValueError("target identities, labels or eligibility change across seed/predictor jobs")
        cohorts.setdefault(key, facts)
        all_metrics.extend(recompute_metrics(frame))
        mask = np.zeros(len(frame), dtype=bool)
        for alpha in job["alphas"]:
            keep_methods = {m for p, model, m, a in needed if p == job["phase"] and model == job["model"] and a == float(alpha)}
            mask |= (frame.alpha == float(alpha)) & frame.method.isin(keep_methods)
        extra = [c for c in ["probability_1", "max_train_tanimoto"] if c in frame]
        inference.append(frame.loc[mask, IDENTITY + ["sample_id", "structure_id", "y", "eligible"] + FLAGS + extra])
    if failures:
        import datetime
        failure_dir = run_root / "analysis_failures"
        failure_dir.mkdir(exist_ok=True)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        failure_file = failure_dir / (stamp + "_incomplete_matrix.csv")
        pd.DataFrame(failures).to_csv(failure_file, index=False)
        raise ValueError(f"{len(failures)} declared endpoint/seed/alpha/method entries failed verification; complete failure list: {failure_file}")
    synthetic_frame, synthetic = read_synthetic(run_root, protocol)
    cells = pd.DataFrame(all_metrics)
    aggregate = aggregate_metrics(cells, panels)
    infer = pd.concat(inference, ignore_index=True)
    del inference
    # Low-cardinality categories reduce memory without changing the molecular
    # identity strings used by the bootstrap or the endpoint group structure.
    for column in ["phase", "model", "method", "sample_id", "structure_id"]:
        infer[column] = infer[column].astype("category")
    n_boot = replicates if replicates is not None else int(protocol["statistical_analysis"]["bootstrap_replicates"])
    if n_boot != int(protocol["statistical_analysis"]["bootstrap_replicates"]):
        raise ValueError("formal report bootstrap repetitions must match the protocol")
    paired = compute_contrasts(infer, contrasts, panels, n_boot, int(protocol["statistical_analysis"]["bootstrap_seed"]))
    decision = interpret_primary(paired, matrix_complete=True, reference_complete=True)
    # No output is created until every input and the complete matrix passed.
    output.mkdir(parents=True)
    cells.to_csv(output / "all_cell_metrics.csv", index=False)
    aggregate.to_csv(output / "all_panel_metrics.csv", index=False)
    paired_df = pd.DataFrame(paired)
    paired_df.to_csv(output / "all_paired_contrasts.csv", index=False)
    gate_stratified_targeting(cells, gates).to_csv(output / "gate_stratified_own_budget.csv", index=False)
    write_json(output / "comparison_plan.json", [asdict(c) for c in contrasts])
    write_json(output / "decision.json", decision)
    from .mechanism_diagnostics import analyze_predictions, write_diagnostics
    mechanisms = {}
    for phase, spec in phase_specs.items():
        # Invalid structures have no molecular identity and are excluded from
        # all analyses; keep every eligible row, including unknown outcomes.
        selected = infer[(infer.phase == phase) & (infer.eligible == 1) & (infer.model == "stack")]
        diagnostic = analyze_predictions(selected, phase=phase, endpoints=spec["endpoints"])
        mechanisms[phase] = write_diagnostics(diagnostic, output / "mechanism" / phase)
    figures = plot_results(aggregate, paired_df, output)
    plot_synthetic(synthetic_frame, output)
    figures += ["figures/synthetic_mechanisms.pdf", "figures/synthetic_mechanisms.png"]
    manuscript = write_manuscript(output, aggregate, paired_df, decision, compile_pdf, synthetic)
    report = {"status": "COMPLETE_MATRIX_ANALYZED", "matrix_jobs": len(jobs), "phases": phases,
              "protocol_sha256": sha256(protocol_path), "required_jobs_sha256": sha256(manifest_path),
              "reporting_source_sha256": sha256(Path(__file__)), "inference_source_sha256": sha256(HERE / "summarize.py"),
              "input_receipts": receipts, "decision": decision, "figures": figures, "manuscript": manuscript,
              "mechanism_diagnostics": mechanisms, "phase_specs": phase_specs,
              "synthetic_mechanisms": synthetic,
              "interpretation": "Retrospective EPA analysis. One primary comparison; secondary intervals are pointwise exploratory. Seeds are averaged as repeated fits, not independent observations."}
    write_json(output / "report_complete.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args(argv)
    result = build_report(args.run_root, args.output, compile_pdf=not args.no_compile)
    print(json.dumps({"status": result["status"], "decision": result["decision"]["decision"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
