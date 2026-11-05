"""Prespecified query-weight alignment controls, never using target outcomes.

These controls destroy the association of a query's density-ratio weight with
its identity within fixed probability bins. They do NOT destroy the estimated
target distribution, calibration reweighting, or the cross-view dependence.
The same permutation is used across views and all rules get their own exact
class-specific matched-budget probability comparator.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np

from paper2_admet_benchmark.revision_2026 import core
from paper2_admet_benchmark.scripts.racer_c4.racer_c4_core import (
    binary_nonconformity, transport_envelope, weighted_mondrian_sets,
)


def query_permutation(probability_1, ids, *, seed, edges=(0, .2, .4, .6, .8, 1)):
    """Return row-order-invariant donor indices, stratified by fixed p bins."""
    p = np.asarray(probability_1, dtype=float)
    names = np.asarray(ids, dtype=str)
    edges = np.asarray(edges, dtype=float)
    if p.ndim != 1 or names.shape != p.shape or len(set(names)) != len(names):
        raise ValueError("probabilities and unique IDs must be aligned vectors")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be finite and in [0,1]")
    if len(edges) < 2 or edges[0] != 0 or edges[-1] != 1 or np.any(np.diff(edges) <= 0):
        raise ValueError("fixed bin edges must increase from 0 to 1")
    strata = np.searchsorted(edges[1:-1], p, side="right")
    result = np.arange(len(p))
    for group in range(len(edges) - 1):
        indices = np.flatnonzero(strata == group)
        canonical = sorted(indices, key=lambda i: names[i])
        # Hash ordering avoids dependence on input order and NumPy RNG version.
        donors = sorted(canonical, key=lambda i: hashlib.sha256(
            f"query-weight-null:{int(seed)}:{names[i]}".encode()).digest())
        result[canonical] = donors
    return result, strata


def method_names(replicates=5):
    output = []
    for index in range(int(replicates)):
        for mode in ("", "ungated_"):
            method = f"null_{mode}query_permutation_{index}"
            output.extend((method, f"matched_{method}_omitted_label_probability"))
    return output


def generate_controls(cal_p, cal_y, target_p, ids, baseline, ratios, audits,
                      views, *, alpha, seed, replicates=5,
                      edges=(0, .2, .4, .6, .8, 1)):
    """Return method->sets and diagnostics; no query labels accepted."""
    if int(replicates) < 1:
        raise ValueError("replicates must be positive")
    p = np.asarray(target_p, dtype=float)
    base = np.asarray(baseline, dtype=bool)
    if base.shape != (len(p), 2) or np.any(base.sum(axis=1) == 0):
        raise ValueError("baseline must be aligned and empty-repaired")
    if len(views) != 2:
        raise ValueError("TAME control requires the two locked views")
    priority = np.where(base[:, 0], p, 1 - p)
    output, diagnostic = {}, []
    for index in range(int(replicates)):
        permutation, strata = query_permutation(p, ids, seed=int(seed) + 79000 + index, edges=edges)
        weighted = [weighted_mondrian_sets(binary_nonconformity(cal_p), cal_y,
            binary_nonconformity(p), ratios[v].source_weights,
            np.asarray(ratios[v].target_weights)[permutation], alpha) for v in views]
        for mode in ("", "ungated_"):
            selected_audits = [audits[v] if not mode else replace(audits[v], active=True,
                reason="query_null_ungated_diagnostic") for v in views]
            name = f"null_{mode}query_permutation_{index}"
            output[name] = transport_envelope(base, weighted, selected_audits,
                protected_labels=[1], quorum="all", minimum_active_views=2).membership
            output[f"matched_{name}_omitted_label_probability"] = core.matched_budget_deferral(
                base, output[name], priority, ids, match_classes=True)
        diagnostic.append({"replicate": index, "seed": int(seed) + 79000 + index,
            "fraction_changed_identity": float(np.mean(permutation != np.arange(len(p)))) if len(p) else 0.,
            "bin_sizes": np.bincount(strata, minlength=len(edges) - 1).tolist()})
    return output, {"control": "query_weight_identity_alignment_within_fixed_probability_bins",
        "not_destroyed": ["source_calibration_weights", "target_density_fit", "joint_view_weight_distribution"],
        "fixed_edges": list(edges), "replicates": diagnostic}
