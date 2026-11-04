"""Numerical comparisons for the CHEMOLAB revision.

The methods here measure a coverage/deferral trade-off; they are not new
coverage theorems. APS uses deterministic cumulative candidate scores (ties
are ordered by label), not a randomized APS implementation. Deferral rules
receive covariates or model predictions only, never evaluation labels.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence

import numpy as np

from paper2_admet_benchmark.scripts.racer_c4.racer_c4_core import (
    finite_sample_quantile,
    ordinary_mondrian_sets,
    transport_envelope,
)


def _probabilities(values: np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be a finite one-dimensional vector in [0,1]")
    return p


def _labels(values: np.ndarray, n: int | None = None) -> np.ndarray:
    y = np.asarray(values)
    if y.ndim != 1 or (n is not None and len(y) != n):
        raise ValueError("labels must be a one-dimensional aligned vector")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be exactly 0 or 1; missing values are not negatives")
    return y.astype(np.int8)


def _sets(values: np.ndarray) -> np.ndarray:
    member = np.asarray(values)
    if member.ndim != 2 or member.shape[1] != 2 or not np.isin(member, [0, 1]).all():
        raise ValueError("sets must be a binary matrix with shape (n,2)")
    return member.astype(bool)


def _alpha(alpha: float | Mapping[int, float]) -> dict[int, float]:
    if isinstance(alpha, Mapping):
        if set(alpha) != {0, 1}:
            raise ValueError("class-specific alpha must have exactly keys 0 and 1")
        values = {k: float(alpha[k]) for k in (0, 1)}
    else:
        values = {0: float(alpha), 1: float(alpha)}
    if any(not math.isfinite(v) or not 0 < v < 1 for v in values.values()):
        raise ValueError("every alpha must be finite and strictly between zero and one")
    return values


def candidate_scores(probability_1: np.ndarray, score: str = "lac") -> np.ndarray:
    """LAC or deterministic APS scores for each candidate, columns [0,1]."""
    p = _probabilities(probability_1)
    probability = np.column_stack((1 - p, p))
    if score == "lac":
        return 1 - probability
    if score != "aps":
        raise ValueError("score must be lac or aps")
    order = np.argsort(-probability, axis=1, kind="stable")
    cumulative = np.cumsum(np.take_along_axis(probability, order, axis=1), axis=1)
    result = np.empty_like(probability)
    np.put_along_axis(result, order, cumulative, axis=1)
    return result


def conformal_sets(
    calibration_probability: np.ndarray,
    calibration_labels: np.ndarray,
    query_probability: np.ndarray,
    alpha: float | Mapping[int, float] = 0.1,
    method: str = "mondrian_lac",
) -> np.ndarray:
    """Split CP with finite-sample rank correction, without empty-set repair.

    Missing calibration classes receive an infinite threshold. Mondrian
    calibration selects a threshold by candidate label, never predicted label.
    Exchangeability assumptions are needed for nominal coverage statements;
    these procedures do not guarantee coverage under arbitrary shift.
    """
    cal = _probabilities(calibration_probability)
    y = _labels(calibration_labels, len(cal))
    query = _probabilities(query_probability)
    alphas = _alpha(alpha)
    if method not in {"marginal_lac", "mondrian_lac", "marginal_aps", "mondrian_aps"}:
        raise ValueError(f"unknown conformal method: {method}")
    scope, score = method.split("_")
    calibration_scores = candidate_scores(cal, score)
    query_scores = candidate_scores(query, score)
    if scope == "mondrian":
        return ordinary_mondrian_sets(calibration_scores, y, query_scores, alphas)
    if alphas[0] != alphas[1]:
        raise ValueError("marginal CP requires a common alpha")
    true_scores = calibration_scores[np.arange(len(y)), y]
    threshold = finite_sample_quantile(true_scores, alphas[0])
    return query_scores <= threshold


def repair_empty(membership: np.ndarray) -> np.ndarray:
    """Map empty sets to the full set while preserving every nonempty set."""
    result = _sets(membership).copy()
    result[result.sum(axis=1) == 0] = True
    return result


def full_sets(n: int) -> np.ndarray:
    if isinstance(n, bool) or int(n) != n or n < 0:
        raise ValueError("n must be a nonnegative integer")
    return np.ones((int(n), 2), dtype=bool)


def _ids(sample_ids: Sequence[str], n: int) -> np.ndarray:
    ids = np.asarray(sample_ids, dtype=str)
    if ids.ndim != 1 or len(ids) != n or len(set(ids)) != n or np.any(ids == ""):
        raise ValueError("sample IDs must be nonempty, unique, and aligned")
    return ids


def deferral_priority(
    probability_1: np.ndarray,
    sample_ids: Sequence[str],
    kind: str = "uncertainty",
    similarity: np.ndarray | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Return priorities, largest first, without using target outcomes.

    Hash-based random priorities are invariant to input row order. AD uses
    maximum training-set similarity in [0,1], supplied by the caller.
    """
    p = _probabilities(probability_1)
    ids = _ids(sample_ids, len(p))
    if kind == "uncertainty":
        return 1 - np.abs(2 * p - 1)
    if kind == "ad":
        if similarity is None:
            raise ValueError("ad priority requires training-set similarity")
        sim = _probabilities(similarity)
        if len(sim) != len(p):
            raise ValueError("similarities must align to probabilities")
        return 1 - sim
    if kind == "random":
        return np.asarray([
            int.from_bytes(hashlib.sha256(f"{int(seed)}\0{s}".encode()).digest()[:8], "big")
            / 2**64 for s in ids
        ])
    raise ValueError("priority kind must be uncertainty, ad, or random")


def _budget(value: int, available: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("a deferral budget must be an integer count")
    if not 0 <= int(value) <= available:
        raise ValueError("deferral budget is outside the available singleton count")
    return int(value)


def fixed_budget_deferral(
    baseline: np.ndarray,
    budget: int | Mapping[int, int],
    priority: np.ndarray,
    sample_ids: Sequence[str],
) -> np.ndarray:
    """Expand exactly a specified number of baseline singletons to {0,1}.

    An integer matches a total budget. A mapping {0:k0,1:k1} separately
    matches expansion counts from the original predicted singleton classes.
    Empty sets are repaired before defining the eligible singleton pool.
    """
    base = repair_empty(baseline)
    ids = _ids(sample_ids, len(base))
    rank = np.asarray(priority, dtype=float)
    if rank.shape != (len(base),) or not np.isfinite(rank).all():
        raise ValueError("priority must be a finite aligned vector")
    singleton = base.sum(axis=1) == 1
    if isinstance(budget, Mapping):
        if set(budget) != {0, 1}:
            raise ValueError("class budgets must have exactly keys 0 and 1")
        pools = [(singleton & base[:, c], budget[c]) for c in (0, 1)]
    else:
        pools = [(singleton, budget)]
    output = base.copy()
    for pool, count in pools:
        candidates = np.flatnonzero(pool)
        k = _budget(count, len(candidates))
        # sample ID is an explicit stable tie breaker, independent of row order.
        ordered = candidates[np.lexsort((ids[candidates], -rank[candidates]))]
        output[ordered[:k]] = True
    return output


def matched_budget_deferral(
    baseline: np.ndarray,
    reference: np.ndarray,
    priority: np.ndarray,
    sample_ids: Sequence[str],
    match_classes: bool = True,
) -> np.ndarray:
    """Match reference expansion counts without accessing target labels.

    Default matching is within each *predicted* singleton class. This prevents
    a label-1 protection rule from being compared with a random rule that
    spends its budget on a different mixture of {0} and {1} predictions.
    """
    base = repair_empty(baseline)
    ref = _sets(reference)
    if ref.shape != base.shape or np.any(base & ~ref):
        raise ValueError("reference must contain the empty-repaired baseline")
    expanded = (base.sum(axis=1) == 1) & (ref.sum(axis=1) == 2)
    budget = ({c: int(np.sum(expanded & base[:, c])) for c in (0, 1)}
              if match_classes else int(expanded.sum()))
    return fixed_budget_deferral(base, budget, priority, sample_ids)


def point_abstention(
    probability_1: np.ndarray,
    budget: int,
    priority: np.ndarray,
    sample_ids: Sequence[str],
) -> np.ndarray:
    """Non-conformal argmax classification with a fixed abstention count."""
    p = _probabilities(probability_1)
    predicted = (p > 0.5).astype(int)  # ties go to 0, consistently with APS.
    member = np.eye(2, dtype=bool)[predicted]
    return fixed_budget_deferral(member, budget, priority, sample_ids)


def tame_variants(
    baseline: np.ndarray,
    weighted_sets: Sequence[np.ndarray],
    audits: Sequence[object],
    quorum: str = "all",
    minimum_active_views: int = 2,
) -> dict[str, np.ndarray]:
    """Historical label 0, assay-active label 1, and symmetric label protection.

    Label 1 denotes assay activity, not an established clinical toxicity or
    screening cost. The label-1 and symmetric variants are new comparisons,
    not retrospectively sealed variants of the original method.
    """
    base = _sets(baseline)
    return {
        name: transport_envelope(
            base, weighted_sets, audits, protected_labels=labels,
            quorum=quorum, minimum_active_views=minimum_active_views,
        ).membership
        for name, labels in (("tame_label0_historical", (0,)),
                             ("tame_label1", (1,)), ("tame_both", (0, 1)))
    }


def decision_metrics(targets: np.ndarray, membership: np.ndarray) -> dict[str, float | int]:
    """Full denominators for set coverage and two illustrative assay decisions.

    'Clear' means output {0}; 'discard' means output {1}. These are analytical
    labels, not clinical recommendations. Empty/full sets both defer action.
    If a true class is absent, all two-class aggregate metrics are NaN.
    """
    member = _sets(membership)
    y = _labels(targets, len(member))
    n = len(y)
    size = member.sum(axis=1)
    singleton = size == 1
    clear = singleton & member[:, 0]
    discard = singleton & member[:, 1]
    correct = singleton & member[np.arange(n), y]
    wrong = singleton & ~member[np.arange(n), y]
    ratio = lambda count, denom: float(count / denom) if denom else math.nan
    out: dict[str, float | int] = {
        "n": n, "n_singleton": int(singleton.sum()), "n_clear": int(clear.sum()),
        "n_discard": int(discard.sum()), "n_deferred": int((~singleton).sum()),
        "n_empty": int((size == 0).sum()), "n_ambiguous": int((size == 2).sum()),
        "overall_coverage": ratio(member[np.arange(n), y].sum(), n),
        "singleton_rate": ratio(singleton.sum(), n), "deferred_rate": ratio((~singleton).sum(), n),
        "empty_rate": ratio((size == 0).sum(), n), "ambiguous_rate": ratio((size == 2).sum(), n),
        "mean_set_size": ratio(size.sum(), n), "clear_yield": ratio(clear.sum(), n),
        "discard_yield": ratio(discard.sum(), n),
        "clearance_contamination": ratio(np.sum(clear & (y == 1)), clear.sum()),
        "discard_contamination": ratio(np.sum(discard & (y == 0)), discard.sum()),
        "accepted_error_rate": ratio(wrong.sum(), singleton.sum()),
        "correct_singleton_yield": ratio(correct.sum(), n),
        "wrong_singleton_exposure": ratio(wrong.sum(), n),
        "singleton_tn": int(np.sum(clear & (y == 0))),
        "singleton_fn": int(np.sum(clear & (y == 1))),
        "singleton_tp": int(np.sum(discard & (y == 1))),
        "singleton_fp": int(np.sum(discard & (y == 0))),
    }
    for c in (0, 1):
        mask = y == c
        count = int(mask.sum())
        out.update({
            f"class_{c}_n": count,
            f"class_{c}_coverage": ratio(member[mask, c].sum(), count),
            f"class_{c}_csy": ratio(np.sum(correct & mask), count),
            f"class_{c}_wrong_singleton_exposure": ratio(np.sum(wrong & mask), count),
            f"class_{c}_singleton_rate": ratio(np.sum(singleton & mask), count),
            f"class_{c}_accepted_error_rate": ratio(np.sum(wrong & mask), np.sum(singleton & mask)),
        })
    present = all(out[f"class_{c}_n"] > 0 for c in (0, 1))
    cov = [out[f"class_{c}_coverage"] for c in (0, 1)]
    csy = [out[f"class_{c}_csy"] for c in (0, 1)]
    out.update({
        "minimum_class_coverage": float(min(cov)) if present else math.nan,
        "macro_coverage": float(np.mean(cov)) if present else math.nan,
        "macro_csy": float(np.mean(csy)) if present else math.nan,
        "worst_csy": float(min(csy)) if present else math.nan,
        "false_clearance": out["class_1_wrong_singleton_exposure"],
        "false_discard": out["class_0_wrong_singleton_exposure"],
    })
    return out
