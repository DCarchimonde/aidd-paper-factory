"""Label-free strata, certified decision rules and audit allocation policies.

Only observed CountCS states and frozen source-model probabilities enter the
allocation API. The caller draws the NEXT item uniformly without replacement
inside the chosen stratum and records every observed prefix. Forecasts used
for allocation are heuristics; the count certificate does not trust them.
"""
from __future__ import annotations

import copy
import math
from fractions import Fraction
from typing import Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.stats import binom

from .core import (CountCS, InconsistentConfidenceSet, OptimizationFailure,
                   certify_triage, optimize_triage, update_count_interval)


def _positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if not math.isfinite(number) or number < 1 or number != math.floor(number):
        raise ValueError(f"{name} must be a positive integer")
    return int(number)


def make_strata(score, support, ids, mode="score_support", score_bins=4,
                support_bins=2) -> np.ndarray:
    """Global rank-quantile cross cells, without looking at any assay label.

    Scores and support values are independently ranked over the fixed pool.
    Equal values are ordered by the unique string sample ID, so assignments are
    invariant to input row order. Cartesian empty cells are dropped; remaining
    cell IDs are ordered by score bin then support bin and remapped to 0..H-1.
    score_only uses score_bins * support_bins rank bins (eight by default),
    matching the maximum number of cells in the default cross stratification.
    """
    if mode not in ("score_support", "score_only"):
        raise ValueError("mode must be score_support or score_only")
    score_bins = _positive_integer(score_bins, "score_bins")
    support_bins = _positive_integer(support_bins, "support_bins")
    probabilities = np.asarray(score, dtype=float)
    identities = np.asarray(ids, dtype=str)
    if probabilities.ndim != 1 or identities.shape != probabilities.shape:
        raise ValueError("scores and IDs must be aligned one-dimensional vectors")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("frozen scores must be finite probabilities")
    if len(set(identities.tolist())) != len(identities) or np.any(identities == ""):
        raise ValueError("strata require unique nonempty sample IDs")
    if not len(probabilities):
        return np.empty(0, dtype=np.int64)

    def ranks(values, bins):
        ordering = np.lexsort((identities, values))
        out = np.empty(len(values), dtype=np.int64)
        out[ordering] = np.arange(len(values), dtype=np.int64) * bins // len(values)
        return out

    if mode == "score_only":
        raw = ranks(probabilities, score_bins * support_bins)
    else:
        similarities = np.asarray(support, dtype=float)
        if similarities.shape != probabilities.shape or not np.isfinite(similarities).all():
            raise ValueError("support must be aligned and finite")
        if np.any((similarities < 0) | (similarities > 1)):
            raise ValueError("support must lie in [0,1]")
        raw = ranks(probabilities, score_bins) * support_bins + ranks(similarities, support_bins)
    return np.unique(raw, return_inverse=True)[1].astype(np.int64)


def _state_arrays(states: Sequence[CountCS]):
    if any(not isinstance(state, CountCS) for state in states):
        raise TypeError("policies require observed CountCS objects")
    remaining = np.asarray([state.N - state.n for state in states], dtype=np.int64)
    bounds = [state.remaining_bounds() for state in states]
    lower = np.asarray([pair[0] for pair in bounds], dtype=np.int64)
    upper = np.asarray([pair[1] for pair in bounds], dtype=np.int64)
    positive = sum(state.s for state in states)
    negative = sum(state.n - state.s for state in states)
    return remaining, lower, upper, positive, negative


def _order(order, h: int) -> np.ndarray:
    if order is None:
        return np.arange(h, dtype=np.int64)
    raw = np.asarray(order)
    if raw.shape != (h,) or not np.isin(raw, np.arange(h)).all() or len(set(raw.tolist())) != h:
        raise ValueError("order must be a permutation of all stratum indices")
    if raw.dtype.kind == "b":
        raise ValueError("order cannot be a boolean vector")
    return raw.astype(np.int64)


def _result(M, L, U, a, b, clear, discard, eps_fn, eps_fp, solver: str):
    certificate = certify_triage(M, L, U, a, b, clear, discard, eps_fn, eps_fp)
    if not certificate["certified"]:
        raise OptimizationFailure("policy failed the independent exact count-box certificate")
    saved = int(M[clear | discard].sum())
    return {"status": "optimal", "clear_mask": clear, "discard_mask": discard,
            "defer_mask": ~(clear | discard),
            "actions": np.where(clear, 0, np.where(discard, 1, 2)).astype(np.int8),
            "saved_cost": float(saved), "saved_items": saved,
            "certificate": certificate, "solver": solver}


def _loose_optimum(M, L, U, a, b, eps_fn, eps_fp):
    """Conservative separate upper numerator / lower denominator benchmark."""
    h = len(M)
    values = M.astype(float)
    active = M > 0
    if not h or not np.any(active):
        return _result(M, L, U, a, b, active, active, eps_fn, eps_fp, "trivial_empty")
    matrix = np.zeros((h + 2, 2 * h))
    matrix[np.arange(h), np.arange(h)] = 1
    matrix[np.arange(h), h + np.arange(h)] = 1
    matrix[h, :h] = U
    matrix[h + 1, h:] = M - L
    denominator_positive = a + int(L.sum())
    denominator_negative = b + int((M - U).sum())
    upper = np.concatenate((np.ones(h), [eps_fn * denominator_positive,
                                             eps_fp * denominator_negative]))
    solution = milp(c=-np.concatenate((values, values)), integrality=np.ones(2 * h),
                    bounds=Bounds(np.zeros(2 * h), np.concatenate((active, active)).astype(float)),
                    constraints=LinearConstraint(matrix, np.full(h + 2, -np.inf), upper),
                    options={"mip_rel_gap": 0.0, "presolve": True})
    if solution.status != 0 or solution.x is None or not np.isfinite(solution.x).all():
        raise OptimizationFailure(f"loose MILP failed: {solution.status}, {solution.message}")
    rounded = np.rint(solution.x)
    if not np.isin(rounded, [0, 1]).all() or np.max(np.abs(solution.x - rounded)) > 1e-6:
        raise OptimizationFailure("loose MILP returned nonbinary actions")
    clear, discard = rounded[:h].astype(bool), rounded[h:].astype(bool)
    if np.any((clear | discard) & ~active):
        raise OptimizationFailure("loose MILP selected an empty stratum")
    # Exact arithmetic checks the loose constraints themselves, not just the
    # potentially more permissive joint-risk certificate.
    for numerator, denominator, tolerance in ((int(U[clear].sum()), denominator_positive, eps_fn),
                                              (int((M - L)[discard].sum()), denominator_negative, eps_fp)):
        cap = Fraction(str(tolerance))
        if numerator * cap.denominator > denominator * cap.numerator:
            raise OptimizationFailure("loose MILP violated its independent exact fraction check")
    result = _result(M, L, U, a, b, clear, discard, eps_fn, eps_fp, "loose_scipy_milp")
    if not math.isclose(result["saved_cost"], -float(solution.fun), rel_tol=1e-10, abs_tol=1e-8):
        raise OptimizationFailure("loose objective disagrees with action savings")
    return result


def optimize_policy(states, eps_fn, eps_fp, mode="joint", order=None) -> dict:
    """Certified optimum under joint, score-threshold or loose constraints.

    order is a frozen permutation of ALL strata by mean source-model score,
    increasing. Threshold rules clear a prefix and discard a disjoint suffix;
    intervening strata defer. A support cell can therefore have a different
    action from the other support cell in its score bin. Equal objective values
    use the first (prefix length, suffix start) in deterministic enumeration.
    Unit assay costs are common to all policies in this comparison.
    """
    M, L, U, a, b = _state_arrays(states)
    empty = np.zeros(len(M), dtype=bool)
    # Public core validation also checks risk levels and count consistency.
    certify_triage(M, L, U, a, b, empty, empty, eps_fn, eps_fp)
    if mode == "joint":
        return optimize_triage(M, L, U, a, b, eps_fn, eps_fp)
    if mode == "loose":
        return _loose_optimum(M, L, U, a, b, eps_fn, eps_fp)
    if mode != "thresholds":
        raise ValueError("optimizer mode must be joint, thresholds or loose")
    ordering = _order(order, len(M))
    best = _result(M, L, U, a, b, empty, empty, eps_fn, eps_fp, "exhaustive_ordered_thresholds")
    for prefix in range(len(M) + 1):
        for suffix in range(prefix, len(M) + 1):
            clear, discard = empty.copy(), empty.copy()
            clear[ordering[:prefix]] = True
            discard[ordering[suffix:]] = True
            clear &= M > 0
            discard &= M > 0
            saved = int(M[clear | discard].sum())
            if saved <= best["saved_items"]:
                continue
            certificate = certify_triage(M, L, U, a, b, clear, discard, eps_fn, eps_fp)
            if certificate["certified"]:
                best = _result(M, L, U, a, b, clear, discard, eps_fn, eps_fp,
                               "exhaustive_ordered_thresholds")
    return best


def _posterior_q(states, prior_p):
    prior = np.asarray(prior_p, dtype=float)
    if prior.shape != (len(states),) or not np.isfinite(prior).all() or np.any((prior < 0) | (prior > 1)):
        raise ValueError("prior_p must contain one frozen probability per stratum")
    return np.asarray([(state.s + 2.0 * p) / (state.n + 2.0) for state, p in zip(states, prior)])


def _tied_max(values, eligible, rng) -> int:
    candidates = np.flatnonzero(eligible)
    if not len(candidates):
        raise ValueError("no unaudited stratum remains")
    best = np.max(values[candidates])
    tied = candidates[np.isclose(values[candidates], best, rtol=1e-12, atol=1e-12)]
    return int(rng.choice(tied))


def _lookahead_values(states, q, optimizer_mode, order, block_size, eps_fn, eps_fp):
    """Return present savings and model-forecast future REMAINING savings.

    The Binomial model is a heuristic, not the true finite-population sampling
    law. Every k=0..b is integrated for b<=8. A virtual state intersects only
    the known current CS with the future endpoint inversion; unknown internal
    orders are not fabricated. Actual observations must update every prefix.
    Empty forecast intersections conservatively receive zero release savings.
    """
    current = optimize_policy(states, eps_fn, eps_fp, optimizer_mode, order)["saved_cost"]
    future = np.full(len(states), -np.inf)
    for h, state in enumerate(states):
        batch = min(block_size, state.N - state.n)
        if batch == 0:
            continue
        expected = 0.0
        for extra_positive in range(batch + 1):
            probability = float(binom.pmf(extra_positive, batch, q[h]))
            if probability == 0:
                continue
            hypothetical = copy.copy(state)
            hypothetical.n = state.n + batch
            hypothetical.s = state.s + extra_positive
            try:
                lo, hi = update_count_interval(state.N, hypothetical.n, hypothetical.s,
                                              state.lower_total, state.upper_total, state.delta,
                                              prior=state.prior)
                hypothetical.lower_total, hypothetical.upper_total = lo, hi
                proposal = list(states)
                proposal[h] = hypothetical
                saved = optimize_policy(proposal, eps_fn, eps_fp, optimizer_mode, order)["saved_cost"]
            except InconsistentConfidenceSet:
                saved = 0.0
            expected += probability * saved
        # Saved remaining items already exclude the newly paid audit batch.
        # Subtracting that batch again would double-charge acquisition cost.
        future[h] = expected
    return float(current), future


def choose_stratum(states, prior_p, rng, policy="lookahead", optimizer_mode="joint",
                   order=None, block_size=4, eps_fn=.1, eps_fp=.1) -> int:
    """Choose a stratum only; the operator owns its hidden random permutation.

    uniform: random remaining-item allocation; entropy: random weights M*H(q);
    neyman: largest deficit to N*sqrt(q(1-q)) target audit allocation;
    lookahead: largest expected increase of certified remaining assay savings.
    If no positive forecast gain exists, explore the widest remaining positive
    count interval; ties use the supplied RNG. No target-label array is accepted.
    """
    if policy not in ("uniform", "entropy", "neyman", "lookahead"):
        raise ValueError("unknown allocation policy")
    block_size = _positive_integer(block_size, "block_size")
    if block_size > 8:
        raise ValueError("exact small-block outcome enumeration supports block_size <= 8")
    M, L, U, _, _ = _state_arrays(states)
    eligible = M > 0
    if not np.any(eligible):
        raise ValueError("no unaudited stratum remains")
    q = _posterior_q(states, prior_p)
    if policy == "uniform":
        return int(rng.choice(len(states), p=M / M.sum()))
    if policy == "entropy":
        entropy = np.zeros(len(states))
        interior = (q > 0) & (q < 1)
        entropy[interior] = -q[interior] * np.log(q[interior]) - (1 - q[interior]) * np.log1p(-q[interior])
        weights = M * entropy
        if weights.sum() == 0:
            weights = M.astype(float)
        return int(rng.choice(len(states), p=weights / weights.sum()))
    if policy == "neyman":
        weights = np.asarray([state.N for state in states]) * np.sqrt(q * (1 - q))
        weights[~eligible] = 0
        if weights.sum() == 0:
            return _tied_max((U - L).astype(float), eligible, rng)
        audited = np.asarray([state.n for state in states])
        desired = (audited.sum() + block_size) * weights / weights.sum()
        return _tied_max(desired - audited, eligible, rng)
    current, forecast = _lookahead_values(states, q, optimizer_mode, order, block_size, eps_fn, eps_fp)
    if np.max(forecast[eligible]) > current + 1e-12:
        return _tied_max(forecast, eligible, rng)
    return _tied_max((U - L).astype(float), eligible, rng)
