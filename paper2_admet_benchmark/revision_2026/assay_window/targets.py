"""Operational paired-response onsets and concentration-resolution bounds.

An onset is the first crossing in the piecewise-linear interpolant of observed
responses on log10(M) doses, recorded only as its enclosing dose bracket. Its
bracket records the tested-concentration resolution, not a confidence interval
for an error-free biological potency. Every interval is ``(lower, upper]``;
``None`` encodes the corresponding unbounded side in JSON records. A curve
that never crosses stays right censored instead of receiving a maximum-dose
point estimate. Nonmonotonic recrossings remain visible in the returned record.

The observation mask coarsens these brackets to a declared model grid. Bounds
on window probability obtained from grid cells are model-resolution bounds,
not distribution-free confidence or risk guarantees.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

DEFAULT_EDGES = np.arange(-10.0, -2.99, 0.5)
DEFAULT_MARGIN_LOG10 = math.log10(3.0)


def _vector(values: Sequence, name: str) -> np.ndarray:
    try:
        result = np.asarray([np.nan if x is None or x == "" else float(x)
                             for x in values], dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a one-dimensional numeric sequence") from error
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return result


def _edges(values) -> np.ndarray:
    edges = np.asarray(values, dtype=float)
    if edges.ndim != 1 or not len(edges) or not np.isfinite(edges).all():
        raise ValueError("grid edges must be a nonempty finite vector")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("grid edges must be strictly increasing")
    return edges


def _bounds(interval: Mapping) -> tuple[float, float]:
    if not isinstance(interval, Mapping):
        raise ValueError("an onset interval must be a mapping")
    if not {"lower_log_m", "upper_log_m"}.issubset(interval):
        raise ValueError("interval requires lower_log_m and upper_log_m")
    lower = -math.inf if interval["lower_log_m"] is None else float(interval["lower_log_m"])
    upper = math.inf if interval["upper_log_m"] is None else float(interval["upper_log_m"])
    if math.isnan(lower) or math.isnan(upper) or not lower < upper:
        raise ValueError("interval must have lower < upper, with unbounded sides represented by None")
    return lower, upper


def _margin(value: float) -> float:
    margin = float(value)
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin_log10 must be finite and nonnegative")
    return margin


def onset_interval(concentrations_m: Sequence, responses_pct: Sequence,
                   threshold_pct: float, direction: int = -1) -> dict:
    """Bracket the first log-dose linear-interpolant threshold crossing.

    Responses are oriented as ``direction * responses_pct``; antagonist and
    viability-loss readouts use ``direction=-1``. Missing/nonfinite pairs are
    omitted. Repeated concentrations with the same response are deduplicated;
    conflicting repeated responses require an explicit replicate decision and
    are rejected. Crossing equality counts as a crossing. Missing pairs define
    the effective interpolation grid. Later recrossings do not change the first
    crossing; onset separation does not imply a sustained response window.
    """
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or +1")
    threshold = float(threshold_pct)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold_pct must be positive and finite")
    concentrations = _vector(concentrations_m, "concentrations_m")
    responses = _vector(responses_pct, "responses_pct")
    if concentrations.shape != responses.shape:
        raise ValueError("concentrations and responses must be aligned")
    if np.any(np.isfinite(concentrations) & (concentrations <= 0)):
        raise ValueError("concentrations in M must be positive")
    keep = np.isfinite(concentrations) & np.isfinite(responses)
    conc, response = concentrations[keep], responses[keep]
    order = np.argsort(conc, kind="stable")
    conc, response = conc[order], response[order]
    unique_c, unique_r = [], []
    for c, r in zip(conc, response):
        if unique_c and c == unique_c[-1]:
            if r != unique_r[-1]:
                raise ValueError("conflicting responses at the same concentration; preserve replicates separately")
            continue
        unique_c.append(float(c))
        unique_r.append(float(r))
    conc, response = np.asarray(unique_c), np.asarray(unique_r)
    result = {
        "lower_log_m": None, "upper_log_m": None,
        "censoring": "unobserved", "n_points": len(conc),
        "n_input_points": len(concentrations), "n_missing_pairs": int((~keep).sum()),
        "concentration_min_log_m": None, "concentration_max_log_m": None,
        "recrossings": 0, "n_below_after_first_crossing": 0,
        "first_crossing_index": None, "threshold_pct": threshold,
        "direction": int(direction), "interval_convention": "(lower, upper]",
        "target_definition": "first_observed_threshold_crossing",
    }
    if not len(conc):
        return result
    logs = np.log10(conc)
    result["concentration_min_log_m"] = float(logs[0])
    result["concentration_max_log_m"] = float(logs[-1])
    above = direction * response >= threshold
    result["recrossings"] = int(np.sum(above[:-1] & ~above[1:]))
    crossed = np.flatnonzero(above)
    if not len(crossed):
        result.update(lower_log_m=float(logs[-1]), censoring="right")
        return result
    first = int(crossed[0])
    result.update(upper_log_m=float(logs[first]),
                  first_crossing_index=first,
                  n_below_after_first_crossing=int((~above[first + 1:]).sum()))
    if first == 0:
        result["censoring"] = "left"
    else:
        result.update(lower_log_m=float(logs[first - 1]), censoring="interval")
    return result


def interval_bin_mask(interval: Mapping, edges=DEFAULT_EDGES) -> np.ndarray:
    """Return bins intersecting ``(lower, upper]`` without endpoint double counts.

    Bins are ``(-inf,e0], (e0,e1], ..., (e_last,+inf)``. The intersection of
    two such intervals is nonempty iff its largest lower endpoint is strictly
    below its smallest upper endpoint. Coarsening expands a bracket only to
    the grid cells it actually intersects.
    """
    edges = _edges(edges)
    lower, upper = _bounds(interval)
    lows = np.r_[-np.inf, edges]
    highs = np.r_[edges, np.inf]
    mask = np.maximum(lows, lower) < np.minimum(highs, upper)
    if not mask.any():
        raise ValueError("interval does not intersect the grid")
    return mask


def pair_allowed_mask(target_interval: Mapping, viability_interval: Mapping,
                      edges=DEFAULT_EDGES) -> np.ndarray:
    """Rectangular observation mask for the two paired onset brackets."""
    target = interval_bin_mask(target_interval, edges)
    viability = interval_bin_mask(viability_interval, edges)
    return target[:, None] & viability[None, :]


def _window_rect(lower_t, upper_t, lower_v, upper_v, horizon_t, horizon_v, margin):
    # No inf-inf operations: margin and tested horizons are finite.
    upper_allowed_t = np.minimum(np.minimum(upper_t, horizon_t),
                                 np.minimum(horizon_v - margin, upper_v - margin))
    possible = lower_t < upper_allowed_t
    guaranteed = ((upper_t <= horizon_t) & (upper_t + margin <= horizon_v)
                  & (upper_t + margin <= lower_v))
    return guaranteed, possible


def window_cell_masks(edges, target_horizon: float, viability_horizon: float,
                      margin_log10: float = DEFAULT_MARGIN_LOG10) -> tuple[np.ndarray, np.ndarray]:
    """Cells that guarantee/can contain a three-condition response window.

    E = {T + margin <= V, T <= target_horizon,
         T + margin <= viability_horizon}. Horizons and onsets are log10(M).
    A probability model's mass on guaranteed cells is a lower resolution bound;
    its mass on possible cells is an upper resolution bound.
    """
    edges = _edges(edges)
    margin = _margin(margin_log10)
    ht, hv = float(target_horizon), float(viability_horizon)
    if not math.isfinite(ht) or not math.isfinite(hv):
        raise ValueError("tested horizons must be finite log10(M) values")
    lows, highs = np.r_[-np.inf, edges], np.r_[edges, np.inf]
    guaranteed, possible = _window_rect(lows[:, None], highs[:, None],
                                        lows[None, :], highs[None, :], ht, hv, margin)
    if np.any(guaranteed & ~possible):
        raise RuntimeError("inconsistent window-resolution masks")
    return guaranteed.astype(bool), possible.astype(bool)


def window_reference_bounds(target_interval: Mapping, viability_interval: Mapping,
                            target_horizon: float | None = None,
                            viability_horizon: float | None = None,
                            margin_log10: float = DEFAULT_MARGIN_LOG10) -> dict:
    """Classify what the observed onset rectangle establishes about E.

    The within-range event and the unrestricted separation margin are reported
    separately. In particular, two right-censored curves can establish that no
    window was observed in the tested range while leaving global ordering
    unresolved. Missing horizons make the within-range reference unresolved.
    """
    lt, ut = _bounds(target_interval)
    lv, uv = _bounds(viability_interval)
    margin = _margin(margin_log10)
    ht = (target_interval.get("concentration_max_log_m")
          if target_horizon is None else target_horizon)
    hv = (viability_interval.get("concentration_max_log_m")
          if viability_horizon is None else viability_horizon)
    global_yes = ut + margin <= lv
    global_possible = lt < uv - margin
    global_status = "definite_yes" if global_yes else "unresolved" if global_possible else "definite_no"
    result = {"status": "unresolved", "within_range_status": "unresolved",
              "global_margin_status": global_status,
              "definite_yes": False, "definite_no": False, "possible": True,
              "target_horizon_log_m": None, "viability_horizon_log_m": None,
              "margin_log10": margin, "scope": "window_within_tested_concentration_range"}
    if ht is None or hv is None:
        result["reason"] = "missing_tested_horizon"
        return result
    ht, hv = float(ht), float(hv)
    if not math.isfinite(ht) or not math.isfinite(hv):
        raise ValueError("tested horizons must be finite log10(M) values")
    yes, possible = _window_rect(lt, ut, lv, uv, ht, hv, margin)
    status = "definite_yes" if yes else "unresolved" if possible else "definite_no"
    result.update(status=status, within_range_status=status,
                  definite_yes=bool(yes), definite_no=not bool(possible),
                  possible=bool(possible), target_horizon_log_m=ht,
                  viability_horizon_log_m=hv)
    return result
