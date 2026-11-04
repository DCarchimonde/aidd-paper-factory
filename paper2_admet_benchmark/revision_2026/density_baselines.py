"""Optional, explicitly limited density-reweighting comparison components.

These are NOT full reproductions of CoDrug or KMM-CP. The KDE component omits
CoDrug's energy-trained representation. The KMM component uses a fixed RBF
bandwidth, exact mean-one normalization by default, and candidate-label
Mondrian quantiles; it omits KMM-CP bandwidth selection and selective gating.

KDE uses estimated density ratios and query mass at infinity. KMM uses the
weighted empirical quantile in KMM-CP, without query mass or a finite-sample
rank correction. Neither component establishes finite-sample validity under
arbitrary molecular shift. All density fitting is outcome-label-free but
transductive in the supplied calibration and target covariates.

References:
  Tibshirani et al., Conformal Prediction Under Covariate Shift (NeurIPS 2019).
  KMM-CP, arXiv:2603.26415v1, Sections 3.1-3.2 and Appendix E.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

from paper2_admet_benchmark.revision_2026.core import (
    _alpha, _labels, _probabilities, candidate_scores,
)
from paper2_admet_benchmark.scripts.racer_c4.racer_c4_core import weighted_mondrian_sets


class DensityEstimationError(RuntimeError):
    """A numerical density fit is invalid; do not silently replace its output."""


def _features(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(source, dtype=float)
    z = np.asarray(target, dtype=float)
    if x.ndim != 2 or z.ndim != 2 or x.shape[1] != z.shape[1] or x.shape[1] == 0:
        raise ValueError("source and target must be 2D arrays with equal nonzero feature dimension")
    if min(len(x), len(z)) < 2:
        raise ValueError("each domain requires at least two rows")
    if not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError("features must be finite")
    return x, z


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _ess(weights: np.ndarray) -> float:
    denom = float(np.dot(weights, weights))
    return float(weights.sum() ** 2 / denom) if denom else 0.0


def kde_density_ratio(
    source_features: np.ndarray,
    target_features: np.ndarray,
    bandwidth: float = 1.0,
    weight_clip: tuple[float, float] = (0.05, 20.0),
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Gaussian-KDE p_target(x)/p_source(x), fitted on the supplied covariates.

    Source-only standardization avoids silently using a target-dependent
    feature scale. KDE automatically divides by each domain's sample count;
    no additional n_source/n_target factor belongs in this density ratio.
    The same fixed bandwidth is used in both KDEs. Values are clipped in log
    space to avoid overflow and to expose, rather than hide, saturation.
    """
    x, z = _features(source_features, target_features)
    bw = _positive(bandwidth, "bandwidth")
    if len(weight_clip) != 2:
        raise ValueError("weight_clip must be a pair")
    lower, upper = (_positive(v, "weight bound") for v in weight_clip)
    if lower > upper:
        raise ValueError("weight lower bound exceeds upper bound")
    scaler = StandardScaler().fit(x)
    xs, zs = scaler.transform(x), scaler.transform(z)
    source_kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(xs)
    target_kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(zs)
    query = np.vstack((xs, zs))
    log_ratio = target_kde.score_samples(query) - source_kde.score_samples(query)
    if not np.isfinite(log_ratio).all():
        raise DensityEstimationError("KDE returned non-finite log density ratios")
    lo, hi = math.log(lower), math.log(upper)
    ratio = np.exp(np.clip(log_ratio, lo, hi))
    sw, tw = ratio[:len(x)], ratio[len(x):]
    diagnostics = {
        "method": "kde_mondrian_component", "full_codrug_reproduction": False,
        "standardization": "source_only_standard_scaler", "bandwidth": bw,
        "source_n": len(x), "target_n": len(z), "feature_dimension": x.shape[1],
        "weight_min": lower, "weight_max": upper,
        "source_ess": _ess(sw), "target_ess": _ess(tw),
        "source_clip_fraction": float(np.mean((log_ratio[:len(x)] <= lo) | (log_ratio[:len(x)] >= hi))),
        "target_clip_fraction": float(np.mean((log_ratio[len(x):] <= lo) | (log_ratio[len(x):] >= hi))),
        "query_mass": "estimated_query_weight_at_infinity",
        "finite_sample_shift_guarantee": False,
    }
    return sw, tw, diagnostics


def kernel_mean_matching(
    source_features: np.ndarray,
    target_features: np.ndarray,
    bandwidth: float = 1.0,
    weight_bound: float = 30.0,
    epsilon: float = 0.0,
    max_source: int = 2500,
    max_target: int = 2500,
    maxiter: int = 1000,
    feasibility_tolerance: float = 1e-6,
) -> tuple[np.ndarray, dict]:
    """Solve bounded RBF KMM with SLSQP; reject failed/infeasible solutions.

    Minimize 0.5 beta^T K beta - kappa^T beta, where
    kappa_i=(n/m) sum_j k(x_i,z_j), with 0<=beta<=B and
    abs(mean(beta)-1)<=epsilon. Default epsilon=0 enforces sum(beta)=n;
    this is a disclosed component choice, not an asserted KMM-CP default.
    A 1e-8 diagonal ridge is disclosed in diagnostics for numerical stability.
    Input caps fail explicitly; no hidden subsampling changes the comparator.
    """
    x, z = _features(source_features, target_features)
    bw = _positive(bandwidth, "bandwidth")
    bound = _positive(weight_bound, "weight_bound")
    if bound < 1:
        raise ValueError("weight_bound must be at least 1 so the mean-one initial point is feasible")
    eps = float(epsilon)
    if not math.isfinite(eps) or not 0 <= eps < 1:
        raise ValueError("epsilon must be finite and in [0,1)")
    tolerance = _positive(feasibility_tolerance, "feasibility_tolerance")
    for value, name in ((max_source, "max_source"), (max_target, "max_target"), (maxiter, "maxiter")):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    n, m = len(x), len(z)
    if n > max_source or m > max_target:
        raise DensityEstimationError(
            f"KMM size cap exceeded: source={n}/{max_source}, target={m}/{max_target}; "
            "no subsampling was performed. Increase the explicit caps or report this cell unavailable."
        )
    scaler = StandardScaler().fit(x)
    xs, zs = scaler.transform(x), scaler.transform(z)
    gamma = 1 / (2 * bw * bw)
    gram = rbf_kernel(xs, gamma=gamma)
    ridge = 1e-8
    gram.flat[::n+1] += ridge
    kappa = (n / m) * rbf_kernel(xs, zs, gamma=gamma).sum(axis=1)
    ones = np.ones(n)
    lower_sum, upper_sum = n * (1-eps), n * (1+eps)
    # Scaling the objective by n does not change the optimizer.
    objective = lambda beta: float((0.5 * beta @ gram @ beta - kappa @ beta) / n)
    gradient = lambda beta: (gram @ beta - kappa) / n
    if eps == 0:
        constraints = [{"type": "eq", "fun": lambda beta: beta.sum()-n,
                        "jac": lambda beta: ones}]
    else:
        constraints = [
            {"type": "ineq", "fun": lambda beta: beta.sum()-lower_sum,
             "jac": lambda beta: ones},
            {"type": "ineq", "fun": lambda beta: upper_sum-beta.sum(),
             "jac": lambda beta: -ones},
        ]
    result = minimize(objective, ones, method="SLSQP", jac=gradient,
                      bounds=[(0.0, bound)] * n, constraints=constraints,
                      options={"maxiter": maxiter, "ftol": 1e-9, "disp": False})
    beta = np.asarray(result.x, dtype=float)
    if beta.shape != (n,) or not np.isfinite(beta).all():
        raise DensityEstimationError("KMM solver returned invalid weights")
    sum_residual = max(0.0, lower_sum-beta.sum(), beta.sum()-upper_sum)
    box_residual = max(0.0, float(-beta.min()), float(beta.max()-bound))
    if not result.success or max(sum_residual, box_residual) > tolerance:
        raise DensityEstimationError(
            f"KMM solver failed validation: success={result.success}, status={result.status}, "
            f"sum_residual={sum_residual:.6g}, box_residual={box_residual:.6g}; {result.message}"
        )
    projected = bool(np.any((beta < 0) | (beta > bound)))
    beta = np.clip(beta, 0, bound)  # only round-off already checked above.
    diagnostics = {
        "method": "kmm_mondrian_component", "full_kmm_cp_reproduction": False,
        "standardization": "source_only_standard_scaler", "bandwidth": bw,
        "rbf_gamma": gamma, "source_n": n, "target_n": m, "feature_dimension": x.shape[1],
        "weight_bound": bound, "epsilon": eps, "ridge": ridge,
        "solver": "scipy_slsqp", "solver_success": True, "solver_status": int(result.status),
        "solver_iterations": int(result.nit), "sum_residual": float(sum_residual),
        "box_residual": float(box_residual), "feasibility_tolerance": tolerance,
        "roundoff_projection_applied": projected,
        "source_ess": _ess(beta), "weight_mean": float(beta.mean()),
        "zero_weight_fraction": float(np.mean(beta == 0)),
        "objective_initial": objective(ones), "objective_final": objective(beta),
        "query_mass": "none_empirical_weighted_quantile", "finite_sample_shift_guarantee": False,
    }
    return beta, diagnostics


def weighted_empirical_mondrian(
    calibration_probability: np.ndarray,
    calibration_labels: np.ndarray,
    query_probability: np.ndarray,
    calibration_weights: np.ndarray,
    alpha: float | Mapping[int, float] = 0.1,
) -> np.ndarray:
    """Candidate-label weighted empirical quantiles (no query infinity atom).

    This is the KMM-CP-style asymptotic quantile component, with candidate-
    label Mondrian indexing. It is not finite-sample corrected split CP.
    A missing or zero-total-weight calibration class has infinite threshold.
    """
    cp = _probabilities(calibration_probability)
    qp = _probabilities(query_probability)
    y = _labels(calibration_labels, len(cp))
    alpha_by_class = _alpha(alpha)
    w = np.asarray(calibration_weights, dtype=float)
    if w.shape != (len(cp),) or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("calibration weights must be a finite nonnegative aligned vector")
    cs, qs = candidate_scores(cp), candidate_scores(qp)
    thresholds = np.full(2, math.inf)
    for label in (0, 1):
        active = (y == label) & (w > 0)
        score, weight = cs[active, label], w[active]
        if not len(score):
            continue
        order = np.argsort(score, kind="stable")
        cumulative = np.cumsum(weight[order])
        required = (1-alpha_by_class[label]) * cumulative[-1]
        index = int(np.searchsorted(cumulative, required, side="left"))
        thresholds[label] = score[order[index]]
    return qs <= thresholds


def _component_inputs(cal_x, cal_y, query_x, cal_p, query_p):
    x, z = _features(cal_x, query_x)
    cp, qp = _probabilities(cal_p), _probabilities(query_p)
    y = _labels(cal_y, len(x))
    if len(cp) != len(x) or len(qp) != len(z):
        raise ValueError("probabilities and feature rows must align")
    return x, y, z, cp, qp


def kde_mondrian_component(
    cal_x, cal_y, query_x, cal_p, query_p, alpha=0.1,
    bandwidth=1.0, weight_clip=(0.05, 20.0),
) -> tuple[np.ndarray, dict]:
    x, y, z, cp, qp = _component_inputs(cal_x, cal_y, query_x, cal_p, query_p)
    alphas = _alpha(alpha)
    sw, tw, diagnostics = kde_density_ratio(x, z, bandwidth, weight_clip)
    for c in (0, 1):
        diagnostics[f"class_{c}_source_ess"] = _ess(sw[y == c])
    member = weighted_mondrian_sets(candidate_scores(cp), y, candidate_scores(qp), sw, tw, alphas)
    return member, diagnostics


def kmm_mondrian_component(
    cal_x, cal_y, query_x, cal_p, query_p, alpha=0.1,
    bandwidth=1.0, weight_bound=30.0, epsilon=0.0,
    max_source=2500, max_target=2500, maxiter=1000,
) -> tuple[np.ndarray, dict]:
    x, y, z, cp, qp = _component_inputs(cal_x, cal_y, query_x, cal_p, query_p)
    alphas = _alpha(alpha)
    weights, diagnostics = kernel_mean_matching(
        x, z, bandwidth, weight_bound, epsilon, max_source, max_target, maxiter,
    )
    for c in (0, 1):
        diagnostics[f"class_{c}_source_ess"] = _ess(weights[y == c])
        diagnostics[f"class_{c}_zero_total_weight"] = bool(weights[y == c].sum() == 0)
    return weighted_empirical_mondrian(cp, y, qp, weights, alphas), diagnostics
