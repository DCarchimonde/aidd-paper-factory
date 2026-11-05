"""Paper-based KMM-CP and Selective KMM-CP (UAI 2026).

Implements the complete post-prediction procedures: permutation-standardized
MMD bandwidth selection, bounded KMM, joint source/target selection QP,
thresholding, standard-KMM refit, and empirical weighted calibration.
See KMM_IMPLEMENTATION.md for paper locations, unspecified implementation
choices, the predicted-group pseudocode discrepancy, and backbone requirements.
Target outcomes are deliberately absent from every fitting API.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence
import math

import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

from paper2_admet_benchmark.revision_2026.core import (
    _labels, _probabilities, candidate_scores, decision_metrics,
)

PAPER_URL = "https://proceedings.mlr.press/v337/laghuvarapu26a.html"
PAPER_PDF_SHA256 = "dbcc53ccd8a594b076f698f1fb2516cae701af803be8979eef3af788bbdae5ec"
LOCKED_OSQP_VERSION = "1.0.4"
METHOD_NAMES = (
    "kmm_cp_global", "kmm_cp_mondrian_candidate", "kmm_cp_mondrian_paper_predicted",
    "skmm_cp_global", "skmm_cp_mondrian_candidate", "skmm_cp_mondrian_paper_predicted",
)


class KMMError(RuntimeError):
    """A fit failed numerical validation; no substitute method is returned."""


@dataclass(frozen=True)
class KMMConfig:
    weight_bound: float = 30.0
    mass_tolerance: float = 0.0
    selection_fraction: float = 0.5
    selection_threshold: float = 0.2
    bandwidth_multipliers: tuple[float, ...] = (0.01, 0.1, 0.5, 1.0, 2.0)
    permutations: int = 100
    feature_scaling: str = "none"
    zero_distance_bandwidth: float = 1.0
    solver_eps_abs: float = 1e-7
    solver_eps_rel: float = 1e-7
    feasibility_tolerance: float = 5e-6
    solver_max_iter: int = 100000
    max_total_rows: int = 10000


def default_config() -> dict:
    """Explicit serializable defaults, including paper-unspecified choices."""
    return asdict(KMMConfig())


def backbone_config() -> dict:
    """KMM Appendix E.3 settings for the shared independent AttentiveFP trainer.

    The paper specifies three message-passing steps but not readout iterations;
    readout_steps=2 is an explicit implementation default. Penultimate output
    is 64D after the classifier hidden layer, distinct from the 512D graph.
    """
    return {
        "graph_size": 512, "hidden_sizes": [64], "gnn_layers": 3,
        "readout_steps": 2, "learning_rate": 0.001, "batch_size": 128,
        "max_epochs": 50, "patience": 5, "energy_weight": 0.0,
        "class_weight": "inverse_frequency", "scheduler_step_size": 1,
        "scheduler_gamma": 0.8, "early_stop_on": "train_loss",
        "density_representation": "embedding", "compute_conformal": False,
    }


def _config(value: Mapping | KMMConfig | None) -> KMMConfig:
    config = value if isinstance(value, KMMConfig) else KMMConfig(**(dict(value) if value else {}))
    if not np.isfinite(config.weight_bound) or config.weight_bound < 1:
        raise ValueError("weight_bound must be finite and >=1")
    if not np.isfinite(config.mass_tolerance) or not 0 <= config.mass_tolerance < 1:
        raise ValueError("mass_tolerance must lie in [0,1)")
    if not 0 < config.selection_fraction < 1:
        raise ValueError("selection_fraction must lie in (0,1)")
    if not 0 <= config.selection_threshold <= 1:
        raise ValueError("selection_threshold must lie in [0,1]")
    if config.feature_scaling not in {"none", "source_standard"}:
        raise ValueError("feature_scaling must be none or source_standard")
    for name in ("zero_distance_bandwidth", "solver_eps_abs", "solver_eps_rel", "feasibility_tolerance"):
        if not math.isfinite(getattr(config, name)) or getattr(config, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("permutations", "solver_max_iter", "max_total_rows"):
        v = getattr(config, name)
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 2:
            raise ValueError(f"{name} must be an integer >=2")
    if not len(config.bandwidth_multipliers) or any(not math.isfinite(v) or v <= 0 for v in config.bandwidth_multipliers):
        raise ValueError("bandwidth multipliers must be finite positive values")
    return config


def _features(source, target, minimum_target=2):
    x, z = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if x.ndim != 2 or z.ndim != 2 or x.shape[1] == 0 or x.shape[1] != z.shape[1]:
        raise ValueError("source and target features must have equal nonzero dimension")
    if len(x) < 2 or len(z) < minimum_target or not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError("features must be finite and domains must contain enough rows")
    return x, z


def _kernel(x, z, bandwidth):
    return np.exp(-cdist(x, z, metric="sqeuclidean") / (2 * bandwidth * bandwidth))


def _mmd_for_assignments(kernel: np.ndarray, assignments: np.ndarray, n_source: int) -> np.ndarray:
    """Unbiased two-sample MMD² for fixed-size source assignments."""
    n, m = n_source, kernel.shape[0]-n_source
    a = assignments.astype(float, copy=False)
    left = a @ kernel
    ss = np.sum(left*a, axis=1)
    cross = a @ kernel.sum(axis=1) - ss
    tt = kernel.sum() - ss - 2*cross
    source_diag = a @ np.diag(kernel)
    target_diag = np.trace(kernel) - source_diag
    return ((ss-source_diag)/(n*(n-1)) + (tt-target_diag)/(m*(m-1)) - 2*cross/(n*m))


def select_bandwidth(source_features, target_features, seed=0, config=None):
    """Appendix E.5 grid; maximize (observed MMD²-null mean)/null SD.

    Cross-domain Euclidean distances define the median. Permutations preserve
    domain sizes and are shared across candidates. Counts, statistic choice,
    zero-variance handling, and tie handling are fixed implementation choices
    because the paper does not specify them. No outcome labels are inputs.
    """
    cfg = _config(config)
    x, z = _features(source_features, target_features)
    n, m = len(x), len(z)
    if n+m > cfg.max_total_rows:
        raise KMMError(f"explicit max_total_rows={cfg.max_total_rows} exceeded; no subsampling performed")
    distances = cdist(x, z, metric="euclidean")
    median_distance = float(np.median(distances))
    if median_distance <= np.finfo(float).eps:
        positive = distances[distances > np.finfo(float).eps]
        base = float(np.median(positive)) if len(positive) else cfg.zero_distance_bandwidth
        median_policy = "positive_distance_median" if len(positive) else "all_identical_fixed_bandwidth"
    else:
        base, median_policy = median_distance, "cross_domain_median"
    candidates = np.array(cfg.bandwidth_multipliers, float)*base
    combined = np.vstack((x, z))
    assignments = np.zeros((cfg.permutations+1, n+m), dtype=float)
    assignments[0, :n] = 1
    rng = np.random.default_rng(int(seed))
    for row in assignments[1:]:
        row[rng.permutation(n+m)[:n]] = 1
    records = []
    for bandwidth in candidates:
        statistics = _mmd_for_assignments(_kernel(combined, combined, bandwidth), assignments, n)
        mean = float(statistics[1:].mean())
        sd = float(statistics[1:].std(ddof=1))
        usable = sd > 1e-12
        standardized = float((statistics[0]-mean)/sd) if usable else None
        records.append({"bandwidth": float(bandwidth), "mmd2_observed": float(statistics[0]),
                        "null_mean": mean, "null_std": sd, "standardized_mmd": standardized,
                        "usable_null_variance": usable})
    valid = [i for i, r in enumerate(records) if r["usable_null_variance"]]
    if valid:
        # First candidate wins an exact tie; no use of calibration outcomes.
        best = max(valid, key=lambda i: records[i]["standardized_mmd"])
        reason = "maximum_permutation_standardized_mmd2"
    else:
        best = int(np.argmin(np.abs(np.asarray(cfg.bandwidth_multipliers)-1)))
        reason = "all_null_variances_degenerate_use_nearest_median_multiplier"
    return float(candidates[best]), {
        "median_cross_domain_distance": median_distance, "median_policy": median_policy,
        "permutations": cfg.permutations, "permutation_seed": int(seed),
        "mmd_estimator": "unbiased_squared_mmd", "records": records,
        "selected_index": best, "selected_bandwidth": float(candidates[best]), "reason": reason,
    }


def _solve_qp(P, q, A, lower, upper, initial, cfg):
    try:
        import osqp
    except ImportError as exc:
        raise KMMError("OSQP is required: install osqp==1.0.4 through the revision installer") from exc
    solver = osqp.OSQP()
    solver.setup(P=sparse.csc_matrix(np.triu(P)), q=np.asarray(q, float),
                 A=sparse.csc_matrix(A), l=np.asarray(lower, float), u=np.asarray(upper, float),
                 eps_abs=cfg.solver_eps_abs, eps_rel=cfg.solver_eps_rel,
                 max_iter=cfg.solver_max_iter, polishing=True, verbose=False,
                 adaptive_rho_interval=50)
    solver.warm_start(x=initial)
    result = solver.solve(raise_error=False)
    if result.x is None or result.info.status_val != 1 or not np.isfinite(result.x).all():
        raise KMMError(f"OSQP failed: {result.info.status}; iterations={result.info.iter}")
    constraint = A @ result.x
    residual = float(max(np.max(np.maximum(np.asarray(lower)-constraint, 0)),
                         np.max(np.maximum(constraint-np.asarray(upper), 0))))
    # Constraints are expressed as means, so mass residuals are not inflated
    # by cohort size; box residuals remain in weight units.
    if residual > cfg.feasibility_tolerance:
        raise KMMError(f"OSQP constraint residual {residual:g} exceeds {cfg.feasibility_tolerance:g}")
    return result.x, {
        "solver": "osqp", "solver_version": osqp.__version__,
        "locked_solver_version": LOCKED_OSQP_VERSION,
        "matches_locked_solver": osqp.__version__ == LOCKED_OSQP_VERSION, "status": result.info.status,
        "status_value": int(result.info.status_val), "iterations": int(result.info.iter),
        "primal_residual": float(result.info.prim_res), "dual_residual": float(result.info.dual_res),
        "constraint_residual": residual, "objective": float(result.info.obj_val),
        "objective_ridge": 0.0,
    }


def _ess(weights):
    return float(weights.sum()**2 / np.dot(weights, weights)) if np.any(weights) else 0.0


def fit_standard_kmm(source_features, target_features, bandwidth, config=None):
    """Paper equations (5)-(6), solved without a ridge-altered objective."""
    cfg = _config(config)
    x, z = _features(source_features, target_features, minimum_target=1)
    n, m = len(x), len(z)
    if n+m > cfg.max_total_rows:
        raise KMMError("explicit size cap exceeded; no subsampling performed")
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be finite and positive")
    Kss, Kst = _kernel(x, x, bandwidth), _kernel(x, z, bandwidth)
    # QP is one half of n² times MMD², up to a constant term.
    P, q = Kss, -(n/m)*Kst.sum(axis=1)
    A = sparse.vstack([sparse.eye(n), sparse.csr_matrix(np.full((1,n), 1/n))], format="csc")
    lower = np.r_[np.zeros(n), 1-cfg.mass_tolerance]
    upper = np.r_[np.full(n, cfg.weight_bound), 1+cfg.mass_tolerance]
    raw, diagnostics = _solve_qp(P, q, A, lower, upper, np.ones(n), cfg)
    weights = np.clip(raw, 0, cfg.weight_bound)
    if abs(weights.mean()-1) > cfg.mass_tolerance+cfg.feasibility_tolerance:
        raise KMMError("roundoff projection violates KMM mean-mass constraint")
    diagnostics.update({"source_n": n, "target_n": m, "bandwidth": float(bandwidth),
                        "weight_bound": cfg.weight_bound, "mass_tolerance": cfg.mass_tolerance,
                        "weight_mean": float(weights.mean()), "ess": _ess(weights),
                        "roundoff_projection_max": float(np.max(np.abs(raw-weights))),
                        "source_weight_min": float(weights.min()), "source_weight_max": float(weights.max())})
    return weights, diagnostics


def fit_joint_selection(source_features, target_features, bandwidth, config=None):
    """Paper equations (7)-(8), jointly solve for w in [0,B], a in [0,1]."""
    cfg = _config(config)
    x, z = _features(source_features, target_features)
    n, m = len(x), len(z)
    if n+m > cfg.max_total_rows:
        raise KMMError("explicit size cap exceeded; no subsampling performed")
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be finite and positive")
    Kss, Kst, Ktt = _kernel(x,x,bandwidth), _kernel(x,z,bandwidth), _kernel(z,z,bandwidth)
    ratio = n/m
    P = np.block([[Kss, -ratio*Kst], [-ratio*Kst.T, ratio*ratio*Ktt]])
    # Equal mean source/selection mass (within epsilon), and mean(a)>=tau.
    mass = np.r_[np.full(n,1/n), np.full(m,-1/m)]
    retained = np.r_[np.zeros(n), np.full(m,1/m)]
    A = sparse.vstack([sparse.eye(n+m), sparse.csr_matrix(mass[None,:]),
                       sparse.csr_matrix(retained[None,:])], format="csc")
    lower = np.r_[np.zeros(n+m), -cfg.mass_tolerance, cfg.selection_fraction]
    upper = np.r_[np.full(n,cfg.weight_bound), np.ones(m), cfg.mass_tolerance, np.inf]
    initial = np.full(n+m, cfg.selection_fraction)
    raw, diagnostics = _solve_qp(P, np.zeros(n+m), A, lower, upper, initial, cfg)
    weights, selection = np.clip(raw[:n],0,cfg.weight_bound), np.clip(raw[n:],0,1)
    if (abs(weights.mean()-selection.mean()) > cfg.mass_tolerance+cfg.feasibility_tolerance
            or selection.mean() < cfg.selection_fraction-cfg.feasibility_tolerance):
        raise KMMError("roundoff projection violates joint selection constraints")
    diagnostics.update({"source_n": n, "target_n": m, "bandwidth": float(bandwidth),
                        "weight_mean": float(weights.mean()), "selection_mass": float(selection.mean()),
                        "minimum_selection_mass": cfg.selection_fraction,
                        "mass_difference": float(weights.mean()-selection.mean()),
                        "ess": _ess(weights)})
    return weights, selection, diagnostics


def _weighted_quantile(scores, weights, alpha):
    positive = weights > 0
    if not positive.any():
        return math.inf
    scores, weights = np.asarray(scores)[positive], np.asarray(weights)[positive]
    order = np.argsort(scores, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, (1-alpha)*cumulative[-1], side="left"))
    return float(scores[order[index]])


def calibrate_sets(cal_probability, cal_labels, target_probability, weights, alpha, mode):
    """Empirical weighted LAC; global / proper candidate / literal paper rule.

    The paper-predicted rule is explicitly a pseudocode fidelity control and
    is not described as class-conditional valid. No query atom is added.
    """
    cp, tp = _probabilities(cal_probability), _probabilities(target_probability)
    y = _labels(cal_labels, len(cp))
    w = np.asarray(weights, float)
    if w.shape != (len(cp),) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be finite, nonnegative, aligned, and have positive total mass")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    cs, ts = candidate_scores(cp), candidate_scores(tp)
    true_scores = cs[np.arange(len(y)),y]
    if mode == "global":
        q = _weighted_quantile(true_scores,w,alpha)
        return ts <= q, {"global": q}
    thresholds = np.array([_weighted_quantile(true_scores[y==c],w[y==c],alpha) for c in (0,1)])
    if mode == "mondrian_candidate":
        member = ts <= thresholds
    elif mode == "mondrian_paper_predicted":
        predicted = (tp > .5).astype(int)  # class 0 wins exact probability tie.
        member = ts <= thresholds[predicted,None]
    else:
        raise ValueError(f"unknown calibration mode {mode}")
    return member, {str(c): float(thresholds[c]) for c in (0,1)}


def fit_predict(cal_x, cal_y, cal_p, target_x, target_p,
                alphas: Sequence[float] = (0.1,), seed: int = 0, config=None):
    """Run standard and selective KMM-CP with one density fit per target batch.

    Return raw predictions, full-population defer policies, and selection masks
    separately. Unselected SKMM rows are full sets in `predictions`. Their
    `raw_predictions` are threshold extrapolations for storage only, never
    evidence of valid predictions on rejected points. Source y is used only
    after every KMM/selection/bandwidth operation has finished.
    """
    cfg = _config(config)
    x,z = _features(cal_x,target_x)
    cp,tp = _probabilities(cal_p),_probabilities(target_p)
    y = _labels(cal_y,len(x))
    if len(cp)!=len(x) or len(tp)!=len(z):
        raise ValueError("feature and probability rows are not aligned")
    levels = tuple(float(a) for a in alphas)
    if not levels or len(set(levels)) != len(levels) or any(not math.isfinite(a) or not 0<a<1 for a in levels):
        raise ValueError("alphas must be unique values strictly between zero and one")
    if cfg.feature_scaling == "source_standard":
        scaler = StandardScaler().fit(x)
        x,z = scaler.transform(x),scaler.transform(z)
    bandwidth,bw_audit = select_bandwidth(x,z,seed=seed,config=cfg)
    standard_w,standard_audit = fit_standard_kmm(x,z,bandwidth,config=cfg)
    joint_w,selection,selection_audit = fit_joint_selection(x,z,bandwidth,config=cfg)
    selected = selection >= cfg.selection_threshold
    if not selected.any():
        raise KMMError("selective threshold retained zero targets; no unannounced top-k or baseline fallback")
    # Algorithm 1 uses the same kernel k for initial selection and KMM refit.
    selective_w,refit_audit = fit_standard_kmm(x,z[selected],bandwidth,config=cfg)
    predictions,raw_predictions,thresholds = {},{},{}
    method_selected = {name: (selected.copy() if name.startswith("skmm_") else np.ones(len(z),bool))
                       for name in METHOD_NAMES}
    for alpha in levels:
        predictions[alpha],raw_predictions[alpha],thresholds[alpha] = {},{},{}
        for name in METHOD_NAMES:
            selective = name.startswith("skmm_")
            mode = name.split("_cp_",1)[1]
            raw,q = calibrate_sets(cp,y,tp,selective_w if selective else standard_w,alpha,mode)
            full = raw.copy()
            if selective:
                full[~selected] = True
            predictions[alpha][name],raw_predictions[alpha][name],thresholds[alpha][name] = full,raw,q
    diagnostics = {
        "implementation": "independent_paper_based_full_postprediction_KMM_CP",
        "paper_url": PAPER_URL, "paper_pdf_sha256": PAPER_PDF_SHA256,
        "config": asdict(cfg), "seed": int(seed), "feature_dimension": x.shape[1],
        "bandwidth_selection": bw_audit, "standard_kmm": standard_audit,
        "joint_selection": selection_audit, "selected_refit": refit_audit,
        "selected_count": int(selected.sum()), "target_count": len(z),
        "retained_fraction": float(selected.mean()), "thresholds": thresholds,
        "source_classes": {str(c):int(np.sum(y==c)) for c in (0,1)},
        "zero_weight_classes": {k:[c for c in (0,1) if w[y==c].sum()==0]
                                for k,w in (("kmm",standard_w),("skmm",selective_w))},
        "query_atom_at_infinity": False, "finite_sample_arbitrary_shift_guarantee": False,
        "unselected_policy": "full_set_deferral_with_separate_selected_only_metrics",
        "backbone_requirement": "caller_supplied_probabilities_and_penultimate_embeddings; record actual backbone",
    }
    return {"predictions":predictions,"raw_predictions":raw_predictions,
            "selected":selected,"method_selected":method_selected,"selection_weights":selection,
            "source_weights":{"kmm":standard_w,"skmm":selective_w,"joint_selection":joint_w},
            "diagnostics":diagnostics}


def evaluate_selection(target_labels, raw_membership, selected):
    """Descriptive selected-only and all-population outcomes with denominators."""
    y = _labels(target_labels)
    raw = np.asarray(raw_membership)
    keep = np.asarray(selected)
    if (raw.shape != (len(y),2) or keep.shape != (len(y),)
            or not np.isin(keep,[0,1]).all() or not np.isin(raw,[0,1]).all()):
        raise ValueError("selection, sets, and labels must align")
    keep = keep.astype(bool)
    full = raw.copy()
    full[~keep] = True
    result = {"all_population":decision_metrics(y,full),
              "selected_only":decision_metrics(y[keep],raw[keep]),
              "selected_n":int(keep.sum()),"total_n":len(y),
              "selected_fraction":float(keep.mean()) if len(y) else math.nan}
    result["class_selection_fraction"] = {
        str(c): float(keep[y==c].mean()) if np.any(y==c) else math.nan for c in (0,1)}
    return result
