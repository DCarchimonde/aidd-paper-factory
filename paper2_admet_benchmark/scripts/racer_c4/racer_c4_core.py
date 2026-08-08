from __future__ import annotations

"""Pure numerical core for RACER-C4/TAME.

TAME is a transport-audited multi-view conformal envelope.  It estimates
calibration-to-target density ratios without target labels, audits each ratio
view, and unions every accepted weighted Mondrian set with the ordinary
Mondrian set.  The union gives a simple, testable set-inclusion invariant:
TAME can never remove a label included by its ordinary baseline.  When too few
transport views pass the label-free audit, transport is disabled and the
implementation returns the ordinary set, repairing only empty sets to the full
binary set.

Estimated density ratios do not create an exact finite-sample guarantee under
arbitrary shift.  The audit and set-inclusion invariant are operational safety
properties, not a new coverage theorem.
"""

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EPSILON = 1.0e-6


def clipped(values: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), epsilon, 1.0 - epsilon)


def binary_nonconformity(probability_1: np.ndarray) -> np.ndarray:
    """Return candidate-label LAC scores in columns ``[label 0, label 1]``."""

    p1 = clipped(probability_1)
    return np.column_stack([p1, 1.0 - p1])


def finite_sample_quantile(values: np.ndarray, alpha: float) -> float:
    scores = np.sort(np.asarray(values, dtype=float))
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if not len(scores):
        return math.inf
    rank = math.ceil((len(scores) + 1) * (1.0 - float(alpha)))
    return math.inf if rank > len(scores) else float(scores[rank - 1])


def mondrian_thresholds(
    calibration_scores: np.ndarray,
    calibration_targets: np.ndarray,
    alpha: float | Mapping[int, float],
) -> dict[int, float]:
    scores = np.asarray(calibration_scores, dtype=float)
    target = np.asarray(calibration_targets, dtype=np.int8)
    if scores.shape != (len(target), 2):
        raise ValueError("calibration_scores must have shape (n,2)")
    if np.any((target != 0) & (target != 1)):
        raise ValueError("calibration_targets must be binary")
    alpha_by_label = (
        {label: float(alpha[label]) for label in (0, 1)}
        if isinstance(alpha, Mapping)
        else {0: float(alpha), 1: float(alpha)}
    )
    return {
        label: finite_sample_quantile(
            scores[target == label, label], alpha_by_label[label]
        )
        for label in (0, 1)
    }


def threshold_membership(
    query_scores: np.ndarray, thresholds: Mapping[int, float]
) -> np.ndarray:
    scores = np.asarray(query_scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != 2:
        raise ValueError("query_scores must have shape (n,2)")
    return np.column_stack(
        [scores[:, label] <= float(thresholds[label]) for label in (0, 1)]
    )


def ordinary_mondrian_sets(
    calibration_scores: np.ndarray,
    calibration_targets: np.ndarray,
    query_scores: np.ndarray,
    alpha: float | Mapping[int, float],
) -> np.ndarray:
    return threshold_membership(
        query_scores,
        mondrian_thresholds(calibration_scores, calibration_targets, alpha),
    )


def weighted_test_thresholds(
    calibration_scores: np.ndarray,
    calibration_targets: np.ndarray,
    calibration_weights: np.ndarray,
    query_weights: np.ndarray,
    alpha: float | Mapping[int, float],
) -> np.ndarray:
    """Candidate-specific weighted Mondrian thresholds.

    The query weight is mass at infinity.  This matches the conservative
    weighted split-conformal quantile construction used in the frozen Paper 2
    shift diagnostic.
    """

    scores = np.asarray(calibration_scores, dtype=float)
    target = np.asarray(calibration_targets, dtype=np.int8)
    source_weight = np.asarray(calibration_weights, dtype=float)
    query_weight = np.asarray(query_weights, dtype=float)
    if scores.shape != (len(target), 2):
        raise ValueError("calibration_scores must have shape (n,2)")
    if source_weight.shape != (len(target),):
        raise ValueError("calibration_weights must align to calibration rows")
    if query_weight.ndim != 1:
        raise ValueError("query_weights must be a vector")
    if (
        not np.isfinite(source_weight).all()
        or not np.isfinite(query_weight).all()
        or np.any(source_weight <= 0.0)
        or np.any(query_weight <= 0.0)
    ):
        raise ValueError("all weights must be finite and positive")
    alpha_by_label = (
        {label: float(alpha[label]) for label in (0, 1)}
        if isinstance(alpha, Mapping)
        else {0: float(alpha), 1: float(alpha)}
    )
    output = np.full((len(query_weight), 2), math.inf, dtype=float)
    for label in (0, 1):
        mask = target == label
        if not mask.any():
            continue
        label_scores = scores[mask, label]
        label_weights = source_weight[mask]
        order = np.argsort(label_scores, kind="stable")
        ordered_scores = label_scores[order]
        cumulative = np.cumsum(label_weights[order])
        required_mass = (1.0 - alpha_by_label[label]) * (
            float(cumulative[-1]) + query_weight
        )
        positions = np.searchsorted(cumulative, required_mass, side="left")
        finite = positions < len(ordered_scores)
        output[finite, label] = ordered_scores[positions[finite]]
    return output


def weighted_mondrian_sets(
    calibration_scores: np.ndarray,
    calibration_targets: np.ndarray,
    query_scores: np.ndarray,
    calibration_weights: np.ndarray,
    query_weights: np.ndarray,
    alpha: float | Mapping[int, float],
) -> np.ndarray:
    scores = np.asarray(query_scores, dtype=float)
    thresholds = weighted_test_thresholds(
        calibration_scores,
        calibration_targets,
        calibration_weights,
        query_weights,
        alpha,
    )
    if scores.shape != thresholds.shape:
        raise ValueError("query score and threshold matrices differ")
    return scores <= thresholds


def effective_sample_size(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or not len(values):
        return 0.0
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        return 0.0
    return float(values.sum() ** 2 / np.square(values).sum())


def standardized_mean_gap(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_weights: np.ndarray | None = None,
) -> float:
    source = np.asarray(source_features, dtype=float)
    target = np.asarray(target_features, dtype=float)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("source/target feature dimensions differ")
    combined = np.vstack([source, target])
    scale = np.std(combined, axis=0)
    active = scale > 1.0e-12
    if not active.any():
        return 0.0
    if source_weights is None:
        source_mean = np.mean(source[:, active], axis=0)
    else:
        weight = np.asarray(source_weights, dtype=float)
        if weight.shape != (len(source),):
            raise ValueError("source_weights do not align to source features")
        source_mean = np.average(source[:, active], axis=0, weights=weight)
    target_mean = np.mean(target[:, active], axis=0)
    standardized = (source_mean - target_mean) / scale[active]
    return float(np.sqrt(np.mean(np.square(standardized))))


@dataclass(frozen=True)
class DensityRatioResult:
    source_weights: np.ndarray
    target_weights: np.ndarray
    domain_auc: float
    source_ess: float
    class_0_ess: float
    class_1_ess: float
    lower_clip_fraction: float
    upper_clip_fraction: float
    mean_gap_before: float
    mean_gap_after: float


def cross_fitted_density_ratio(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_targets: np.ndarray,
    *,
    seed: int,
    folds: int = 5,
    c_value: float = 1.0,
    weight_min: float = 0.05,
    weight_max: float = 20.0,
) -> DensityRatioResult:
    """Estimate label-free target/source ratios with an honest domain audit.

    Cross-fitted probabilities supply the domain-AUC diagnostic.  The fixed
    regularized model is then refit on all unlabeled domain covariates to obtain
    the actual ratios.  Calibration outcome labels never enter either fit.
    """

    source = np.asarray(source_features, dtype=np.float32)
    target = np.asarray(target_features, dtype=np.float32)
    labels = np.asarray(source_targets, dtype=np.int8)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("source/target feature dimensions differ")
    if labels.shape != (len(source),) or np.any((labels != 0) & (labels != 1)):
        raise ValueError("source_targets must be an aligned binary vector")
    if min(len(source), len(target)) < 2:
        raise ValueError("each domain requires at least two rows")
    if not 0.0 < weight_min <= weight_max:
        raise ValueError("require 0 < weight_min <= weight_max")
    domain_x = np.vstack([source, target])
    domain_y = np.concatenate(
        [np.zeros(len(source), dtype=np.int8), np.ones(len(target), dtype=np.int8)]
    )
    n_splits = min(int(folds), int(np.bincount(domain_y).min()))
    if n_splits < 2:
        raise ValueError("domain cross-fitting requires at least two folds")
    oof_probability = np.empty(len(domain_y), dtype=float)
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=int(seed) + 9100
    )
    def make_model() -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler(with_mean=False)),
                (
                    "domain",
                    LogisticRegression(
                        C=float(c_value),
                        solver="liblinear",
                        max_iter=2000,
                        class_weight=None,
                        random_state=int(seed) + 9100,
                    ),
                ),
            ]
        )

    for fit, query in splitter.split(domain_x, domain_y):
        model = make_model()
        model.fit(domain_x[fit], domain_y[fit])
        oof_probability[query] = model.predict_proba(domain_x[query])[:, 1]
    final_model = make_model()
    final_model.fit(domain_x, domain_y)
    probability = clipped(final_model.predict_proba(domain_x)[:, 1])
    raw_ratio = probability / (1.0 - probability) * len(source) / len(target)
    ratio = np.clip(raw_ratio, float(weight_min), float(weight_max))
    source_weight = ratio[: len(source)]
    target_weight = ratio[len(source) :]
    return DensityRatioResult(
        source_weights=source_weight,
        target_weights=target_weight,
        domain_auc=float(roc_auc_score(domain_y, oof_probability)),
        source_ess=effective_sample_size(source_weight),
        class_0_ess=effective_sample_size(source_weight[labels == 0]),
        class_1_ess=effective_sample_size(source_weight[labels == 1]),
        lower_clip_fraction=float(np.mean(raw_ratio <= float(weight_min))),
        upper_clip_fraction=float(np.mean(raw_ratio >= float(weight_max))),
        mean_gap_before=standardized_mean_gap(source, target),
        mean_gap_after=standardized_mean_gap(source, target, source_weight),
    )


@dataclass(frozen=True)
class TransportAudit:
    view: str
    active: bool
    reason: str
    source_n: int
    target_n: int
    class_0_n: int
    class_1_n: int
    source_ess: float
    class_0_ess: float
    class_1_ess: float
    domain_auc: float
    lower_clip_fraction: float
    upper_clip_fraction: float
    mean_gap_before: float
    mean_gap_after: float


def audit_density_ratio(
    view: str,
    result: DensityRatioResult,
    source_targets: np.ndarray,
    target_n: int,
    *,
    minimum_source_n: int = 100,
    minimum_target_n: int = 100,
    minimum_class_n: int = 20,
    minimum_ess_fraction: float = 0.25,
    maximum_clip_fraction: float = 0.30,
    maximum_domain_auc: float = 0.99,
    maximum_gap_ratio: float = 1.05,
) -> TransportAudit:
    labels = np.asarray(source_targets, dtype=np.int8)
    counts = [int(np.sum(labels == label)) for label in (0, 1)]
    failures: list[str] = []
    if len(labels) < int(minimum_source_n):
        failures.append("source_n")
    if int(target_n) < int(minimum_target_n):
        failures.append("target_n")
    if any(value < int(minimum_class_n) for value in counts):
        failures.append("class_n")
    class_ess = [result.class_0_ess, result.class_1_ess]
    if any(
        ess < max(2.0, float(minimum_ess_fraction) * count)
        for ess, count in zip(class_ess, counts)
    ):
        failures.append("class_ess")
    if result.source_ess < float(minimum_ess_fraction) * len(labels):
        failures.append("source_ess")
    if max(result.lower_clip_fraction, result.upper_clip_fraction) > float(
        maximum_clip_fraction
    ):
        failures.append("weight_clipping")
    if result.domain_auc > float(maximum_domain_auc):
        failures.append("domain_separation")
    allowed_gap = max(1.0e-12, result.mean_gap_before * float(maximum_gap_ratio))
    if result.mean_gap_after > allowed_gap:
        failures.append("balance")
    reason = "active" if not failures else "+".join(failures)
    return TransportAudit(
        view=str(view),
        active=not failures,
        reason=reason,
        source_n=len(labels),
        target_n=int(target_n),
        class_0_n=counts[0],
        class_1_n=counts[1],
        source_ess=result.source_ess,
        class_0_ess=result.class_0_ess,
        class_1_ess=result.class_1_ess,
        domain_auc=result.domain_auc,
        lower_clip_fraction=result.lower_clip_fraction,
        upper_clip_fraction=result.upper_clip_fraction,
        mean_gap_before=result.mean_gap_before,
        mean_gap_after=result.mean_gap_after,
    )


@dataclass(frozen=True)
class EnvelopeDecision:
    membership: np.ndarray
    active_views: tuple[str, ...]
    failed_closed: bool


def transport_envelope(
    ordinary_sets: np.ndarray,
    weighted_sets: Sequence[np.ndarray],
    audits: Sequence[TransportAudit],
    *,
    protected_labels: Sequence[int] = (0, 1),
    quorum: str = "any",
    minimum_active_views: int = 1,
) -> EnvelopeDecision:
    """Augment a baseline using audited views, or fail closed to the baseline.

    ``quorum='any'`` is a conservative union across views.  ``quorum='all'``
    only adds a protected label when every active view includes it.  In either
    case baseline-empty rows become full sets, so augmentation cannot create a
    new singleton or a new wrong-singleton exposure.
    """

    baseline = np.asarray(ordinary_sets, dtype=bool)
    if baseline.ndim != 2 or baseline.shape[1] != 2:
        raise ValueError("ordinary_sets must have shape (n,2)")
    if len(weighted_sets) != len(audits):
        raise ValueError("weighted_sets and audits must have equal length")
    protected = tuple(sorted(set(int(value) for value in protected_labels)))
    if any(value not in (0, 1) for value in protected):
        raise ValueError("protected_labels may contain only zero and one")
    if quorum not in {"any", "all"}:
        raise ValueError("quorum must be 'any' or 'all'")
    if int(minimum_active_views) <= 0:
        raise ValueError("minimum_active_views must be positive")
    candidates: list[np.ndarray] = []
    active: list[str] = []
    for candidate, audit in zip(weighted_sets, audits):
        values = np.asarray(candidate, dtype=bool)
        if values.shape != baseline.shape:
            raise ValueError("a weighted set matrix has the wrong shape")
        if audit.active:
            candidates.append(values)
            active.append(audit.view)
    if len(active) < int(minimum_active_views):
        # Close the transport route without pretending that an unaudited view
        # is safe.  Preserve the ordinary method exactly except that an empty
        # baseline set becomes the non-actionable full set.
        output = baseline.copy()
        output[baseline.sum(axis=1) == 0] = True
        return EnvelopeDecision(output, tuple(active), True)
    output = baseline.copy()
    stacked = np.stack(candidates, axis=0)
    proposed = np.any(stacked, axis=0) if quorum == "any" else np.all(stacked, axis=0)
    for label in protected:
        output[:, label] |= proposed[:, label]
    # A baseline empty set is never allowed to become a new singleton.  Making
    # it full is the deterministic, label-free fail-closed action.
    output[baseline.sum(axis=1) == 0] = True
    if np.any(baseline & ~output):
        raise RuntimeError("transport envelope violated baseline set inclusion")
    if np.any(output.sum(axis=1) == 0):
        raise RuntimeError("transport envelope emitted an empty set")
    return EnvelopeDecision(output, tuple(active), False)


def set_metrics(targets: np.ndarray, membership: np.ndarray) -> dict[str, float | int]:
    target = np.asarray(targets, dtype=np.int8)
    member = np.asarray(membership, dtype=bool)
    if member.shape != (len(target), 2):
        raise ValueError("membership must have shape (n,2)")
    if np.any((target != 0) & (target != 1)):
        raise ValueError("targets must be binary")
    size = member.sum(axis=1)
    output: dict[str, float | int] = {"n": int(len(target))}
    class_csy: list[float] = []
    class_coverage: list[float] = []
    for label in (0, 1):
        mask = target == label
        count = int(mask.sum())
        output[f"class_{label}_n"] = count
        if not count:
            output[f"class_{label}_coverage"] = math.nan
            output[f"class_{label}_csy"] = math.nan
            output[f"class_{label}_wrong_singleton_exposure"] = math.nan
            continue
        coverage = float(np.mean(member[mask, label]))
        correct = float(np.mean((size[mask] == 1) & member[mask, label]))
        wrong = float(np.mean((size[mask] == 1) & member[mask, 1 - label]))
        output[f"class_{label}_coverage"] = coverage
        output[f"class_{label}_csy"] = correct
        output[f"class_{label}_wrong_singleton_exposure"] = wrong
        class_coverage.append(coverage)
        class_csy.append(correct)
    output["macro_coverage"] = float(np.mean(class_coverage))
    output["minimum_class_coverage"] = float(np.min(class_coverage))
    output["macro_csy"] = float(np.mean(class_csy))
    output["worst_csy"] = float(np.min(class_csy))
    output["ambiguous_rate"] = float(np.mean(size == 2))
    output["empty_rate"] = float(np.mean(size == 0))
    output["singleton_rate"] = float(np.mean(size == 1))
    return output
