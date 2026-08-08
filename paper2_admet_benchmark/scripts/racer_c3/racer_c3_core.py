from __future__ import annotations

"""Numerical core for the development-only RACER-C3 candidate.

The module contains no file I/O and never accepts deployment labels when it
constructs the score or chooses the batch route.  A separate retrospective
runner is responsible for applying the fixed candidate to already-known v1
outcomes.
"""

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


PROBABILITY_COLUMNS = (
    "ecfp_p",
    "dmpnn_p",
    "molformer_p",
    "stack_p",
    "unrestricted_p",
)
RISK_COLUMNS = (
    "disagreement",
    "ecfp_distance",
    "local_oof_brier_loss",
    "bri",
    "risk_percentile",
)


def clipped(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1.0e-6, 1.0 - 1.0e-6)


def validate_probability_matrix(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(PROBABILITY_COLUMNS):
        raise ValueError(
            f"probabilities must have shape (n,{len(PROBABILITY_COLUMNS)})"
        )
    if not np.isfinite(values).all():
        raise ValueError("probabilities contain a non-finite value")
    return clipped(values)


def validate_risk_matrix(risk: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(risk, dtype=float)
    if values.shape != (n, len(RISK_COLUMNS)):
        raise ValueError(f"risk must have shape ({n},{len(RISK_COLUMNS)})")
    if not np.isfinite(values).all():
        raise ValueError("risk contains a non-finite value")
    return values


def temper_support(
    support_probability: np.ndarray,
    risk_percentile: np.ndarray,
    t_max: float,
) -> np.ndarray:
    """Continuously attenuate candidate support toward one half at high risk."""

    if t_max < 1.0:
        raise ValueError("t_max must be at least one")
    support = clipped(support_probability)
    risk = np.asarray(risk_percentile, dtype=float)
    if support.shape != risk.shape:
        raise ValueError("support_probability and risk_percentile shapes differ")
    if not np.isfinite(risk).all() or np.any((risk < 0.0) | (risk > 1.0)):
        raise ValueError("risk_percentile must be finite and lie in [0,1]")
    temperature = 1.0 + (float(t_max) - 1.0) * risk
    return clipped(expit(logit(support) / temperature))


def fallback_scores(
    stack_probability: np.ndarray,
    risk_percentile: np.ndarray,
    t_max: float = 1.5,
) -> np.ndarray:
    """Exact score fallback to the v1 no-gate RACER attenuation."""

    p1 = temper_support(stack_probability, risk_percentile, t_max)
    return np.column_stack([p1, 1.0 - p1])


def robust_logit_mean_probability(probabilities: np.ndarray) -> np.ndarray:
    """Equal-weight logit pool of ECFP, D-MPNN, and MoLFormer views."""

    values = validate_probability_matrix(probabilities)
    return clipped(expit(np.mean(logit(values[:, :3]), axis=1)))


def candidate_features(
    probabilities: np.ndarray,
    risk: np.ndarray,
    candidate_label: int,
) -> np.ndarray:
    """Features for the shared candidate-correctness expert."""

    values = validate_probability_matrix(probabilities)
    risk_values = validate_risk_matrix(risk, len(values))
    if candidate_label not in (0, 1):
        raise ValueError("candidate_label must be zero or one")
    candidate_probability = values if candidate_label == 1 else 1.0 - values
    candidate_logits = logit(clipped(candidate_probability))
    first_three = candidate_probability[:, :3]
    mean_probability = np.mean(first_three, axis=1, keepdims=True)
    minimum_probability = np.min(first_three, axis=1, keepdims=True)
    maximum_probability = np.max(first_three, axis=1, keepdims=True)
    spread = maximum_probability - minimum_probability
    stack_nonconformity = 1.0 - candidate_probability[:, 3:4]
    label_column = np.full((len(values), 1), float(candidate_label))
    return np.column_stack(
        [
            candidate_logits,
            mean_probability,
            minimum_probability,
            maximum_probability,
            spread,
            risk_values,
            risk_values * stack_nonconformity,
            label_column,
            label_column * risk_values,
        ]
    )


def candidate_training_block(
    probabilities: np.ndarray,
    risk: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.asarray(targets, dtype=np.int8)
    if target.shape != (len(probabilities),) or np.any((target != 0) & (target != 1)):
        raise ValueError("targets must be a binary vector aligned to probabilities")
    features = np.vstack(
        [candidate_features(probabilities, risk, label) for label in (0, 1)]
    )
    correctness = np.concatenate([(target == label).astype(np.int8) for label in (0, 1)])
    labels = np.concatenate(
        [np.full(len(target), label, dtype=np.int8) for label in (0, 1)]
    )
    return features, correctness, labels


def balanced_candidate_weights(
    correctness: np.ndarray, candidate_labels: np.ndarray
) -> np.ndarray:
    correct = np.asarray(correctness, dtype=np.int8)
    labels = np.asarray(candidate_labels, dtype=np.int8)
    if correct.shape != labels.shape:
        raise ValueError("correctness and candidate_labels shapes differ")
    keys = labels * 2 + correct
    counts = np.bincount(keys, minlength=4)
    if np.any(counts == 0):
        raise ValueError("all candidate-label/correctness cells require positive count")
    return len(keys) / (4.0 * counts[keys])


def make_candidate_model(c_value: float, seed: int) -> Pipeline:
    if c_value <= 0.0:
        raise ValueError("c_value must be positive")
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=float(c_value), max_iter=2500, random_state=int(seed)),
    )


@dataclass(frozen=True)
class CandidateCorrectnessEnsemble:
    models: tuple[Pipeline, ...]

    def predict_correctness(
        self, probabilities: np.ndarray, risk: np.ndarray
    ) -> np.ndarray:
        if not self.models:
            raise ValueError("candidate correctness ensemble is empty")
        predictions = []
        for model in self.models:
            predictions.append(
                np.column_stack(
                    [
                        model.predict_proba(candidate_features(probabilities, risk, label))[
                            :, 1
                        ]
                        for label in (0, 1)
                    ]
                )
            )
        return clipped(np.mean(np.stack(predictions), axis=0))


@dataclass(frozen=True)
class CandidateCorrectnessCrossfit:
    oof_correctness: np.ndarray
    ensemble: CandidateCorrectnessEnsemble


def crossfit_candidate_correctness(
    probabilities: np.ndarray,
    risk: np.ndarray,
    targets: np.ndarray,
    folds: np.ndarray,
    seed: int,
    c_value: float = 0.1,
) -> CandidateCorrectnessCrossfit:
    values = validate_probability_matrix(probabilities)
    risk_values = validate_risk_matrix(risk, len(values))
    target = np.asarray(targets, dtype=np.int8)
    fold_values = np.asarray(folds, dtype=int)
    if target.shape != (len(values),) or fold_values.shape != (len(values),):
        raise ValueError("targets/folds must align to probabilities")
    unique_folds = sorted(set(fold_values.tolist()))
    if len(unique_folds) < 2:
        raise ValueError("at least two development folds are required")
    oof = np.full((len(values), 2), np.nan, dtype=float)
    models: list[Pipeline] = []
    for position, heldout in enumerate(unique_folds):
        fit = fold_values != heldout
        query = ~fit
        x_fit, y_fit, label_fit = candidate_training_block(
            values[fit], risk_values[fit], target[fit]
        )
        model = make_candidate_model(c_value, int(seed) + 100 * position)
        model.fit(
            x_fit,
            y_fit,
            logisticregression__sample_weight=balanced_candidate_weights(
                y_fit, label_fit
            ),
        )
        for label in (0, 1):
            oof[query, label] = model.predict_proba(
                candidate_features(values[query], risk_values[query], label)
            )[:, 1]
        models.append(model)
    if not np.isfinite(oof).all():
        raise RuntimeError("candidate correctness OOF predictions are incomplete")
    return CandidateCorrectnessCrossfit(
        oof_correctness=clipped(oof),
        ensemble=CandidateCorrectnessEnsemble(tuple(models)),
    )


def frontier_scores(
    probabilities: np.ndarray,
    risk: np.ndarray,
    candidate_correctness: np.ndarray,
    risk_percentile: np.ndarray,
    t_max_0: float = 1.5,
    t_max_1: float = 1.5,
) -> np.ndarray:
    """Construct the asymmetric frontier score without using a hard class."""

    values = validate_probability_matrix(probabilities)
    validate_risk_matrix(risk, len(values))
    correctness = clipped(candidate_correctness)
    if correctness.shape != (len(values), 2):
        raise ValueError("candidate_correctness must have shape (n,2)")
    robust_p1 = robust_logit_mean_probability(values)
    support_0 = 1.0 - robust_p1
    support_1 = correctness[:, 1]
    tempered_0 = temper_support(support_0, risk_percentile, t_max_0)
    tempered_1 = temper_support(support_1, risk_percentile, t_max_1)
    return np.column_stack([1.0 - tempered_0, 1.0 - tempered_1])


@dataclass(frozen=True)
class FrontierGateDecision:
    active: bool
    overlap_fraction: float
    median_ecfp_distance: float
    valid_union_n: int
    reason: str


def symmetric_frontier_gate(
    development_scaffolds: Sequence[str],
    calibration_and_batch_scaffolds: Sequence[str],
    calibration_and_batch_ecfp_distance: np.ndarray,
    overlap_fraction_max: float = 0.05,
    median_ecfp_distance_min: float = 0.57,
    minimum_valid_union_n: int = 100,
) -> FrontierGateDecision:
    """Choose one route from a permutation-invariant unlabeled union summary."""

    if not 0.0 <= overlap_fraction_max <= 1.0:
        raise ValueError("overlap_fraction_max must lie in [0,1]")
    if minimum_valid_union_n <= 0:
        raise ValueError("minimum_valid_union_n must be positive")
    development = {str(value).strip() for value in development_scaffolds if str(value).strip()}
    union = np.asarray(
        [str(value).strip() for value in calibration_and_batch_scaffolds], dtype=object
    )
    distance = np.asarray(calibration_and_batch_ecfp_distance, dtype=float)
    if union.shape != distance.shape:
        raise ValueError("union scaffolds and distances shapes differ")
    valid = np.asarray([bool(value) for value in union]) & np.isfinite(distance)
    valid_n = int(valid.sum())
    if not development:
        return FrontierGateDecision(False, math.nan, math.nan, valid_n, "empty_development_reference")
    if valid_n < minimum_valid_union_n:
        return FrontierGateDecision(False, math.nan, math.nan, valid_n, "insufficient_union")
    valid_scaffolds = union[valid]
    overlap = float(np.mean(np.asarray([value in development for value in valid_scaffolds])))
    median_distance = float(np.median(distance[valid]))
    active = overlap <= overlap_fraction_max and median_distance >= median_ecfp_distance_min
    reason = "frontier_active" if active else "fallback_not_jointly_frontier"
    return FrontierGateDecision(active, overlap, median_distance, valid_n, reason)


def routed_scores(
    fallback: np.ndarray,
    frontier: np.ndarray,
    decision: FrontierGateDecision,
) -> np.ndarray:
    fallback_values = np.asarray(fallback, dtype=float)
    frontier_values = np.asarray(frontier, dtype=float)
    if fallback_values.shape != frontier_values.shape or fallback_values.ndim != 2:
        raise ValueError("fallback and frontier score matrices must have equal shape")
    return frontier_values.copy() if decision.active else fallback_values.copy()


def finite_sample_quantile(values: np.ndarray, alpha: float) -> float:
    scores = np.sort(np.asarray(values, dtype=float))
    if not len(scores):
        return math.inf
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    rank = math.ceil((len(scores) + 1) * (1.0 - float(alpha)))
    return math.inf if rank > len(scores) else float(scores[rank - 1])


def mondrian_thresholds(
    calibration_scores: np.ndarray,
    calibration_targets: np.ndarray,
    alpha: float | dict[int, float],
) -> dict[int, float]:
    scores = np.asarray(calibration_scores, dtype=float)
    target = np.asarray(calibration_targets, dtype=np.int8)
    if scores.shape != (len(target), 2):
        raise ValueError("calibration_scores must have shape (n,2)")
    alpha_by_label = (
        {label: float(alpha[label]) for label in (0, 1)}
        if isinstance(alpha, dict)
        else {0: float(alpha), 1: float(alpha)}
    )
    return {
        label: finite_sample_quantile(
            scores[target == label, label], alpha_by_label[label]
        )
        for label in (0, 1)
    }


def prediction_membership(
    query_scores: np.ndarray, thresholds: dict[int, float]
) -> np.ndarray:
    scores = np.asarray(query_scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != 2:
        raise ValueError("query_scores must have shape (n,2)")
    return np.column_stack(
        [scores[:, label] <= float(thresholds[label]) for label in (0, 1)]
    )


def set_metrics(targets: np.ndarray, membership: np.ndarray) -> dict[str, float]:
    target = np.asarray(targets, dtype=np.int8)
    member = np.asarray(membership, dtype=bool)
    if member.shape != (len(target), 2):
        raise ValueError("membership must have shape (n,2)")
    singleton_0 = member[:, 0] & ~member[:, 1]
    singleton_1 = member[:, 1] & ~member[:, 0]
    ambiguous = member[:, 0] & member[:, 1]
    empty = ~member[:, 0] & ~member[:, 1]
    output: dict[str, float] = {}
    class_csy = []
    class_coverage = []
    for label, singleton in ((0, singleton_0), (1, singleton_1)):
        mask = target == label
        if not mask.any():
            raise ValueError(f"target class {label} is empty")
        csy = float(np.mean(singleton[mask]))
        coverage = float(np.mean(member[mask, label]))
        wrong_singleton = singleton_1 if label == 0 else singleton_0
        output[f"class_{label}_csy"] = csy
        output[f"class_{label}_coverage"] = coverage
        output[f"class_{label}_wrong_singleton_exposure"] = float(
            np.mean(wrong_singleton[mask])
        )
        class_csy.append(csy)
        class_coverage.append(coverage)
    output["macro_csy"] = float(np.mean(class_csy))
    output["worst_csy"] = float(np.min(class_csy))
    output["macro_coverage"] = float(np.mean(class_coverage))
    output["ambiguous_rate"] = float(np.mean(ambiguous))
    output["empty_rate"] = float(np.mean(empty))
    return output
