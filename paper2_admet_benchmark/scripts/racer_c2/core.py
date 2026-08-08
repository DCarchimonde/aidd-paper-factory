from __future__ import annotations

"""Pure numerical components for RACER-C2 development.

RACER-C2 replaces the v1 predicted-class gate with candidate-label reliability
tilting and certifies the actual set-valued decisions.  This file deliberately
contains no test-result discovery, endpoint selection, or filesystem
orchestration.
"""

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.special import expit, logit
from scipy.stats import beta
from sklearn.ensemble import HistGradientBoostingRegressor


EPSILON = 1.0e-6


@dataclass(frozen=True, order=True)
class ScoreConfiguration:
    """One member of the development-only RACER-C2 score family."""

    t_max: float
    gamma_0: float
    gamma_1: float
    counterfactual_blend: float = 0.0

    def validate(self) -> None:
        if self.t_max < 1.0:
            raise ValueError("t_max must be at least one")
        if not math.isfinite(self.gamma_0) or not math.isfinite(self.gamma_1):
            raise ValueError("candidate-label reliability tilts must be finite")
        if not 0.0 <= self.counterfactual_blend <= 1.0:
            raise ValueError("counterfactual_blend must lie in [0,1]")


@dataclass(frozen=True)
class ActionCertificateConstraints:
    """Bounds applied to final conformal states, never base hard classes."""

    familywise_alpha: float = 0.05
    coverage_floor: float | None = None
    wrong_singleton_ceiling: float | None = 0.10
    empty_exposure_ceiling: float | None = None
    critical_class: int = 1
    critical_csy_floor: float | None = None
    minimum_true_class_n: int = 25

    def validate(self) -> None:
        if self.critical_class not in {0, 1}:
            raise ValueError("critical_class must be binary")
        if not 0.0 < self.familywise_alpha < 1.0:
            raise ValueError("familywise_alpha must lie in (0,1)")
        if self.minimum_true_class_n < 1:
            raise ValueError("minimum_true_class_n must be positive")
        for name in (
            "coverage_floor",
            "wrong_singleton_ceiling",
            "empty_exposure_ceiling",
            "critical_csy_floor",
        ):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")


def clipped(values: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), epsilon, 1.0 - epsilon)


def attenuated_probability(
    stack_probability: np.ndarray,
    reliability_percentile: np.ndarray,
    t_max: float,
) -> np.ndarray:
    """The v1-compatible safe baseline used inside the v2 score family."""

    if t_max < 1.0:
        raise ValueError("t_max must be at least one")
    probability = clipped(stack_probability)
    risk = np.asarray(reliability_percentile, dtype=float)
    if probability.shape != risk.shape or np.any((risk < 0.0) | (risk > 1.0)):
        raise ValueError("probability and reliability percentiles are incompatible")
    temperature = 1.0 + (float(t_max) - 1.0) * risk
    return clipped(expit(logit(probability) / temperature))


def candidate_nonconformity(probability_class_1: np.ndarray) -> np.ndarray:
    """Return scores for candidate labels 0 and 1, in that order."""

    probability = clipped(probability_class_1)
    return np.column_stack([probability, 1.0 - probability])


def tilt_candidate_scores(
    candidate_scores: np.ndarray,
    reliability_percentile: np.ndarray,
    gamma_0: float,
    gamma_1: float,
) -> np.ndarray:
    """Apply class-specific exponential reliability tilting.

    For candidate label ``y``, RACER-C2 uses

    ``s_y(x) = a_y(x) * exp(gamma_y * r(x))``.

    A positive ``gamma_y`` makes high-risk rows less conforming for candidate
    label ``y``; a negative value protects that candidate label as reliability
    risk rises.  ``gamma_0=gamma_1=0`` is an exact, bitwise-safe fallback.
    """

    scores = np.asarray(candidate_scores, dtype=float)
    risk = np.asarray(reliability_percentile, dtype=float)
    gammas = np.asarray([gamma_0, gamma_1], dtype=float)
    if scores.ndim != 2 or scores.shape[1] != 2:
        raise ValueError("candidate scores must have shape (n,2)")
    if risk.shape != (len(scores),):
        raise ValueError("reliability percentile must have one value per row")
    if not np.isfinite(scores).all() or not np.isfinite(risk).all():
        raise ValueError("candidate scores and reliability percentiles must be finite")
    if np.any((risk < 0.0) | (risk > 1.0)):
        raise ValueError("reliability percentiles must lie in [0,1]")
    if not np.isfinite(gammas).all():
        raise ValueError("candidate-label reliability tilts must be finite")
    if gamma_0 == 0.0 and gamma_1 == 0.0:
        return scores.copy()
    tilted = scores * np.exp(risk[:, None] * gammas[None, :])
    if not np.isfinite(tilted).all():
        raise ValueError("candidate-label reliability tilting overflowed")
    return tilted


def candidate_feature_matrix(
    probability_columns: np.ndarray,
    reliability_columns: np.ndarray,
    candidate_label: int,
) -> np.ndarray:
    """Build monotone counterfactual-error features for one candidate label.

    Every probability column is converted to nonconformity for the candidate
    label.  Larger values of every returned feature must therefore be
    interpretable as no safer than smaller values.
    """

    if candidate_label not in {0, 1}:
        raise ValueError("candidate_label must be binary")
    probabilities = clipped(np.asarray(probability_columns, dtype=float))
    reliability = np.asarray(reliability_columns, dtype=float)
    if probabilities.ndim != 2 or reliability.ndim != 2:
        raise ValueError("candidate features require two-dimensional arrays")
    if len(probabilities) != len(reliability):
        raise ValueError("probability/reliability row counts differ")
    if not np.isfinite(reliability).all():
        raise ValueError("reliability features must be finite")
    candidate_probability = probabilities if candidate_label == 1 else 1.0 - probabilities
    return np.column_stack([1.0 - candidate_probability, reliability])


def balanced_binary_weights(targets: np.ndarray) -> np.ndarray:
    values = np.asarray(targets, dtype=np.int8)
    counts = Counter(int(value) for value in values)
    if set(counts) != {0, 1}:
        raise ValueError("candidate-error fitting requires both outcome classes")
    return np.asarray(
        [len(values) / (2.0 * counts[int(value)]) for value in values],
        dtype=float,
    )


def fit_candidate_error_model(
    features: np.ndarray,
    candidate_errors: np.ndarray,
    seed: int,
) -> HistGradientBoostingRegressor:
    """Fit the low-capacity monotone counterfactual error index."""

    x = np.asarray(features, dtype=float)
    error = np.asarray(candidate_errors, dtype=np.int8)
    if x.ndim != 2 or len(x) != len(error) or not np.isfinite(x).all():
        raise ValueError("candidate-error training arrays are invalid")
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=80,
        max_leaf_nodes=7,
        min_samples_leaf=20,
        l2_regularization=1.0,
        monotonic_cst=[1] * x.shape[1],
        random_state=int(seed),
    )
    model.fit(x, error, sample_weight=balanced_binary_weights(error))
    return model


def predict_candidate_error(model: object, features: np.ndarray) -> np.ndarray:
    values = np.asarray(model.predict(np.asarray(features, dtype=float)), dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("candidate-error predictions must be finite and one-dimensional")
    return np.clip(values, 0.0, 1.0)


def crossfit_candidate_error_scores(
    probability_columns: np.ndarray,
    reliability_columns: np.ndarray,
    targets: np.ndarray,
    fold_ids: np.ndarray,
    seed: int,
    external_probability_columns: np.ndarray | None = None,
    external_reliability_columns: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return honest development scores and an optional fold ensemble.

    A development row is scored only by a model that excluded its meta-fold.
    External rows are averaged over the same fold-specific models.  The target
    is counterfactual: for candidate label ``y`` it is ``1[Y != y]``.
    """

    probability = np.asarray(probability_columns, dtype=float)
    reliability = np.asarray(reliability_columns, dtype=float)
    y = np.asarray(targets, dtype=np.int8)
    folds = np.asarray(fold_ids)
    if len(probability) != len(y) or len(reliability) != len(y) or len(folds) != len(y):
        raise ValueError("development arrays have inconsistent row counts")
    unique_folds = sorted(set(folds.tolist()))
    if len(unique_folds) < 2:
        raise ValueError("at least two development folds are required")
    use_external = external_probability_columns is not None or external_reliability_columns is not None
    if use_external and (
        external_probability_columns is None or external_reliability_columns is None
    ):
        raise ValueError("both external feature blocks are required")
    external_probability = (
        np.asarray(external_probability_columns, dtype=float) if use_external else None
    )
    external_reliability = (
        np.asarray(external_reliability_columns, dtype=float) if use_external else None
    )
    if use_external and len(external_probability) != len(external_reliability):
        raise ValueError("external feature blocks have inconsistent row counts")

    oof = np.full((len(y), 2), np.nan, dtype=float)
    external_by_fold: list[np.ndarray] = []
    for fold_position, heldout_fold in enumerate(unique_folds):
        fit = folds != heldout_fold
        heldout = folds == heldout_fold
        fold_external: list[np.ndarray] = []
        for label in (0, 1):
            fit_features = candidate_feature_matrix(probability[fit], reliability[fit], label)
            heldout_features = candidate_feature_matrix(
                probability[heldout], reliability[heldout], label
            )
            errors = (y[fit] != label).astype(np.int8)
            model = fit_candidate_error_model(
                fit_features,
                errors,
                int(seed) + fold_position * 10 + label,
            )
            oof[heldout, label] = predict_candidate_error(model, heldout_features)
            if use_external:
                fold_external.append(
                    predict_candidate_error(
                        model,
                        candidate_feature_matrix(
                            external_probability, external_reliability, label
                        ),
                    )
                )
        if use_external:
            external_by_fold.append(np.column_stack(fold_external))
    if not np.isfinite(oof).all():
        raise RuntimeError("counterfactual development scores are incomplete")
    external = (
        np.mean(np.stack(external_by_fold, axis=0), axis=0)
        if use_external
        else None
    )
    return np.clip(oof, 0.0, 1.0), external


def reference_midrank_percentiles(
    reference_scores: np.ndarray,
    query_scores: np.ndarray,
) -> np.ndarray:
    """Map both candidate-label scores against a fixed development reference."""

    reference = np.asarray(reference_scores, dtype=float)
    query = np.asarray(query_scores, dtype=float)
    if reference.ndim != 2 or query.ndim != 2 or reference.shape[1] != 2 or query.shape[1] != 2:
        raise ValueError("candidate score matrices must have shape (n,2)")
    if not len(reference) or not np.isfinite(reference).all() or not np.isfinite(query).all():
        raise ValueError("candidate score matrices must be nonempty and finite")
    output = np.empty_like(query, dtype=float)
    for label in (0, 1):
        ordered = np.sort(reference[:, label])
        left = np.searchsorted(ordered, query[:, label], side="left")
        right = np.searchsorted(ordered, query[:, label], side="right")
        output[:, label] = (left + right) / (2.0 * len(ordered))
    return np.clip(output, 0.0, 1.0)


def compose_candidate_scores(
    stack_probability: np.ndarray,
    reliability_percentile: np.ndarray,
    counterfactual_percentiles: np.ndarray | None,
    configuration: ScoreConfiguration,
) -> np.ndarray:
    """Compose the finite RACER-C2 score family.

    The primary mechanism is candidate-label reliability tilting.  The learned
    counterfactual score is retained only as an explicitly optional ablation;
    the current development lock fixes its weight to zero.
    """

    configuration.validate()
    base_probability = attenuated_probability(
        stack_probability,
        reliability_percentile,
        configuration.t_max,
    )
    base_scores = candidate_nonconformity(base_probability)
    weight = configuration.counterfactual_blend
    if counterfactual_percentiles is None:
        if weight != 0.0:
            raise ValueError("a nonzero counterfactual blend requires percentiles")
        combined = base_scores
    else:
        counterfactual = np.asarray(counterfactual_percentiles, dtype=float)
        if counterfactual.shape != base_scores.shape:
            raise ValueError("counterfactual percentiles must have shape (n,2)")
        if not np.isfinite(counterfactual).all() or np.any(
            (counterfactual < 0.0) | (counterfactual > 1.0)
        ):
            raise ValueError("counterfactual percentiles must lie in [0,1]")
        combined = (1.0 - weight) * base_scores + weight * counterfactual
    return tilt_candidate_scores(
        combined,
        reliability_percentile,
        configuration.gamma_0,
        configuration.gamma_1,
    )


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    values = np.sort(np.asarray(scores, dtype=float))
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    if not len(values):
        return math.inf
    k = math.ceil((len(values) + 1) * (1.0 - alpha))
    return math.inf if k > len(values) else float(values[k - 1])


def mondrian_thresholds(
    candidate_scores: np.ndarray,
    targets: np.ndarray,
    alpha: float,
) -> dict[int, float]:
    scores = np.asarray(candidate_scores, dtype=float)
    y = np.asarray(targets, dtype=np.int8)
    if scores.shape != (len(y), 2):
        raise ValueError("candidate scores must have shape (n,2)")
    return {
        label: finite_sample_quantile(scores[y == label, label], alpha)
        for label in (0, 1)
    }


def prediction_sets(
    candidate_scores: np.ndarray,
    thresholds: Mapping[int, float],
) -> np.ndarray:
    scores = np.asarray(candidate_scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != 2:
        raise ValueError("candidate scores must have shape (n,2)")
    return np.column_stack(
        [scores[:, label] <= float(thresholds[label]) for label in (0, 1)]
    )


def state_labels(sets: np.ndarray) -> list[str]:
    membership = np.asarray(sets, dtype=bool)
    if membership.ndim != 2 or membership.shape[1] != 2:
        raise ValueError("prediction sets must have shape (n,2)")
    output = []
    for includes_zero, includes_one in membership:
        if includes_zero and includes_one:
            output.append("Ambiguous")
        elif includes_zero:
            output.append("Accept(0)")
        elif includes_one:
            output.append("Accept(1)")
        else:
            output.append("Defer-empty")
    return output


def set_metric_record(targets: np.ndarray, sets: np.ndarray) -> dict[str, float | int | None]:
    y = np.asarray(targets, dtype=np.int8)
    membership = np.asarray(sets, dtype=bool)
    if membership.shape != (len(y), 2):
        raise ValueError("target and set row counts differ")
    result: dict[str, float | int | None] = {"n": len(y)}
    csy: list[float] = []
    for label in (0, 1):
        mask = y == label
        n = int(mask.sum())
        correct = membership[:, label] & ~membership[:, 1 - label]
        wrong = membership[:, 1 - label] & ~membership[:, label]
        covered = membership[:, label]
        empty = ~membership.any(axis=1)
        result.update(
            {
                f"class_{label}_n": n,
                f"class_{label}_correct_singleton_n": int((mask & correct).sum()),
                f"class_{label}_wrong_singleton_n": int((mask & wrong).sum()),
                f"class_{label}_covered_n": int((mask & covered).sum()),
                f"class_{label}_empty_n": int((mask & empty).sum()),
                f"class_{label}_coverage": float((mask & covered).sum() / n) if n else None,
                f"class_{label}_csy": float((mask & correct).sum() / n) if n else None,
                f"class_{label}_wrong_singleton_exposure": (
                    float((mask & wrong).sum() / n) if n else None
                ),
                f"class_{label}_empty_exposure": float((mask & empty).sum() / n) if n else None,
            }
        )
        if n:
            csy.append(float((mask & correct).sum() / n))
    result["macro_csy"] = float(np.mean(csy)) if csy else None
    result["worst_csy"] = float(np.min(csy)) if csy else None
    result["ambiguous_n"] = int(np.sum(membership[:, 0] & membership[:, 1]))
    result["empty_n"] = int(np.sum(~membership.any(axis=1)))
    return result


def one_sided_exact_lower(successes: int, n: int, alpha: float) -> float:
    if n <= 0 or successes <= 0:
        return 0.0
    return float(beta.ppf(alpha, successes, n - successes + 1))


def one_sided_exact_upper(successes: int, n: int, alpha: float) -> float:
    if n <= 0 or successes >= n:
        return 1.0
    return float(beta.ppf(1.0 - alpha, successes + 1, n - successes))


def certify_final_sets(
    targets: np.ndarray,
    sets: np.ndarray,
    constraints: ActionCertificateConstraints,
) -> dict[str, object]:
    """Certify the observable final states on an independent policy role.

    The family contains only explicitly enabled class/estimand bounds.  There is
    no search over gate thresholds, so the correction is not multiplied by a
    hidden policy grid.
    """

    constraints.validate()
    y = np.asarray(targets, dtype=np.int8)
    membership = np.asarray(sets, dtype=bool)
    if membership.shape != (len(y), 2) or set(np.unique(y)) != {0, 1}:
        raise ValueError("policy targets/sets are incomplete")
    enabled_constraints = []
    if constraints.coverage_floor is not None:
        enabled_constraints.extend((label, "coverage") for label in (0, 1))
    if constraints.wrong_singleton_ceiling is not None:
        enabled_constraints.extend((label, "wrong_singleton") for label in (0, 1))
    if constraints.empty_exposure_ceiling is not None:
        enabled_constraints.extend((label, "empty") for label in (0, 1))
    if constraints.critical_csy_floor is not None:
        enabled_constraints.append((constraints.critical_class, "critical_csy"))
    if not enabled_constraints:
        raise ValueError("at least one action certificate constraint is required")
    simultaneous_alpha = constraints.familywise_alpha / len(enabled_constraints)
    class_rows: dict[str, object] = {}
    all_pass = True
    for label in (0, 1):
        mask = y == label
        n = int(mask.sum())
        covered = int((mask & membership[:, label]).sum())
        wrong = int(
            (mask & membership[:, 1 - label] & ~membership[:, label]).sum()
        )
        empty = int((mask & ~membership.any(axis=1)).sum())
        correct = int(
            (mask & membership[:, label] & ~membership[:, 1 - label]).sum()
        )
        ready = n >= constraints.minimum_true_class_n
        coverage_lower = one_sided_exact_lower(covered, n, simultaneous_alpha)
        wrong_upper = one_sided_exact_upper(wrong, n, simultaneous_alpha)
        empty_upper = one_sided_exact_upper(empty, n, simultaneous_alpha)
        csy_lower = one_sided_exact_lower(correct, n, simultaneous_alpha)
        checks: dict[str, bool] = {}
        if constraints.coverage_floor is not None:
            checks["coverage"] = coverage_lower >= constraints.coverage_floor
        if constraints.wrong_singleton_ceiling is not None:
            checks["wrong_singleton"] = (
                wrong_upper <= constraints.wrong_singleton_ceiling
            )
        if constraints.empty_exposure_ceiling is not None:
            checks["empty"] = empty_upper <= constraints.empty_exposure_ceiling
        if label == constraints.critical_class and constraints.critical_csy_floor is not None:
            checks["critical_csy"] = csy_lower >= constraints.critical_csy_floor
        passed = ready and all(checks.values())
        all_pass = all_pass and passed
        class_rows[str(label)] = {
            "n": n,
            "count_ready": ready,
            "covered_n": covered,
            "correct_singleton_n": correct,
            "wrong_singleton_n": wrong,
            "empty_n": empty,
            "coverage_lower": coverage_lower,
            "csy_lower": csy_lower,
            "wrong_singleton_upper": wrong_upper,
            "empty_exposure_upper": empty_upper,
            "checks": checks,
            "passed": passed,
        }
    return {
        "status": "certified" if all_pass else "certificate-failed-closed",
        "simultaneous_alpha": simultaneous_alpha,
        "tested_constraint_count": len(enabled_constraints),
        "selection_grid_test_count": 0,
        "classes": class_rows,
    }


def select_development_configuration(
    evaluations: Iterable[Mapping[str, object]],
    mean_coverage_shortfall_margin: float,
    minimum_cell_class_coverage: float,
) -> tuple[ScoreConfiguration, list[dict[str, object]]]:
    """Choose one global score configuration from honest development metrics.

    Each input row represents one cell/configuration and must include baseline
    and candidate class coverages plus candidate MacroCSY.  Selection is
    endpoint/cell-equal; it cannot inspect policy, conformal, or test labels.
    """

    if mean_coverage_shortfall_margin < 0.0:
        raise ValueError("mean_coverage_shortfall_margin must be nonnegative")
    if not 0.0 <= minimum_cell_class_coverage <= 1.0:
        raise ValueError("minimum_cell_class_coverage must lie in [0,1]")
    materialized = [dict(row) for row in evaluations]
    if not materialized:
        raise ValueError("development evaluations are empty")
    grouped: dict[ScoreConfiguration, list[dict[str, object]]] = {}
    for row in materialized:
        configuration = ScoreConfiguration(
            t_max=float(row["t_max"]),
            gamma_0=float(row["gamma_0"]),
            gamma_1=float(row["gamma_1"]),
            counterfactual_blend=float(row["counterfactual_blend"]),
        )
        configuration.validate()
        grouped.setdefault(configuration, []).append(row)
    expected_cells = max(len(rows) for rows in grouped.values())
    summary: list[dict[str, object]] = []
    for configuration, rows in grouped.items():
        complete = len(rows) == expected_cells
        coverage_summary = {
            label: {
                "baseline_mean": float(
                    np.mean(
                        [float(row[f"baseline_class_{label}_coverage"]) for row in rows]
                    )
                ),
                "candidate_mean": float(
                    np.mean(
                        [float(row[f"candidate_class_{label}_coverage"]) for row in rows]
                    )
                ),
                "candidate_minimum": float(
                    np.min(
                        [float(row[f"candidate_class_{label}_coverage"]) for row in rows]
                    )
                ),
            }
            for label in (0, 1)
        }
        feasible = complete and all(
            coverage_summary[label]["candidate_mean"]
            >= coverage_summary[label]["baseline_mean"]
            - mean_coverage_shortfall_margin
            and coverage_summary[label]["candidate_minimum"]
            >= minimum_cell_class_coverage
            for label in (0, 1)
        )
        mean_baseline_macro_csy = float(
            np.mean([float(row["baseline_macro_csy"]) for row in rows])
        )
        mean_candidate_macro_csy = float(
            np.mean([float(row["candidate_macro_csy"]) for row in rows])
        )
        summary.append(
            {
                "t_max": configuration.t_max,
                "gamma_0": configuration.gamma_0,
                "gamma_1": configuration.gamma_1,
                "counterfactual_blend": configuration.counterfactual_blend,
                "cell_count": len(rows),
                "complete": complete,
                "feasible": feasible,
                "class_0_baseline_mean_coverage": coverage_summary[0][
                    "baseline_mean"
                ],
                "class_0_candidate_mean_coverage": coverage_summary[0][
                    "candidate_mean"
                ],
                "class_0_candidate_minimum_coverage": coverage_summary[0][
                    "candidate_minimum"
                ],
                "class_1_baseline_mean_coverage": coverage_summary[1][
                    "baseline_mean"
                ],
                "class_1_candidate_mean_coverage": coverage_summary[1][
                    "candidate_mean"
                ],
                "class_1_candidate_minimum_coverage": coverage_summary[1][
                    "candidate_minimum"
                ],
                "mean_baseline_macro_csy": mean_baseline_macro_csy,
                "mean_macro_csy": mean_candidate_macro_csy,
                "mean_macro_csy_gain": (
                    mean_candidate_macro_csy - mean_baseline_macro_csy
                ),
            }
        )
    feasible_rows = [row for row in summary if bool(row["feasible"])]
    if not feasible_rows:
        raise RuntimeError("no RACER-C2 score configuration passes the development gate")
    chosen = min(
        feasible_rows,
        key=lambda row: (
            -float(row["mean_macro_csy"]),
            float(row["counterfactual_blend"]),
            abs(float(row["gamma_0"])) + abs(float(row["gamma_1"])),
            float(row["t_max"]),
            float(row["gamma_0"]),
            float(row["gamma_1"]),
        ),
    )
    configuration = ScoreConfiguration(
        t_max=float(chosen["t_max"]),
        gamma_0=float(chosen["gamma_0"]),
        gamma_1=float(chosen["gamma_1"]),
        counterfactual_blend=float(chosen["counterfactual_blend"]),
    )
    return configuration, sorted(
        summary,
        key=lambda row: (
            float(row["t_max"]),
            float(row["gamma_0"]),
            float(row["gamma_1"]),
            float(row["counterfactual_blend"]),
        ),
    )
