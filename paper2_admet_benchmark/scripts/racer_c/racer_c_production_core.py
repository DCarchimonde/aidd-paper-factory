from __future__ import annotations

"""Pure numerical contracts used by the frozen RACER-C production runner.

This module deliberately contains no file-system orchestration or test-result
inspection.  Keeping score construction, policy-independent conformal logic,
and metric denominators here makes them testable without a GPU.
"""

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression


EPSILON = 1.0e-6


def clipped(values: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), epsilon, 1.0 - epsilon)


def logits(values: np.ndarray) -> np.ndarray:
    return logit(clipped(values))


def fit_platt(probabilities: np.ndarray, targets: np.ndarray, seed: int) -> LogisticRegression:
    y = np.asarray(targets, dtype=np.int8)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Platt calibration requires both classes")
    return LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    ).fit(logits(probabilities).reshape(-1, 1), y)


def apply_platt(model: LogisticRegression, probabilities: np.ndarray) -> np.ndarray:
    return clipped(model.predict_proba(logits(probabilities).reshape(-1, 1))[:, 1])


def fit_stacker(block_probabilities: np.ndarray, targets: np.ndarray, seed: int) -> LogisticRegression:
    x = logits(np.asarray(block_probabilities, dtype=float))
    y = np.asarray(targets, dtype=np.int8)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("the frozen stacker requires exactly three predictor blocks")
    return LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    ).fit(x, y)


def stack_probability(model: LogisticRegression, block_probabilities: np.ndarray) -> np.ndarray:
    return clipped(model.predict_proba(logits(block_probabilities))[:, 1])


def tanimoto_topk_local_loss(
    query: np.ndarray,
    reference: np.ndarray,
    reference_losses: np.ndarray,
    k: int,
    query_groups: Sequence[str] | None = None,
    reference_groups: Sequence[str] | None = None,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 1-max similarity and mean loss among honest nearest neighbours."""

    q = np.asarray(query, dtype=np.float32)
    r = np.asarray(reference, dtype=np.float32)
    losses = np.asarray(reference_losses, dtype=float)
    if q.ndim != 2 or r.ndim != 2 or q.shape[1] != r.shape[1]:
        raise ValueError("query/reference fingerprint dimensions differ")
    if len(r) != len(losses) or not len(r):
        raise ValueError("reference losses are incomplete")
    if (query_groups is None) != (reference_groups is None):
        raise ValueError("both group sequences are required for group exclusion")
    r_sum = r.sum(axis=1)
    distances = np.empty(len(q), dtype=float)
    local = np.empty(len(q), dtype=float)
    for start in range(0, len(q), batch_size):
        stop = min(start + batch_size, len(q))
        qb = q[start:stop]
        intersection = qb @ r.T
        union = qb.sum(axis=1, keepdims=True) + r_sum[None, :] - intersection
        similarity = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=np.float32),
            where=union > 0,
        )
        if query_groups is not None and reference_groups is not None:
            for offset, group in enumerate(query_groups[start:stop]):
                mask = np.asarray([value == group for value in reference_groups])
                similarity[offset, mask] = -np.inf
        for offset in range(stop - start):
            valid = np.flatnonzero(np.isfinite(similarity[offset]))
            if not len(valid):
                raise ValueError("group exclusion removed every development reference")
            take = min(k, len(valid))
            candidates = valid[np.argpartition(similarity[offset, valid], -take)[-take:]]
            ordered = candidates[np.argsort(-similarity[offset, candidates], kind="stable")]
            row = start + offset
            distances[row] = 1.0 - float(similarity[offset, ordered[0]])
            local[row] = float(np.mean(losses[ordered]))
    return distances, local


def reliability_features(
    block_probabilities: np.ndarray,
    stack_probabilities: np.ndarray,
    distance: np.ndarray,
    local_loss: np.ndarray,
) -> np.ndarray:
    block_logits = logits(block_probabilities)
    margin = np.abs(logits(stack_probabilities))
    disagreement = np.var(block_logits, axis=1)
    return np.column_stack([margin, disagreement, distance, local_loss]).astype(float)


def balanced_weights(targets: np.ndarray) -> np.ndarray:
    y = np.asarray(targets, dtype=np.int8)
    counts = Counter(int(value) for value in y)
    if set(counts) != {0, 1}:
        raise ValueError("class-balanced weights require both classes")
    return np.asarray([len(y) / (2.0 * counts[int(value)]) for value in y], dtype=float)


def fit_bri(features: np.ndarray, squared_losses: np.ndarray, targets: np.ndarray, seed: int):
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=7,
        min_samples_leaf=20,
        l2_regularization=1.0,
        monotonic_cst=[-1, 1, 1, 1],
        random_state=seed,
    )
    model.fit(features, squared_losses, sample_weight=balanced_weights(targets))
    return model


def bri_predict(model: object, features: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(model.predict(features), dtype=float), 0.0, 1.0)


def class_midrank_percentiles(
    reference_risk: np.ndarray,
    reference_predicted_class: np.ndarray,
    query_risk: np.ndarray,
    query_predicted_class: np.ndarray,
) -> np.ndarray:
    output = np.empty(len(query_risk), dtype=float)
    for label in (0, 1):
        ref = np.sort(np.asarray(reference_risk)[np.asarray(reference_predicted_class) == label])
        indices = np.flatnonzero(np.asarray(query_predicted_class) == label)
        if not len(ref) or not len(indices):
            if len(indices):
                raise ValueError(f"no development risk reference for predicted class {label}")
            continue
        values = np.asarray(query_risk)[indices]
        left = np.searchsorted(ref, values, side="left")
        right = np.searchsorted(ref, values, side="right")
        # Empirical mid-rank CDF: P(R < r) + 0.5 P(R = r).  Do not add a
        # pseudo-observation here; the development reference itself is fixed.
        output[indices] = (left + right) / (2.0 * len(ref))
    return np.clip(output, 0.0, 1.0)


def attenuate(probabilities: np.ndarray, risk_percentiles: np.ndarray, t_max: float) -> np.ndarray:
    if t_max < 1.0:
        raise ValueError("T_max must be at least one")
    temperature = 1.0 + (float(t_max) - 1.0) * np.asarray(risk_percentiles)
    return clipped(expit(logits(probabilities) / temperature))


def nonconformity(probabilities: np.ndarray) -> np.ndarray:
    p = clipped(probabilities)
    return np.column_stack([p, 1.0 - p])


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    values = np.sort(np.asarray(scores, dtype=float))
    if not len(values):
        return math.inf
    k = math.ceil((len(values) + 1) * (1.0 - alpha))
    return math.inf if k > len(values) else float(values[k - 1])


def conformal_thresholds(
    probabilities: np.ndarray,
    targets: np.ndarray,
    alpha: float,
    selected: np.ndarray | None = None,
    class_conditional: bool = True,
    score_multiplier: np.ndarray | None = None,
) -> dict[int, float]:
    y = np.asarray(targets, dtype=np.int8)
    scores = nonconformity(probabilities)[np.arange(len(y)), y]
    if score_multiplier is not None:
        scores = scores * np.asarray(score_multiplier, dtype=float)
    mask = np.ones(len(y), dtype=bool) if selected is None else np.asarray(selected, dtype=bool)
    if class_conditional:
        return {label: finite_sample_quantile(scores[mask & (y == label)], alpha) for label in (0, 1)}
    q = finite_sample_quantile(scores[mask], alpha)
    return {0: q, 1: q}


def prediction_sets(
    probabilities: np.ndarray,
    thresholds: Mapping[int, float],
    score_multiplier: np.ndarray | None = None,
) -> np.ndarray:
    scores = nonconformity(probabilities)
    if score_multiplier is not None:
        multiplier = np.asarray(score_multiplier, dtype=float)
        if multiplier.shape != scores.shape:
            raise ValueError("candidate-label multiplier must have shape (n,2)")
        scores = scores * multiplier
    return np.column_stack(
        [scores[:, label] <= float(thresholds[label]) for label in (0, 1)]
    )


def state_labels(sets: np.ndarray, selected: np.ndarray | None = None) -> list[str]:
    membership = np.asarray(sets, dtype=bool)
    gate = np.ones(len(membership), dtype=bool) if selected is None else np.asarray(selected, dtype=bool)
    output: list[str] = []
    for keep, values in zip(gate, membership):
        if not keep:
            output.append("Defer-risk/domain")
        elif values[0] and values[1]:
            output.append("Ambiguous")
        elif values[0]:
            output.append("Accept(0)")
        elif values[1]:
            output.append("Accept(1)")
        else:
            output.append("Defer-empty")
    return output


def metric_record(targets: np.ndarray, states: Sequence[str]) -> dict[str, float | int | None]:
    y = np.asarray(targets, dtype=np.int8)
    if len(y) != len(states):
        raise ValueError("state/target row counts differ")
    result: dict[str, float | int | None] = {"n": len(y)}
    csy: list[float] = []
    for label in (0, 1):
        mask = y == label
        n = int(mask.sum())
        accepted = np.asarray([value == f"Accept({label})" for value in states])
        wrong = np.asarray([value == f"Accept({1-label})" for value in states])
        selected = np.asarray([value not in {"Defer-risk/domain"} for value in states])
        singleton = accepted | wrong
        correct_n = int((mask & accepted).sum())
        wrong_n = int((mask & wrong).sum())
        selected_n = int((mask & selected).sum())
        singleton_n = int((mask & singleton).sum())
        result.update(
            {
                f"class_{label}_n": n,
                f"class_{label}_correct_singleton_n": correct_n,
                f"class_{label}_wrong_singleton_n": wrong_n,
                f"class_{label}_selected_n": selected_n,
                f"class_{label}_singleton_n": singleton_n,
                f"class_{label}_coverage": (correct_n + int((mask & np.asarray([s == 'Ambiguous' for s in states])).sum())) / n if n else None,
                f"class_{label}_csy": correct_n / n if n else None,
                f"class_{label}_wrong_singleton_exposure": wrong_n / n if n else None,
                f"class_{label}_gate_retention": selected_n / n if n else None,
                f"class_{label}_singleton_formation": singleton_n / selected_n if selected_n else None,
                f"class_{label}_singleton_correctness": correct_n / singleton_n if singleton_n else None,
            }
        )
        if n:
            csy.append(correct_n / n)
    result["macro_csy"] = float(np.mean(csy)) if csy else None
    result["worst_csy"] = float(np.min(csy)) if csy else None
    result["ambiguous_n"] = sum(value == "Ambiguous" for value in states)
    result["empty_n"] = sum(value == "Defer-empty" for value in states)
    result["risk_defer_n"] = sum(value == "Defer-risk/domain" for value in states)
    return result


@dataclass(frozen=True)
class RcpTransform:
    models: tuple[object, object]
    floor: float = 1.0e-3

    def multipliers(self, features: np.ndarray) -> np.ndarray:
        quantiles = np.column_stack(
            [np.maximum(np.asarray(model.predict(features), dtype=float), self.floor) for model in self.models]
        )
        return 1.0 / quantiles


def fit_rcp_transform(
    features: np.ndarray,
    probabilities: np.ndarray,
    targets: np.ndarray,
    alpha: float,
    seed: int,
) -> RcpTransform:
    y = np.asarray(targets, dtype=np.int8)
    true_scores = nonconformity(probabilities)[np.arange(len(y)), y]
    models: list[object] = []
    for label in (0, 1):
        mask = y == label
        if int(mask.sum()) < 35:
            raise ValueError(f"RCP class {label} development cell has fewer than 35 rows")
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=1.0 - alpha,
            learning_rate=0.05,
            max_iter=100,
            max_leaf_nodes=7,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=seed + label,
        ).fit(features[mask], true_scores[mask])
        models.append(model)
    return RcpTransform((models[0], models[1]))
