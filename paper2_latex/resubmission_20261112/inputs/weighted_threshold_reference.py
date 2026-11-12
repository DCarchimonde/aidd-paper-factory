"""Unmodified weighted_test_thresholds from the versioned TAME core."""
from __future__ import annotations
import math
from typing import Mapping
import numpy as np

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
