"""Exact count certificates and finite-batch triage optimization.

This module does not learn strata or select which labels to inspect. Its
guarantee requires strata fixed before audit outcomes and uniform sampling
without replacement *within* the adaptively chosen stratum. All inspected
items are handled by their observed labels, never counted as automated calls.

The count CS uses the prior/posterior e-process construction of Waudby-Smith
and Ramdas (NeurIPS 2020), not a new confidence-sequence theorem. The uniform
total prior gives e_t(K)=1/((t+1)*HypergeomPMF(s;N,K,t)). The prespecified
edge-mixture prior concentrates some mass at all-zero/all-one strata; its
remaining mass is a Jeffreys beta-binomial prior. Priors cannot be selected or
changed after examining audit outcomes.
Inversion and intersection over time yield a count CS. Per-stratum failure
probabilities must sum to the intended overall failure probability.

Remaining-count bounds describe a simultaneous rectangular confidence set.
Worst-case class-denominator errors over this box have exact corner solutions.
The MILP optimizes saved remaining assay costs over disjoint stratum unions;
its certificate remains valid after arbitrary outcome-adaptive union selection.
This is finite-current-batch inference, not a guarantee on future compounds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import math
from typing import Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.special import logsumexp
from scipy.stats import betabinom, hypergeom


PRIORS = ("uniform", "edge_mixture")


class InconsistentConfidenceSet(RuntimeError):
    """The running CS intersection is empty; release must fail closed."""


class OptimizationFailure(RuntimeError):
    """No independently checked optimal triage solution was returned."""


def _integer(value, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not a boolean")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite integer") from exc
    if not math.isfinite(number) or number != math.floor(number) or number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    # Exact numerical arithmetic downstream relies on float-representable counts.
    if number > 2**52:
        raise ValueError(f"{name} exceeds the supported exact-integer range")
    return int(number)


def _probability(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a probability, not a boolean")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must lie strictly between zero and one") from exc
    if not math.isfinite(number) or not 0 < number < 1:
        raise ValueError(f"{name} must lie strictly between zero and one")
    return number


def _counts(values, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional count vector")
    return np.asarray([_integer(v, name) for v in raw], dtype=np.int64)


def _mask(values, size: int, name: str, *, default: bool | None = None) -> np.ndarray:
    if values is None:
        if default is None:
            raise ValueError(f"{name} is required")
        return np.full(size, default, dtype=bool)
    raw = np.asarray(values)
    if raw.shape != (size,) or not np.isin(raw, [0, 1]).all():
        raise ValueError(f"{name} must be an aligned binary vector")
    return raw.astype(bool)


def _audit_state(N, n, s):
    N, n, s = _integer(N, "N"), _integer(n, "n"), _integer(s, "s")
    if s > n or n > N:
        raise ValueError("audit counts must satisfy 0 <= s <= n <= N")
    return N, n, s


def _log_mixture_mass(n: int, s: int, prior: str) -> float:
    if prior not in PRIORS:
        raise ValueError(f"prior must be one of {PRIORS}")
    if prior == "uniform":
        return -math.log(n + 1)
    # The fixed original-total prior is .4 at K=0, .4 at K=N, and .2 times
    # BetaBinomial(N,.5,.5). Its count predictive law has this same mixture
    # with n in place of N, including n=0 where both edge atoms contribute.
    terms = [math.log(.2) + float(betabinom.logpmf(s, n, .5, .5))]
    if s == 0:
        terms.append(math.log(.4))
    if s == n:
        terms.append(math.log(.4))
    return float(logsumexp(terms))


def log_e_value(N: int, n: int, s: int, K: int, *, prior: str = "uniform") -> float:
    """Log mixture/point likelihood ratio for one proposed original total K."""
    N, n, s = _audit_state(N, n, s)
    K = _integer(K, "K")
    if K > N:
        raise ValueError("K cannot exceed N")
    log_mixture = _log_mixture_mass(n, s, prior)
    if N == 0:
        return 0.0
    return float(log_mixture - hypergeom.logpmf(s, N, K, n))


def finite_population_cs(N: int, n: int, s: int, delta: float, *,
                         prior: str = "uniform") -> tuple[int, int]:
    """Invert the count e-process at a particular internal sample size.

    The threshold is time-uniform; it does not spend delta afresh per look.
    Use CountCS to additionally intersect these sets over observed prefixes.
    Closed inclusion at the boundary is conservative. A small downward log
    threshold guard also makes floating-point boundary evaluation conservative.
    """
    N, n, s = _audit_state(N, n, s)
    delta = _probability(delta, "delta")
    log_mixture = _log_mixture_mass(n, s, prior)
    if n == 0:
        return 0, N
    if n == N:
        return s, s
    totals = np.arange(s, N - n + s + 1, dtype=np.int64)
    log_mass = hypergeom.logpmf(s, N, totals, n)
    if np.isnan(log_mass).any():
        raise FloatingPointError("hypergeometric evaluation produced NaN")
    allowed = totals[log_mass >= math.log(delta) + log_mixture - 1e-12]
    if not len(allowed):
        raise InconsistentConfidenceSet("count e-process inversion is empty")
    # Taking the interval hull is conservative even if a numerical library
    # were to violate the mathematical unimodality of the likelihood in K.
    return int(allowed[0]), int(allowed[-1])


def update_count_interval(N: int, n: int, s: int, previous_lower: int,
                          previous_upper: int, delta: float, *,
                          prior: str = "uniform") -> tuple[int, int]:
    """Stateless running-intersection update for a recorded audit prefix.

    previous_lower/upper bound the ORIGINAL stratum positive total. This helper
    does not validate the sampling ledger; the caller must preserve every
    previous prefix and the fixed within-stratum random sampling order.
    """
    N, n, s = _audit_state(N, n, s)
    previous_lower = _integer(previous_lower, "previous_lower")
    previous_upper = _integer(previous_upper, "previous_upper")
    if previous_lower > previous_upper or previous_upper > N:
        raise ValueError("previous interval must lie inside [0,N]")
    lo, hi = finite_population_cs(N, n, s, delta, prior=prior)
    lo, hi = max(lo, previous_lower), min(hi, previous_upper)
    if lo > hi:
        raise InconsistentConfidenceSet("running count CS is empty; defer the remainder")
    return lo, hi


@dataclass
class CountCS:
    """A single frozen stratum's sequential without-replacement count state."""

    N: int
    delta: float
    prior: str = "uniform"
    n: int = field(default=0, init=False)
    s: int = field(default=0, init=False)
    lower_total: int = field(default=0, init=False)
    upper_total: int = field(default=0, init=False)
    valid: bool = field(default=True, init=False)
    _frozen_configuration: tuple = field(default=(), init=False, repr=False)

    def __post_init__(self):
        self.N = _integer(self.N, "N")
        self.delta = _probability(self.delta, "delta")
        _log_mixture_mass(0, 0, self.prior)
        self.upper_total = self.N
        self._frozen_configuration = (self.N, self.delta, self.prior)

    def _check_configuration(self):
        if (self.N, self.delta, self.prior) != self._frozen_configuration:
            raise ValueError("stratum size, error budget and prior must remain fixed after construction")

    def observe(self, label: int) -> tuple[int, int]:
        """Add the next uniformly drawn, previously unaudited stratum label."""
        self._check_configuration()
        if not self.valid:
            raise InconsistentConfidenceSet("cannot resume an empty running CS")
        if isinstance(label, (bool, np.bool_)):
            label = int(label)
        label = _integer(label, "label")
        if label not in (0, 1):
            raise ValueError("audit labels must be exactly 0 or 1")
        if self.n >= self.N:
            raise ValueError("stratum has no unaudited items")
        n, s = self.n + 1, self.s + label
        lo, hi = finite_population_cs(self.N, n, s, self.delta, prior=self.prior)
        lo, hi = max(lo, self.lower_total), min(hi, self.upper_total)
        # Record the actual observed label even if a rare empty-CS event occurs.
        self.n, self.s = n, s
        if lo > hi:
            self.valid = False
            raise InconsistentConfidenceSet("running count CS is empty; defer the remainder")
        self.lower_total, self.upper_total = lo, hi
        return self.remaining_bounds()

    def observe_many(self, labels: Sequence[int]) -> tuple[int, int]:
        """Validate a batch atomically, then process every prefix in order."""
        raw = np.asarray(labels)
        if raw.ndim != 1 or not np.isin(raw, [0, 1]).all():
            raise ValueError("audit labels must be a binary vector")
        if len(raw) > self.N - self.n:
            raise ValueError("batch exceeds the remaining stratum size")
        for label in raw:
            self.observe(int(label))
        return self.remaining_bounds()

    def remaining_bounds(self) -> tuple[int, int]:
        self._check_configuration()
        if not self.valid:
            raise InconsistentConfidenceSet("no release from an empty confidence set")
        return max(0, self.lower_total - self.s), min(self.N - self.n, self.upper_total - self.s)

    def snapshot(self) -> dict:
        lo, hi = self.remaining_bounds()
        return {"N": self.N, "n": self.n, "s": self.s, "delta": self.delta, "prior": self.prior,
                "lower_total": self.lower_total, "upper_total": self.upper_total,
                "remaining": self.N - self.n, "lower_remaining": lo,
                "upper_remaining": hi, "valid": self.valid}


def _triage_inputs(M, L, U, a, b, eps_fn, eps_fp):
    M, L, U = _counts(M, "M"), _counts(L, "L"), _counts(U, "U")
    if not (M.shape == L.shape == U.shape):
        raise ValueError("remaining size/lower/upper vectors must align")
    if np.any(L > U) or np.any(U > M):
        raise ValueError("remaining count bounds must satisfy 0 <= L <= U <= M")
    a, b = _integer(a, "a"), _integer(b, "b")
    if sum(map(int, M)) + a + b > 2**52:
        raise ValueError("total batch size exceeds exact-integer range")
    return M, L, U, a, b, _probability(eps_fn, "eps_fn"), _probability(eps_fp, "eps_fp")


def _fraction_at_most(numerator: int, denominator: int, tolerance: float) -> bool:
    if denominator == 0:
        return numerator == 0
    # Integer cross-multiplication avoids tolerance-based acceptance of a
    # scientifically infeasible policy at a MILP floating-point boundary.
    target = Fraction(str(tolerance))
    return numerator * target.denominator <= denominator * target.numerator


def certify_triage(M, L, U, a, b, clear_mask, discard_mask,
                   eps_fn: float, eps_fp: float) -> dict:
    """Exact worst-case class errors over a remaining-positive-count box.

    clear_mask denotes automatic label 0; discard_mask automatic label 1.
    Unselected strata are deferred. a/b count ALL already audited positives /
    negatives, including labels from strata now selected for either action.
    A zero actual class denominator has zero error by the operational convention;
    the separate absence flags must accompany any class-coverage interpretation.
    """
    M, L, U, a, b, eps_fn, eps_fp = _triage_inputs(M, L, U, a, b, eps_fn, eps_fp)
    clear = _mask(clear_mask, len(M), "clear_mask")
    discard = _mask(discard_mask, len(M), "discard_mask")
    if np.any(clear & discard):
        raise ValueError("a stratum cannot receive both automated labels")
    # These two corners need not be attained by the same label assignment.
    # Each is an exact supremum for its own risk; both must pass.
    fn_corner = np.where(clear, U, L)
    fp_corner = np.where(discard, L, U)
    fn_num = sum(map(int, U[clear]))
    fn_den = a + sum(map(int, fn_corner))
    fp_num = sum(map(int, (M - L)[discard]))
    fp_den = b + sum(map(int, M - fp_corner))
    fn_ok = _fraction_at_most(fn_num, fn_den, eps_fn)
    fp_ok = _fraction_at_most(fp_num, fp_den, eps_fp)
    return {"certified": fn_ok and fp_ok, "fn_certified": fn_ok, "fp_certified": fp_ok,
            "fn_upper": fn_num / fn_den if fn_den else 0.0,
            "fp_upper": fp_num / fp_den if fp_den else 0.0,
            "fn_worst_numerator": fn_num, "fn_worst_denominator": fn_den,
            "fp_worst_numerator": fp_num, "fp_worst_denominator": fp_den,
            "fn_corner_remaining_positive": fn_corner,
            "fp_corner_remaining_positive": fp_corner,
            "class_1_absence_possible": a + int(L.sum()) == 0,
            "class_0_absence_possible": b + int((M - U).sum()) == 0,
            "audited_positive": a, "audited_negative": b,
            "remaining_items": int(M.sum()),
            "released_negative_items": int(M[clear].sum()),
            "released_positive_items": int(M[discard].sum()),
            "deferred_items": int(M[~(clear | discard)].sum()),
            "eps_fn": eps_fn, "eps_fp": eps_fp}


def optimize_triage(M, L, U, a, b, eps_fn: float, eps_fp: float, *,
                    eligible_clear=None, eligible_discard=None, costs=None,
                    time_limit: float | None = None) -> dict:
    """Maximize saved remaining assay cost subject to both exact certificates.

    costs[h] is the predeclared cost per unaudited item in stratum h. The audit
    already spent a+b labels; objective savings never count those inspected
    items. The primary objective is unperturbed: a tied optimum is whatever
    the fixed SciPy/HiGHS implementation returns, not a post-outcome tie search.
    Solver failure/non-optimal status raises; the caller must then defer all.
    """
    M, L, U, a, b, eps_fn, eps_fp = _triage_inputs(M, L, U, a, b, eps_fn, eps_fp)
    h = len(M)
    can_clear = _mask(eligible_clear, h, "eligible_clear", default=True) & (M > 0)
    can_discard = _mask(eligible_discard, h, "eligible_discard", default=True) & (M > 0)
    unit_cost = np.ones(h) if costs is None else np.asarray(costs, dtype=float)
    if unit_cost.shape != (h,) or not np.isfinite(unit_cost).all() or np.any(unit_cost <= 0):
        raise ValueError("costs must be aligned, finite and strictly positive")
    options = {"mip_rel_gap": 0.0, "presolve": True}
    if time_limit is not None:
        if not math.isfinite(float(time_limit)) or float(time_limit) <= 0:
            raise ValueError("time_limit must be finite and positive")
        options["time_limit"] = float(time_limit)
    values = M * unit_cost
    if not np.isfinite(values).all():
        raise ValueError("stratum savings overflowed")
    if h == 0 or not np.any(can_clear | can_discard):
        clear, discard = np.zeros(h, bool), np.zeros(h, bool)
        certificate = certify_triage(M, L, U, a, b, clear, discard, eps_fn, eps_fp)
        return {"status": "optimal", "clear_mask": clear, "discard_mask": discard,
                "defer_mask": ~(clear | discard), "actions": np.full(h, 2, dtype=np.int8), "saved_cost": 0.0,
                "saved_items": 0, "certificate": certificate, "solver": "trivial_no_eligible_items"}
    matrix = np.zeros((h + 2, 2 * h))
    matrix[np.arange(h), np.arange(h)] = 1
    matrix[np.arange(h), h + np.arange(h)] = 1
    matrix[h, :h] = (1 - eps_fn) * U + eps_fn * L
    matrix[h + 1, h:] = (1 - eps_fp) * (M - L) + eps_fp * (M - U)
    upper = np.concatenate((np.ones(h), [eps_fn * (a + int(L.sum())),
                            eps_fp * (b + int((M - U).sum()))]))
    result = milp(c=-np.concatenate((values, values)), integrality=np.ones(2 * h),
                  bounds=Bounds(np.zeros(2 * h), np.concatenate((can_clear, can_discard)).astype(float)),
                  constraints=LinearConstraint(matrix, np.full(h + 2, -np.inf), upper),
                  options=options)
    if result.status != 0 or result.x is None or not np.isfinite(result.x).all():
        raise OptimizationFailure(f"MILP did not certify an optimum: status={result.status}, {result.message}")
    rounded = np.rint(result.x)
    if np.max(np.abs(result.x - rounded)) > 1e-6 or not np.isin(rounded, [0, 1]).all():
        raise OptimizationFailure("MILP returned a nonbinary solution")
    clear, discard = rounded[:h].astype(bool), rounded[h:].astype(bool)
    if np.any(clear & ~can_clear) or np.any(discard & ~can_discard):
        raise OptimizationFailure("MILP violated allowed release actions")
    certificate = certify_triage(M, L, U, a, b, clear, discard, eps_fn, eps_fp)
    if not certificate["certified"]:
        raise OptimizationFailure("MILP solution failed the independent exact risk certificate")
    saved = float(values[clear | discard].sum())
    if not math.isclose(saved, -float(result.fun), rel_tol=1e-10, abs_tol=1e-8):
        raise OptimizationFailure("MILP objective disagrees with the selected strata")
    return {"status": "optimal", "clear_mask": clear, "discard_mask": discard,
            "defer_mask": ~(clear | discard),
            "actions": np.where(clear, 0, np.where(discard, 1, 2)).astype(np.int8), "saved_cost": saved,
            "saved_items": int(M[clear | discard].sum()), "certificate": certificate,
            "solver": "scipy.optimize.milp/HiGHS", "solver_status": int(result.status),
            "solver_message": str(result.message),
            "mip_gap": float(result.mip_gap) if getattr(result, "mip_gap", None) is not None else None,
            "mip_node_count": int(result.mip_node_count) if getattr(result, "mip_node_count", None) is not None else None}
