"""Independent finite-population red-team checks.

The oracle below enumerates actual label counts and evaluates the two ratios
directly with Fraction.  It deliberately does not use the implemented corner
formula, linear constraints, or certificate as its optimizer oracle.
These are exhaustive numerical checks of tiny populations, not a substitute
for the probability argument and not chemical performance experiments.
"""
from fractions import Fraction
from functools import lru_cache
import itertools
import math
import unittest

import numpy as np

from paper2_admet_benchmark.revision_2026.batch_triage.core import (
    CountCS, InconsistentConfidenceSet, certify_triage, optimize_triage,
)


def _ratio(numerator, denominator):
    return Fraction(int(numerator), int(denominator)) if denominator else Fraction(0)


def _actual_risks(M, positive, a, b, actions):
    """Evaluate error among ALL true-class batch members, including audits."""
    fn = sum(k for k, action in zip(positive, actions) if action == 0)
    fp = sum(m - k for m, k, action in zip(M, positive, actions) if action == 1)
    return (_ratio(fn, a + sum(positive)),
            _ratio(fp, b + sum(m - k for m, k in zip(M, positive))))


def _enumerated_risk_maxima(M, L, U, a, b, actions):
    possible = itertools.product(*(range(l, u + 1) for l, u in zip(L, U)))
    risks = [_actual_risks(M, positive, a, b, actions) for positive in possible]
    return max(r[0] for r in risks), max(r[1] for r in risks)


def _binary_orders(N, K):
    for indices in itertools.combinations(range(N), K):
        selected = set(indices)
        yield tuple(int(i in selected) for i in range(N))


class IndependentCountBoxTests(unittest.TestCase):
    def test_all_two_stratum_tiny_boxes_and_actions(self):
        # Every possible interval for stratum sizes 0,1,2; absent classes and
        # empty strata occur naturally. No floating tolerance decides feasibility.
        intervals = [(m, l, u) for m in range(3) for l in range(m + 1)
                     for u in range(l, m + 1)]
        eps_fn, eps_fp = Fraction("0.25"), Fraction("0.5")
        for first, second in itertools.product(intervals, repeat=2):
            M, L, U = (tuple(v[j] for v in (first, second)) for j in range(3))
            for a, b in itertools.product((0, 2), repeat=2):
                for actions in itertools.product(range(3), repeat=2):
                    fn, fp = _enumerated_risk_maxima(M, L, U, a, b, actions)
                    got = certify_triage(M, L, U, a, b,
                                         np.array(actions) == 0,
                                         np.array(actions) == 1,
                                         float(eps_fn), float(eps_fp))
                    self.assertEqual(got["fn_upper"], float(fn))
                    self.assertEqual(got["fp_upper"], float(fp))
                    self.assertEqual(got["certified"], fn <= eps_fn and fp <= eps_fp)
                    self.assertEqual(got["class_1_absence_possible"], a + sum(L) == 0)
                    self.assertEqual(got["class_0_absence_possible"],
                                     b + sum(m - u for m, u in zip(M, U)) == 0)

    def test_optimizer_matches_actual_label_enumeration_with_costs_and_eligibility(self):
        rng = np.random.default_rng(93821)
        for _ in range(48):
            M = rng.integers(0, 5, size=3).tolist()
            L = [int(rng.integers(m + 1)) for m in M]
            U = [int(rng.integers(l, m + 1)) for l, m in zip(L, M)]
            a, b = rng.integers(0, 7, size=2).tolist()
            can_clear, can_discard = rng.integers(0, 2, size=(2, 3)).astype(bool)
            costs = rng.integers(1, 5, size=3).tolist()
            eps_fn, eps_fp = Fraction("0.25"), Fraction("0.4")
            best = 0
            for actions in itertools.product(range(3), repeat=3):
                if any(action == 0 and not can_clear[h] or
                       action == 1 and not can_discard[h]
                       for h, action in enumerate(actions)):
                    continue
                fn, fp = _enumerated_risk_maxima(M, L, U, a, b, actions)
                if fn <= eps_fn and fp <= eps_fp:
                    best = max(best, sum(m * c for m, c, action in zip(M, costs, actions)
                                         if action != 2))
            got = optimize_triage(M, L, U, a, b, float(eps_fn), float(eps_fp),
                                  costs=costs, eligible_clear=can_clear,
                                  eligible_discard=can_discard)
            self.assertEqual(got["saved_cost"], best)
            fn, fp = _enumerated_risk_maxima(M, L, U, a, b, got["actions"])
            self.assertLessEqual(fn, eps_fn)
            self.assertLessEqual(fp, eps_fp)

    def test_audited_labels_are_not_counted_as_saved_assays(self):
        result = optimize_triage([4, 6], [0, 6], [0, 6], 7, 3, .1, .1)
        self.assertEqual(result["saved_items"], 10)
        # Original population20 = assays already10 + future0 + saved10.
        certificate = result["certificate"]
        spent = certificate["audited_positive"] + certificate["audited_negative"]
        self.assertEqual(spent + certificate["deferred_items"] + result["saved_items"], 20)

    def test_zero_observed_errors_is_not_a_release_certificate(self):
        # Seeing one negative does not rule out all nine remaining items being
        # positive. A zero-error plug-in would release nine unsafe negative calls.
        state = CountCS(10, .05)
        state.observe(0)
        lo, hi = state.remaining_bounds()
        unsafe = certify_triage([9], [lo], [hi], 0, 1, [1], [0], .1, .1)
        self.assertFalse(unsafe["certified"])
        self.assertEqual(_actual_risks([9], [9], 0, 1, [0])[0], 1)
        self.assertEqual(optimize_triage([9], [lo], [hi], 0, 1, .1, .1)["saved_items"], 0)


class AdaptiveWithoutReplacementTests(unittest.TestCase):
    def test_adaptive_stratum_selection_and_anytime_postselection(self):
        for prior in ("uniform", "edge_mixture"):
            with self.subTest(prior=prior):
                self._check_adaptive_stratum_selection_and_anytime_postselection(prior)

    def _check_adaptive_stratum_selection_and_anytime_postselection(self, prior):
        """Enumerate every random order for every fixed two-stratum population.

        At each prefix an adversary may choose ANY certified three-action
        assignment. Thus the event below includes optional stopping, selecting
        either release direction after labels, and expanding/retracting unions.
        Cross-stratum choices depend on observed labels, but never peek at the
        next label within a stratum. Probability is exact over all local orders.
        """
        sizes = (4, 4)
        delta_each, eps_fn, eps_fp = .2, .4, .4
        tolerance_fn, tolerance_fp = Fraction(str(eps_fn)), Fraction(str(eps_fp))

        @lru_cache(maxsize=None)
        def certified_actions(signature):
            M = tuple(s[0] - s[1] for s in signature)
            L = tuple(max(0, s[3] - s[2]) for s in signature)
            U = tuple(min(s[0] - s[1], s[4] - s[2]) for s in signature)
            a = sum(s[2] for s in signature)
            b = sum(s[1] - s[2] for s in signature)
            out = []
            for actions in itertools.product(range(3), repeat=2):
                result = certify_triage(M, L, U, a, b,
                                        np.array(actions) == 0,
                                        np.array(actions) == 1, eps_fn, eps_fp)
                if result["certified"]:
                    out.append(actions)
            return tuple(out)

        release_seen = False
        for true_totals in itertools.product(*(range(N + 1) for N in sizes)):
            unsafe_paths = 0
            path_count = math.prod(math.comb(N, K) for N, K in zip(sizes, true_totals))
            for orders in itertools.product(*[_binary_orders(N, K)
                                              for N, K in zip(sizes, true_totals)]):
                states = [CountCS(N, delta_each, prior=prior) for N in sizes]
                unsafe = False
                for t in range(sum(sizes) + 1):
                    signature = tuple((s.N, s.n, s.s, s.lower_total, s.upper_total)
                                      for s in states)
                    M = tuple(s.N - s.n for s in states)
                    remaining_true = tuple(K - s.s for K, s in zip(true_totals, states))
                    a, b = sum(s.s for s in states), sum(s.n - s.s for s in states)
                    for actions in certified_actions(signature):
                        release_seen |= any(m and action != 2 for m, action in zip(M, actions))
                        fn, fp = _actual_risks(M, remaining_true, a, b, actions)
                        unsafe |= fn > tolerance_fn or fp > tolerance_fp
                    if t == sum(sizes):
                        break
                    # A predictable, outcome-dependent allocation rule. Each
                    # local random order was selected uniformly before audits.
                    chosen = (a + t) % 2
                    if states[chosen].n == sizes[chosen]:
                        chosen = 1 - chosen
                    state = states[chosen]
                    try:
                        state.observe(orders[chosen][state.n])
                    except InconsistentConfidenceSet:
                        break  # Fail closed; no later certificate is available.
                unsafe_paths += unsafe
            self.assertLessEqual(Fraction(unsafe_paths, path_count),
                                 Fraction(str(2 * delta_each)),
                                 msg=f"fixed population totals {true_totals}")
        self.assertTrue(release_seen, "the check must exercise nonempty release decisions")


if __name__ == "__main__":
    unittest.main()
