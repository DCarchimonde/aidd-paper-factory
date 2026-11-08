"""Tests of dose-resolution semantics, censoring and observed paired evidence."""
import json
import math
import unittest

import numpy as np

from .targets import (DEFAULT_EDGES, interval_bin_mask, onset_interval,
                      pair_allowed_mask, window_cell_masks, window_reference_bounds)


def interval(lower, upper, horizon=0.0):
    return dict(lower_log_m=lower, upper_log_m=upper,
                concentration_max_log_m=horizon)


class OnsetTests(unittest.TestCase):
    def test_sort_missing_and_identical_duplicates(self):
        r = onset_interval([1e-5, 1e-7, 1e-6, 1e-6, None],
                           [-40, -2, -31, -31, None], 30)
        self.assertEqual((r['lower_log_m'], r['upper_log_m']), (-7., -6.))
        self.assertEqual(r['censoring'], 'interval')
        self.assertEqual(r['n_points'], 3)
        self.assertEqual(r['n_missing_pairs'], 1)
        json.dumps(r, allow_nan=False)

    def test_right_left_and_unobserved(self):
        right = onset_interval([1e-7, 1e-6], [0, -5], 20)
        self.assertEqual((right['lower_log_m'], right['upper_log_m']), (-6., None))
        self.assertEqual(right['censoring'], 'right')
        left = onset_interval([1e-7, 1e-6], [-20, -40], 20)
        self.assertEqual((left['lower_log_m'], left['upper_log_m']), (None, -7.))
        self.assertEqual(left['censoring'], 'left')
        missing = onset_interval([None, np.nan], ['', 10], 20)
        self.assertEqual(missing['censoring'], 'unobserved')
        self.assertEqual(missing['n_points'], 0)
        self.assertTrue(interval_bin_mask(missing).all())
        json.dumps(missing, allow_nan=False)

    def test_recrossings_are_not_silently_smoothed(self):
        r = onset_interval([1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
                           [0, -35, -5, -40, -10], 30)
        self.assertEqual((r['lower_log_m'], r['upper_log_m']), (-8., -7.))
        self.assertEqual(r['recrossings'], 2)
        self.assertEqual(r['n_below_after_first_crossing'], 2)

    def test_direction_and_equality(self):
        r = onset_interval([1e-8, 1e-7], [10, 30], 30, direction=1)
        self.assertEqual(r['upper_log_m'], -7.)
        self.assertEqual(r['first_crossing_index'], 1)

    def test_invalid_inputs(self):
        examples = [([1e-5, 1e-5], [-5, -6], 20, -1),
                    ([0], [-5], 20, -1), ([1e-5], [-5], 0, -1),
                    ([1e-5], [], 20, -1), ([1e-5], [-5], 20, 0)]
        for args in examples:
            with self.subTest(args=args), self.assertRaises(ValueError):
                onset_interval(*args)

    def test_real_ncats_paired_fixture(self):
        # Official NCATS AR-MDA-KB2 antagonist p1, Nitrofurazone
        # SAMPLE_ID NCGC00090686-07, antagonist3 SDID153069292,
        # viability3 SDID153083496; observed raw CONC are M and DATA are %.
        c = [1.18e-9, 2.638e-9, 5.8989999999e-9, 1.319e-8,
             2.9489999999e-8, 6.5949999999e-8, 1.475e-7, 3.297e-7,
             7.3729999999e-7, 1.649e-6, 3.6869999999e-6, 8.244e-6,
             1.8429999999e-5, 4.1219999999e-5, 9.217e-5]
        target = [-9.51200358, 13.38308097, 2.78173374, -1.35206729,
                  7.29560965, -.64921595, 7.99301431, -2.63702871,
                  -12.60750318, -6.80200633, -.18404792, -6.53223936,
                  -8.44575007, -30.16887527, -71.48245184]
        viability = [-.34270022, 3.95403144, 4.43516775, -.06008404,
                     .15810338, .62539555, 3.56904129, 3.21261397,
                     -2.10401915, -1.01020442, -1.3003137, -3.70776842,
                     -2.85731726, -15.13098512, -47.00840546]
        t, v = onset_interval(c, target, 30), onset_interval(c, viability, 20)
        self.assertEqual(t['first_crossing_index'], 13)
        self.assertEqual(v['first_crossing_index'], 14)
        self.assertEqual(t['recrossings'], 0)
        self.assertEqual(v['recrossings'], 0)
        # Earlier first observed target crossing alone does not establish 3x separation.
        self.assertEqual(window_reference_bounds(t, v)['status'], 'unresolved')
        self.assertTrue(pair_allowed_mask(t, v).any())


class ResolutionTests(unittest.TestCase):
    def test_exact_grid_edges_do_not_double_count(self):
        edges = np.array([-2., -1., 0.])
        np.testing.assert_array_equal(interval_bin_mask(interval(-2., -1.), edges),
                                      [False, True, False, False])
        np.testing.assert_array_equal(interval_bin_mask(interval(None, -2.), edges),
                                      [True, False, False, False])
        np.testing.assert_array_equal(interval_bin_mask(interval(0., None), edges),
                                      [False, False, False, True])
        np.testing.assert_array_equal(interval_bin_mask(interval(-1.5, -.5), edges),
                                      [False, True, True, False])

    def test_missing_pair_gives_full_mask(self):
        unknown = interval(None, None)
        self.assertTrue(pair_allowed_mask(unknown, unknown).all())
        self.assertEqual(pair_allowed_mask(unknown, unknown).shape,
                         (len(DEFAULT_EDGES) + 1,) * 2)

    def test_invalid_grid_and_interval(self):
        for edges in ([0, 0], [1, 0], [np.nan], []):
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                interval_bin_mask(interval(-2, -1), edges)
        for bounds in ((0, 0), (1, -1), (np.nan, 1)):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                interval_bin_mask(interval(*bounds))

    def test_three_window_reference_outcomes(self):
        yes = window_reference_bounds(interval(-8, -7), interval(-6, -5),
                                      target_horizon=-4, viability_horizon=-4, margin_log10=.5)
        no = window_reference_bounds(interval(-5, -4), interval(-8, -7),
                                     target_horizon=-3, viability_horizon=-3, margin_log10=.5)
        maybe = window_reference_bounds(interval(-7, -6), interval(-6.5, -5),
                                        target_horizon=-4, viability_horizon=-4, margin_log10=.5)
        self.assertEqual(yes['status'], 'definite_yes')
        self.assertEqual(no['status'], 'definite_no')
        self.assertEqual(maybe['status'], 'unresolved')

    def test_open_lower_endpoint_at_horizon_is_impossible(self):
        # T > -5 cannot satisfy T <= -5 even if the viability margin is large.
        r = window_reference_bounds(interval(-5, None), interval(-3, None),
                                    target_horizon=-5, viability_horizon=-3, margin_log10=0)
        self.assertEqual(r['status'], 'definite_no')

    def test_both_right_censored_scope_is_explicit(self):
        r = window_reference_bounds(interval(-5, None, -5), interval(-5, None, -5))
        self.assertEqual(r['status'], 'definite_no')
        self.assertEqual(r['global_margin_status'], 'unresolved')
        self.assertEqual(r['scope'], 'window_within_tested_concentration_range')

    def test_missing_horizon_stays_unresolved(self):
        r = window_reference_bounds(interval(None, None, None), interval(-5, None, -5))
        self.assertEqual(r['status'], 'unresolved')
        self.assertEqual(r['reason'], 'missing_tested_horizon')

    def test_window_masks_match_rectangle_references(self):
        edges = np.array([-2., -1., 0.])
        lower, upper = [None, *edges], [*edges, None]
        for ht, hv, margin in ((0, 0, 0), (0, 0, .5), (-1, -.5, .5)):
            guaranteed, possible = window_cell_masks(edges, ht, hv, margin)
            self.assertFalse(np.any(guaranteed & ~possible))
            for i in range(4):
                for j in range(4):
                    r = window_reference_bounds(interval(lower[i], upper[i]),
                                                interval(lower[j], upper[j]), ht, hv, margin)
                    self.assertEqual(bool(guaranteed[i, j]), r['definite_yes'])
                    self.assertEqual(bool(possible[i, j]), r['possible'])

    def test_rectangle_bounds_cover_concrete_points(self):
        edges = np.array([-2., -1., 0.])
        guaranteed, possible = window_cell_masks(edges, -.2, .2, .5)
        # Include upper endpoints exactly and points arbitrarily above open lowers.
        samples = [[-20, -3, -2], [-1.99999, -1.5, -1],
                   [-.99999, -.5, 0], [.00001, .5, 10]]
        for i, ts in enumerate(samples):
            for j, vs in enumerate(samples):
                outcomes = [t + .5 <= v and t <= -.2 and t + .5 <= .2
                            for t in ts for v in vs]
                if guaranteed[i, j]:
                    self.assertTrue(all(outcomes))
                if not possible[i, j]:
                    self.assertFalse(any(outcomes))

    def test_tails_and_horizon_validation(self):
        with np.errstate(invalid='raise'):
            yes, possible = window_cell_masks(DEFAULT_EDGES, -4, -4)
        self.assertEqual(yes.dtype, np.bool_)
        self.assertEqual(possible.dtype, np.bool_)
        for ht, hv, margin in ((np.inf, 0, .5), (0, np.nan, .5), (0, 0, -.1)):
            with self.subTest(values=(ht, hv, margin)), self.assertRaises(ValueError):
                window_cell_masks(DEFAULT_EDGES, ht, hv, margin)


if __name__ == '__main__':
    unittest.main()
