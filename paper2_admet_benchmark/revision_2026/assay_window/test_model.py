"""Numerical and observation-likelihood checks; no molecular data training."""
import unittest

import numpy as np

from .model import CensoredPairModel


def exact_masks(target, viability, k):
    masks = np.zeros((len(target), k, k), dtype=bool)
    masks[np.arange(len(target)), target, viability] = True
    return masks


class CensoredPairModelTests(unittest.TestCase):
    def test_analytic_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(71)
        design = np.column_stack((np.ones(7), rng.normal(size=(7, 2))))
        mask = rng.random((7, 3, 3)) < .4
        mask[:, 0, 0] = True
        mask[-1] = True
        weights = np.arange(1, 8, dtype=float)
        weights /= weights.sum()
        for association in (False, True):
            model = CensoredPairModel(3, association=association, l2=.03)
            theta = rng.normal(scale=.2, size=3 * (6 + int(association)))
            _, gradient = model._loss_gradient(theta, design, mask, weights)
            numerical = np.empty_like(theta)
            step = 1e-5
            for j in range(len(theta)):
                delta = np.zeros_like(theta)
                delta[j] = step
                numerical[j] = (model._loss_gradient(theta + delta, design, mask, weights)[0]
                                - model._loss_gradient(theta - delta, design, mask, weights)[0]) / (2 * step)
            np.testing.assert_allclose(gradient, numerical, atol=2e-8, rtol=2e-6)

    def test_unrestricted_observations_contribute_zero(self):
        model = CensoredPairModel(3, l2=0)
        design = np.array([[1., -1.], [1., 2.]])
        theta = np.linspace(-.8, .9, 14)
        loss, gradient = model._loss_gradient(theta, design, np.ones((2, 3, 3), bool), np.array([.4, .6]))
        self.assertEqual(loss, 0.0)
        np.testing.assert_array_equal(gradient, np.zeros_like(theta))
        with self.assertRaisesRegex(ValueError, "informative positive-weight"):
            CensoredPairModel(3).fit(np.array([[1.], [3.]]), np.ones((2, 3, 3), bool))
        fitted = CensoredPairModel(3).fit(np.array([[1.], [3.]]), exact_masks([0, 2], [1, 2], 3))
        self.assertEqual(fitted.negative_log_likelihood([[8.]], np.ones((1, 3, 3), bool)), 0.0)

    def test_probabilities_and_exact_independent_factorization(self):
        X = np.array([[-2.], [-1.], [0.], [1.], [2.], [3.]])
        mask = exact_masks([0, 0, 1, 1, 2, 2], [0, 1, 2, 0, 1, 2], 3)
        for associated in (False, True):
            model = CensoredPairModel(3, association=associated, l2=.03).fit(X, mask)
            joint = model.predict_joint([[-1.5], [2.5]])
            self.assertEqual(joint.shape, (2, 3, 3))
            self.assertTrue(np.all(joint >= 0))
            np.testing.assert_allclose(joint.sum(axis=(1, 2)), 1.0, atol=1e-14)
            if not associated:
                np.testing.assert_allclose(joint, joint.sum(axis=2)[:, :, None] * joint.sum(axis=1)[:, None, :], atol=1e-14)

    def test_only_training_features_determine_standardization(self):
        X = np.array([[2., 5.], [4., 5.], [8., 5.]])
        model = CensoredPairModel(2).fit(X, exact_masks([0, 0, 1], [0, 1, 1], 2), sample_weight=[1, 1, 2])
        np.testing.assert_allclose(model.mean_, [5.5, 5.0])
        self.assertEqual(model.scale_[1], 1.0)
        before = (model.mean_.copy(), model.scale_.copy())
        model.predict_joint([[1000., -1000.]])
        np.testing.assert_array_equal(model.mean_, before[0])
        np.testing.assert_array_equal(model.scale_, before[1])

    def test_molecule_weights_remove_replication_count_effect(self):
        X = np.array([[-1.], [0.], [1.], [2.]])
        mask = exact_masks([0, 0, 1, 1], [0, 1, 0, 1], 2)
        repeated = [0, 0, 0, 1, 2, 3]
        a = CensoredPairModel(2, l2=.01).fit(X, mask)
        b = CensoredPairModel(2, l2=.01).fit(X[repeated], mask[repeated], [1/3, 1/3, 1/3, 1, 1, 1])
        np.testing.assert_allclose(a.predict_joint(X), b.predict_joint(X), atol=1e-8)

    def test_censored_likelihood_sums_allowed_cells(self):
        model = CensoredPairModel(3).fit([[0.], [1.]], exact_masks([0, 2], [1, 2], 3))
        mask = np.zeros((1, 3, 3), bool)
        mask[0, 1:, :2] = True
        probability = model.predict_joint([[.3]])[0][mask[0]].sum()
        self.assertAlmostEqual(model.negative_log_likelihood([[.3]], mask), -np.log(probability), places=14)

    def test_observation_losses_equal_weighted_summary(self):
        model = CensoredPairModel(3).fit([[-1.], [0.], [1.]], exact_masks([0, 1, 2], [0, 2, 2], 3))
        X = np.array([[-.5], [.2], [1.5]])
        masks = exact_masks([0, 1, 2], [0, 2, 2], 3)
        masks[1] = True
        masks[2, 1:, 1:] = True
        losses = model.observation_losses(X, masks)
        weights = np.array([2., 3., 5.])
        self.assertEqual(losses.shape, (3,))
        self.assertEqual(losses[1], 0.0)
        self.assertTrue(np.isfinite(losses).all())
        self.assertAlmostEqual(model.negative_log_likelihood(X, masks, weights),
                               np.average(losses, weights=weights), places=14)
        self.assertAlmostEqual(model.negative_log_likelihood(X, masks), losses.mean(), places=14)

    def test_uninformative_and_zero_weight_rows_cannot_change_fit(self):
        X = np.array([[-2.], [-1.], [1.], [2.]])
        masks = exact_masks([0, 0, 2, 2], [0, 0, 2, 2], 3)
        extra_X = np.full((101, 1), 50.)
        extra_masks = np.ones((101, 3, 3), bool)
        extra_masks[-1] = exact_masks([1], [1], 3)[0]
        weights = np.concatenate((np.ones(4), np.ones(100), [0.]))
        for association in (False, True):
            original = CensoredPairModel(3, association=association, l2=.01).fit(X, masks)
            augmented = CensoredPairModel(3, association=association, l2=.01).fit(
                np.concatenate((X, extra_X)), np.concatenate((masks, extra_masks)), weights)
            self.assertEqual(augmented.n_informative_, 4)
            self.assertEqual(augmented.n_uninformative_, 101)
            self.assertEqual(original.n_uninformative_, 0)
            np.testing.assert_array_equal(original.mean_, augmented.mean_)
            np.testing.assert_array_equal(original.scale_, augmented.scale_)
            np.testing.assert_allclose(original.theta_, augmented.theta_, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(original.predict_joint([[1.], [3.]]),
                                       augmented.predict_joint([[1.], [3.]]), atol=1e-12, rtol=1e-12)

    def test_only_positive_weight_uninformative_rows_reject_training(self):
        masks = np.concatenate((exact_masks([0], [1], 2), np.ones((1, 2, 2), bool)))
        with self.assertRaisesRegex(ValueError, "informative positive-weight"):
            CensoredPairModel(2).fit([[0.], [50.]], masks, sample_weight=[0., 1.])

    def test_identifiable_dependence_on_heldout_synthetic_pairs(self):
        # Marginals are balanced; dependence changes sign with X. An independent
        # model cannot recover the paired table from marginals alone.
        rng = np.random.default_rng(814)
        def make(n):
            x = rng.choice([-1., 1.], size=n)
            t = rng.integers(0, 2, size=n)
            same = rng.random(n) < np.where(x > 0, .9, .1)
            v = np.where(same, t, 1 - t)
            return x[:, None], exact_masks(t, v, 2)
        X, masks = make(1200)
        test_X, test_masks = make(2000)
        independent = CensoredPairModel(2, association=False, l2=.005).fit(X, masks)
        associated = CensoredPairModel(2, association=True, l2=.005).fit(X, masks)
        self.assertTrue(independent.converged_)
        self.assertTrue(associated.converged_)
        gain = independent.negative_log_likelihood(test_X, test_masks) - associated.negative_log_likelihood(test_X, test_masks)
        self.assertGreater(gain, .20)
        joint = associated.predict_joint([[-1.], [1.]])
        self.assertLess(np.trace(joint[0]), .25)
        self.assertGreater(np.trace(joint[1]), .75)

    def test_rejects_invalid_or_empty_observation_masks(self):
        with self.assertRaises(ValueError):
            CensoredPairModel(2).fit([[0.]], np.zeros((1, 2, 2), bool))
        with self.assertRaises(ValueError):
            CensoredPairModel(2).fit([[0.]], np.ones((1, 2, 2), int))
        with self.assertRaises(ValueError):
            CensoredPairModel(2).fit([[0.]], np.ones((2, 2, 2), bool))
        with self.assertRaises(ValueError):
            CensoredPairModel(2).fit([[np.nan]], np.ones((1, 2, 2), bool))
        with self.assertRaises(ValueError):
            CensoredPairModel(2).fit([[0.]], np.ones((1, 2, 2), bool), [0.])
        with self.assertRaises(ValueError):
            CensoredPairModel(2).fit([[0.]], np.ones((1, 2, 2), bool), [-1.])
        with self.assertRaises(RuntimeError):
            CensoredPairModel(2).predict_joint([[0.]])


if __name__ == "__main__":
    unittest.main()
