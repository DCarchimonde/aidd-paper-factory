"""Paired discrete-onset models with interval/coarsened-label likelihood.

The bins describe protocol-defined response and viability onsets. They are not
survival times or latent causal target engagement. For an allowed set A, the
data loss is -log P((T,V) in A | X), without imputing censored observations.
This is an observation likelihood when the coarsening mechanism is ignorable
conditional on the recorded design/covariates. A fitted curve's confidence
interval is not automatically such an exact coarsening event.

Both models use additive linear heads. The associated model adds one linear
coefficient rho(X) times the fixed ordered-bin product z_T*z_V. This is a
small parametric association model, not an unrestricted dependence model.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


class CensoredPairModel:
    """CPU-only paired categorical regression on boolean observation masks.

    Parameters are fitted using a weighted mean negative log-likelihood plus
    ``l2 / 2 * ||theta||^2``. The penalty includes intercepts. Standardization
    is fitted only on informative, positive-weight training rows, using their
    renormalized sample weights. Completely unrestricted masks and zero-weight
    rows do not affect preprocessing, loss normalization, or regularization.
    A training set without informative observations is rejected. Setting
    association=False yields exactly factorized target/viability predictions,
    with the same coarsened observation likelihood and preprocessing.
    """

    def __init__(self, n_bins, association=True, l2=1e-3, max_iter=300):
        if isinstance(n_bins, (bool, np.bool_)) or not isinstance(n_bins, (int, np.integer)) or n_bins < 2:
            raise ValueError("n_bins must be an integer at least 2")
        if not isinstance(association, (bool, np.bool_)):
            raise ValueError("association must be boolean")
        if not np.isscalar(l2) or not np.isfinite(l2) or l2 < 0:
            raise ValueError("l2 must be finite and nonnegative")
        if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        self.n_bins = int(n_bins)
        self.association = bool(association)
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.z_ = np.linspace(-1.0, 1.0, self.n_bins)
        self.association_basis_ = np.outer(self.z_, self.z_)

    @staticmethod
    def _features(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a nonempty two-dimensional matrix")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        return X

    def _mask(self, allowed_mask, n):
        mask = np.asarray(allowed_mask)
        if mask.dtype != np.bool_:
            raise ValueError("allowed_mask must have boolean dtype")
        if mask.shape != (n, self.n_bins, self.n_bins):
            raise ValueError("allowed_mask must have shape (n_samples, n_bins, n_bins)")
        if not mask.any(axis=(1, 2)).all():
            raise ValueError("each observation must allow at least one bin pair")
        return mask

    @staticmethod
    def _weights(sample_weight, n):
        if sample_weight is None:
            return np.full(n, 1.0 / n)
        weights = np.asarray(sample_weight, dtype=float)
        if weights.shape != (n,) or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("sample_weight must be a finite nonnegative length-n vector")
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("sample_weight must have finite positive total")
        return weights / total

    def _unpack(self, theta, n_design):
        count = n_design * self.n_bins
        target = theta[:count].reshape(n_design, self.n_bins)
        viability = theta[count:2 * count].reshape(n_design, self.n_bins)
        association = theta[2 * count:] if self.association else None
        return target, viability, association

    def _logits(self, theta, design):
        target, viability, association = self._unpack(theta, design.shape[1])
        logits = (design @ target)[:, :, None] + (design @ viability)[:, None, :]
        if self.association:
            logits = logits + (design @ association)[:, None, None] * self.association_basis_[None]
        return logits

    def _loss_gradient(self, theta, design, mask, weights):
        """Analytic gradient of the complete penalized training objective."""
        logits = self._logits(theta, design)
        log_z = logsumexp(logits, axis=(1, 2))
        allowed_logits = np.where(mask, logits, -np.inf)
        log_allowed_z = logsumexp(allowed_logits, axis=(1, 2))
        losses = log_z - log_allowed_z
        # Exactly zero contribution for observations that supply no restriction.
        unrestricted = mask.all(axis=(1, 2))
        losses[unrestricted] = 0.0
        p = np.exp(logits - log_z[:, None, None])
        p_given_allowed = np.exp(allowed_logits - log_allowed_z[:, None, None])
        residual = p - p_given_allowed
        residual[unrestricted] = 0.0
        residual *= weights[:, None, None]
        target_grad = design.T @ residual.sum(axis=2)
        viability_grad = design.T @ residual.sum(axis=1)
        parts = [target_grad.ravel(), viability_grad.ravel()]
        if self.association:
            association_grad = design.T @ np.sum(residual * self.association_basis_[None], axis=(1, 2))
            parts.append(association_grad)
        gradient = np.concatenate(parts) + self.l2 * theta
        loss = float(weights @ losses + 0.5 * self.l2 * (theta @ theta))
        return loss, gradient

    def fit(self, X, allowed_mask, sample_weight=None):
        X = self._features(X)
        mask = self._mask(allowed_mask, len(X))
        weights = self._weights(sample_weight, len(X))
        informative = ~mask.all(axis=(1, 2)) & (weights > 0)
        n_informative = int(informative.sum())
        if n_informative == 0:
            raise ValueError("fit requires at least one informative positive-weight observation")
        n_uninformative = int(len(X) - n_informative)
        X, mask, weights = X[informative], mask[informative], weights[informative]
        weights = weights / weights.sum()
        self.n_features_in_ = X.shape[1]
        self.mean_ = weights @ X
        variance = weights @ np.square(X - self.mean_)
        self.scale_ = np.sqrt(variance)
        self.scale_[self.scale_ <= np.finfo(float).eps] = 1.0
        design = np.column_stack((np.ones(len(X)), (X - self.mean_) / self.scale_))
        count = design.shape[1] * (2 * self.n_bins + int(self.association))
        initial = np.zeros(count)
        result = minimize(self._loss_gradient, initial, args=(design, mask, weights),
                          method="L-BFGS-B", jac=True,
                          options={"maxiter": self.max_iter, "ftol": 1e-11, "gtol": 1e-7,
                                   "maxls": 40})
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise RuntimeError("optimization returned nonfinite parameters or loss")
        self.theta_ = result.x.copy()
        self.converged_ = bool(result.success)
        self.optimization_message_ = str(result.message)
        self.n_iter_ = int(result.nit)
        self.training_objective_ = float(result.fun)
        self.n_informative_ = n_informative
        # Excluded rows: unrestricted masks, zero weights, or both.
        self.n_uninformative_ = n_uninformative
        target, viability, association = self._unpack(self.theta_, design.shape[1])
        self.coef_target_ = target.copy()
        self.coef_viability_ = viability.copy()
        self.coef_association_ = association.copy() if self.association else None
        return self

    def _design(self, X):
        if not hasattr(self, "theta_"):
            raise RuntimeError("fit must be called before prediction or evaluation")
        X = self._features(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("prediction features differ from the fitted feature count")
        return np.column_stack((np.ones(len(X)), (X - self.mean_) / self.scale_))

    def predict_joint(self, X):
        """Return joint bin probabilities with shape (n_samples, K, K)."""
        design = self._design(X)
        logits = self._logits(self.theta_, design)
        return np.exp(logits - logsumexp(logits, axis=(1, 2), keepdims=True))

    def observation_losses(self, X, allowed_mask):
        """Return one stable masked negative log-likelihood per observation.

        These are unweighted data losses, with exactly zero for unrestricted
        masks. They contain no training parameter penalty.
        """
        design = self._design(X)
        mask = self._mask(allowed_mask, len(design))
        logits = self._logits(self.theta_, design)
        losses = logsumexp(logits, axis=(1, 2)) - logsumexp(np.where(mask, logits, -np.inf), axis=(1, 2))
        losses[mask.all(axis=(1, 2))] = 0.0
        return losses

    def negative_log_likelihood(self, X, allowed_mask, sample_weight=None):
        """Weighted evaluation loss, without the training parameter penalty."""
        losses = self.observation_losses(X, allowed_mask)
        weights = self._weights(sample_weight, len(losses))
        return float(weights @ losses)
