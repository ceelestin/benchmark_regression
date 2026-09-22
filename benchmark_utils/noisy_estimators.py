"""Predictors with controllable fold-to-fold variability.

Both wrappers fit a base scikit-learn regressor and then perturb it so that the
*training-driven* component of the out-of-fold error can be dialled up
independently of the study set, the base learner and its bias.  They are the
levers of the redundancy-score calibration experiments:

* :class:`NoisyPredictionRegressor` adds zero-mean noise to the predictions.
  The noise is a deterministic function of the input row and of a per-fit
  seed, so the same sample receives the same perturbation whichever array it
  is predicted in (test fold, study set, bench set, outer chunks) while
  different fits (different CV folds) draw independent perturbations.  This
  drives the per-sample error correlation across folds towards zero without
  touching the base predictor.

* :class:`NoisyCoefRegressor` perturbs the fitted linear coefficients once
  per fit.  Each fold then carries its own smooth bias function of ``x``, the
  way an unstable learner trained on a different subsample would, which
  correlates errors across *different* samples within a fold and de-correlates
  them across folds.

Noise scales are relative: ``pred_noise`` multiplies the standard deviation
of the training targets, ``coef_noise`` multiplies the root-mean-square
coefficient.  ``0`` recovers the base learner exactly.
"""
import hashlib

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone


def _fit_seed(X, y):
    """Deterministic per-fit seed from the training data (differs across
    CV folds, identical for identical training sets)."""
    h = hashlib.blake2b(digest_size=8)
    h.update(np.ascontiguousarray(X, dtype=float).tobytes())
    h.update(np.ascontiguousarray(y, dtype=float).tobytes())
    return int.from_bytes(h.digest(), "little")


def _hashed_normal(X, seed, n_features_rff=32, frequency_scale=50.0):
    """Approximately standard-normal noise that is a deterministic function
    of each input row.

    Uses random Fourier features with very high frequencies: for rows that
    differ by more than ~1/frequency_scale in any direction the phases are
    effectively independent uniform variables, so the average of
    ``n_features_rff`` cosines is close to Gaussian (CLT) with unit variance
    and is uncorrelated with any smooth function of ``x``.  Rows are hashed
    only through ``x``, so predictions for the same sample agree across calls.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=frequency_scale, size=(X.shape[1], n_features_rff))
    b = rng.uniform(0.0, 2.0 * np.pi, size=n_features_rff)
    Z = np.cos(X @ W + b)                      # each column: mean 0, var 1/2
    return Z.sum(axis=1) * np.sqrt(2.0 / n_features_rff)


class NoisyPredictionRegressor(BaseEstimator, RegressorMixin):
    """Base regressor plus per-sample noise ``pred_noise * std(y_train)``."""

    def __init__(self, base_estimator, pred_noise=0.0, n_features_rff=32,
                 frequency_scale=50.0):
        self.base_estimator = base_estimator
        self.pred_noise = pred_noise
        self.n_features_rff = n_features_rff
        self.frequency_scale = frequency_scale

    def fit(self, X, y):
        self.model_ = clone(self.base_estimator).fit(X, y)
        self.noise_scale_ = float(self.pred_noise) * float(np.std(y))
        self.seed_ = _fit_seed(X, y)
        return self

    def predict(self, X):
        pred = self.model_.predict(X)
        if self.noise_scale_ > 0:
            pred = pred + self.noise_scale_ * _hashed_normal(
                X, self.seed_, self.n_features_rff, self.frequency_scale
            )
        return pred


class NoisyCoefRegressor(BaseEstimator, RegressorMixin):
    """Linear base regressor whose fitted ``coef_`` (and intercept) are
    perturbed once per fit by ``coef_noise * rms(coef_) * N(0, 1)``."""

    def __init__(self, base_estimator, coef_noise=0.0):
        self.base_estimator = base_estimator
        self.coef_noise = coef_noise

    def fit(self, X, y):
        model = clone(self.base_estimator).fit(X, y)
        coef = np.asarray(model.coef_, dtype=float)
        rng = np.random.default_rng(_fit_seed(X, y))
        scale = float(self.coef_noise) * float(
            np.sqrt(np.mean(coef ** 2)) if coef.size else 0.0
        )
        if scale > 0:
            model.coef_ = coef + scale * rng.standard_normal(coef.shape)
            if hasattr(model, "intercept_"):
                model.intercept_ = model.intercept_ + scale * rng.standard_normal()
        self.model_ = model
        return self

    def predict(self, X):
        return self.model_.predict(X)
