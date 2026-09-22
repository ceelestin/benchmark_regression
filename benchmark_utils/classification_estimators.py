"""Binary classifiers exposed as probability regressors, with controllable
fold-to-fold variability (classification analog of ``noisy_estimators``).

The objective evaluates ``model.predict(X)`` as a real-valued prediction: for a
binary target in {0, 1} these wrappers return P(y = 1 | x), so the objective's
squared error is the Brier score and, when it detects a binary target, it also
derives the 0-1 error (threshold 0.5) and the negative log-likelihood.

* :class:`ProbaRegressor`  -- any scikit-learn classifier -> predict = proba[:, 1].
* :class:`NoisyLogitRegressor` -- adds per-sample noise ``logit_noise * N(0,1)`` to
  the LOGIT of a base classifier (fresh per fit, deterministic per sample: the
  same hashed random-Fourier construction as ``NoisyPredictionRegressor``).
* :class:`NoisyCoefClassifier` -- perturbs the fitted ``coef_``/``intercept_`` of a
  linear classifier once per fit by ``coef_noise * rms(coef) * N(0,1)``.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone

from benchmark_utils.noisy_estimators import _fit_seed, _hashed_normal

_EPS = 1e-6


def _proba(model, X):
    p = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    if len(classes) == 1:                       # degenerate training fold
        return np.full(len(X), float(classes[0]))
    return p[:, classes.index(1)] if 1 in classes else p[:, -1]


class ProbaRegressor(BaseEstimator, RegressorMixin):
    """Classifier -> P(y=1|x) as the prediction."""

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y):
        self.model_ = clone(self.base_estimator).fit(X, np.asarray(y).astype(int))
        return self

    def predict(self, X):
        return _proba(self.model_, X)


class NoisyLogitRegressor(BaseEstimator, RegressorMixin):
    """Classifier -> sigmoid(logit(P) + logit_noise * eta(x)), eta ~ N(0,1) per
    sample, independent across fits."""

    def __init__(self, base_estimator, logit_noise=0.0, n_features_rff=32,
                 frequency_scale=50.0):
        self.base_estimator = base_estimator
        self.logit_noise = logit_noise
        self.n_features_rff = n_features_rff
        self.frequency_scale = frequency_scale

    def fit(self, X, y):
        self.model_ = clone(self.base_estimator).fit(X, np.asarray(y).astype(int))
        self.seed_ = _fit_seed(X, y)
        return self

    def predict(self, X):
        p = np.clip(_proba(self.model_, X), _EPS, 1 - _EPS)
        if self.logit_noise > 0:
            logit = np.log(p / (1 - p)) + float(self.logit_noise) * _hashed_normal(
                X, self.seed_, self.n_features_rff, self.frequency_scale
            )
            p = 1.0 / (1.0 + np.exp(-logit))
        return p


class NoisyCoefClassifier(BaseEstimator, RegressorMixin):
    """Linear classifier whose fitted coefficients are perturbed once per fit."""

    def __init__(self, base_estimator, coef_noise=0.0):
        self.base_estimator = base_estimator
        self.coef_noise = coef_noise

    def fit(self, X, y):
        model = clone(self.base_estimator).fit(X, np.asarray(y).astype(int))
        coef = np.asarray(model.coef_, dtype=float)
        rng = np.random.default_rng(_fit_seed(X, y))
        scale = float(self.coef_noise) * float(np.sqrt(np.mean(coef ** 2)))
        if scale > 0:
            model.coef_ = coef + scale * rng.standard_normal(coef.shape)
            model.intercept_ = model.intercept_ + scale * rng.standard_normal(
                np.shape(model.intercept_)
            )
        self.model_ = model
        return self

    def predict(self, X):
        return _proba(self.model_, X)
