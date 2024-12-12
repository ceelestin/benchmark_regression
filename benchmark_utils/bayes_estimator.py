from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.base import BaseEstimator, RegressorMixin


class BayesEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, n_features, noise, beta, random_state=None):
        # The following beta must be the same as in the dataset simbayes.py)
        self.beta = beta
        self.noise = noise
        self.rng = np.random.RandomState(random_state)

    def fit(self, X, y):
        pass

    def predict(self, X):
        return X @ self.beta + self.noise * self.rng.randn(X.shape[0])
