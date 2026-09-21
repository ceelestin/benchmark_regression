from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):
    name = "sim_classification"

    # Binary classification via a logistic model:
    # logit = X @ beta / sqrt(n_features)   (unit-variance logit)
    # y ~ Bernoulli(sigmoid(logit))          (cast to float for R² scoring)
    #
    # The logit is scaled by 1/sqrt(n_features) so that its variance is 1
    # regardless of dimensionality, yielding a balanced and non-trivial
    # classification problem. R² on {0, 1} targets remains a valid measure of
    # explained variance and is equivalent to 1 - Brier_score / Var(y).
    parameters = {
        "n_samples": [100],
        "n_features": [5],
        "seed": [0],
        "fixed": [True, False],
    }

    def get_data(self):
        rng = np.random.RandomState(0 if self.fixed else self.seed)
        beta = rng.randn(self.n_features)
        X = rng.randn(self.n_samples, self.n_features)

        logit = X @ beta / np.sqrt(self.n_features)
        prob = 1.0 / (1.0 + np.exp(-logit))
        y = (rng.rand(self.n_samples) < prob).astype(float)

        cat_indicator = [False] * X.shape[1]

        return dict(X=X, y=y, categorical_indicator=cat_indicator, beta=beta)
