from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):
    name = "sim_nonlinear"

    # Generalized additive model with sinusoidal nonlinearity:
    # y = sum_j beta_j * sin(X_j) + noise
    # A linear model cannot recover this signal, making it a harder benchmark.
    parameters = {
        "n_samples": [100],
        "n_features": [5],
        "noise": [0.2],
        "seed": [0],
        "fixed": [True, False],
    }

    def get_data(self):
        rng = np.random.RandomState(0 if self.fixed else self.seed)
        beta = rng.randn(self.n_features)
        X = rng.randn(self.n_samples, self.n_features)
        y = np.sin(4 * X) @ beta + self.noise * rng.randn(self.n_samples)
        cat_indicator = [False] * X.shape[1]

        return dict(X=X, y=y, categorical_indicator=cat_indicator, beta=beta)
