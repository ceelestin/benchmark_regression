from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):
    name = "sim_interactions"

    # Linear model with all pairwise interaction effects:
    # y = X @ beta + sum_{i<j} gamma_{ij} * X_i * X_j + noise
    # gamma is scaled by 1/sqrt(n_pairs) so the interaction term has unit
    # variance, matching the scale of the linear term.
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
        n_pairs = self.n_features * (self.n_features - 1) // 2
        gamma = rng.randn(n_pairs) / np.sqrt(n_pairs)

        X = rng.randn(self.n_samples, self.n_features)

        y = X @ beta
        k = 0
        for i in range(self.n_features):
            for j in range(i + 1, self.n_features):
                y = y + gamma[k] * X[:, i] * X[:, j]
                k += 1
        y = y + self.noise * rng.randn(self.n_samples)

        cat_indicator = [False] * X.shape[1]

        return dict(X=X, y=y, categorical_indicator=cat_indicator, beta=beta)
