from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):
    # Name to select the dataset in the CLI and to display the results.
    name = "simbayes"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        # "n_samples": [100, 200, 500, 1000, 5000, 10000, 50000, 100000]
        # "n_features": [5],
        # "noise": [0.2, 0.3, 0.5, 1, 2],
        # "seed": list(range(200)),
        "n_samples": [100],
        "n_features": [5],
        "noise": [0.2],
        "seed": [0],
    }

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # Generate pseudorandom data using `numpy`.
        # The following seed must be the same as in the BayesEstimator utils
        rng = np.random.RandomState(self.seed)
        beta = rng.randn(self.n_features)
        X = rng.randn(self.n_samples, self.n_features)
        y = X @ beta + self.noise * rng.randn(self.n_samples)
        cat_indicator = [False] * X.shape[1]

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(
            X=X, y=y, categorical_indicator=cat_indicator, beta=beta
            )
