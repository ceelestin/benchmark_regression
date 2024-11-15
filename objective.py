from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.dummy import DummyRegressor
    from sklearn.model_selection import (
        KFold, RepeatedKFold, ShuffleSplit, train_test_split
    )


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):
    # Name to select the objective in the CLI and to display the results.
    name = "Regression"
    url = "https://github.com/tommoral/benchmark_classification"

    is_convex = False

    requirements = ["pip:scikit-learn"]

    # List of parameters for the objective. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    # This means the OLS objective will have a parameter `self.whiten_y`.
    parameters = {
        # "seed": list(range(200)),
        "test_size": [0.20],
        "val_size": [0.20],
        "procedure": ["train_test_split", "KFold", "RepeatedKFold",
                      "ShuffleSplit"],
        "n_repeats": [1, 2, 3],
        "n_splits": list(range(1, 11)),
    }

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.5"

    def skip(self, **data):
        if self.procedure == "train_test_split" and self.n_splits != 1:
            return True, "train_test_split does not require n_splits"
        if self.n_splits != int(1.0 / self.test_size):
            if self.procedure == "KFold":
                return True, "KFold's n_splits must be 1/test_size"
            if self.procedure == "RepeatedKFold":
                return True, "RepeatedKFold's n_splits must be 1/test_size"
        if self.procedure == "RepeatedKFold" and self.n_repeats == 1:
            return True, "RepeatedKFold with 1 repeat is equivalent to KFold"
        if self.procedure != "RepeatedKFold" and self.n_repeats != 1:
            return True, f"{self.procedure} does not require n_repeats"
        # if self.procedure == "ShuffleSplit" and self.n_splits is None:
        #     return True, "ShuffleSplit requires n_splits to be defined"
        # return False, None

    def set_data(self, X, y, categorical_indicator, seed):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        rng = np.random.RandomState(seed)

        if self.procedure == "train_test_split":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=rng
            )
        elif self.procedure == "KFold":
            folding = KFold(
                n_splits=self.n_splits, shuffle=True, random_state=rng
            )
        elif self.procedure == "RepeatedKFold":
            folding = RepeatedKFold(
                n_splits=self.n_splits, n_repeats=self.n_repeats,
                random_state=rng
            )
        elif self.procedure == "ShuffleSplit":
            folding = ShuffleSplit(
                n_splits=self.n_splits, test_size=self.test_size,
                random_state=rng
            )

        if self.n_splits != 1:
            X_train, y_train = [], []
            X_val, y_val = [], []
            X_test, y_test = [], []
            for train_index, test_index in folding.split(X):
                X_train_tmp, X_val_tmp, y_train_tmp, y_val_tmp = \
                    train_test_split(
                        X[train_index], y[train_index],
                        test_size=self.val_size/(1-self.test_size),
                        random_state=rng
                        )
                X_train.append(X_train_tmp)
                X_val.append(X_val_tmp)
                X_test.append(X[test_index])
                y_train.append(y_train_tmp)
                y_val.append(y_val_tmp)
                y_test.append(y[test_index])
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=self.val_size/(1-self.test_size),
                random_state=rng
                )

        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        self.categorical_indicator = categorical_indicator
        self.seed = seed

    def evaluate_result(self, model):
        # The arguments of this function are the outputs of the
        # `Solver.get_result`. This defines the benchmark's API to pass
        # solvers' result. This is customizable for each benchmark.
        if self.n_splits == 1:
            score_train = model.score(self.X_train, self.y_train)
            score_test = model.score(self.X_test, self.y_test)
            score_val = model.score(self.X_val, self.y_val)
        else:
            score_train, score_test, score_val = 0, 0, 0
            for i in range(self.n_repeats):
                for j in range(self.n_splits):
                    k = i * self.n_splits + j
                    score_train += model.score(
                        self.X_train[k], self.y_train[k]
                        )
                    score_test += model.score(self.X_test[k], self.y_test[k])
                    score_val += model.score(self.X_val[k], self.y_val[k])
            score_train /= (self.n_splits * self.n_repeats)
            score_test /= (self.n_splits * self.n_repeats)
            score_val /= (self.n_splits * self.n_repeats)

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(
            score_test=score_test,
            score_train=score_train,
            score_val=score_val,
            value=1 - score_test,
        )

    def get_one_result(self):
        # Return one solution. The return value should be an object compatible
        # with `self.compute`. This is mainly for testing purposes.
        if self.n_splits == 1:
            return dict(model=DummyRegressor().fit(self.X_train, self.y_train))
        else:
            return dict(model=DummyRegressor().fit(
                self.X_train[0], self.y_train[0]
                ))

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.
        return dict(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            categorical_indicator=self.categorical_indicator,
            seed=self.seed,
        )
