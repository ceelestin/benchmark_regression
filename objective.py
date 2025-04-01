from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from sklearn.dummy import DummyRegressor
    from sklearn.model_selection import (RepeatedKFold, ShuffleSplit,
                                         train_test_split)


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
        "n_repeats": [1, 2, 3, 4, 5, 10, 20],
        "n_splits": list(range(1, 11), 15, 20, 25, 50, 100),
        "procedure": ["train_test_split", "RepeatedKFold", "ShuffleSplit"],
        "study_size": [10, 18, 32, 56, 100, 178, 316, 562, 1000, 1778, 3162,
                       5623, 10000],
        "test_size": [0.20],
        "val_size": [0.20],
    }

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.5"

    def skip(self, **data):
        if self.procedure == "train_test_split" and self.n_splits != 1:
            return True, "train_test_split does not require n_splits"
        if self.procedure == "RepeatedKFold":
            if self.n_splits != int(1.0 / self.test_size):
                return True, "RepeatedKFold's n_splits must be 1/test_size"
        if self.procedure != "RepeatedKFold" and self.n_repeats != 1:
            return True, f"{self.procedure} does not require n_repeats"
        return False, None

    def set_data(self, X, y, categorical_indicator, beta):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        self.X, self.y = X, y

        if self.procedure == "train_test_split":
            self.cv_bool = False
        elif self.procedure == "RepeatedKFold":
            self.cv_bool = True
            self.cv = RepeatedKFold(
                n_splits=self.n_splits, n_repeats=self.n_repeats
            )
        elif self.procedure == "ShuffleSplit":
            self.cv_bool = True
            self.cv = ShuffleSplit(
                n_splits=self.n_splits, test_size=self.test_size
            )

        self.categorical_indicator = categorical_indicator
        self.beta = beta

    def evaluate_result(self, model):
        # The arguments of this function are the outputs of the
        # `Solver.get_result`. This defines the benchmark's API to pass
        # solvers' result. This is customizable for each benchmark.
        score_train = model.score(self.X_train, self.y_train)
        score_test = model.score(self.X_test, self.y_test)
        score_val = model.score(self.X_val, self.y_val)
        score_bench = model.score(self.X_bench, self.y_bench)

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(
            score_test=score_test,
            score_train=score_train,
            score_val=score_val,
            score_bench=score_bench,
            value=1 - score_test,
        )

    def get_one_result(self):
        # Return one solution. The return value should be an object compatible
        # with `self.compute`. This is mainly for testing purposes.
        return dict(model=DummyRegressor().fit(self.X_train, self.y_train))

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.
        self.X_study, self.X_bench, self.y_study, self.y_bench = \
            train_test_split(
                self.X, self.y, test_size=100000, random_state=0
            )

        if self.study_size != 10000:
            _, self.X_study, _, self.y_study = train_test_split(
                self.X_study, self.y_study, test_size=self.study_size
                )

        if self.cv_bool:
            self.X_train, self.X_test, self.y_train, self.y_test = \
                self.get_split(self.X_study, self.y_study)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(
                    self.X_study, self.y_study, test_size=self.test_size
                )

        self.X_train, self.X_val, self.y_train, self.y_val = \
            train_test_split(
                self.X_train, self.y_train,
                test_size=self.val_size/(1-self.test_size)
                )

        return dict(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            X_bench=self.X_bench,
            y_bench=self.y_bench,
            categorical_indicator=self.categorical_indicator,
            beta=self.beta
        )
