from benchopt import BaseSolver, safe_import_context

from benchmark_utils.bayes_estimator import BayesEstimator

with safe_import_context() as import_ctx:
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder as OHE


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):
    # Name to select the solver in the CLI and to display the results.
    name = "Bayes"
    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        # 'noise': [0, 0.1, 0.2, 0.3, 0.5, 0.75, 1],
        "noise": [0],
    }

    # Force solver to run only once if you don't want to record training steps
    sampling_strategy = "run_once"

    def set_objective(
        self, X_train, y_train, X_val, y_val, categorical_indicator, seed
    ):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.cat_ind = categorical_indicator
        self.seed = seed

        if type(self.X_train) is list:
            size = len(self.X_train[0][0])
        else:
            size = self.X_train.shape[1]
        preprocessor = ColumnTransformer(
            [
                (
                    "one_hot",
                    OHE(
                        categories="auto",
                        handle_unknown="ignore",
                    ),
                    [i for i in range(size) if self.cat_ind[i]],
                ),
                (
                    "numerical",
                    "passthrough",
                    [i for i in range(size) if not self.cat_ind[i]],
                ),
            ]
        )

        self.model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    BayesEstimator(
                        n_features=size, noise=self.noise, seed=self.seed
                        ),
                ),
            ]
        )

    def run(self, n_iter):
        # This is the function that is called to fit the model.
        # The param n_iter is defined if you change the sample strategy to
        # other value than "run_once"
        # https://benchopt.github.io/performance_curves.html
        if type(self.X_train) is list:
            self.model.fit(self.X_train[0], self.y_train[0])
        else:
            self.model.fit(self.X_train, self.y_train)

    def get_result(self):
        # Returns the model after fitting.
        # The outputs of this function is a dictionary which defines the
        # keyword arguments for `Objective.evaluate_result`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return dict(model=self.model)
