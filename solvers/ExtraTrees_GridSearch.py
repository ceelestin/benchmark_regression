from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.model_selection import GridSearchCV


# ExtraTrees with in-solver hyperparameter tuning by GRID SEARCH.
# The search is run with internal cross-validation on X_train only; the refit
# best_estimator_ is what the objective evaluates on the held-out X_test/bench/outer.
# n_estimators is deliberately NOT tuned (more trees only reduces variance and then
# plateaus): we fix it and tune the parameters that actually trade off bias/variance.
class Solver(BaseSolver):

    name = 'ExtraTrees_GridSearch'

    # Only scalar parameters are exposed to benchopt: a dict/list parameter value
    # (like the grid itself) becomes an un-serializable results column and forces the
    # parquet writer to fall back to CSV. The grid is built in set_objective instead.
    parameters = {
        "n_estimators": [200],
        "cv": [3],
    }

    # Force solver to run only once if you don't want to record training steps
    sampling_strategy = "run_once"

    def set_objective(
        self, X_train, y_train, X_bench, y_bench,
        X_outer, y_outer, categorical_indicator, beta
    ):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        self.X_train, self.y_train = X_train, y_train
        self.X_bench, self.y_bench = X_bench, y_bench
        self.X_outer, self.y_outer = X_outer, y_outer
        self.cat_ind = categorical_indicator
        self.beta = beta

        # Grid over max_features / min_samples_leaf / max_depth -> 3 * 2 * 2 = 12 points.
        # Its 12 candidates match RandomizedSearchCV's n_iter for an equal number of fits.
        param_grid = {
            "max_features": [1.0, 0.5, "sqrt"],
            "min_samples_leaf": [1, 5],
            "max_depth": [None, 10],
        }

        # n_jobs=1: benchopt already parallelizes across solver instances and pins
        # BLAS/OMP threads to 1, so internal parallelism would oversubscribe.
        # random_state left unset, matching the ExtraTrees_noHPO baseline, so each run
        # draws fresh trees and tuned-vs-untuned stays apples-to-apples.
        self.search = GridSearchCV(
            ExtraTreesRegressor(n_estimators=self.n_estimators),
            param_grid=param_grid,
            cv=self.cv,
            n_jobs=1,
        )

    def run(self, n_iter):
        # Fits the grid search (internal CV on X_train) and refits best_estimator_.
        self.search.fit(self.X_train, self.y_train)

    def get_result(self):
        # Return the tuned, refit estimator under the `model` key expected by
        # `Objective.evaluate_result`.
        return dict(model=self.search.best_estimator_)
