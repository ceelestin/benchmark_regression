from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV


# Ridge with in-solver hyperparameter tuning by GRID SEARCH, built like MLP_GridSearch.
# The search runs internal cross-validation on X_train only; the refit
# best_estimator_ is what the objective evaluates on the held-out X_test/bench/outer.
class Solver(BaseSolver):

    name = 'Ridge_GridSearch'

    # The grid is built in set_objective (a list parameter would become an
    # un-serializable results column, see MLP_GridSearch).
    parameters = {
        "cv": [3],
    }

    sampling_strategy = "run_once"

    def set_objective(
        self, X_train, y_train, X_bench, y_bench,
        X_outer, y_outer, categorical_indicator, beta
    ):
        self.X_train, self.y_train = X_train, y_train
        self.X_bench, self.y_bench = X_bench, y_bench
        self.X_outer, self.y_outer = X_outer, y_outer
        self.cat_ind = categorical_indicator
        self.beta = beta

        # 12 log-spaced values over [1e-4, 1e3] (CE, 2026-10-30): the same number of
        # candidates as the MLP and ExtraTrees searches.
        param_grid = {"alpha": np.logspace(-4, 3, 12)}

        # n_jobs=1: benchopt already parallelizes across solver instances.
        self.search = GridSearchCV(
            Ridge(),
            param_grid=param_grid,
            cv=self.cv,
            n_jobs=1,
        )

    def run(self, n_iter):
        self.search.fit(self.X_train, self.y_train)

    def get_result(self):
        return dict(model=self.search.best_estimator_)
