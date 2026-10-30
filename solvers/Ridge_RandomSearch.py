from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from scipy.stats import loguniform
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import RandomizedSearchCV


# Ridge with in-solver hyperparameter tuning by RANDOM SEARCH.
# Same setup as Ridge_GridSearch, but alpha is drawn log-uniformly over the grid's
# range. n_iter equals the grid size (12), so both strategies use the same number
# of model fits, as for MLP_RandomSearch.
class Solver(BaseSolver):

    name = 'Ridge_RandomSearch'

    parameters = {
        "n_iter": [12],
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

        # alpha is a scale parameter: sampled log-uniformly over [1e-4, 1e3]
        # (CE, 2026-10-30), the range of Ridge_GridSearch's grid.
        param_distributions = {"alpha": loguniform(1e-4, 1e3)}

        # random_state left unset so each split draws its own candidates.
        self.search = RandomizedSearchCV(
            Ridge(),
            param_distributions=param_distributions,
            n_iter=self.n_iter,
            cv=self.cv,
            n_jobs=1,
        )

    def run(self, n_iter):
        self.search.fit(self.X_train, self.y_train)

    def get_result(self):
        return dict(model=self.search.best_estimator_)
