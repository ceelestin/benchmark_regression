from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.ensemble import GradientBoostingRegressor


class Solver(BaseSolver):

    name = 'GradientBoosting'

    # Stochasticity-controlling hyperparameters are fixed:
    #   subsample=0.8  : stochastic GB (fraction of samples drawn per tree)
    #   max_features=1.0: use all features per split (tree stochasticity comes
    #                     only from subsample, not feature subsampling)
    #   learning_rate=0.1, max_depth=3: standard fixed capacity settings
    # Only n_estimators is varied.
    parameters = {
        "n_estimators": [100],
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

        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=0.1,   # fixed
            max_depth=3,         # fixed
            subsample=0.8,       # fixed: stochastic gradient boosting
            max_features=1.0,    # fixed: no feature subsampling
        )

    def run(self, n_iter):
        self.model.fit(self.X_train, self.y_train)

    def get_result(self):
        return dict(model=self.model)
