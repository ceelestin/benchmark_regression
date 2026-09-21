from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.ensemble import RandomForestRegressor


class Solver(BaseSolver):

    name = 'RandomForest'

    # max_features='sqrt' and bootstrap=True are the standard RF defaults
    # that control stochasticity; they are fixed here (no HPO).
    # Only n_estimators is varied.
    parameters = {
        "n_estimators": [100],
        "max_depth": [100],
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

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features='sqrt',   # fixed: standard RF feature subsampling
            bootstrap=True,        # fixed: standard bagging
        )

    def run(self, n_iter):
        self.model.fit(self.X_train, self.y_train)

    def get_result(self):
        return dict(model=self.model)
