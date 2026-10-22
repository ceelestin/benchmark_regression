from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.tree import DecisionTreeRegressor


class Solver(BaseSolver):

    name = 'AdaBoost'

    # AdaBoost.R2 (sklearn AdaBoostRegressor, linear loss, learning_rate 1):
    # each round refits a tree of depth max_depth on a weighted bootstrap of
    # the training set. random_state is left unset so the resampling differs
    # across fits.
    parameters = {
        "n_estimators": [50],
        "max_depth": [3],
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

        self.model = AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=self.max_depth),
            n_estimators=self.n_estimators,
        )

    def run(self, n_iter):
        self.model.fit(self.X_train, self.y_train)

    def get_result(self):
        return dict(model=self.model)
