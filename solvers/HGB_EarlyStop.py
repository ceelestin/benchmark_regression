from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.ensemble import HistGradientBoostingRegressor


class Solver(BaseSolver):

    name = 'HGB_EarlyStop'

    # HistGradientBoosting with early stopping on a random 10% validation split
    # of the training set (n_iter_no_change=10), up to max_iter rounds; other
    # settings are sklearn defaults. random_state is left unset so the
    # validation split differs across fits.
    parameters = {
        "max_iter": [1000],
        "learning_rate": [0.1],
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

        self.model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
        )

    def run(self, n_iter):
        self.model.fit(self.X_train, self.y_train)

    def get_result(self):
        return dict(model=self.model)
