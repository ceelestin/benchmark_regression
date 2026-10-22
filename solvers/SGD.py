from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.linear_model import SGDRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler


class Solver(BaseSolver):

    name = 'SGD'

    # Linear model fitted by stochastic gradient descent (sklearn defaults:
    # squared loss, l2 penalty, invscaling learning rate, max_iter=1000).
    # random_state is left unset so each fit draws a fresh sample ordering:
    # the optimisation noise is the source of fit-to-fit variability.
    parameters = {
        "alpha": [1e-4],
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

        self.model = make_pipeline(
            StandardScaler(), SGDRegressor(alpha=self.alpha, max_iter=1000, tol=1e-3)
        )

    def run(self, n_iter):
        self.model.fit(self.X_train, self.y_train)

    def get_result(self):
        return dict(model=self.model)
