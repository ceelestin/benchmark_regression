from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.svm import SVR


class Solver(BaseSolver):

    name = "SVR"

    parameters = {
        "C": [10.0],
        "epsilon": [0.1],
        "gamma": ["scale"],
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

        self.model = SVR(
            C=self.C,
            epsilon=self.epsilon,
            gamma=self.gamma,
        )

    def run(self, n_iter):
        self.model.fit(self.X_train, self.y_train)

    def get_result(self):
        return dict(model=self.model)
