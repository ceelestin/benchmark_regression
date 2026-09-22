from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.linear_model import LogisticRegression
    from benchmark_utils.classification_estimators import NoisyLogitRegressor


class Solver(BaseSolver):
    """Logistic regression with per-sample logit noise (fresh per fold, deterministic per sample): classification analog of RidgeNoisyPred. logit_noise=0 is plain LogReg."""

    name = 'LogRegNoisyLogit'
    parameters = {
        "C": [1.0],
        "logit_noise": [0.0, 0.5, 1.0, 2.0, 4.0],
    }
    sampling_strategy = "run_once"

    def set_objective(
        self, X_train, y_train, X_bench, y_bench,
        X_outer, y_outer, categorical_indicator, beta
    ):
        self.X_train, self.y_train = X_train, y_train
        self.model = NoisyLogitRegressor(LogisticRegression(C=self.C, max_iter=1000), logit_noise=self.logit_noise)

    def run(self, n_iter):
        self.model.fit(self.X_train, self.y_train)

    def get_result(self):
        return dict(model=self.model)
