from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.linear_model import LogisticRegression
    from benchmark_utils.classification_estimators import NoisyCoefClassifier


class Solver(BaseSolver):
    """Logistic regression with per-fit coefficient perturbation: classification analog of RidgeNoisyCoef. coef_noise=0 is plain LogReg."""

    name = 'LogRegNoisyCoef'
    parameters = {
        "C": [1.0],
        "coef_noise": [0.0, 0.1, 0.3, 1.0],
    }
    sampling_strategy = "run_once"

    def set_objective(
        self, X_train, y_train, X_bench, y_bench,
        X_outer, y_outer, categorical_indicator, beta
    ):
        self.X_train, self.y_train = X_train, y_train
        self.model = NoisyCoefClassifier(LogisticRegression(C=self.C, max_iter=1000), coef_noise=self.coef_noise)

    def run(self, n_iter):
        self.model.fit(self.X_train, self.y_train)

    def get_result(self):
        return dict(model=self.model)
