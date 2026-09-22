from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.linear_model import LogisticRegression
    from benchmark_utils.classification_estimators import ProbaRegressor


class Solver(BaseSolver):
    """Logistic regression; predict() returns P(y=1|x) (classification analog of Ridge)."""

    name = 'LogReg'
    parameters = {
        "C": [1.0],
    }
    sampling_strategy = "run_once"

    def set_objective(
        self, X_train, y_train, X_bench, y_bench,
        X_outer, y_outer, categorical_indicator, beta
    ):
        self.X_train, self.y_train = X_train, y_train
        self.model = ProbaRegressor(LogisticRegression(C=self.C, max_iter=1000))

    def run(self, n_iter):
        self.model.fit(self.X_train, self.y_train)

    def get_result(self):
        return dict(model=self.model)
