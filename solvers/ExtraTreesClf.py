from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.ensemble import ExtraTreesClassifier
    from benchmark_utils.classification_estimators import ProbaRegressor


class Solver(BaseSolver):
    """ExtraTrees classifier; predict() = mean leaf class-1 frequency over trees."""

    name = 'ExtraTreesClf'
    parameters = {
        "n_estimators": [200],
    }
    sampling_strategy = "run_once"

    def set_objective(
        self, X_train, y_train, X_bench, y_bench,
        X_outer, y_outer, categorical_indicator, beta
    ):
        self.X_train, self.y_train = X_train, y_train
        self.model = ProbaRegressor(ExtraTreesClassifier(n_estimators=self.n_estimators))

    def run(self, n_iter):
        self.model.fit(self.X_train, self.y_train)

    def get_result(self):
        return dict(model=self.model)
