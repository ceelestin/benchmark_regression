from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.tree import DecisionTreeClassifier
    from benchmark_utils.classification_estimators import ProbaRegressor


class Solver(BaseSolver):
    """Unpruned decision tree classifier; predict() = leaf class-1 frequency (0/1 for pure leaves)."""

    name = 'DecisionTreeClf'
    parameters = {
        "max_depth": [100],
        "min_samples_leaf": [1],
    }
    sampling_strategy = "run_once"

    def set_objective(
        self, X_train, y_train, X_bench, y_bench,
        X_outer, y_outer, categorical_indicator, beta
    ):
        self.X_train, self.y_train = X_train, y_train
        self.model = ProbaRegressor(DecisionTreeClassifier(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf))

    def run(self, n_iter):
        self.model.fit(self.X_train, self.y_train)

    def get_result(self):
        return dict(model=self.model)
