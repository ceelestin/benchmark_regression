from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler


# Random Fourier features (the main-text RandomFeatures, i.e. KernelApprox kind="rff") with
# in-solver hyperparameter tuning by GRID SEARCH, built like MLP_GridSearch.
# n_components is fixed (as n_estimators for ExtraTrees); the RBF width gamma and the
# Ridge penalty alpha are tuned. The search runs internal CV on X_train only.
class Solver(BaseSolver):

    name = 'RandomFeatures_GridSearch'

    parameters = {
        "n_components": [50],
        "cv": [3],
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

        # Grid over gamma / alpha -> 4 * 3 = 12 points, around the fixed learner's
        # (gamma 0.1, alpha 1e-3); the same number of candidates as the other searches.
        param_grid = {
            "rff__gamma": [0.01, 0.03, 0.1, 0.3],
            "ridge__alpha": [1e-4, 1e-3, 1e-2],
        }

        # random_state left unset so each fit draws fresh random features.
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("rff", RBFSampler(n_components=self.n_components)),
            ("ridge", Ridge()),
        ])

        # n_jobs=1: benchopt already parallelizes across solver instances.
        self.search = GridSearchCV(
            base,
            param_grid=param_grid,
            cv=self.cv,
            n_jobs=1,
        )

    def run(self, n_iter):
        self.search.fit(self.X_train, self.y_train)

    def get_result(self):
        return dict(model=self.search.best_estimator_)
