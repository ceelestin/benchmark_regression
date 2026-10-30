from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from scipy.stats import loguniform
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler


# Random Fourier features with in-solver hyperparameter tuning by RANDOM SEARCH.
# Same setup as RandomFeatures_GridSearch, but gamma and alpha are drawn log-uniformly
# over the grid's range. n_iter equals the grid size (12), so both strategies use the
# same number of model fits, as for MLP_RandomSearch.
class Solver(BaseSolver):

    name = 'RandomFeatures_RandomSearch'

    parameters = {
        "n_components": [50],
        "n_iter": [12],
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

        # gamma and alpha are scale parameters: sampled log-uniformly over the grid's range.
        param_distributions = {
            "rff__gamma": loguniform(0.01, 0.3),
            "ridge__alpha": loguniform(1e-4, 1e-2),
        }

        # random_state left unset so each split draws its own candidates and features.
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("rff", RBFSampler(n_components=self.n_components)),
            ("ridge", Ridge()),
        ])

        self.search = RandomizedSearchCV(
            base,
            param_distributions=param_distributions,
            n_iter=self.n_iter,
            cv=self.cv,
            n_jobs=1,
        )

    def run(self, n_iter):
        self.search.fit(self.X_train, self.y_train)

    def get_result(self):
        return dict(model=self.search.best_estimator_)
