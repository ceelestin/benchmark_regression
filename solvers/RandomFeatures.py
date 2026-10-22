from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler


class Solver(BaseSolver):

    name = 'RandomFeatures'

    # Kernel ridge approximated with random Fourier features (Rahimi & Recht):
    # standardise, map to n_components random RBF features, then Ridge.
    # gamma=0.1 and alpha=1.0 are fixed; random_state is left unset so each
    # fit draws fresh random features.
    parameters = {
        "n_components": [100],
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
            StandardScaler(),
            RBFSampler(gamma=0.1, n_components=self.n_components),
            Ridge(alpha=1.0),
        )

    def run(self, n_iter):
        self.model.fit(self.X_train, self.y_train)

    def get_result(self):
        return dict(model=self.model)
