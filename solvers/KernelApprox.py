from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.kernel_approximation import Nystroem, RBFSampler
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler


class Solver(BaseSolver):

    name = 'KernelApprox'

    # Approximate RBF kernel ridge: standardise, map to n_components random
    # features ("rff": random Fourier features, Rahimi & Recht; "nystroem":
    # Nystroem landmarks drawn from the training set), then Ridge(alpha).
    # random_state is left unset so each fit draws fresh features/landmarks.
    parameters = {
        "kind": ["rff"],
        "n_components": [300],
        "gamma": [0.1],
        "alpha": [1e-3],
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

        feats = RBFSampler if self.kind == "rff" else Nystroem
        self.model = make_pipeline(
            StandardScaler(),
            feats(gamma=self.gamma, n_components=self.n_components),
            Ridge(alpha=self.alpha),
        )

    def run(self, n_iter):
        self.model.fit(self.X_train, self.y_train)

    def get_result(self):
        return dict(model=self.model)
