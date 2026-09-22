from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.linear_model import Ridge
    from benchmark_utils.noisy_estimators import NoisyCoefRegressor


class Solver(BaseSolver):
    """Ridge whose fitted coefficients are perturbed once per fit by
    ``coef_noise`` x rms(coef) x N(0, 1): every CV fold gets its own smooth
    bias function of x, mimicking an unstable learner.

    Lever for the redundancy-score calibration: correlates errors across
    samples within a fold and de-correlates them across folds.
    ``coef_noise=0`` is plain Ridge.
    """

    name = 'RidgeNoisyCoef'
    parameters = {
        "alpha": [1.0],
        "coef_noise": [0.0, 0.1, 0.3, 1.0],
    }
    sampling_strategy = "run_once"

    def set_objective(
        self, X_train, y_train, X_bench, y_bench,
        X_outer, y_outer, categorical_indicator, beta
    ):
        self.X_train, self.y_train = X_train, y_train
        self.model = NoisyCoefRegressor(
            Ridge(alpha=self.alpha), coef_noise=self.coef_noise
        )

    def run(self, n_iter):
        self.model.fit(self.X_train, self.y_train)

    def get_result(self):
        return dict(model=self.model)
