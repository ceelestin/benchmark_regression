from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.linear_model import Ridge
    from benchmark_utils.noisy_estimators import NoisyPredictionRegressor


class Solver(BaseSolver):
    """Ridge whose predictions carry per-sample noise of relative scale
    ``pred_noise`` (x std of the training targets), drawn independently on
    every CV fold but deterministically per sample within a fold.

    Lever for the redundancy-score calibration: increases the fold-driven
    (training-like) component of the OOF error without changing the base
    predictor or its bias.  ``pred_noise=0`` is plain Ridge.
    """

    name = 'RidgeNoisyPred'
    parameters = {
        "alpha": [1.0],
        "pred_noise": [0.0, 0.1, 0.3, 1.0, 3.0],
    }
    sampling_strategy = "run_once"

    def set_objective(
        self, X_train, y_train, X_bench, y_bench,
        X_outer, y_outer, categorical_indicator, beta
    ):
        self.X_train, self.y_train = X_train, y_train
        self.model = NoisyPredictionRegressor(
            Ridge(alpha=self.alpha), pred_noise=self.pred_noise
        )

    def run(self, n_iter):
        self.model.fit(self.X_train, self.y_train)

    def get_result(self):
        return dict(model=self.model)
