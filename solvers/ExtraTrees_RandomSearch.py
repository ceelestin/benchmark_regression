from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from scipy.stats import randint, uniform
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.model_selection import RandomizedSearchCV


# ExtraTrees with in-solver hyperparameter tuning by RANDOM SEARCH.
# Same setup as ExtraTrees_GridSearch, but candidates are drawn from finer/continuous
# distributions so random search can reach values the grid cannot. n_iter equals the
# grid size (12), so the two strategies use the SAME number of model fits -- the classic
# Bergstra & Bengio fair comparison, isolating the search strategy itself.
class Solver(BaseSolver):

    name = 'ExtraTrees_RandomSearch'

    # n_iter matches the 12-point grid of ExtraTrees_GridSearch for equal compute.
    # The sampling distributions are built in set_objective (scipy dists are not
    # convenient to express as a benchopt cross-product).
    parameters = {
        "n_estimators": [200],
        "n_iter": [12],
        "cv": [3],
    }

    # Force solver to run only once if you don't want to record training steps
    sampling_strategy = "run_once"

    def set_objective(
        self, X_train, y_train, X_bench, y_bench,
        X_outer, y_outer, categorical_indicator, beta
    ):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        self.X_train, self.y_train = X_train, y_train
        self.X_bench, self.y_bench = X_bench, y_bench
        self.X_outer, self.y_outer = X_outer, y_outer
        self.cat_ind = categorical_indicator
        self.beta = beta

        # Finer than the grid: max_features anywhere in [0.1, 1.0], min_samples_leaf
        # in [1, 20), plus a discrete choice for max_depth.
        param_distributions = {
            "max_features": uniform(0.1, 0.9),
            "min_samples_leaf": randint(1, 20),
            "max_depth": [None, 5, 10, 20, 30],
        }

        # n_jobs=1: benchopt already parallelizes across solver instances and pins
        # BLAS/OMP threads to 1, so internal parallelism would oversubscribe.
        # random_state is left unset on both the estimator (matching the
        # ExtraTrees_noHPO baseline) and the search itself, so every run draws fresh
        # trees AND a fresh set of candidates. The measured variance therefore includes
        # random search's own stochasticity, not just the data's.
        self.search = RandomizedSearchCV(
            ExtraTreesRegressor(n_estimators=self.n_estimators),
            param_distributions=param_distributions,
            n_iter=self.n_iter,
            cv=self.cv,
            n_jobs=1,
        )

    def run(self, n_iter):
        # Fits the random search (internal CV on X_train) and refits best_estimator_.
        self.search.fit(self.X_train, self.y_train)

    def get_result(self):
        # Return the tuned, refit estimator under the `model` key expected by
        # `Objective.evaluate_result`.
        return dict(model=self.search.best_estimator_)
