from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from scipy.stats import loguniform
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.neural_network import MLPRegressor


# MLP with in-solver hyperparameter tuning by RANDOM SEARCH.
# Same setup as MLP_GridSearch, but alpha and learning_rate_init are drawn from
# log-uniform distributions so random search can reach scales the grid cannot.
# n_iter equals the grid size (12), so both strategies use the SAME number of model
# fits -- the classic Bergstra & Bengio fair comparison.
class Solver(BaseSolver):

    name = 'MLP_RandomSearch'

    parameters = {
        "max_iter": [1000],
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

        # alpha and learning_rate_init are scale parameters, so they are sampled
        # log-uniformly over a range wider than the grid's two discrete values.
        param_distributions = {
            "hidden_layer_sizes": [(50,), (100,), (200,), (100, 50), (50, 50)],
            "alpha": loguniform(1e-6, 1e-1),
            "learning_rate_init": loguniform(1e-4, 1e-1),
        }

        # activation/solver fixed as in the MLP baseline; random_state left unset so
        # each fit draws fresh weight initialisations and minibatch ordering.
        base = MLPRegressor(
            activation='relu',
            solver='adam',
            max_iter=self.max_iter,
        )

        # n_jobs=1: benchopt already parallelizes across solver instances and pins
        # BLAS/OMP threads to 1, so internal parallelism would oversubscribe.
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
