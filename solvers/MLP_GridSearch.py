from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.model_selection import GridSearchCV
    from sklearn.neural_network import MLPRegressor


# MLP with in-solver hyperparameter tuning by GRID SEARCH.
# The search runs internal cross-validation on X_train only; the refit
# best_estimator_ is what the objective evaluates on the held-out X_test/bench/outer.
class Solver(BaseSolver):

    name = 'MLP_GridSearch'

    # Only scalar parameters are exposed to benchopt: a dict/list parameter value
    # (like the grid itself) becomes an un-serializable results column and forces the
    # parquet writer to fall back to CSV. The grid is built in set_objective instead.
    parameters = {
        "max_iter": [1000],
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

        # Grid over capacity / L2 regularisation / step size -> 3 * 2 * 2 = 12 points.
        # Its 12 candidates match RandomizedSearchCV's n_iter for an equal number of fits.
        param_grid = {
            "hidden_layer_sizes": [(50,), (100,), (100, 50)],
            "alpha": [1e-4, 1e-2],
            "learning_rate_init": [1e-3, 1e-2],
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
