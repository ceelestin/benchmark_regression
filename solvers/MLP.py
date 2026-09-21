from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.neural_network import MLPRegressor


class Solver(BaseSolver):

    name = 'MLP'

    # Stochasticity-controlling hyperparameters are fixed:
    #   solver='adam'                : standard mini-batch optimizer (stochastic)
    #   activation='relu'            : standard hidden-layer non-linearity
    #   learning_rate_init=1e-3      : adam default
    #   max_iter=200                 : enough for adam to converge on small N
    # Only hidden_layer_sizes is varied. random_state is left unset so each
    # run draws fresh weight initialisations and minibatch ordering.
    parameters = {
        "hidden_layer_sizes": [(100,)],
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

        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation='relu',
            solver='adam',
            learning_rate_init=1e-3,
            max_iter=1000,
        )

    def run(self, n_iter):
        self.model.fit(self.X_train, self.y_train)

    def get_result(self):
        return dict(model=self.model)
