from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import optuna
    from sklearn.base import clone
    from sklearn.dummy import DummyRegressor


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class OSolver(BaseSolver):
    stopping_criterion = SufficientProgressCriterion(
        strategy="callback", patience=5
        )

    params = {
        "test_size": 0.25,
        "seed": 42,
    }

    extra_model_params = {}

    def set_objective(
            self, X_train, y_train, X_val, y_val, categorical_indicator, seed
            ):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.cat_ind = categorical_indicator

        self.model = self.get_model()  # Includes preprocessor

    def objective(self, trial):
        param = self.sample_parameters(trial)
        params = self.extra_model_params.copy()
        params.update({f"model__{p}": v for p, v in param.items()})
        model = clone(self.model).set_params(**params)
        res = model.fit(self.X_train, self.y_train)
        trial.set_user_attr("model", res)
        return res.score(self.X_val, self.y_val)

    def run(self, callback):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        sampler = optuna.samplers.RandomSampler()
        self.best_model = DummyRegressor().fit(self.X_train, self.y_train)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        while callback():
            study.optimize(self.objective, n_trials=10)
            self.best_model = study.best_trial.user_attrs["model"]

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return dict(model=self.best_model)

    def get_next(self, n_iter):
        return n_iter + 1

    def warmup_solver(self):
        pass
