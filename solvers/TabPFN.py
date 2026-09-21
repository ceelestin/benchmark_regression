from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from tabpfn import TabPFNRegressor


class Solver(BaseSolver):

    name = 'TabPFN'

    requirements = ["pip:tabpfn"]

    # TabPFN is a pre-trained transformer for tabular data. There are very
    # few hyperparameters to tune: the model is fully determined by the
    # checkpoint. We expose `device` so the user can switch between CPU and
    # GPU; everything else uses package defaults.
    parameters = {
        "device": ["cpu"],
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

        self.model = TabPFNRegressor(device=self.device)

    def run(self, n_iter):
        self.model.fit(self.X_train, self.y_train)

    def get_result(self):
        return dict(model=self.model)
