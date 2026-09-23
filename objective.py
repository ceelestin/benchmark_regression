from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    from sklearn.dummy import DummyRegressor
    from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                 median_absolute_error, r2_score)
    from sklearn.model_selection import (RepeatedKFold, ShuffleSplit,
                                         train_test_split)

    class RepeatedFixedSplit:
        """CV splitter yielding the SAME train/test split for all n_splits folds.

        Used when ``fixed_split=True``: the model is trained and evaluated on one
        identical partition on each of the K folds, so the folds differ only by
        the model's internal training stochasticity (and are identical for
        deterministic models). Contrast with ``ShuffleSplit``, which produces K
        *distinct* partitions.

        With ``random_state=None`` (the default used below) a *single* random
        split is drawn once per objective instance — i.e. once per seed — and
        reused across all K folds. This models the "fixed split" practice
        correctly: each study picks one arbitrary partition and reuses it, and
        the partition *varies from seed to seed*. Pinning ``random_state`` to a
        constant would instead freeze every seed to the identical partition,
        collapsing the Monte Carlo to a single realization.
        """

        def __init__(self, n_splits, test_size, random_state=None):
            self.n_splits = int(n_splits)
            self.test_size = test_size
            self.random_state = random_state

        def split(self, X, y=None, groups=None):
            train, test = next(
                ShuffleSplit(
                    n_splits=1, test_size=self.test_size,
                    random_state=self.random_state,
                ).split(X, y, groups)
            )
            for _ in range(self.n_splits):
                yield train, test

        def get_n_splits(self, X=None, y=None, groups=None):
            return self.n_splits


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):
    # Name to select the objective in the CLI and to display the results.
    name = "Regression"
    url = ""

    is_convex = False

    requirements = ["pip:scikit-learn"]

    # List of parameters for the objective. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    # This means the OLS objective will have a parameter `self.whiten_y`.
    parameters = {
        "n_splits": [5, 200],
        "procedure": ["train_test_split", "RepeatedKFold", "ShuffleSplit"],
        "train_size": [10, 30, 100, 300, 1000, 3000, 10000],
        "test_size": [0.20],
        "fixed_split": [True, False],
        # "study" (default): the usual CV, each fold trains on the study part
        # not in its test fold. "outer": the test fold is still a CV split of
        # the study, but the training set of every fold is a fresh draw of the
        # same size from a pool disjoint from the study (the first half of the
        # outer set; the second half keeps serving the outer chunks). No study
        # sample is then ever a training point for one fold and a test point
        # for another: diagnostic for the train/test coupling across folds.
        "train_source": ["study"],
    }

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.9.0"

    def skip(self, **data):
        if self.procedure == "train_test_split" and self.n_splits != 1:
            return True, "train_test_split does not require n_splits"
        if self.procedure == "RepeatedKFold":
            if self.fixed_split:
                return True, "RepeatedKFold cannot be used with fixed_split"
            folds_per_repeat = int(round(1.0 / self.test_size))
            if self.n_splits % folds_per_repeat != 0:
                return True, (
                    f"RepeatedKFold: n_splits ({self.n_splits}) must be a "
                    f"multiple of 1/test_size ({folds_per_repeat})"
                )
        return False, None

    def set_data(self, X, y, categorical_indicator, beta):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        self.X, self.y = X, y

        if self.procedure == "train_test_split":
            self.cv_bool = False
        elif self.procedure == "RepeatedKFold":
            self.cv_bool = True
            folds_per_repeat = int(round(1.0 / self.test_size))
            self.cv = RepeatedKFold(
                n_splits=folds_per_repeat,
                n_repeats=self.n_splits // folds_per_repeat,
            )
        elif self.procedure == "ShuffleSplit":
            self.cv_bool = True
            if self.fixed_split:
                # One random partition per seed, reused on every one of the K
                # folds (random_state=None => varies across seeds; see
                # RepeatedFixedSplit docstring).
                self.cv = RepeatedFixedSplit(
                    n_splits=self.n_splits, test_size=self.test_size,
                    random_state=None,
                )
            else:
                self.cv = ShuffleSplit(
                    n_splits=self.n_splits, test_size=self.test_size,
                    random_state=None,
                )

        self.categorical_indicator = categorical_indicator
        self.beta = beta
        # Binary target in {0, 1} (e.g. sim_classification): predictions are then
        # read as P(y=1|x); the squared error is the Brier score and the 0-1
        # error / negative log-likelihood are derived as well.
        y_unique = np.unique(np.asarray(y))
        self._is_classification = bool(
            len(y_unique) <= 2 and np.all(np.isin(y_unique, [0.0, 1.0]))
        )

    # Per-run bookkeeping that must not leak from one solver run to the next.
    # benchopt <= 1.9.0 reuses a single Objective instance for every solver
    # of a dataset under ``-j 1`` (and calls ``set_dataset`` once per
    # repetition, so it cannot be used as a per-run hook): without a reset the
    # fold counter and the cumulative per-fold stores of solver A continue
    # into solver B, which shifts ``split_index`` and corrupts every
    # cross-fold/redundancy column of B.  ``-j >= 2`` gives each run its own
    # pickled copy and is unaffected.
    _RUN_STATE_ATTRS = (
        "_eval_count", "_cv", "_study_row_index", "_oof_pair_cache",
        "_stack_icc_cache", "_oof_anova",
        "study_error01_by_fold", "study_nll_by_fold",
        "study_first_train_mask", "study_first_test_mask",
        "study_train_masks_by_fold", "study_test_masks_by_fold",
        "study_predictions_by_fold", "study_errors_by_fold",
        "study_squared_errors_by_fold",
        "bench_predictions_by_fold", "bench_errors_by_fold",
        "bench_squared_errors_by_fold",
        "y_pred_study1", "y_pred_bench1", "y_error_study1", "y_error_bench1",
        "y_var_study1", "y_var_bench1",
        "y_var_squared_study1", "y_var_squared_bench1",
    )

    def _reset_run_state(self):
        for attr in self._RUN_STATE_ATTRS:
            if hasattr(self, attr):
                delattr(self, attr)

    def _study_subset_mask(self, X_subset):
        X_study = np.asarray(self.X_study)
        X_subset = np.asarray(X_subset)
        if X_study.ndim == 1:
            X_study = X_study.reshape(-1, 1)
        if X_subset.ndim == 1:
            X_subset = X_subset.reshape(-1, 1)

        if not hasattr(self, "_study_row_index"):
            self._study_row_index = {
                np.ascontiguousarray(row).tobytes(): idx
                for idx, row in enumerate(X_study)
            }

        mask = np.zeros(len(X_study), dtype=bool)
        for row in X_subset:
            idx = self._study_row_index.get(
                np.ascontiguousarray(row).tobytes()
            )
            if idx is not None:
                mask[idx] = True
        return mask

    def _icc_from_stack(self, q_stack):
        q_stack = np.asarray(q_stack, dtype=float)
        if q_stack.ndim != 2 or q_stack.shape[0] < 2:
            return None, None, None
        cov_matrix = np.cov(q_stack)
        if cov_matrix.ndim != 2:
            return None, None, None
        diag = np.diag(cov_matrix)
        offdiag_mask = ~np.eye(q_stack.shape[0], dtype=bool)
        offdiag = cov_matrix[offdiag_mask]
        finite_offdiag = offdiag[np.isfinite(offdiag)]
        finite_diag = diag[np.isfinite(diag)]
        if finite_offdiag.size == 0 or finite_diag.size == 0:
            return None, None, None
        cov_mean = float(np.mean(finite_offdiag))
        var_mean = float(np.mean(finite_diag))
        rho = cov_mean / var_mean if var_mean > 0 else 0.0
        return cov_mean, var_mean, rho

    def _squared_error_icc_from_stack(self, q_stack):
        return self._icc_from_stack(q_stack)

    def _icc_from_stack_incremental(self, cache_key, new_rows):
        """Same statistic as ``_icc_from_stack(np.vstack(all_rows))`` --
        mean off-diagonal covariance, mean variance (ddof=1), their ratio --
        accumulated one fold at a time.

        ``new_rows`` are the fold rows appended since the previous call for
        this ``cache_key`` (usually one).  Rows never change once appended, so
        the covariances of a new fold with every earlier fold are computed
        once (a matrix-vector product on the stored centered rows) instead of
        recomputing the full k x k covariance matrix at every fold.  This
        turns the O(K^3 n) per-run cost of the ``np.cov`` recomputation into
        O(K^2 n).  Non-finite covariances/variances are dropped from the means
        exactly as in ``_icc_from_stack``.
        """
        if not hasattr(self, "_stack_icc_cache"):
            self._stack_icc_cache = {}
        cache = self._stack_icc_cache.get(cache_key)
        if cache is None:
            cache = {"n_folds": 0, "centered": None, "vars": [], "covs": []}
            self._stack_icc_cache[cache_key] = cache
        for row in new_rows:
            row = np.asarray(row, dtype=float).ravel()
            n = row.size
            centered = row - np.mean(row)
            denom = n - 1 if n > 1 else np.nan
            var = float(centered @ centered / denom)
            k = cache["n_folds"]
            if cache["centered"] is None:
                capacity = max(int(getattr(self, "n_splits", 0) or 0), 8)
                cache["centered"] = np.empty((capacity, n), dtype=float)
            elif k >= cache["centered"].shape[0]:
                grown = np.empty(
                    (2 * cache["centered"].shape[0], n), dtype=float
                )
                grown[:k] = cache["centered"][:k]
                cache["centered"] = grown
            if k > 0:
                covs = cache["centered"][:k] @ centered / denom
                cache["covs"].extend(float(c) for c in covs)
            cache["centered"][k] = centered
            cache["vars"].append(var)
            cache["n_folds"] = k + 1

        if cache["n_folds"] < 2:
            return None, None, None
        covs = np.asarray(cache["covs"], dtype=float)
        vars_ = np.asarray(cache["vars"], dtype=float)
        finite_covs = covs[np.isfinite(covs)]
        finite_vars = vars_[np.isfinite(vars_)]
        if finite_covs.size == 0 or finite_vars.size == 0:
            return None, None, None
        cov_mean = float(np.mean(finite_covs))
        var_mean = float(np.mean(finite_vars))
        rho = cov_mean / var_mean if var_mean > 0 else 0.0
        return cov_mean, var_mean, rho

    def _stack_icc_n_folds(self, cache_key):
        cache = getattr(self, "_stack_icc_cache", {}).get(cache_key)
        return int(cache["n_folds"]) if cache is not None else 0

    def _oof_anova_update(self, quantities, test_mask):
        """Accumulate per-sample running sums of out-of-fold quantities.

        For every study sample the fold-wise OOF values (prediction, error,
        squared error) form an unbalanced one-way layout: sample i is observed
        on the folds where it is in the test part.  Running sums of the
        values and of their squares give, at any fold, the classical
        decomposition of the OOF error into a SAMPLE-driven part (variance
        across samples of the per-sample mean OOF value: noise + shared bias,
        identical in every fold) and a FOLD-driven part (mean across samples
        of the per-sample variance of the OOF value across folds: training-set
        variability of the learner).  O(N_study) per fold.
        """
        if not hasattr(self, "_oof_anova"):
            n_study = len(test_mask)
            self._oof_anova = {
                "count": np.zeros(n_study, dtype=float),
                "sums": {
                    key: np.zeros(n_study, dtype=float) for key in quantities
                },
                "sumsq": {
                    key: np.zeros(n_study, dtype=float) for key in quantities
                },
            }
        acc = self._oof_anova
        mask = np.asarray(test_mask, dtype=bool)
        acc["count"][mask] += 1.0
        for key, values in quantities.items():
            v = np.asarray(values, dtype=float)[mask]
            acc["sums"][key][mask] += v
            acc["sumsq"][key][mask] += v * v

    def _oof_anova_summary(self):
        """Sample- and fold-driven variance components of the OOF quantities.

        Uses the samples observed OOF at least twice.  Returns a dict keyed by
        quantity with ``sample_var`` (variance across samples of the per-sample
        mean), ``fold_var`` (mean across samples of the unbiased per-sample
        variance across folds) and ``icc`` = sample_var / (sample_var +
        fold_var), plus coverage statistics.
        """
        acc = getattr(self, "_oof_anova", None)
        if acc is None:
            return None
        count = acc["count"]
        seen = count >= 2
        out = {
            "n_samples_seen_twice": int(np.sum(seen)),
            "mean_oof_count": float(np.mean(count)) if count.size else None,
        }
        if np.sum(seen) < 2:
            for key in acc["sums"]:
                out[key] = {"sample_var": None, "fold_var": None, "icc": None}
            return out
        c = count[seen]
        for key in acc["sums"]:
            s = acc["sums"][key][seen]
            ss = acc["sumsq"][key][seen]
            means = s / c
            # unbiased per-sample variance across the folds where i is OOF
            within = (ss - c * means ** 2) / (c - 1)
            within = np.clip(within, 0.0, None)
            fold_var = float(np.mean(within))
            # variance across samples of the per-sample mean; the means carry
            # fold noise of order fold_var / c, removed for an unbiased
            # between-sample component (clipped at 0).
            sample_var = float(np.var(means, ddof=1) - np.mean(within / c))
            sample_var = max(sample_var, 0.0)
            total = sample_var + fold_var
            icc = sample_var / total if total > 0 else None
            out[key] = {
                "sample_var": sample_var, "fold_var": fold_var, "icc": icc,
            }
        return out

    def _icc_gain_from_rho(self, rho, n_splits=None):
        if rho is None:
            return None
        if n_splits is None:
            n_splits = int(getattr(self, "n_splits", 0))
        gain_denom = 1 + (n_splits - 1) * rho
        return n_splits / gain_denom if gain_denom != 0 else np.nan

    def _residualize_by_train_membership(self, q_stack, train_mask_stack):
        resid = np.empty_like(q_stack, dtype=float)
        for k, (q, train_mask) in enumerate(zip(q_stack, train_mask_stack)):
            centered = q.astype(float, copy=True)
            for is_train in (False, True):
                mask = train_mask == is_train
                if np.any(mask):
                    centered[mask] -= np.mean(q[mask])
            resid[k] = centered
        return resid

    @staticmethod
    def _oof_pair_stats(q_stack, test_mask_stack, k, ell):
        # Covariance / mean variance of two folds' quantities on the samples
        # that are out-of-fold (in the test part) for BOTH folds.
        mask = test_mask_stack[k] & test_mask_stack[ell]
        n_intersection = int(np.sum(mask))
        if n_intersection < 2:
            return None
        pair_cov = np.cov(q_stack[k, mask], q_stack[ell, mask])
        cov = float(pair_cov[0, 1])
        var = float(0.5 * (pair_cov[0, 0] + pair_cov[1, 1]))
        if not np.isfinite(cov) or not np.isfinite(var):
            return None
        return cov, var, n_intersection

    def _oof_intersection_icc(self, q_stack, test_mask_stack, cache_key=None):
        """Mean pairwise OOF-intersection covariance / variance / ICC.

        The statistic is a plain mean over all fold pairs (k < ell), so it can
        be accumulated incrementally: fold rows never change once appended,
        hence with ``cache_key`` only the pairs involving folds not seen yet
        are computed and the per-pair values are kept on ``self``.  This turns
        the O(K^3) per-run cost of recomputing every pair at every fold into
        O(K^2) while giving the same mean (up to floating-point summation
        order).  Without ``cache_key`` the full recomputation is done.
        """
        n_folds = q_stack.shape[0]
        if cache_key is None:
            covs, vars_, intersection_sizes = [], [], []
            for k in range(n_folds):
                for ell in range(k + 1, n_folds):
                    stats = self._oof_pair_stats(q_stack, test_mask_stack, k, ell)
                    if stats is None:
                        continue
                    covs.append(stats[0])
                    vars_.append(stats[1])
                    intersection_sizes.append(stats[2])
        else:
            if not hasattr(self, "_oof_pair_cache"):
                self._oof_pair_cache = {}
            cache = self._oof_pair_cache.get(cache_key)
            if cache is None or cache["n_folds"] > n_folds:
                cache = {"n_folds": 0, "covs": [], "vars": [], "sizes": []}
                self._oof_pair_cache[cache_key] = cache
            for ell in range(cache["n_folds"], n_folds):
                for k in range(ell):
                    stats = self._oof_pair_stats(q_stack, test_mask_stack, k, ell)
                    if stats is None:
                        continue
                    cache["covs"].append(stats[0])
                    cache["vars"].append(stats[1])
                    cache["sizes"].append(stats[2])
            cache["n_folds"] = n_folds
            covs, vars_, intersection_sizes = (
                cache["covs"], cache["vars"], cache["sizes"]
            )

        if not covs:
            return None, None, None, None, None

        cov_mean = float(np.mean(covs))
        var_mean = float(np.mean(vars_))
        rho = cov_mean / var_mean if var_mean > 0 else 0.0
        return (
            cov_mean,
            var_mean,
            rho,
            int(len(covs)),
            float(np.mean(intersection_sizes)),
        )

    def _oof_intersection_squared_error_icc(self, q_stack, test_mask_stack):
        return self._oof_intersection_icc(q_stack, test_mask_stack)

    def _split_overlap_summary(self, current_mask, reference_mask):
        if reference_mask is None:
            return None, None
        size = int(np.sum(current_mask))
        count = int(np.sum(current_mask & reference_mask))
        frac = float(count / size) if size > 0 else None
        return count, frac

    def _metrics(self, y_true, y_pred):
        # All evaluation metrics, derived from a single set of predictions so that
        # train/test/bench/outer share one implementation.
        #
        # Lower-is-better metrics are stored NEGATED (sklearn's
        # `neg_mean_squared_error` convention) so that "higher is better" holds for
        # every metric column and ranking logic works unchanged whichever is selected.
        # RMSE is recovered downstream as sqrt(-neg_mse).
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        # r2_score on these predictions is numerically identical to model.score().
        metrics = {
            "r2": float(r2_score(y_true, y_pred)),
            "neg_mse": -float(mean_squared_error(y_true, y_pred)),
            "neg_mae": -float(mean_absolute_error(y_true, y_pred)),
            "neg_median_ae": -float(median_absolute_error(y_true, y_pred)),
        }

        # Correlation is undefined for fewer than 3 points or a constant vector
        # (the ranking configs go down to train_size=10, i.e. a 2-3 row test set,
        # and a constant prediction has zero variance). Return NaN rather than
        # letting scipy emit warnings across thousands of runs.
        if (
            len(y_true) < 3
            or np.ptp(y_true) == 0
            or np.ptp(y_pred) == 0
            or not np.all(np.isfinite(y_pred))
        ):
            metrics["spearman"] = np.nan
            metrics["pearson"] = np.nan
        else:
            metrics["spearman"] = float(spearmanr(y_true, y_pred).statistic)
            metrics["pearson"] = float(pearsonr(y_true, y_pred).statistic)

        # Binary classification (predictions = P(y=1|x)): accuracy at threshold
        # 0.5 and the negated mean negative log-likelihood (higher is better).
        if getattr(self, "_is_classification", False):
            p = np.clip(y_pred, 1e-6, 1 - 1e-6)
            metrics["accuracy"] = float(np.mean((p > 0.5) == (y_true > 0.5)))
            metrics["neg_nll"] = float(np.mean(
                y_true * np.log(p) + (1 - y_true) * np.log(1 - p)
            ))
        else:
            metrics["accuracy"] = np.nan
            metrics["neg_nll"] = np.nan

        return metrics

    def evaluate_result(self, model):
        # The arguments of this function are the outputs of the
        # `Solver.get_result`. This defines the benchmark's API to pass
        # solvers' result. This is customizable for each benchmark.
        # Predict once per split and derive every metric from those predictions.
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        y_pred_bench = model.predict(self.X_bench)

        metrics_train = self._metrics(self.y_train, y_pred_train)
        metrics_test = self._metrics(self.y_test, y_pred_test)
        metrics_bench = self._metrics(self.y_bench, y_pred_bench)

        # Preserved names/semantics (R²) so existing analyses keep working.
        score_train = metrics_train["r2"]
        score_test = metrics_test["r2"]
        score_bench = metrics_bench["r2"]

        if not hasattr(self, "_eval_count"):
            self._eval_count = 0
        self._eval_count += 1

        split_index = self._eval_count
        split_train_size = len(self.X_train)
        split_test_size = len(self.X_test)
        split_train_overlap_prev_count = None
        split_test_overlap_prev_count = None
        split_train_overlap_first_count = None
        split_test_overlap_first_count = None
        split_train_overlap_prev_frac = None
        split_test_overlap_prev_frac = None
        split_train_overlap_first_frac = None
        split_test_overlap_first_frac = None

        y_pred_bench_current = None
        y_pred_study_current = None
        if self.cv_bool:
            y_var_bench = np.var(self.y_bench)
            y_var_study = np.var(self.y_study)
            # Reuse the bench predictions computed above (100k rows) instead of
            # re-running inference on the same data.
            y_pred_bench_current = y_pred_bench
            y_pred_study_current = model.predict(self.X_study)
            bench_error_current = y_pred_bench_current - self.y_bench
            study_error_current = y_pred_study_current - self.y_study
            if y_var_bench > 0:
                bench_squared_error = bench_error_current ** 2 / y_var_bench
            else:
                bench_squared_error = np.zeros_like(self.y_bench, dtype=float)
            if y_var_study > 0:
                study_squared_error = study_error_current ** 2 / y_var_study
            else:
                study_squared_error = np.zeros_like(self.y_study, dtype=float)

            # Classification: per-sample 0-1 error and negative log-likelihood
            # (predictions are probabilities of class 1).
            bench_error01 = bench_nll = study_error01 = study_nll = None
            if getattr(self, "_is_classification", False):
                pb = np.clip(np.asarray(y_pred_bench_current, dtype=float), 1e-6, 1 - 1e-6)
                ps = np.clip(np.asarray(y_pred_study_current, dtype=float), 1e-6, 1 - 1e-6)
                bench_error01 = ((pb > 0.5) != (self.y_bench > 0.5)).astype(float)
                study_error01 = ((ps > 0.5) != (self.y_study > 0.5)).astype(float)
                bench_nll = -(self.y_bench * np.log(pb) + (1 - self.y_bench) * np.log(1 - pb))
                study_nll = -(self.y_study * np.log(ps) + (1 - self.y_study) * np.log(1 - ps))

            current_train_mask = self._study_subset_mask(self.X_train)
            current_test_mask = self._study_subset_mask(self.X_test)
            split_train_size = int(np.sum(current_train_mask))
            split_test_size = int(np.sum(current_test_mask))

            if self._eval_count == 1:
                self.bench_predictions_by_fold = []
                self.bench_errors_by_fold = []
                self.bench_squared_errors_by_fold = []
                self.study_predictions_by_fold = []
                self.study_errors_by_fold = []
                self.study_squared_errors_by_fold = []
                self.study_train_masks_by_fold = []
                self.study_test_masks_by_fold = []
                self.study_first_train_mask = current_train_mask.copy()
                self.study_first_test_mask = current_test_mask.copy()

            previous_train_mask = (
                self.study_train_masks_by_fold[-1]
                if self.study_train_masks_by_fold else None
            )
            previous_test_mask = (
                self.study_test_masks_by_fold[-1]
                if self.study_test_masks_by_fold else None
            )
            (
                split_train_overlap_prev_count,
                split_train_overlap_prev_frac,
            ) = self._split_overlap_summary(current_train_mask, previous_train_mask)
            (
                split_test_overlap_prev_count,
                split_test_overlap_prev_frac,
            ) = self._split_overlap_summary(current_test_mask, previous_test_mask)
            (
                split_train_overlap_first_count,
                split_train_overlap_first_frac,
            ) = self._split_overlap_summary(
                current_train_mask,
                self.study_first_train_mask,
            )
            (
                split_test_overlap_first_count,
                split_test_overlap_first_frac,
            ) = self._split_overlap_summary(
                current_test_mask,
                self.study_first_test_mask,
            )

            # Bench rows (100k samples) are folded straight into the
            # incremental ICC caches: no per-fold copy of the bench stack is
            # kept and no k x k covariance matrix is recomputed per fold.
            self._icc_from_stack_incremental(
                "bench_prediction", [y_pred_bench_current]
            )
            self._icc_from_stack_incremental(
                "bench_error", [bench_error_current]
            )
            self._icc_from_stack_incremental(
                "bench_squared_error", [bench_squared_error]
            )
            # Study rows are kept (small: N_study samples) for the pairwise
            # OOF-intersection statistics, and also fed to the incremental
            # caches for the "full" and "resid_train_membership" ICCs.
            self.study_predictions_by_fold.append(y_pred_study_current)
            self.study_errors_by_fold.append(study_error_current)
            self.study_squared_errors_by_fold.append(study_squared_error)
            self.study_train_masks_by_fold.append(current_train_mask)
            self.study_test_masks_by_fold.append(current_test_mask)
            for key, row in (
                ("prediction", y_pred_study_current),
                ("error", study_error_current),
                ("squared_error", study_squared_error),
            ):
                self._icc_from_stack_incremental(f"study_{key}_full", [row])
                resid_row = self._residualize_by_train_membership(
                    np.asarray(row, dtype=float)[None, :],
                    current_train_mask[None, :],
                )[0]
                self._icc_from_stack_incremental(
                    f"study_{key}_resid_train_membership", [resid_row]
                )
            anova_quantities = {
                "prediction": y_pred_study_current,
                "error": study_error_current,
                "squared_error": study_squared_error,
            }
            if study_error01 is not None:
                self._icc_from_stack_incremental("bench_error01", [bench_error01])
                self._icc_from_stack_incremental("bench_nll", [bench_nll])
                if self._eval_count == 1:
                    self.study_error01_by_fold = []
                    self.study_nll_by_fold = []
                self.study_error01_by_fold.append(study_error01)
                self.study_nll_by_fold.append(study_nll)
                anova_quantities["error01"] = study_error01
                anova_quantities["nll"] = study_nll
            self._oof_anova_update(anova_quantities, current_test_mask)

        if self._eval_count == 1:
            self.y_pred_bench1 = y_pred_bench
            self.y_error_bench1 = self.y_pred_bench1 - self.y_bench
            self.y_var_bench1 = np.var(self.y_error_bench1)
            self.y_var_squared_bench1 = np.var(self.y_error_bench1**2)

            self.y_pred_study1 = (
                y_pred_study_current
                if y_pred_study_current is not None
                else model.predict(self.X_study)
            )
            self.y_error_study1 = self.y_pred_study1 - self.y_study
            self.y_var_study1 = np.var(self.y_error_study1)
            self.y_var_squared_study1 = np.var(self.y_error_study1**2)

        # Compute outer scores on the first split only (covers both no-CV and CV cases).
        # `outer_scores` keeps its original meaning (per-chunk R²) so existing
        # analyses are unaffected. Two families of outer output:
        #  * PER-CHUNK lists (`outer_scores`, `outer_neg_mse`, `outer_neg_mae`) --
        #    one value per chunk. These are per-sample means, so the analysis may
        #    combine them with the usual (base + k*mean_k)/(k+1) average.
        #  * CUMULATIVE POOLED lists (`*_cum`) -- element k is the metric recomputed
        #    on the POOLED sample (test split + chunks 0..k). Median AE, Spearman and
        #    Pearson are NOT per-sample means, so averaging chunk-level values does
        #    not estimate the pooled metric; these must be used directly instead of
        #    being fed through the combination formula. `outer_r2_cum` is included
        #    because equal-weight averaging of per-chunk R² is only an approximation.
        outer_scores = []
        outer_neg_mse = []
        outer_neg_mae = []
        outer_accuracy = []
        outer_neg_nll = []
        outer_r2_cum = []
        outer_neg_median_ae_cum = []
        outer_spearman_cum = []
        outer_pearson_cum = []
        if self._eval_count == 1 and getattr(self, "X_outer", None) is not None:
            n_outer = len(self.X_outer)
            test_len = len(self.X_test)
            if test_len > 0:
                n_chunks = min(n_outer // test_len, 200)
                # Pooled buffers seeded with the test split, grown one chunk at a
                # time; preallocated so no repeated concatenation is needed.
                pooled_true = np.empty(test_len * (n_chunks + 1), dtype=float)
                pooled_pred = np.empty(test_len * (n_chunks + 1), dtype=float)
                pooled_true[:test_len] = self.y_test
                pooled_pred[:test_len] = y_pred_test
                filled = test_len
                for i in range(n_chunks):
                    start = i * test_len
                    end = (i + 1) * test_len
                    y_true_chunk = self.y_outer[start:end]
                    y_pred_chunk = model.predict(self.X_outer[start:end])

                    chunk_metrics = self._metrics(y_true_chunk, y_pred_chunk)
                    outer_scores.append(chunk_metrics["r2"])
                    outer_neg_mse.append(chunk_metrics["neg_mse"])
                    outer_neg_mae.append(chunk_metrics["neg_mae"])
                    outer_accuracy.append(chunk_metrics["accuracy"])
                    outer_neg_nll.append(chunk_metrics["neg_nll"])

                    pooled_true[filled:filled + test_len] = y_true_chunk
                    pooled_pred[filled:filled + test_len] = y_pred_chunk
                    filled += test_len
                    pooled_metrics = self._metrics(
                        pooled_true[:filled], pooled_pred[:filled]
                    )
                    outer_r2_cum.append(pooled_metrics["r2"])
                    outer_neg_median_ae_cum.append(
                        pooled_metrics["neg_median_ae"]
                    )
                    outer_spearman_cum.append(pooled_metrics["spearman"])
                    outer_pearson_cum.append(pooled_metrics["pearson"])

        # Compute bench and study covariance exactly once: between split 1 and split 2.
        bench_covariance = None
        bench_error_cov = None
        bench_squared_error_cov = None
        bench_error_corr = None
        bench_squared_error_corr = None
        bench_error_var = None
        bench_squared_error_var = None
        bench_error_rho = None
        bench_squared_error_rho = None

        study_covariance = None
        study_error_cov = None
        study_squared_error_cov = None
        study_error_corr = None
        study_squared_error_corr = None
        study_error_var = None
        study_squared_error_var = None
        study_error_rho = None
        study_squared_error_rho = None

        bench_prediction_cov_all = None
        bench_prediction_var_all = None
        bench_prediction_rho_all = None
        bench_gain_proxy_prediction_all = None
        bench_error_cov_all = None
        bench_error_var_all = None
        bench_error_rho_all = None
        bench_gain_proxy_error_all = None
        bench_squared_error_cov_all = None
        bench_squared_error_var_all = None
        bench_squared_error_rho_all = None
        bench_rho_delta_proxy_squared_error_all = None
        bench_gain_proxy_squared_error_all = None

        study_prediction_cov_all_full = None
        study_prediction_var_all_full = None
        study_prediction_rho_all_full = None
        study_gain_proxy_prediction_all_full = None
        study_error_cov_all_full = None
        study_error_var_all_full = None
        study_error_rho_all_full = None
        study_gain_proxy_error_all_full = None
        study_squared_error_cov_all_full = None
        study_squared_error_var_all_full = None
        study_squared_error_rho_all_full = None
        study_gain_proxy_squared_error_all_full = None
        study_prediction_cov_all_resid_train_membership = None
        study_prediction_var_all_resid_train_membership = None
        study_prediction_rho_all_resid_train_membership = None
        study_gain_proxy_prediction_all_resid_train_membership = None
        study_error_cov_all_resid_train_membership = None
        study_error_var_all_resid_train_membership = None
        study_error_rho_all_resid_train_membership = None
        study_gain_proxy_error_all_resid_train_membership = None
        study_squared_error_cov_all_resid_train_membership = None
        study_squared_error_var_all_resid_train_membership = None
        study_squared_error_rho_all_resid_train_membership = None
        study_gain_proxy_squared_error_all_resid_train_membership = None
        study_prediction_cov_all_oof_intersection = None
        study_prediction_var_all_oof_intersection = None
        study_prediction_rho_all_oof_intersection = None
        study_gain_proxy_prediction_all_oof_intersection = None
        study_error_cov_all_oof_intersection = None
        study_error_var_all_oof_intersection = None
        study_error_rho_all_oof_intersection = None
        study_gain_proxy_error_all_oof_intersection = None
        study_squared_error_cov_all_oof_intersection = None
        study_squared_error_var_all_oof_intersection = None
        study_squared_error_rho_all_oof_intersection = None
        study_gain_proxy_squared_error_all_oof_intersection = None
        study_oof_intersection_pairs_all = None
        study_oof_intersection_mean_size_all = None

        if self.cv_bool and self._eval_count == 2:
            y_error_bench = y_pred_bench - self.y_bench
            bench_covariance = np.cov(self.y_pred_bench1, y_pred_bench)[0, 1]
            bench_error_cov = np.cov(self.y_error_bench1, y_error_bench)[0, 1]
            bench_squared_error_cov = np.cov(
                self.y_error_bench1**2,
                y_error_bench**2,
            )[0, 1]
            bench_error_corr = np.corrcoef(
                self.y_error_bench1,
                y_error_bench,
            )[0, 1]
            bench_squared_error_corr = np.corrcoef(
                self.y_error_bench1**2,
                y_error_bench**2,
            )[0, 1]
            bench_error_var = 0.5 * (
                self.y_var_bench1 + np.var(y_error_bench)
            )
            bench_squared_error_var = 0.5 * (
                self.y_var_squared_bench1 + np.var(y_error_bench**2)
            )
            bench_error_rho = bench_error_cov / bench_error_var if bench_error_var > 0 else 0
            bench_squared_error_rho = (
                bench_squared_error_cov / bench_squared_error_var
                if bench_squared_error_var > 0 else 0
            )

            y_pred_study = (
                y_pred_study_current
                if y_pred_study_current is not None
                else model.predict(self.X_study)
            )
            y_error_study = y_pred_study - self.y_study
            study_covariance = np.cov(self.y_pred_study1, y_pred_study)[0, 1]
            study_error_cov = np.cov(self.y_error_study1, y_error_study)[0, 1]
            study_squared_error_cov = np.cov(
                self.y_error_study1**2,
                y_error_study**2,
            )[0, 1]
            study_error_corr = np.corrcoef(
                self.y_error_study1,
                y_error_study,
            )[0, 1]
            study_squared_error_corr = np.corrcoef(
                self.y_error_study1**2,
                y_error_study**2,
            )[0, 1]
            study_error_var = 0.5 * (
                self.y_var_study1 + np.var(y_error_study)
            )
            study_squared_error_var = 0.5 * (
                self.y_var_squared_study1 + np.var(y_error_study**2)
            )
            study_error_rho = study_error_cov / study_error_var if study_error_var > 0 else 0
            study_squared_error_rho = (
                study_squared_error_cov / study_squared_error_var
                if study_squared_error_var > 0 else 0
            )

        if (
            self.cv_bool
            and self._stack_icc_n_folds("bench_prediction") >= 2
        ):
            n_splits = int(getattr(
                self, "n_splits", self._stack_icc_n_folds("bench_prediction")
            ))
            (
                bench_prediction_cov_all,
                bench_prediction_var_all,
                bench_prediction_rho_all,
            ) = self._icc_from_stack_incremental("bench_prediction", [])
            bench_gain_proxy_prediction_all = self._icc_gain_from_rho(
                bench_prediction_rho_all,
                n_splits=n_splits,
            )

            (
                bench_error_cov_all,
                bench_error_var_all,
                bench_error_rho_all,
            ) = self._icc_from_stack_incremental("bench_error", [])
            bench_gain_proxy_error_all = self._icc_gain_from_rho(
                bench_error_rho_all,
                n_splits=n_splits,
            )

            (
                bench_squared_error_cov_all,
                bench_squared_error_var_all,
                bench_squared_error_rho_all,
            ) = self._icc_from_stack_incremental("bench_squared_error", [])

            n_bench = len(self.X_bench)
            n_test = len(self.X_test)
            denom_size = n_bench + n_test
            if denom_size > 0 and bench_squared_error_rho_all is not None:
                bench_rho_delta_proxy_squared_error_all = (
                    bench_squared_error_rho_all * n_test / denom_size
                )
            else:
                bench_rho_delta_proxy_squared_error_all = 0.0
            bench_gain_proxy_squared_error_all = self._icc_gain_from_rho(
                bench_rho_delta_proxy_squared_error_all,
                n_splits=n_splits,
            )

        # Classification-only cross-fold statistics (None for regression).
        bench_error01_cov_all = bench_error01_var_all = bench_error01_rho_all = None
        bench_nll_cov_all = bench_nll_var_all = bench_nll_rho_all = None
        study_error01_cov_all_oof_intersection = None
        study_error01_var_all_oof_intersection = None
        study_error01_rho_all_oof_intersection = None
        study_nll_cov_all_oof_intersection = None
        study_nll_var_all_oof_intersection = None
        study_nll_rho_all_oof_intersection = None
        if self.cv_bool and self._stack_icc_n_folds("bench_error01") >= 2:
            (
                bench_error01_cov_all,
                bench_error01_var_all,
                bench_error01_rho_all,
            ) = self._icc_from_stack_incremental("bench_error01", [])
            (
                bench_nll_cov_all,
                bench_nll_var_all,
                bench_nll_rho_all,
            ) = self._icc_from_stack_incremental("bench_nll", [])

        if (
            self.cv_bool
            and hasattr(self, "study_predictions_by_fold")
            and len(self.study_predictions_by_fold) >= 2
        ):
            q_study_prediction_stack = np.vstack(self.study_predictions_by_fold)
            q_study_error_stack = np.vstack(self.study_errors_by_fold)
            q_study_squared_stack = np.vstack(self.study_squared_errors_by_fold)
            test_mask_stack = np.vstack(self.study_test_masks_by_fold)
            n_splits = int(getattr(self, "n_splits", q_study_squared_stack.shape[0]))

            (
                study_prediction_cov_all_full,
                study_prediction_var_all_full,
                study_prediction_rho_all_full,
            ) = self._icc_from_stack_incremental("study_prediction_full", [])
            study_gain_proxy_prediction_all_full = self._icc_gain_from_rho(
                study_prediction_rho_all_full,
                n_splits=n_splits,
            )
            (
                study_error_cov_all_full,
                study_error_var_all_full,
                study_error_rho_all_full,
            ) = self._icc_from_stack_incremental("study_error_full", [])
            study_gain_proxy_error_all_full = self._icc_gain_from_rho(
                study_error_rho_all_full,
                n_splits=n_splits,
            )
            (
                study_squared_error_cov_all_full,
                study_squared_error_var_all_full,
                study_squared_error_rho_all_full,
            ) = self._icc_from_stack_incremental("study_squared_error_full", [])
            study_gain_proxy_squared_error_all_full = self._icc_gain_from_rho(
                study_squared_error_rho_all_full,
                n_splits=n_splits,
            )

            # Residualized-by-train-membership ICCs: rows were residualized
            # and cached when appended (a row's residualization depends only
            # on its own fold's train mask).
            (
                study_prediction_cov_all_resid_train_membership,
                study_prediction_var_all_resid_train_membership,
                study_prediction_rho_all_resid_train_membership,
            ) = self._icc_from_stack_incremental(
                "study_prediction_resid_train_membership", []
            )
            study_gain_proxy_prediction_all_resid_train_membership = (
                self._icc_gain_from_rho(
                    study_prediction_rho_all_resid_train_membership,
                    n_splits=n_splits,
                )
            )
            (
                study_error_cov_all_resid_train_membership,
                study_error_var_all_resid_train_membership,
                study_error_rho_all_resid_train_membership,
            ) = self._icc_from_stack_incremental(
                "study_error_resid_train_membership", []
            )
            study_gain_proxy_error_all_resid_train_membership = (
                self._icc_gain_from_rho(
                    study_error_rho_all_resid_train_membership,
                    n_splits=n_splits,
                )
            )
            (
                study_squared_error_cov_all_resid_train_membership,
                study_squared_error_var_all_resid_train_membership,
                study_squared_error_rho_all_resid_train_membership,
            ) = self._icc_from_stack_incremental(
                "study_squared_error_resid_train_membership", []
            )
            study_gain_proxy_squared_error_all_resid_train_membership = (
                self._icc_gain_from_rho(
                    study_squared_error_rho_all_resid_train_membership,
                    n_splits=n_splits,
                )
            )

            (
                study_prediction_cov_all_oof_intersection,
                study_prediction_var_all_oof_intersection,
                study_prediction_rho_all_oof_intersection,
                pred_oof_pairs,
                pred_oof_mean_size,
            ) = self._oof_intersection_icc(
                q_study_prediction_stack,
                test_mask_stack,
                cache_key="prediction",
            )
            study_gain_proxy_prediction_all_oof_intersection = (
                self._icc_gain_from_rho(
                    study_prediction_rho_all_oof_intersection,
                    n_splits=n_splits,
                )
            )
            (
                study_error_cov_all_oof_intersection,
                study_error_var_all_oof_intersection,
                study_error_rho_all_oof_intersection,
                error_oof_pairs,
                error_oof_mean_size,
            ) = self._oof_intersection_icc(
                q_study_error_stack,
                test_mask_stack,
                cache_key="error",
            )
            study_gain_proxy_error_all_oof_intersection = (
                self._icc_gain_from_rho(
                    study_error_rho_all_oof_intersection,
                    n_splits=n_splits,
                )
            )
            (
                study_squared_error_cov_all_oof_intersection,
                study_squared_error_var_all_oof_intersection,
                study_squared_error_rho_all_oof_intersection,
                squared_oof_pairs,
                squared_oof_mean_size,
            ) = self._oof_intersection_icc(
                q_study_squared_stack,
                test_mask_stack,
                cache_key="squared_error",
            )
            study_gain_proxy_squared_error_all_oof_intersection = (
                self._icc_gain_from_rho(
                    study_squared_error_rho_all_oof_intersection,
                    n_splits=n_splits,
                )
            )
            study_oof_intersection_pairs_all = (
                squared_oof_pairs or error_oof_pairs or pred_oof_pairs
            )
            study_oof_intersection_mean_size_all = (
                squared_oof_mean_size or error_oof_mean_size or pred_oof_mean_size
            )
            if len(getattr(self, "study_error01_by_fold", [])) >= 2:
                (
                    study_error01_cov_all_oof_intersection,
                    study_error01_var_all_oof_intersection,
                    study_error01_rho_all_oof_intersection,
                    _, _,
                ) = self._oof_intersection_icc(
                    np.vstack(self.study_error01_by_fold),
                    test_mask_stack,
                    cache_key="error01",
                )
                (
                    study_nll_cov_all_oof_intersection,
                    study_nll_var_all_oof_intersection,
                    study_nll_rho_all_oof_intersection,
                    _, _,
                ) = self._oof_intersection_icc(
                    np.vstack(self.study_nll_by_fold),
                    test_mask_stack,
                    cache_key="nll",
                )

        # Per-sample OOF variance decomposition (sample-driven vs fold-driven
        # components of the out-of-fold prediction / error / squared error),
        # cumulative over the folds seen so far.
        oof_anova = {
            key: {"sample_var": None, "fold_var": None, "icc": None}
            for key in ("prediction", "error", "squared_error", "error01", "nll")
        }
        oof_anova_n_seen_twice = None
        oof_anova_mean_count = None
        if self.cv_bool and self._eval_count >= 2:
            summary = self._oof_anova_summary()
            if summary is not None:
                oof_anova_n_seen_twice = summary["n_samples_seen_twice"]
                oof_anova_mean_count = summary["mean_oof_count"]
                for key in oof_anova:
                    if key in summary:
                        oof_anova[key] = summary[key]

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        # Variance of the study-set target: lets the study-only redundancy score
        # (prediction covariance x error correlation, in y^2 units) be rescaled to
        # a dimensionless value post hoc. Constant within a run.
        study_target_var = (
            float(np.var(self.y_study))
            if getattr(self, "y_study", None) is not None and len(self.y_study) > 1
            else None
        )
        return dict(
            score_test=score_test,
            score_train=score_train,
            score_bench=score_bench,
            # Alternative evaluation metrics. Lower-is-better ones are negated
            # (sklearn's neg_* convention) so higher is better for every column;
            # RMSE is recovered downstream as sqrt(-neg_mse).
            neg_mse_train=metrics_train["neg_mse"],
            neg_mse_test=metrics_test["neg_mse"],
            neg_mse_bench=metrics_bench["neg_mse"],
            neg_mae_train=metrics_train["neg_mae"],
            neg_mae_test=metrics_test["neg_mae"],
            neg_mae_bench=metrics_bench["neg_mae"],
            # Classification only (NaN for regression): accuracy at 0.5 and
            # negated mean negative log-likelihood.
            accuracy_train=metrics_train["accuracy"],
            accuracy_test=metrics_test["accuracy"],
            accuracy_bench=metrics_bench["accuracy"],
            neg_nll_train=metrics_train["neg_nll"],
            neg_nll_test=metrics_test["neg_nll"],
            neg_nll_bench=metrics_bench["neg_nll"],
            neg_median_ae_train=metrics_train["neg_median_ae"],
            neg_median_ae_test=metrics_test["neg_median_ae"],
            neg_median_ae_bench=metrics_bench["neg_median_ae"],
            spearman_train=metrics_train["spearman"],
            spearman_test=metrics_test["spearman"],
            spearman_bench=metrics_bench["spearman"],
            pearson_train=metrics_train["pearson"],
            pearson_test=metrics_test["pearson"],
            pearson_bench=metrics_bench["pearson"],
            # Per-chunk (safe to combine with (base + k*mean_k)/(k+1)).
            outer_scores=outer_scores,
            outer_neg_mse=outer_neg_mse,
            outer_neg_mae=outer_neg_mae,
            outer_accuracy=outer_accuracy,
            outer_neg_nll=outer_neg_nll,
            # Cumulative pooled: element k already IS the metric on test+chunks 0..k,
            # so use these directly -- do NOT feed them through the combination.
            outer_r2_cum=outer_r2_cum,
            outer_neg_median_ae_cum=outer_neg_median_ae_cum,
            outer_spearman_cum=outer_spearman_cum,
            outer_pearson_cum=outer_pearson_cum,
            split_index=split_index,
            split_train_size=split_train_size,
            split_test_size=split_test_size,
            split_train_overlap_prev_frac=split_train_overlap_prev_frac,
            split_test_overlap_prev_frac=split_test_overlap_prev_frac,
            split_train_overlap_first_frac=split_train_overlap_first_frac,
            split_test_overlap_first_frac=split_test_overlap_first_frac,
            split_train_overlap_prev_count=split_train_overlap_prev_count,
            split_test_overlap_prev_count=split_test_overlap_prev_count,
            split_train_overlap_first_count=split_train_overlap_first_count,
            split_test_overlap_first_count=split_test_overlap_first_count,
            bench_covariance=bench_covariance,
            bench_error_cov=bench_error_cov,
            bench_squared_error_cov=bench_squared_error_cov,
            bench_error_corr=bench_error_corr,
            bench_squared_error_corr=bench_squared_error_corr,
            bench_error_var=bench_error_var,
            bench_squared_error_var=bench_squared_error_var,
            bench_error_rho=bench_error_rho,
            bench_squared_error_rho=bench_squared_error_rho,
            bench_prediction_cov_all=bench_prediction_cov_all,
            bench_prediction_var_all=bench_prediction_var_all,
            bench_prediction_rho_all=bench_prediction_rho_all,
            bench_gain_proxy_prediction_all=bench_gain_proxy_prediction_all,
            bench_error_cov_all=bench_error_cov_all,
            bench_error_var_all=bench_error_var_all,
            bench_error_rho_all=bench_error_rho_all,
            bench_gain_proxy_error_all=bench_gain_proxy_error_all,
            bench_squared_error_cov_all=bench_squared_error_cov_all,
            bench_squared_error_var_all=bench_squared_error_var_all,
            bench_squared_error_rho_all=bench_squared_error_rho_all,
            bench_rho_delta_proxy_squared_error_all=(
                bench_rho_delta_proxy_squared_error_all
            ),
            bench_gain_proxy_squared_error_all=bench_gain_proxy_squared_error_all,
            study_prediction_cov_all_full=study_prediction_cov_all_full,
            study_prediction_var_all_full=study_prediction_var_all_full,
            study_prediction_rho_all_full=study_prediction_rho_all_full,
            study_gain_proxy_prediction_all_full=(
                study_gain_proxy_prediction_all_full
            ),
            study_error_cov_all_full=study_error_cov_all_full,
            study_error_var_all_full=study_error_var_all_full,
            study_error_rho_all_full=study_error_rho_all_full,
            study_gain_proxy_error_all_full=study_gain_proxy_error_all_full,
            study_squared_error_cov_all_full=study_squared_error_cov_all_full,
            study_squared_error_var_all_full=study_squared_error_var_all_full,
            study_squared_error_rho_all_full=study_squared_error_rho_all_full,
            study_gain_proxy_squared_error_all_full=(
                study_gain_proxy_squared_error_all_full
            ),
            study_prediction_cov_all_resid_train_membership=(
                study_prediction_cov_all_resid_train_membership
            ),
            study_prediction_var_all_resid_train_membership=(
                study_prediction_var_all_resid_train_membership
            ),
            study_prediction_rho_all_resid_train_membership=(
                study_prediction_rho_all_resid_train_membership
            ),
            study_gain_proxy_prediction_all_resid_train_membership=(
                study_gain_proxy_prediction_all_resid_train_membership
            ),
            study_error_cov_all_resid_train_membership=(
                study_error_cov_all_resid_train_membership
            ),
            study_error_var_all_resid_train_membership=(
                study_error_var_all_resid_train_membership
            ),
            study_error_rho_all_resid_train_membership=(
                study_error_rho_all_resid_train_membership
            ),
            study_gain_proxy_error_all_resid_train_membership=(
                study_gain_proxy_error_all_resid_train_membership
            ),
            study_squared_error_cov_all_resid_train_membership=(
                study_squared_error_cov_all_resid_train_membership
            ),
            study_squared_error_var_all_resid_train_membership=(
                study_squared_error_var_all_resid_train_membership
            ),
            study_squared_error_rho_all_resid_train_membership=(
                study_squared_error_rho_all_resid_train_membership
            ),
            study_gain_proxy_squared_error_all_resid_train_membership=(
                study_gain_proxy_squared_error_all_resid_train_membership
            ),
            study_prediction_cov_all_oof_intersection=(
                study_prediction_cov_all_oof_intersection
            ),
            study_prediction_var_all_oof_intersection=(
                study_prediction_var_all_oof_intersection
            ),
            study_prediction_rho_all_oof_intersection=(
                study_prediction_rho_all_oof_intersection
            ),
            study_gain_proxy_prediction_all_oof_intersection=(
                study_gain_proxy_prediction_all_oof_intersection
            ),
            study_error_cov_all_oof_intersection=(
                study_error_cov_all_oof_intersection
            ),
            study_error_var_all_oof_intersection=(
                study_error_var_all_oof_intersection
            ),
            study_error_rho_all_oof_intersection=(
                study_error_rho_all_oof_intersection
            ),
            study_gain_proxy_error_all_oof_intersection=(
                study_gain_proxy_error_all_oof_intersection
            ),
            study_squared_error_cov_all_oof_intersection=(
                study_squared_error_cov_all_oof_intersection
            ),
            study_squared_error_var_all_oof_intersection=(
                study_squared_error_var_all_oof_intersection
            ),
            study_squared_error_rho_all_oof_intersection=(
                study_squared_error_rho_all_oof_intersection
            ),
            study_gain_proxy_squared_error_all_oof_intersection=(
                study_gain_proxy_squared_error_all_oof_intersection
            ),
            study_oof_intersection_pairs_all=study_oof_intersection_pairs_all,
            study_oof_intersection_mean_size_all=(
                study_oof_intersection_mean_size_all
            ),
            # Sample-driven vs fold-driven variance components of the OOF
            # quantities (cumulative over folds); icc = sample/(sample+fold).
            study_oof_anova_prediction_sample_var=(
                oof_anova["prediction"]["sample_var"]
            ),
            study_oof_anova_prediction_fold_var=(
                oof_anova["prediction"]["fold_var"]
            ),
            study_oof_anova_prediction_icc=oof_anova["prediction"]["icc"],
            study_oof_anova_error_sample_var=oof_anova["error"]["sample_var"],
            study_oof_anova_error_fold_var=oof_anova["error"]["fold_var"],
            study_oof_anova_error_icc=oof_anova["error"]["icc"],
            study_oof_anova_squared_error_sample_var=(
                oof_anova["squared_error"]["sample_var"]
            ),
            study_oof_anova_squared_error_fold_var=(
                oof_anova["squared_error"]["fold_var"]
            ),
            study_oof_anova_squared_error_icc=(
                oof_anova["squared_error"]["icc"]
            ),
            study_oof_anova_n_samples_seen_twice=oof_anova_n_seen_twice,
            study_oof_anova_mean_oof_count=oof_anova_mean_count,
            # Classification only (None for regression).
            bench_error01_cov_all=bench_error01_cov_all,
            bench_error01_var_all=bench_error01_var_all,
            bench_error01_rho_all=bench_error01_rho_all,
            bench_nll_cov_all=bench_nll_cov_all,
            bench_nll_var_all=bench_nll_var_all,
            bench_nll_rho_all=bench_nll_rho_all,
            study_error01_cov_all_oof_intersection=(
                study_error01_cov_all_oof_intersection
            ),
            study_error01_var_all_oof_intersection=(
                study_error01_var_all_oof_intersection
            ),
            study_error01_rho_all_oof_intersection=(
                study_error01_rho_all_oof_intersection
            ),
            study_nll_cov_all_oof_intersection=study_nll_cov_all_oof_intersection,
            study_nll_var_all_oof_intersection=study_nll_var_all_oof_intersection,
            study_nll_rho_all_oof_intersection=study_nll_rho_all_oof_intersection,
            study_oof_anova_error01_sample_var=oof_anova["error01"]["sample_var"],
            study_oof_anova_error01_fold_var=oof_anova["error01"]["fold_var"],
            study_oof_anova_error01_icc=oof_anova["error01"]["icc"],
            study_oof_anova_nll_sample_var=oof_anova["nll"]["sample_var"],
            study_oof_anova_nll_fold_var=oof_anova["nll"]["fold_var"],
            study_oof_anova_nll_icc=oof_anova["nll"]["icc"],
            study_covariance=study_covariance,
            study_target_var=study_target_var,
            study_error_cov=study_error_cov,
            study_squared_error_cov=study_squared_error_cov,
            study_error_corr=study_error_corr,
            study_squared_error_corr=study_squared_error_corr,
            study_error_var=study_error_var,
            study_squared_error_var=study_squared_error_var,
            study_error_rho=study_error_rho,
            study_squared_error_rho=study_squared_error_rho,
            value=1 - score_test,
            hash_bench=self.X_bench.sum()
        )

    def get_one_result(self):
        # Return one solution. The return value should be an object compatible
        # with `self.compute`. This is mainly for testing purposes.
        return dict(model=DummyRegressor().fit(self.X_train, self.y_train))

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.
        # A run is exactly ``n_splits`` repetitions (benchopt derives
        # ``n_repetitions`` from ``cv.get_n_splits()``): once the previous run
        # has consumed them all, the next call starts a new solver run.
        if self.cv_bool and getattr(self, "_eval_count", 0) >= self.n_splits:
            self._reset_run_state()
        self.X_study, self.X_bench, self.y_study, self.y_bench = \
            train_test_split(
                self.X, self.y, test_size=100000, random_state=0
            )

        n_study = self.train_size + round(self.train_size * self.test_size
                                          / (1 - self.test_size))
        self.X_outer, self.X_study, self.y_outer, self.y_study = \
            train_test_split(
                self.X_study, self.y_study, test_size=n_study, random_state=0
            )
        train_source = getattr(self, "train_source", "study")
        if train_source == "outer":
            half = len(self.X_outer) // 2
            self.X_train_pool, self.y_train_pool = (
                self.X_outer[:half], self.y_outer[:half]
            )
            self.X_outer, self.y_outer = self.X_outer[half:], self.y_outer[half:]

        if self.cv_bool:
            self.X_train, self.X_test, self.y_train, self.y_test = \
                self.get_split(self.X_study, self.y_study)
            if train_source == "outer":
                # Same training size as the study-based fold, drawn fresh from
                # the disjoint pool on every fold (independent of the study).
                rng = np.random.default_rng()
                idx = rng.choice(
                    len(self.X_train_pool), size=len(self.X_train), replace=False
                )
                self.X_train, self.y_train = (
                    self.X_train_pool[idx], self.y_train_pool[idx]
                )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(
                    self.X_study, self.y_study, train_size=self.train_size,
                    random_state=0 if self.fixed_split else None
                )

        return dict(
            X_train=self.X_train,
            y_train=self.y_train,
            X_bench=self.X_bench,
            y_bench=self.y_bench,
            X_outer=self.X_outer,
            y_outer=self.y_outer,
            categorical_indicator=self.categorical_indicator,
            beta=self.beta
        )
