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

    # Per-run bookkeeping that must not leak from one solver run to the next.
    # benchopt <= 1.9.0 reuses a single Objective instance for every solver
    # of a dataset under ``-j 1`` (and calls ``set_dataset`` once per
    # repetition, so it cannot be used as a per-run hook): without a reset the
    # fold counter and the cumulative per-fold stores of solver A continue
    # into solver B, which shifts ``split_index`` and corrupts every
    # cross-fold/redundancy column of B.  ``-j >= 2`` gives each run its own
    # pickled copy and is unaffected.
    _RUN_STATE_ATTRS = (
        "_eval_count", "_cv", "_study_row_index",
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

    def _oof_intersection_icc(self, q_stack, test_mask_stack):
        covs = []
        vars_ = []
        intersection_sizes = []
        n_folds = q_stack.shape[0]
        for k in range(n_folds):
            for ell in range(k + 1, n_folds):
                mask = test_mask_stack[k] & test_mask_stack[ell]
                n_intersection = int(np.sum(mask))
                if n_intersection < 2:
                    continue
                pair_cov = np.cov(q_stack[k, mask], q_stack[ell, mask])
                cov = float(pair_cov[0, 1])
                var = float(0.5 * (pair_cov[0, 0] + pair_cov[1, 1]))
                if not np.isfinite(cov) or not np.isfinite(var):
                    continue
                covs.append(cov)
                vars_.append(var)
                intersection_sizes.append(n_intersection)

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

            self.bench_predictions_by_fold.append(y_pred_bench_current)
            self.bench_errors_by_fold.append(bench_error_current)
            self.bench_squared_errors_by_fold.append(bench_squared_error)
            self.study_predictions_by_fold.append(y_pred_study_current)
            self.study_errors_by_fold.append(study_error_current)
            self.study_squared_errors_by_fold.append(study_squared_error)
            self.study_train_masks_by_fold.append(current_train_mask)
            self.study_test_masks_by_fold.append(current_test_mask)

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
            and hasattr(self, "bench_predictions_by_fold")
            and len(self.bench_predictions_by_fold) >= 2
        ):
            n_splits = int(getattr(self, "n_splits", len(self.bench_predictions_by_fold)))
            (
                bench_prediction_cov_all,
                bench_prediction_var_all,
                bench_prediction_rho_all,
            ) = self._icc_from_stack(np.vstack(self.bench_predictions_by_fold))
            bench_gain_proxy_prediction_all = self._icc_gain_from_rho(
                bench_prediction_rho_all,
                n_splits=n_splits,
            )

            (
                bench_error_cov_all,
                bench_error_var_all,
                bench_error_rho_all,
            ) = self._icc_from_stack(np.vstack(self.bench_errors_by_fold))
            bench_gain_proxy_error_all = self._icc_gain_from_rho(
                bench_error_rho_all,
                n_splits=n_splits,
            )

            (
                bench_squared_error_cov_all,
                bench_squared_error_var_all,
                bench_squared_error_rho_all,
            ) = self._icc_from_stack(np.vstack(self.bench_squared_errors_by_fold))

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

        if (
            self.cv_bool
            and hasattr(self, "study_predictions_by_fold")
            and len(self.study_predictions_by_fold) >= 2
        ):
            q_study_prediction_stack = np.vstack(self.study_predictions_by_fold)
            q_study_error_stack = np.vstack(self.study_errors_by_fold)
            q_study_squared_stack = np.vstack(self.study_squared_errors_by_fold)
            train_mask_stack = np.vstack(self.study_train_masks_by_fold)
            test_mask_stack = np.vstack(self.study_test_masks_by_fold)
            n_splits = int(getattr(self, "n_splits", q_study_squared_stack.shape[0]))

            (
                study_prediction_cov_all_full,
                study_prediction_var_all_full,
                study_prediction_rho_all_full,
            ) = self._icc_from_stack(q_study_prediction_stack)
            study_gain_proxy_prediction_all_full = self._icc_gain_from_rho(
                study_prediction_rho_all_full,
                n_splits=n_splits,
            )
            (
                study_error_cov_all_full,
                study_error_var_all_full,
                study_error_rho_all_full,
            ) = self._icc_from_stack(q_study_error_stack)
            study_gain_proxy_error_all_full = self._icc_gain_from_rho(
                study_error_rho_all_full,
                n_splits=n_splits,
            )
            (
                study_squared_error_cov_all_full,
                study_squared_error_var_all_full,
                study_squared_error_rho_all_full,
            ) = self._icc_from_stack(q_study_squared_stack)
            study_gain_proxy_squared_error_all_full = self._icc_gain_from_rho(
                study_squared_error_rho_all_full,
                n_splits=n_splits,
            )

            q_study_prediction_resid = self._residualize_by_train_membership(
                q_study_prediction_stack,
                train_mask_stack,
            )
            q_study_error_resid = self._residualize_by_train_membership(
                q_study_error_stack,
                train_mask_stack,
            )
            q_study_squared_resid = self._residualize_by_train_membership(
                q_study_squared_stack,
                train_mask_stack,
            )
            (
                study_prediction_cov_all_resid_train_membership,
                study_prediction_var_all_resid_train_membership,
                study_prediction_rho_all_resid_train_membership,
            ) = self._icc_from_stack(q_study_prediction_resid)
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
            ) = self._icc_from_stack(q_study_error_resid)
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
            ) = self._icc_from_stack(q_study_squared_resid)
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

        if self.cv_bool:
            self.X_train, self.X_test, self.y_train, self.y_test = \
                self.get_split(self.X_study, self.y_study)
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
