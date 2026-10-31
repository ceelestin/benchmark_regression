"""
compute.py — Core numerical computations for the Validation Crisis experiments.

Provides:
  - to_scalar               helper: extract scalar from length-1 array
  - match_variances         variance-matching kernel (absolute or ratio method)
  - expand_hidden_scores    expand list column to one-row-per-k DataFrame
  - build_nocv_combined_df  merge expanded hidden scores with base study scores
  - build_nocv_cache        pre-process no-CV data into numpy arrays (simulated path)
  - sample_gain_bootstrap   main bootstrap loop (DataFrame mode or Cache mode)
  - agg_ci                  group-wise mean ± 95% CI
  - compute_tau             inter-fold covariance estimator (ANOVA decomposition)
  - sample_gain_at_splits   mean sample gain at one or more split counts
  - prepare_pareto_data     attach time metadata to gains DataFrame (simulated only)
  - compute_retrieval       pairwise ranking retrieval curves and summary table
"""

from __future__ import annotations

import itertools
import warnings
from typing import Literal

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def to_scalar(x):
    """Return x[0] if x is a non-empty list/array, otherwise return x as-is.

    Handles 0-D numpy arrays (``np.array(5)``) without calling ``len()``
    on them, which would raise ``TypeError: len() of unsized object``.
    """
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return x.item()
        return x[0] if x.size > 0 else x
    if isinstance(x, list) and len(x) > 0:
        return x[0]
    return x


# ---------------------------------------------------------------------------
# match_variances
# ---------------------------------------------------------------------------

class _IdentityResampler:
    """Stands in for the random generator to "resample" the original sample itself (full-sample estimate)."""

    def choice(self, a, size=None, replace=True):
        a = np.arange(a) if np.isscalar(a) else np.asarray(a)
        return a if size is None else a[:size]


def _crossing_gain(
    multipliers: np.ndarray,
    nocv_variances: np.ndarray,
    match_idx: int | None,
    target_var: float,
    interpolate: bool,
) -> float:
    """Test-size multiple at which the single-split variance curve reaches *target_var*.

    *multipliers* are the test-size multiples (1 = the test fold alone) of the
    entries of *nocv_variances*; *match_idx* is ``match_variances``' first index
    with ``nocv_variances <= target_var`` (None = never reached: capped at the
    largest multiple). Without interpolation the gain is that multiple, i.e. the
    crossing rounded up to a whole number of test sets. With interpolation it is
    read between the last multiple above the target and the first one below,
    linearly in log(multiple) against log(variance) (variance ~ 1/size).
    """
    if match_idx is None:
        return float(multipliers[-1])
    a1 = float(multipliers[match_idx])
    if not interpolate or match_idx == 0:
        return a1
    a0 = float(multipliers[match_idx - 1])
    v0 = float(nocv_variances[match_idx - 1])
    v1 = float(nocv_variances[match_idx])
    if not (np.isfinite(v0) and np.isfinite(v1) and v1 > 0 and target_var > 0
            and v0 > target_var >= v1 and a1 > a0 > 0):
        return a1
    frac = (np.log(v0) - np.log(target_var)) / (np.log(v0) - np.log(v1))
    return float(np.exp(np.log(a0) + frac * (np.log(a1) - np.log(a0))))


def match_variances(
    nocv_variances: np.ndarray,
    target_cv_var: float,
    method: Literal["absolute", "ratio"],
    baseline_nocv_var: float | None = None,
    baseline_cv_var: float | None = None,
) -> int | None:
    """Find the smallest index k at which the no-CV variance matches the target CV variance.

    Parameters
    ----------
    nocv_variances : 1-D numpy array
        Variance of the no-CV estimator as a function of k (index 0 = k=0 or k=1
        depending on caller convention — see notes below).
    target_cv_var : float
        Variance of the K-split CV estimator for a given (model, size, K).
    method : {'absolute', 'ratio'}
        'absolute' — find first k where ``nocv_variances[k] <= target_cv_var``.
        'ratio'    — normalise both sides by their k=0 / K=1 baselines first,
                     then apply the same criterion.
    baseline_nocv_var : float, optional
        Variance of the no-CV estimator at k=0 (only used when method='ratio').
    baseline_cv_var : float, optional
        Variance of the CV estimator at K=1 (only used when method='ratio').

    Returns
    -------
    int or None
        Index of the first match, or None if no match exists within the array.

    Notes
    -----
    The caller is responsible for the k-indexing convention and for capping the
    return value when None is returned (e.g. use max_k as the cap).
    """
    if pd.isna(target_cv_var):
        return None

    if method == "absolute":
        # Relative tolerance: with a paired bootstrap, the single-split variance
        # at the test-fold size and the K = 1 CV variance are the same number
        # computed by two code paths; floating-point noise must not break the tie.
        matches = nocv_variances <= target_cv_var * (1 + 1e-9)
        if np.any(matches):
            return int(np.argmax(matches))
        return None

    # ratio method
    if baseline_nocv_var is None or baseline_cv_var is None:
        raise ValueError(
            "baseline_nocv_var and baseline_cv_var must be provided when method='ratio'"
        )
    if baseline_nocv_var <= 0 or baseline_cv_var <= 0:
        raise ValueError(
            "Baselines must be strictly positive for method='ratio'; "
            f"got baseline_nocv_var={baseline_nocv_var}, baseline_cv_var={baseline_cv_var}"
        )

    nocv_ratios = nocv_variances / baseline_nocv_var
    cv_ratio = target_cv_var / baseline_cv_var
    matches = nocv_ratios <= cv_ratio
    if np.any(matches):
        return int(np.argmax(matches))
    return None


# ---------------------------------------------------------------------------
# sigma2_nocv_within_seed
# ---------------------------------------------------------------------------

def sigma2_nocv_within_seed(
    df_nocv: pd.DataFrame,
    scores_col: str,
    group_cols: list[str],
) -> pd.DataFrame:
    """σ²_noCV estimator robust to seed-level training instability.

    For each seed, compute the variance of the hidden-test-split scores
    (conditional on the trained model), then average these per-seed
    variances across seeds within each group. Excludes the seed-level
    variance component that captures training failures.
    """
    def _within(scores):
        if not isinstance(scores, (list, np.ndarray)):
            return np.nan
        arr = np.asarray(scores, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size < 2:
            return np.nan
        return float(np.var(arr, ddof=1))

    work = df_nocv.copy()
    work['_v'] = work[scores_col].apply(_within)
    out = (
        work.groupby(group_cols, dropna=False)['_v']
        .mean()
        .reset_index()
        .rename(columns={'_v': 'sigma2_nocv_within'})
    )
    return out


# ---------------------------------------------------------------------------
# expand_hidden_scores
# ---------------------------------------------------------------------------

def expand_hidden_scores(
    df_nocv: pd.DataFrame,
    scores_col: str = "hidden_test_accuracies",
    id_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Expand a list column into one row per k with expanding cumulative mean.

    For each row in *df_nocv* where *scores_col* contains a non-empty list or
    array, emits one output row per index k (1-indexed) with
    ``mean_score_k = mean(scores_col[:k])``.

    Parameters
    ----------
    df_nocv : DataFrame
        Each row has a list/array in *scores_col*.
    scores_col : str
        Name of the column containing hidden scores.
    id_cols : list of str, optional
        Columns to carry through. Defaults to every column except *scores_col*.

    Returns
    -------
    DataFrame with columns ``id_cols + ['k', 'mean_score_k']``.
    """
    if id_cols is None:
        id_cols = [c for c in df_nocv.columns if c != scores_col]

    empty = pd.DataFrame(columns=id_cols + ["k", "mean_score_k"])

    # Keep only rows with non-empty list/array scores
    valid_mask = df_nocv[scores_col].apply(
        lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0
    )
    work = df_nocv.loc[valid_mask, id_cols + [scores_col]].copy()
    if work.empty:
        return empty

    # Convert to plain lists of float for explode
    work[scores_col] = work[scores_col].apply(
        lambda x: list(np.asarray(x, dtype=float))
    )
    # Tag each original row so we can groupby after explode
    work["_row_id"] = np.arange(len(work))

    exploded = work.explode(scores_col, ignore_index=True)
    exploded[scores_col] = exploded[scores_col].astype(float)

    # k = 1-indexed fold position within each original row
    grp = exploded.groupby("_row_id", sort=False)
    exploded["k"] = grp.cumcount() + 1
    exploded["mean_score_k"] = grp[scores_col].cumsum() / exploded["k"]

    return exploded[id_cols + ["k", "mean_score_k"]]


# ---------------------------------------------------------------------------
# build_nocv_combined_df
# ---------------------------------------------------------------------------

def build_nocv_combined_df(
    df_nocv_k: pd.DataFrame,
    df_nocv_base: pd.DataFrame,
    model_col: str = "model",
    score_raw_col: str = "objective_score_test",
    bench_col: str = "objective_score_bench",
    include_k0: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge expanded hidden scores with base study scores.

    Produces two DataFrames ready for ``sample_gain_bootstrap`` in DataFrame mode:

    * ``df_acc``   — combined accuracy score at each k
    * ``df_delta`` — accuracy minus benchmark at each k

    Parameters
    ----------
    df_nocv_k : DataFrame
        Output of ``expand_hidden_scores``.
        Must have columns ``['seed', model_col, 'study_size', 'k', 'mean_score_k']``.
    df_nocv_base : DataFrame
        Single-split no-CV rows (n_splits == 1) with columns
        ``['seed', model_col, 'study_size', score_raw_col, bench_col]``.
    model_col : str
        Column identifying the model.
    score_raw_col : str
        Column in *df_nocv_base* holding the base test accuracy (may be length-1 array).
    bench_col : str
        Column in *df_nocv_base* holding the benchmark accuracy.
    include_k0 : bool
        If True, prepend k=0 rows (base score only) required for ratio-method baselines.

    Returns
    -------
    (df_acc, df_delta) : tuple of DataFrames
    """
    merge_keys = ["seed", model_col, "study_size"]

    base = df_nocv_base[merge_keys + [score_raw_col, bench_col]].copy()
    base["test_accuracy"] = base[score_raw_col].apply(to_scalar)
    base["benchmark_val"] = base[bench_col].apply(to_scalar)
    base = base.drop(columns=[score_raw_col, bench_col])

    merged = df_nocv_k.merge(base, on=merge_keys, how="left")
    merged["combined_score_acc"] = (
        merged["mean_score_k"] * merged["k"] + merged["test_accuracy"]
    ) / (merged["k"] + 1)

    if include_k0:
        k0 = base.copy()
        k0["k"] = 0
        k0["mean_score_k"] = np.nan
        k0["combined_score_acc"] = k0["test_accuracy"]
        merged = pd.concat([k0, merged], ignore_index=True)
        merged = merged.sort_values(["seed", model_col, "study_size", "k"]).reset_index(
            drop=True
        )

    merged["delta_score_combined"] = merged["combined_score_acc"] - merged["benchmark_val"]

    df_acc = merged.drop(columns=["delta_score_combined"])
    df_delta = merged.copy()
    return df_acc, df_delta


# ---------------------------------------------------------------------------
# build_nocv_cache  (simulated path only)
# ---------------------------------------------------------------------------

def build_nocv_cache(
    df_nocv: pd.DataFrame,
    model_col: str = "solver_name",
    size_col: str = "train_size",
    dataset_col: str | None = "dataset_name",
    hidden_col: str = "objective_outer_scores",
    test_col: str = "objective_score_test",
    bench_col: str = "objective_score_bench",
    use_delta: bool = True,
    extra_key_cols: list[str] | None = None,
    combination: str = "average",
    seed_col: str = "seed",
) -> dict:
    """Pre-process no-CV data into numpy arrays for fast bootstrap sampling.

    When *seed_col* is present, each entry also stores the seed of every row
    (``"seeds"``) so that ``sample_gain_bootstrap(paired=True)`` can resample the
    single-split side with the same seeds as the CV side.

    Used by the simulated-data path where hidden scores are a list column.

    Parameters
    ----------
    df_nocv : DataFrame
        One row per (seed, model, size, dataset).
    model_col, size_col, dataset_col : str
        Column names. Set *dataset_col* to None if no dataset dimension.
    hidden_col : str
        Column containing a list of outer/hidden test scores.
    test_col, bench_col : str
        Columns for the base test score and benchmark score.
    use_delta : bool
        If True, subtract benchmark from hidden and test scores.
    extra_key_cols : list of str, optional
        Additional columns to include in the cache key tuple after the
        standard (dataset, model, size) components. E.g. ``['fixed_split']``
        produces keys of the form ``(dataset, model, size, fixed_split_value)``.
    combination : {'average', 'pooled'}
        How the no-CV growth curve is reconstructed from *hidden_col*, which
        ``_cache_mode_iter`` honours via the stored ``'combination'`` tag:

        * ``'average'`` (default, backward-compatible) — the hidden list holds
          **per-chunk** metric values; element k is one chunk's score. The
          curve is the running mean of test + chunks 0..k (valid for metrics
          that are means over instances, e.g. R², MSE, MAE). Stored as
          ``'cum_means'``.
        * ``'pooled'`` — the hidden list is **already cumulative**: element k is
          the metric pooled on test + chunks 0..k (needed for non-averageable
          metrics like Spearman/Pearson/median-AE). Stored verbatim as
          ``'cum_values'``; the k=0 point is the test-only score.

    Returns
    -------
    dict keyed by (dataset_str, model_str, size_int[, extra...]) or
    (model_str, size_int[, extra...]), each value being::

        {
            # 'average' mode:
            'cum_means':   ndarray of shape (N_seeds, max_k),
            # 'pooled' mode:
            'cum_values':  ndarray of shape (N_seeds, max_k),
            'test_scores': ndarray of shape (N_seeds,),
            'max_k':       int,
            'combination': str,
        }
    """
    if combination not in ("average", "pooled"):
        raise ValueError(
            f"combination must be 'average' or 'pooled', got {combination!r}"
        )
    cache: dict = {}
    group_cols = [model_col, size_col]
    if dataset_col is not None:
        group_cols = [dataset_col] + group_cols
    if extra_key_cols:
        group_cols = group_cols + extra_key_cols

    groups = df_nocv.groupby(group_cols, dropna=False)

    for key, subset in groups:
        # Ensure key is always a tuple
        if not isinstance(key, tuple):
            key = (key,)

        hidden_lists = subset[hidden_col].values
        bench_scores = subset[bench_col].values
        test_scores_raw = subset[test_col].values

        # Vectorized validity mask (avoids per-element Python loop)
        has_list = np.array([
            isinstance(L, (list, np.ndarray)) and len(L) > 0
            for L in hidden_lists
        ])
        bench_ok = ~pd.isna(bench_scores)
        t_scalars = np.array([to_scalar(t) for t in test_scores_raw],
                             dtype=float)
        test_ok = ~np.isnan(t_scalars)
        valid = has_list & bench_ok & test_ok

        if not valid.any():
            continue

        lists = hidden_lists[valid]
        test_arr = t_scalars[valid]
        bench_vals = bench_scores[valid].astype(float)

        max_len = max(len(L) for L in lists)
        n_seeds = len(lists)

        hidden_matrix = np.full((n_seeds, max_len), np.nan)
        for i, L in enumerate(lists):
            arr = np.asarray(L, dtype=float)
            hidden_matrix[i, : len(arr)] = arr

        bench_arr = bench_vals.reshape(-1, 1)

        if use_delta:
            hidden_matrix = hidden_matrix - bench_arr
            test_arr = test_arr - bench_vals

        lengths = np.array([len(L) for L in lists], dtype=int)
        position_mask = np.arange(max_len).reshape(1, -1) < lengths.reshape(-1, 1)

        if combination == "pooled":
            # The hidden list is already cumulative (element k = metric pooled on
            # test + chunks 0..k); use it verbatim. Pad positions stay NaN.
            cum_values = hidden_matrix.copy()
            cum_values[~position_mask] = np.nan
            cache[key] = {
                "cum_values": cum_values,
                "test_scores": test_arr,
                "max_k": max_len,
                "combination": "pooled",
            }
            if seed_col in subset.columns:
                cache[key]["seeds"] = subset[seed_col].to_numpy()[valid]
        else:
            # Vectorised cumulative means with ragged-list support.
            # Positions beyond the true hidden-score length for a seed remain NaN
            # so later variance computations do not treat padded values as real
            # observations.
            valid_mask = ~np.isnan(hidden_matrix)
            cumsum = np.cumsum(np.where(valid_mask, hidden_matrix, 0.0), axis=1)
            valid_counts = np.cumsum(valid_mask, axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                cum_means = cumsum / valid_counts
            cum_means[~position_mask] = np.nan
            cache[key] = {
                "cum_means": cum_means,
                "test_scores": test_arr,
                "max_k": max_len,
                "combination": "average",
            }
            if seed_col in subset.columns:
                cache[key]["seeds"] = subset[seed_col].to_numpy()[valid]

    return cache


# ---------------------------------------------------------------------------
# Small helpers for bootstrap key handling
# ---------------------------------------------------------------------------

def _index_key_to_dict(key, index_names: list[str]) -> dict[str, object] | None:
    """Map a pandas (possibly scalar) groupby key to its named index levels."""
    if not isinstance(key, tuple):
        key = (key,)
    if len(key) != len(index_names):
        return None
    return dict(zip(index_names, key))


def _config_signature(
    key_map: dict[str, object],
    index_names: list[str],
    exclude: set[str] | None = None,
) -> tuple[tuple[str, object], ...]:
    """Stable identifier for filtered/skipped configurations."""
    if exclude is None:
        exclude = set()
    return tuple(
        (name, key_map[name])
        for name in index_names
        if name not in exclude and name in key_map
    )


def _resolve_split_col(df: pd.DataFrame, split_col: str) -> str:
    """Return the preferred split-count column, with legacy fallback."""
    if split_col in df.columns:
        return split_col
    if split_col == "K" and "s" in df.columns:
        return "s"
    raise KeyError(f"Split-count column {split_col!r} not found.")


# ---------------------------------------------------------------------------
# sample_gain_bootstrap  — the central function
# ---------------------------------------------------------------------------

def sample_gain_bootstrap(
    data_cv: pd.DataFrame,
    score_col_cv: str,
    model_col: str,
    n_bootstraps: int = 100,
    method: Literal["absolute", "ratio"] = "absolute",
    test_size: float = 0.2,
    # --- DataFrame mode (PCam / NLP) ---
    df_nocv_combined: pd.DataFrame | None = None,
    score_col_nocv: str | None = None,
    # --- Cache mode (Simulated) ---
    nocv_cache: dict | None = None,
    # --- Filters ---
    models_to_analyze: list[str] | None = None,
    study_sizes: list[int] | None = None,
    size_col: str = "study_size",
    random_state: int | None = None,
    # --- Dataset dimension (Cache mode only) ---
    dataset_col: str | None = None,
    # --- Extra groupby columns for cv_var_series (e.g. fixed_split) ---
    extra_cv_cols: list[str] | None = None,
    # --- train_size derivation ---
    compute_train_size: bool = True,
    # --- estimator options ---
    paired: bool = True,
    interpolate: bool = True,
    plugin_estimate: bool = False,
) -> pd.DataFrame:
    """Bootstrapped variance-matching analysis returning Variance-Equivalent Test Sample Gain.

    *paired* (default True): each bootstrap draw resamples seeds once and uses
    the same seeds for the CV variance and for the single-split variance curve.
    The single-split baseline is the first split of the same runs, so both
    variances are computed on the same experiments, as in the definition; in
    particular the gain at K = 1 is exactly 1. Falls back to independent draws
    (with a warning) when the single-split seeds are not a subset of the CV seeds.
    With ``paired=False`` the two sides are resampled independently (the
    behaviour before 2026-10-27).

    *plugin_estimate* (default False): also return the estimate on the original seeds (no resampling) as the
    rows with ``bootstrap_iter == -1``. It is computed before the bootstrap draws and does not use the random
    generator, so the draws are identical with or without it (CE, 2026-10-31: the figures' thick line is this
    estimate; the draws give the thin lines and the percentile intervals).
    *interpolate* (default True): the crossing is read between the two
    bracketing whole multiples of the test-set size (log-log interpolation, see
    ``_crossing_gain``) instead of being rounded up to the next whole multiple
    (the behaviour before 2026-10-27, ``interpolate=False``). ``k_equiv`` keeps
    the integer index of the first crossing (used by the censoring rules).

    Two mutually exclusive operating modes:

    **DataFrame mode** (*df_nocv_combined* + *score_col_nocv* provided) — PCam / NLP.
      Resamples seeds; computes var_cv and var_nocv via groupby.var();
      calls ``match_variances`` per (model, size, K).

    **Cache mode** (*nocv_cache* provided) — Simulated data.
      Resamples CV seeds; for each cache entry resamples no-CV indices
      from numpy arrays; computes ``nocv_variances`` vectorised over k.

    Parameters
    ----------
    data_cv : DataFrame
        CV data with columns ``['seed', model_col, 'study_size', 'n_splits', score_col_cv]``.
        May optionally have a ``'procedure'`` column.
    score_col_cv : str
        Column in *data_cv* to compute variance over (must already be scalar).
    model_col : str
        Model/solver column name.
    n_bootstraps : int
        Bootstrap iterations.
    method : {'absolute', 'ratio'}
        Variance-matching method.
    test_size : float
        Fraction of study set used as test; used to compute sample_gain.
    df_nocv_combined : DataFrame, optional
        (DataFrame mode) Combined score DataFrame with
        ``['seed', model_col, 'study_size', 'k', score_col_nocv]``.
    score_col_nocv : str, optional
        (DataFrame mode) Column to compute variance over.
    nocv_cache : dict, optional
        (Cache mode) Output of ``build_nocv_cache``.
    models_to_analyze : list of str, optional
        Restrict to these models. If None, uses all in data_cv.
    study_sizes : list of int, optional
        Restrict to these study sizes. If None, uses all in data_cv.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    DataFrame with columns:
        ``[model_col, 'N', 'K', 'k_equiv', 'sample_gain', 'train_size',
           'bootstrap_iter']``
        plus optional grouping columns propagated from the CV table
        (``dataset_col`` / ``extra_cv_cols`` in both modes, plus
        ``'fixed_split'`` in Cache mode when the cache keys include one) and
        ``'cv_procedure'`` if ``'procedure'`` is in *data_cv*.

    Notes
    -----
    Sample gain convention: ``sample_gain = k_equiv + 1`` in both modes when
    ``interpolate=False``; with interpolation it lies in ``(k_equiv, k_equiv + 1]``.
    """
    # --- Validate mode ---
    df_mode = df_nocv_combined is not None
    cache_mode = nocv_cache is not None
    if df_mode == cache_mode:
        raise ValueError(
            "Provide exactly one of df_nocv_combined (DataFrame mode) "
            "or nocv_cache (Cache mode)."
        )
    if df_mode and score_col_nocv is None:
        raise ValueError("score_col_nocv must be provided in DataFrame mode.")

    rng = np.random.default_rng(random_state)
    has_procedure = "procedure" in data_cv.columns
    result_extra_cols: list[str] = []
    nocv_match_cols = [model_col, size_col]

    if df_mode:
        if dataset_col is not None and dataset_col in data_cv.columns:
            if dataset_col not in df_nocv_combined.columns:
                raise ValueError(
                    f"dataset_col={dataset_col!r} must also be present in "
                    "df_nocv_combined in DataFrame mode."
                )
            nocv_match_cols.append(dataset_col)
            result_extra_cols.append(dataset_col)

        if extra_cv_cols:
            missing = [
                col for col in extra_cv_cols
                if col in data_cv.columns and col not in df_nocv_combined.columns
            ]
            if missing:
                raise ValueError(
                    "extra_cv_cols must also be present in df_nocv_combined "
                    f"in DataFrame mode; missing: {missing}"
                )
            for col in extra_cv_cols:
                if col in data_cv.columns and col not in nocv_match_cols:
                    nocv_match_cols.append(col)
                if col in data_cv.columns and col != "procedure" and col not in result_extra_cols:
                    result_extra_cols.append(col)
    else:
        if dataset_col is not None:
            result_extra_cols.append(dataset_col)
        if extra_cv_cols:
            for col in extra_cv_cols:
                if col in data_cv.columns and col != "procedure" and col not in result_extra_cols:
                    result_extra_cols.append(col)

    if models_to_analyze is None:
        models_to_analyze = list(data_cv[model_col].unique())
    if study_sizes is not None:
        data_cv = data_cv[data_cv[size_col].isin(study_sizes)].copy()
        if df_mode:
            df_nocv_combined = df_nocv_combined[
                df_nocv_combined[size_col].isin(study_sizes)
            ].copy()

    unique_seeds_cv = data_cv["seed"].unique()
    if df_mode:
        unique_seeds_nocv = df_nocv_combined["seed"].unique()
    pair_df = False
    if paired:
        cv_seed_set = set(unique_seeds_cv.tolist())
        if df_mode:
            pair_df = set(unique_seeds_nocv.tolist()) <= cv_seed_set
            if not pair_df:
                warnings.warn(
                    "sample_gain_bootstrap: single-split seeds are not a subset of the "
                    "CV seeds; resampling the two sides independently.",
                    stacklevel=2,
                )
        else:
            for _cache in nocv_cache.values():
                seeds = _cache.get("seeds")
                _cache["_seed_index"] = None
                if seeds is None:
                    continue
                idx = pd.Index(seeds)
                if idx.is_unique and set(idx.tolist()) <= cv_seed_set:
                    _cache["_seed_index"] = idx

    all_results: list[dict] = []
    filtered_configs: dict = {}

    for i_boot in ([-1] if plugin_estimate else []) + list(range(n_bootstraps)):
        r = _IdentityResampler() if i_boot < 0 else rng
        # --- Resample ---
        boot_seeds_cv = r.choice(unique_seeds_cv, size=len(unique_seeds_cv), replace=True)
        boot_df_cv = (
            pd.DataFrame({"seed": boot_seeds_cv})
            .merge(data_cv, on="seed", how="left")
        )

        if df_mode:
            boot_seeds_nocv = (
                boot_seeds_cv if pair_df else
                r.choice(unique_seeds_nocv, size=len(unique_seeds_nocv), replace=True)
            )
            boot_df_nocv = (
                pd.DataFrame({"seed": boot_seeds_nocv})
                .merge(df_nocv_combined, on="seed", how="left")
            )

        # --- CV variances ---
        cv_group_cols = []
        if dataset_col is not None and dataset_col in boot_df_cv.columns:
            cv_group_cols.append(dataset_col)
        cv_group_cols += [model_col, size_col]
        if has_procedure:
            cv_group_cols.append("procedure")
        if extra_cv_cols:
            for col in extra_cv_cols:
                if col in boot_df_cv.columns:
                    cv_group_cols.append(col)
        cv_group_cols.append("n_splits")

        cv_var_series = (
            boot_df_cv.groupby(cv_group_cols)[score_col_cv]
            .var()
        )

        # --- Per-mode variance computation and matching ---
        if df_mode:
            _df_mode_iter(
                i_boot, cv_var_series, boot_df_nocv, score_col_nocv,
                model_col, size_col, models_to_analyze, has_procedure,
                method, all_results, filtered_configs,
                nocv_match_cols, result_extra_cols,
                interpolate=interpolate,
            )
        else:
            _cache_mode_iter(
                i_boot, cv_var_series, nocv_cache,
                model_col, models_to_analyze, has_procedure,
                method, all_results, filtered_configs, r,
                dataset_col,
                boot_seeds=boot_seeds_cv if paired else None,
                interpolate=interpolate,
            )

    if filtered_configs:
        n_skip = len(filtered_configs)
        warnings.warn(
            f"sample_gain_bootstrap: {n_skip} configurations skipped. "
            "Use the returned 'filtered_configs' attribute for details.",
            stacklevel=2,
        )

    if not all_results:
        cols = [model_col, "N", "K", "k_equiv", "sample_gain", "bootstrap_iter"]
        for col in result_extra_cols:
            if col not in cols:
                cols.append(col)
        if compute_train_size:
            cols.append("train_size")
        if has_procedure:
            cols.insert(4, "cv_procedure")
        empty_df = pd.DataFrame(columns=cols)
        empty_df.attrs["filtered_configs"] = filtered_configs
        return empty_df

    result_df = pd.DataFrame(all_results)
    if compute_train_size:
        result_df["train_size"] = (result_df["N"] * (1 - test_size)).astype(int)
    # Attach filtered_configs as an attribute (non-standard but accessible)
    result_df.attrs["filtered_configs"] = filtered_configs
    return result_df


def _df_mode_iter(
    i_boot, cv_var_series, boot_df_nocv, score_col_nocv,
    model_col, size_col, models_to_analyze, has_procedure,
    method, all_results, filtered_configs,
    nocv_match_cols, result_extra_cols,
    interpolate: bool = False,
):
    """Inner loop body for DataFrame mode bootstrap iteration."""
    nocv_var_series = (
        boot_df_nocv.groupby(nocv_match_cols + ["k"], dropna=False)[score_col_nocv]
        .var()
    )
    cv_index_names = list(cv_var_series.index.names)

    # Iterate over CV variance entries
    for cv_key, target_var in cv_var_series.items():
        key_map = _index_key_to_dict(cv_key, cv_index_names)
        if key_map is None:
            continue

        model = key_map.get(model_col)
        size = key_map.get(size_col)
        proc = key_map.get("procedure")
        K_cv = key_map.get("n_splits")
        config_key = _config_signature(key_map, cv_index_names, exclude={"n_splits"})

        if model not in models_to_analyze:
            continue
        if pd.isna(target_var):
            continue

        nocv_lookup_cols = [col for col in nocv_match_cols if col in key_map]
        nocv_lookup = tuple(key_map[col] for col in nocv_lookup_cols)
        try:
            if len(nocv_lookup_cols) == 1:
                nocv_curve = nocv_var_series.xs(
                    nocv_lookup[0], level=nocv_lookup_cols[0]
                )
            else:
                nocv_curve = nocv_var_series.xs(
                    nocv_lookup, level=nocv_lookup_cols
                )
        except KeyError:
            if config_key not in filtered_configs:
                filtered_configs[config_key] = "No no-CV data found"
            continue

        # nocv_curve is a Series indexed by k
        nocv_k = np.array(nocv_curve.index, dtype=int)
        nocv_vars = nocv_curve.values

        # Baselines for ratio method
        baseline_nocv_var = None
        baseline_cv_var = None
        if method == "ratio":
            # baseline no-CV: k=0 row
            if 0 in nocv_curve.index:
                baseline_nocv_var = float(nocv_curve.loc[0])
            else:
                if config_key not in filtered_configs:
                    filtered_configs[config_key] = "k=0 row missing for ratio method"
                continue
            # baseline CV: K=1 row
            try:
                baseline_key = tuple(
                    1 if name == "n_splits" else key_map[name]
                    for name in cv_index_names
                )
                baseline_cv_var = float(cv_var_series.loc[baseline_key])
            except KeyError:
                if config_key not in filtered_configs:
                    filtered_configs[config_key] = "K=1 CV row missing for ratio method"
                continue
            if baseline_nocv_var is None or baseline_nocv_var <= 0:
                if config_key not in filtered_configs:
                    filtered_configs[config_key] = "Invalid nocv baseline for ratio method"
                continue
            if baseline_cv_var is None or baseline_cv_var <= 0:
                if config_key not in filtered_configs:
                    filtered_configs[config_key] = "Invalid cv baseline for ratio method"
                continue

        k_match_idx = match_variances(
            nocv_vars, target_var, method,
            baseline_nocv_var=baseline_nocv_var,
            baseline_cv_var=baseline_cv_var,
        )
        target_abs = (
            target_var if method == "absolute"
            else target_var / baseline_cv_var * baseline_nocv_var
        )
        sample_gain = _crossing_gain(
            nocv_k + 1, nocv_vars, k_match_idx, target_abs, interpolate
        )
        if k_match_idx is None:
            k_match_idx = len(nocv_vars) - 1  # cap at max available

        # k_equiv = the actual k value at that index
        k_equiv = int(nocv_k[k_match_idx])

        entry = {
            "bootstrap_iter": i_boot,
            model_col: model,
            "N": int(size),
            "K": int(K_cv),
            "k_equiv": k_equiv,
            "sample_gain": sample_gain,
        }
        for col in result_extra_cols:
            if col in key_map:
                entry[col] = key_map[col]
        if has_procedure:
            entry["cv_procedure"] = proc
        all_results.append(entry)


def _cache_mode_iter(
    i_boot, cv_var_series, nocv_cache,
    model_col, models_to_analyze, has_procedure,
    method, all_results, filtered_configs, rng,
    dataset_col,
    boot_seeds=None,
    interpolate: bool = False,
):
    """Inner loop body for Cache mode bootstrap iteration.

    With *boot_seeds* (paired bootstrap), the single-split rows are those of the
    resampled seeds, in the same multiplicity as on the CV side.
    """
    for cache_key, cache in nocv_cache.items():
        # cache_key may be:
        #   (dataset, model, size)            — 3-tuple
        #   (dataset, model, size, fixed_split) — 4-tuple (new format)
        #   (model, size)                      — 2-tuple (no dataset dim)
        fixed_split = None
        if len(cache_key) == 4:
            dataset, model, size, fixed_split = cache_key
        elif len(cache_key) == 3:
            dataset, model, size = cache_key
        elif len(cache_key) == 2:
            dataset = None
            model, size = cache_key
        else:
            continue

        if model not in models_to_analyze:
            continue

        n_nocv = len(cache["test_scores"])
        if n_nocv < 2:
            key = cache_key
            if key not in filtered_configs:
                filtered_configs[key] = "Insufficient no-CV samples"
            continue

        # Resample no-CV: same seeds as the CV side when paired, else independently
        seed_index = cache.get("_seed_index")
        if boot_seeds is not None and seed_index is not None:
            indices = seed_index.get_indexer(boot_seeds)
            indices = indices[indices >= 0]
            if len(indices) < 2:
                continue
        else:
            indices = rng.choice(n_nocv, size=n_nocv, replace=True)
        boot_test_scores = cache["test_scores"][indices]  # (n,)

        max_k = cache["max_k"]
        test_scores_2d = boot_test_scores.reshape(-1, 1)

        if cache.get("combination", "average") == "pooled":
            # Hidden list already cumulative: element k is the metric pooled on
            # test + chunks 0..k, so use it directly (no re-averaging).
            boot_cum_values = cache["cum_values"][indices, :]  # (n, max_k)
            combined_scores_k_ge_1 = boot_cum_values
        else:
            boot_cum_means = cache["cum_means"][indices, :]    # (n, max_k)
            k_values = np.arange(1, max_k + 1).reshape(1, -1)
            # combined_scores[:,j] = mean of (test + first j hidden) for each seed
            combined_scores_k_ge_1 = (
                boot_cum_means * k_values + test_scores_2d
            ) / (k_values + 1)
        # index 0 = test only (k=0), indices 1..max_k = combined with k hidden sets
        combined_scores = np.hstack([test_scores_2d, combined_scores_k_ge_1])

        nocv_variances = np.nanvar(combined_scores, axis=0, ddof=1)
        baseline_nocv_var = float(nocv_variances[0]) if method == "ratio" else None

        if pd.isna(nocv_variances[0]):
            key = cache_key
            if key not in filtered_configs:
                filtered_configs[key] = "Invalid nocv baseline variance"
            continue

        # Iterate over matching CV variance entries
        procedures = [None]
        if has_procedure:
            procedures = cv_var_series.index.get_level_values("procedure").unique()

        has_fs_in_index = (
            fixed_split is not None
            and "fixed_split" in cv_var_series.index.names
        )

        for proc in procedures:
            try:
                if has_procedure:
                    if dataset is not None:
                        key_parts = (dataset, model, int(size), proc)
                    else:
                        key_parts = (model, int(size), proc)
                else:
                    if dataset is not None:
                        key_parts = (dataset, model, int(size))
                    else:
                        key_parts = (model, int(size))

                if has_fs_in_index:
                    key_parts = key_parts + (fixed_split,)

                target_cv_vars = cv_var_series.loc[key_parts]
            except KeyError:
                key = (*cache_key, proc) if proc else cache_key
                if key not in filtered_configs:
                    filtered_configs[key] = "No CV data found"
                continue

            baseline_cv_var = None
            if method == "ratio":
                try:
                    baseline_cv_var = float(target_cv_vars.loc[1])
                except (KeyError, TypeError):
                    key = (*cache_key, proc) if proc else cache_key
                    if key not in filtered_configs:
                        filtered_configs[key] = "K=1 CV row missing for ratio"
                    continue
                if baseline_cv_var is None or baseline_cv_var <= 0:
                    continue

            for K_split, target_var in target_cv_vars.items():
                if pd.isna(target_var):
                    continue

                k_match_idx = match_variances(
                    nocv_variances, target_var, method,
                    baseline_nocv_var=baseline_nocv_var,
                    baseline_cv_var=baseline_cv_var,
                )
                target_abs = (
                    target_var if method == "absolute"
                    else target_var / baseline_cv_var * baseline_nocv_var
                )
                # In cache mode, index 0 = k=0 (test only): multiple = index + 1
                sample_gain = _crossing_gain(
                    np.arange(1, len(nocv_variances) + 1), nocv_variances,
                    k_match_idx, target_abs, interpolate,
                )
                if k_match_idx is None:
                    k_match_idx = max_k  # cap

                entry = {
                    "bootstrap_iter": i_boot,
                    model_col: model,
                    "N": int(size),
                    "K": int(K_split),
                    "k_equiv": k_match_idx,
                    "sample_gain": sample_gain,
                }
                if has_procedure:
                    entry["cv_procedure"] = proc
                if dataset is not None and dataset_col is not None:
                    entry[dataset_col] = dataset
                if fixed_split is not None:
                    entry["fixed_split"] = fixed_split
                all_results.append(entry)


# ---------------------------------------------------------------------------
# agg_ci
# ---------------------------------------------------------------------------

def agg_ci(
    df: pd.DataFrame,
    group_col: str = "K",
    value_col: str = "sample_gain",
) -> pd.DataFrame:
    """Aggregate *value_col* by *group_col* with 95% CI.

    Returns DataFrame with columns:
    ``[group_col, 'mean', 'std', 'count', 'ci95', 'lower', 'upper', 'normalized_time']``
    where ``normalized_time == group_col`` values (for backward compatibility).
    """
    if df.empty:
        return pd.DataFrame()
    group_col = _resolve_split_col(df, group_col)
    agg = (
        df.groupby(group_col)[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["ci95"] = 1.96 * agg["std"] / np.sqrt(agg["count"])
    agg["lower"] = agg["mean"] - agg["ci95"]
    agg["upper"] = agg["mean"] + agg["ci95"]
    agg["normalized_time"] = agg[group_col]
    return agg


# ---------------------------------------------------------------------------
# compute_tau  — inter-fold covariance estimator (Nadeau–Bengio / ANOVA)
# ---------------------------------------------------------------------------

_TAU_COLS = ["K", "n_seeds", "W", "B", "tau", "sigma2",
             "tau_lo", "tau_hi", "B_lo", "B_hi"]


def _tau_from_per_seed(means: np.ndarray, vars_: np.ndarray, K: int
                       ) -> tuple[float, float, float, float]:
    """(W, B, tau, sigma2) from per-seed means / within-seed sample variances."""
    W = float(np.mean(vars_))
    B = float(np.var(means, ddof=1)) if len(means) >= 2 else float("nan")
    tau = B - W / K
    sigma2 = W + tau
    return W, B, tau, sigma2


def compute_tau(
    df: pd.DataFrame,
    score_col: str,
    seed_col: str,
    group_cols: list[str],
    min_folds: int = 2,
    min_seeds: int = 2,
    n_bootstraps: int = 100,
    ci_level: float = 0.95,
    random_state: int | None = 0,
) -> pd.DataFrame:
    """Estimate τ̂ = B̂ − Ŵ / K per group with bootstrap-over-seeds CIs.

    Model: within each group, seeds are assumed iid realisations; within a
    seed, the K fold scores are exchangeable with marginal variance σ² and
    pairwise covariance τ. The estimators are

        Ŵ  = mean_s(per-seed sample variance across folds)   ≈ σ² − τ
        B̂  = var_s(per-seed mean across folds, ddof=1)        ≈ (σ² + (K−1)τ)/K
        τ̂  = B̂ − Ŵ / K
        σ̂² = Ŵ + τ̂

    B̂ is itself the variance of the K-fold CV estimator ``Var(Ē_K)``, and
    τ̂ is its irreducible floor as K → ∞.

    A non-parametric bootstrap over seeds produces percentile CIs for both
    ``tau`` and ``B``. Set ``n_bootstraps=0`` to skip the bootstrap (CI
    columns are returned as NaN).

    Parameters
    ----------
    df : DataFrame
        One row per (group_cols + [seed_col] + fold). The column `score_col`
        holds the per-fold score (scalar).
    score_col, seed_col : str
    group_cols : list of str
        Columns identifying one algorithm × configuration.
    min_folds : int
    min_seeds : int
    n_bootstraps : int
        Seed-level bootstrap iterations for the CI. Set to 0 to disable.
    ci_level : float
        Target coverage; e.g. 0.95 gives the 2.5%–97.5% percentile interval.
    random_state : int, optional
        Seed for the bootstrap RNG. None draws fresh entropy.

    Returns
    -------
    DataFrame with columns
        ``group_cols + ['K', 'n_seeds', 'W', 'B', 'tau', 'sigma2',
                         'tau_lo', 'tau_hi', 'B_lo', 'B_hi']``.
    """
    if df.empty:
        return pd.DataFrame(columns=group_cols + _TAU_COLS)

    work = df[group_cols + [seed_col, score_col]].copy()
    # Fast path: skip apply if already numeric
    if not pd.api.types.is_numeric_dtype(work[score_col]):
        work[score_col] = work[score_col].apply(to_scalar)
    work = work.dropna(subset=[score_col])

    per_seed = (
        work.groupby(group_cols + [seed_col])[score_col]
        .agg(["mean", "var", "count"])
        .reset_index()
    )

    rng = np.random.default_rng(random_state)
    alpha = (1.0 - ci_level) / 2.0
    q_lo, q_hi = 100 * alpha, 100 * (1.0 - alpha)

    rows = []
    skipped: list[tuple] = []
    for key, sub in per_seed.groupby(group_cols):
        sub = sub[sub["count"] >= min_folds]
        if len(sub) < min_seeds:
            skipped.append((key, "insufficient seeds"))
            continue
        K_vals = sub["count"].unique()
        if len(K_vals) > 1:
            skipped.append((key, f"non-constant K across seeds: {sorted(K_vals)}"))
            continue
        K = int(K_vals[0])
        means = sub["mean"].to_numpy(dtype=float)
        vars_ = sub["var"].to_numpy(dtype=float)
        W, B, tau, sigma2 = _tau_from_per_seed(means, vars_, K)

        tau_lo = tau_hi = B_lo = B_hi = float("nan")
        if n_bootstraps > 0 and len(means) >= 2:
            n = len(means)
            idx = rng.integers(0, n, size=(n_bootstraps, n))
            boot_means = means[idx]
            boot_vars = vars_[idx]
            # Vectorised per-replicate W, B
            W_b = boot_vars.mean(axis=1)
            B_b = boot_means.var(axis=1, ddof=1)
            tau_b = B_b - W_b / K
            tau_lo, tau_hi = np.nanpercentile(tau_b, [q_lo, q_hi])
            B_lo, B_hi = np.nanpercentile(B_b, [q_lo, q_hi])

        entry = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        entry.update({
            "K": K,
            "n_seeds": int(len(sub)),
            "W": W,
            "B": B,
            "tau": tau,
            "sigma2": sigma2,
            "tau_lo": float(tau_lo),
            "tau_hi": float(tau_hi),
            "B_lo": float(B_lo),
            "B_hi": float(B_hi),
        })
        rows.append(entry)

    if skipped:
        warnings.warn(
            f"compute_tau: skipped {len(skipped)} group(s). "
            f"Example: {skipped[0]}",
            stacklevel=2,
        )

    return pd.DataFrame(rows, columns=group_cols + _TAU_COLS)


# ---------------------------------------------------------------------------
# sample_gain_at_splits
# ---------------------------------------------------------------------------

def sample_gain_at_splits(
    results_df: pd.DataFrame,
    group_cols: list[str],
    target_splits=(20,),
    value_col: str = "sample_gain",
    split_col: str = "K",
) -> pd.DataFrame:
    """Mean sample gain per group at each split in *target_splits*.

    Filters *results_df* to rows where ``split_col ∈ target_splits`` and
    returns ``mean(value_col)`` grouped by ``group_cols + [split_col]``.
    The resulting DataFrame has one row per (group, target split).
    """
    if results_df.empty:
        return pd.DataFrame(columns=list(group_cols) + [split_col, value_col])

    split_col_resolved = _resolve_split_col(results_df, split_col)
    target_splits = list(target_splits)
    sub = results_df[results_df[split_col_resolved].isin(target_splits)]
    if sub.empty:
        return pd.DataFrame(columns=list(group_cols) + [split_col, value_col])

    out = (
        sub.groupby(list(group_cols) + [split_col_resolved], dropna=False)[value_col]
        .mean()
        .reset_index()
    )
    if split_col_resolved != split_col:
        out = out.rename(columns={split_col_resolved: split_col})
    return out


# ---------------------------------------------------------------------------
# prepare_pareto_data  (simulated only)
# ---------------------------------------------------------------------------

def prepare_pareto_data(
    gains_df: pd.DataFrame,
    cv_data: pd.DataFrame,
    model_col: str = "solver_name",
    size_col: str = "study_size",
    dataset_col: str = "dataset_name",
) -> pd.DataFrame:
    """Attach average time-per-split to *gains_df* (simulated data only).

    Joins with mean time at n_splits=1 from *cv_data*.
    Sets ``normalized_time = K`` (number of splits).

    Returns the augmented DataFrame.
    """
    group_cols = [model_col, size_col]
    if dataset_col in cv_data.columns:
        group_cols.append(dataset_col)

    base_times = (
        cv_data[cv_data["n_splits"] == 1]
        .groupby(group_cols)["time"]
        .mean()
        .reset_index()
        .rename(columns={"time": "time_per_split"})
    )

    left_on = [model_col, "N"]
    right_on = [model_col, size_col]
    if dataset_col in cv_data.columns and dataset_col in gains_df.columns:
        left_on.append(dataset_col)
        right_on.append(dataset_col)

    merged = gains_df.merge(base_times, left_on=left_on, right_on=right_on, how="left")
    split_col = "K" if "K" in merged.columns else "s"
    merged["normalized_time"] = merged[split_col]
    return merged


# ---------------------------------------------------------------------------
# compute_retrieval
# ---------------------------------------------------------------------------

def compute_retrieval(
    df_nocv_all: pd.DataFrame,
    df_cv: pd.DataFrame,
    gt_ranking: pd.DataFrame,
    models_list: list[str],
    study_sizes: list[int],
    target_splits: list[int],
    model_col: str = "model",
    score_col_nocv: str = "combined_score",
    score_col_cv: str = "score_scalar",
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise ranking retrieval curves and a summary table.

    For each (study_size, model_pair) combination determines which model is
    "best" using ground-truth mean benchmark scores, then computes the fraction
    of seeds / bootstrap draws where the best model's score exceeds the worst
    model's score, as a function of the test multiplier (no-CV path) or at
    fixed CV split counts.

    Parameters
    ----------
    df_nocv_all : DataFrame
        Columns ``['seed', model_col, 'study_size', 'multiplier', score_col_nocv]``.
        Include k=0 rows (multiplier=1).
    df_cv : DataFrame
        Columns ``['seed', model_col, 'study_size', 'n_splits', score_col_cv]``
        with scalar scores.
    gt_ranking : DataFrame
        Columns ``['study_size', model_col, 'objective_score_bench']``
        (mean benchmark score across seeds).
    models_list : list of str
        Models to include; all pairs are evaluated.
    study_sizes : list of int
    target_splits : list of int
        CV split counts to use as reference.
    model_col : str
    score_col_nocv, score_col_cv : str
    test_size : float
        Used to compute ``train_size`` in the summary table.

    Returns
    -------
    curves_df : DataFrame
        Columns ``['study_size', 'best_model', 'worst_model', 'multiplier', 'correct_pick']``.
    table_df : DataFrame
        Columns ``['train_size', 'best_model', 'worst_model',
                   'mean_gt_score_diff', 'std_single_split_score_diff', 'n_seeds']``.
    """
    model_pairs = list(itertools.combinations(models_list, 2))
    curve_rows: list[dict] = []
    table_rows: list[dict] = []

    for n_size in study_sizes:
        train_size_est = int(n_size * (1 - test_size))

        for m1, m2 in model_pairs:
            # Ground truth ranking
            r1 = gt_ranking[
                (gt_ranking[model_col] == m1) & (gt_ranking["study_size"] == n_size)
            ]["objective_score_bench"].values
            r2 = gt_ranking[
                (gt_ranking[model_col] == m2) & (gt_ranking["study_size"] == n_size)
            ]["objective_score_bench"].values

            if len(r1) == 0 or len(r2) == 0:
                continue
            if pd.isna(r1[0]) or pd.isna(r2[0]):
                continue

            if r1[0] >= r2[0]:
                best_model, worst_model = m1, m2
                score_diff = float(r1[0] - r2[0])
            else:
                best_model, worst_model = m2, m1
                score_diff = float(r2[0] - r1[0])

            # Std of per-seed single-split score differences.
            base_cv = df_cv[df_cv["study_size"] == n_size]
            # Use n_splits==1 as the single-split baseline if available
            base_cv_1 = base_cv[base_cv["n_splits"] == 1] if "n_splits" in base_cv.columns else base_cv
            m1_bench = (
                base_cv_1[base_cv_1[model_col] == m1][["seed", score_col_cv]]
                .rename(columns={score_col_cv: f"score_{m1}"})
            )
            m2_bench = (
                base_cv_1[base_cv_1[model_col] == m2][["seed", score_col_cv]]
                .rename(columns={score_col_cv: f"score_{m2}"})
            )
            merged_bench = m1_bench.merge(m2_bench, on="seed")
            merged_bench["diff"] = (
                merged_bench[f"score_{best_model}"] - merged_bench[f"score_{worst_model}"]
            )
            std_diff = float(merged_bench["diff"].std(ddof=1)) if len(merged_bench) > 1 else np.nan
            n_seeds = len(merged_bench)

            table_rows.append({
                "train_size": train_size_est,
                "best_model": best_model,
                "worst_model": worst_model,
                "mean_gt_score_diff": score_diff,
                "std_single_split_score_diff": std_diff,
                "n_seeds": n_seeds,
            })

            # --- No-CV retrieval curve ---
            subset_nocv = df_nocv_all[
                (df_nocv_all["study_size"] == n_size)
                & (df_nocv_all[model_col].isin([m1, m2]))
            ]

            if not subset_nocv.empty:
                # Paired by seed
                pivot = subset_nocv.pivot_table(
                    index=["seed", "multiplier"],
                    columns=model_col,
                    values=score_col_nocv,
                ).dropna()

                if best_model in pivot.columns and worst_model in pivot.columns:
                    pivot["correct_pick"] = pivot[best_model] > pivot[worst_model]
                    curve = (
                        pivot.groupby("multiplier")["correct_pick"]
                        .mean()
                        .reset_index()
                    )
                    part = curve[["multiplier", "correct_pick"]].copy()
                    part["study_size"] = n_size
                    part["best_model"] = best_model
                    part["worst_model"] = worst_model
                    curve_rows.append(part)

    _curve_cols = ["study_size", "best_model", "worst_model", "multiplier", "correct_pick"]
    curves_df = (
        pd.concat(curve_rows, ignore_index=True)[_curve_cols]
        if curve_rows
        else pd.DataFrame(columns=_curve_cols)
    )

    table_df = pd.DataFrame(
        table_rows,
        columns=["train_size", "best_model", "worst_model",
                 "mean_gt_score_diff", "std_single_split_score_diff", "n_seeds"],
    ) if table_rows else pd.DataFrame(
        columns=["train_size", "best_model", "worst_model",
                 "mean_gt_score_diff", "std_single_split_score_diff", "n_seeds"]
    )

    return curves_df, table_df


# ---------------------------------------------------------------------------
# Significance-decision replicability curves (rebuttal experiment)
# ---------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float, itmax: int = 200, eps: float = 3e-12) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    import math
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b), scipy-free."""
    import math
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def two_sided_t_pvalue(t_stat: float, df: float) -> float:
    """Two-sided p-value of a Student-t statistic, without scipy.

    Equals ``2 * (1 - CDF_t(|t|, df))`` via the incomplete beta identity
    ``P(|T| > |t|) = I_x(df/2, 1/2)`` with ``x = df / (df + t^2)``.
    """
    if df <= 0 or not np.isfinite(t_stat):
        return np.nan
    x = df / (df + t_stat * t_stat)
    return _betai(0.5 * df, 0.5, x)


def _t_pvalues(t_stats: np.ndarray, df: int) -> np.ndarray:
    """Vectorised two-sided p-values (loops the scalar helper; small K/n here)."""
    out = np.full(len(t_stats), np.nan)
    for i, tv in enumerate(t_stats):
        out[i] = two_sided_t_pvalue(tv, df)
    return out


def _paired_decision(diffs: np.ndarray, var_factor: float, df: int,
                     alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-row paired-t decision on a (n_rows, K) array of differences.

    ``var_factor`` is ``1/K`` for the standard paired t-test and
    ``1/K + rho`` for the Nadeau-Bengio corrected resampled t-test. Returns
    ``(reject, sign)`` boolean / sign arrays; rows with zero/NaN variance do not
    reject.
    """
    dm = np.nanmean(diffs, axis=1)
    dv = np.nanvar(diffs, axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        t_stat = dm / np.sqrt(var_factor * dv)
    t_stat = np.where((dv > 0) & np.isfinite(t_stat), t_stat, np.nan)
    p = _t_pvalues(t_stat, df)
    reject = np.isfinite(p) & (p < alpha)
    return reject, np.sign(dm)


def _decisions_to_labels(reject: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """Map (reject, sign) to ternary labels: 'A>B' / 'B>A' / 'not-sig'."""
    lab = np.full(len(reject), "not-sig", dtype=object)
    lab[reject & (sign > 0)] = "A>B"
    lab[reject & (sign < 0)] = "B>A"
    return lab


def significance_replicability_curves(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    sizes: list[int],
    k_grid,
    *,
    model_col: str = "model",
    seed_col: str = "seed",
    size_col: str = "study_size",
    fold_col: str = "fold",
    fold_score_col: str = "fold_score",
    bench_score_col: str = "bench_score",
    n_test_col: str | None = "n_test",
    n_train_col: str | None = "n_train",
    default_rho: float = 0.25,
    alpha: float = 0.05,
    self_null_kmax: int = 10,
    dataset: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Significance-decision replicability curves (Bouckaert-Frank style).

    For each ``(A, B)`` pair and study size, the *oracle decision* is the paired
    t-test on per-seed **benchmark** scores over all seeds (large benchmark set ≈
    oracle). Then, per split-count K and per seed, the K per-fold **study-test**
    paired differences are tested with the Nadeau-Bengio corrected resampled
    t-test (which accounts for the train-set overlap between MCCV folds), and the
    decision compared to the oracle. Returns:

    * ``rep_df`` — one row per (dataset, pair, N, K, test_type) with
      ``rep`` = fraction of seeds whose decision matches the oracle, plus the
      oracle decision/significance and n_seeds.
    * ``power_fpr_df`` — same keys with ``reject_rate``, ``power`` (reject-rate
      when the oracle is significant), ``fpr_oracle`` (reject-rate when the
      oracle is non-significant) and ``fpr_self`` (exact Δ=0 rejection rate from
      the ``fold_i`` vs ``fold_{i+K}`` self-comparison, averaged over the pair's
      two models; only for K ≤ ``self_null_kmax``).
    """
    rep_rows: list[dict] = []
    pf_rows: list[dict] = []
    k_grid = list(k_grid)

    # ---- exact-null (A-vs-A) rejection rate, cached per (model, size, K) -----
    def _self_null(sub_size: pd.DataFrame, model: str, rho: float) -> dict:
        piv = sub_size[sub_size[model_col] == model].pivot_table(
            index=seed_col, columns=fold_col, values=fold_score_col)
        folds = sorted(piv.columns)
        out: dict[tuple[int, str], float] = {}
        for K in k_grid:
            if K > self_null_kmax or 2 * K > len(folds):
                continue
            first = piv[folds[:K]].to_numpy()
            second = piv[folds[K:2 * K]].to_numpy()
            d = first - second                      # true mean 0 (same algorithm)
            mask = np.isfinite(d).all(axis=1)
            d = d[mask]
            if len(d) < 2:
                continue
            for test_type, vf in (("corrected", 1.0 / K + rho),):
                reject, _ = _paired_decision(d, vf, K - 1, alpha)
                out[(K, test_type)] = float(np.mean(reject))
        return out

    for N in sizes:
        sub = df[df[size_col] == N]
        if sub.empty:
            continue
        if n_test_col and n_train_col and n_test_col in sub and n_train_col in sub:
            ratio = (pd.to_numeric(sub[n_test_col], errors="coerce")
                     / pd.to_numeric(sub[n_train_col], errors="coerce"))
            rho = float(np.nanmedian(ratio))
            if not np.isfinite(rho) or rho <= 0:
                rho = default_rho
        else:
            rho = default_rho

        self_null_cache: dict[str, dict] = {}

        for A, B in pairs:
            sA = sub[sub[model_col] == A]
            sB = sub[sub[model_col] == B]
            if sA.empty or sB.empty:
                continue

            # -- Oracle: paired t-test on per-seed benchmark means --
            bA = sA.groupby(seed_col)[bench_score_col].mean()
            bB = sB.groupby(seed_col)[bench_score_col].mean()
            common = bA.index.intersection(bB.index)
            Dbench = (bA.loc[common] - bB.loc[common]).to_numpy()
            S = len(Dbench)
            if S < 2:
                continue
            dm = float(np.mean(Dbench))
            sd = float(np.std(Dbench, ddof=1))
            t_oracle = dm / (sd / np.sqrt(S)) if sd > 0 else np.nan
            p_oracle = two_sided_t_pvalue(t_oracle, S - 1)
            oracle_sig = bool(np.isfinite(p_oracle) and p_oracle < alpha)
            oracle_dec = ("A>B" if (oracle_sig and dm > 0)
                          else "B>A" if (oracle_sig and dm < 0) else "not-sig")

            # -- Per-seed fold-level paired study-test differences --
            pA = sA.pivot_table(index=seed_col, columns=fold_col, values=fold_score_col)
            pB = sB.pivot_table(index=seed_col, columns=fold_col, values=fold_score_col)
            seeds = pA.index.intersection(pB.index)
            folds = sorted(set(pA.columns).intersection(pB.columns))
            diff_all = (pA.loc[seeds, folds] - pB.loc[seeds, folds])

            if A not in self_null_cache:
                self_null_cache[A] = _self_null(sub, A, rho)
            if B not in self_null_cache:
                self_null_cache[B] = _self_null(sub, B, rho)

            for K in k_grid:
                if K > len(folds):
                    continue
                d = diff_all[folds[:K]].to_numpy()
                keep = np.isfinite(d).all(axis=1)
                d = d[keep]
                n_seeds = len(d)
                if n_seeds < 2:
                    continue
                for test_type, vf in (("corrected", 1.0 / K + rho),):
                    reject, sign = _paired_decision(d, vf, K - 1, alpha)
                    labels = _decisions_to_labels(reject, sign)
                    rep = float(np.mean(labels == oracle_dec))
                    reject_rate = float(np.mean(reject))
                    fs = [self_null_cache[m].get((K, test_type)) for m in (A, B)]
                    fs = [v for v in fs if v is not None]
                    fpr_self = float(np.mean(fs)) if fs else np.nan
                    row = dict(dataset=dataset, pair=f"{A} vs {B}",
                               model_A=A, model_B=B, N=int(N), K=int(K),
                               test_type=test_type, n_seeds=n_seeds,
                               oracle_decision=oracle_dec,
                               oracle_significant=oracle_sig,
                               oracle_p=float(p_oracle))
                    rep_rows.append({**row, "rep": rep})
                    pf_rows.append({**row, "reject_rate": reject_rate,
                                    "power": reject_rate if oracle_sig else np.nan,
                                    "fpr_oracle": reject_rate if not oracle_sig else np.nan,
                                    "fpr_self": fpr_self})

    rep_df = pd.DataFrame(rep_rows)
    power_fpr_df = pd.DataFrame(pf_rows)
    return rep_df, power_fpr_df
