"""Shared helpers for the NLP / PCam validation pipelines.

This module keeps the numerical kernels in ``compute.py`` generic while
providing a paper-aligned vocabulary at the pipeline layer. Column names
track the estimator notation from the paper:

- raw fold-score ANOVA: ``tau``, ``W``, ``B``
- raw single-fold variance estimators (one symbol per estimator):
    * ``sigma2_HO_K``     -- from ANOVA on a $K$-fold CV run ($\\hat\\sigma^2_{HO|K}$)
    * ``sigma2_HO_1``     -- across-seed variance of single-HO replicates
                             ($\\hat\\sigma^2_{HO|1}$)
    * ``sigma2_HO_trxi``  -- within-seed variance, averaged over seeds
                             ($\\hat\\sigma^2_{HO|tr,xi}$)
- bench-adjusted (delta) variants use ``te`` in place of ``HO``:
    * ``tau_te``, ``sigma2_te_K``, ``sigma2_te_1``
"""

from __future__ import annotations

import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import compute
import plot as P


CANONICAL_TAU_GAIN_COLUMNS = [
    "pipeline",
    "algorithm",
    "dataset",
    "study_size",
    "train_size",
    "procedure",
    "fixed_split",
    "tau",
    "sigma2_HO_K",
    "sigma2_HO_1",
    "sigma2_HO_trxi",
    "sigma2_te_1",
    "tau_te",
    "sigma2_te_K",
    "tau_te_lo",
    "tau_te_hi",
    "W",
    "B",
    "K",
    "n_seeds",
    "tau_over_sigma2_HO_K",
    "W_over_B",
    "sample_gain_K20",
    "sample_gain_K20_log1mr2",
    "mean_bench_score",
    "bench_cov",
    "study_cov",
    "bench_error_cov",
    "bench_error_corr",
    "bench_error_var",
    "bench_error_rho",
    "bench_squared_error_cov",
    "bench_squared_error_corr",
    "bench_squared_error_var",
    "bench_squared_error_rho",
    "bench_squared_error_cov_all",
    "bench_squared_error_var_all",
    "bench_squared_error_rho_all",
    "bench_rho_delta_proxy_squared_error_all",
    "bench_gain_proxy_squared_error_all",
    "bench_prediction_cov_all",
    "bench_prediction_var_all",
    "bench_prediction_rho_all",
    "bench_gain_proxy_prediction_all",
    "bench_error_cov_all",
    "bench_error_var_all",
    "bench_error_rho_all",
    "bench_gain_proxy_error_all",
    "study_error_cov",
    "study_error_corr",
    "study_error_var",
    "study_error_rho",
    "study_squared_error_cov",
    "study_squared_error_corr",
    "study_squared_error_var",
    "study_squared_error_rho",
    "study_prediction_cov_all_full",
    "study_prediction_var_all_full",
    "study_prediction_rho_all_full",
    "study_gain_proxy_prediction_all_full",
    "study_error_cov_all_full",
    "study_error_var_all_full",
    "study_error_rho_all_full",
    "study_gain_proxy_error_all_full",
    "study_squared_error_cov_all_full",
    "study_squared_error_var_all_full",
    "study_squared_error_rho_all_full",
    "study_gain_proxy_squared_error_all_full",
    "study_prediction_cov_all_resid_train_membership",
    "study_prediction_var_all_resid_train_membership",
    "study_prediction_rho_all_resid_train_membership",
    "study_gain_proxy_prediction_all_resid_train_membership",
    "study_error_cov_all_resid_train_membership",
    "study_error_var_all_resid_train_membership",
    "study_error_rho_all_resid_train_membership",
    "study_gain_proxy_error_all_resid_train_membership",
    "study_squared_error_cov_all_resid_train_membership",
    "study_squared_error_var_all_resid_train_membership",
    "study_squared_error_rho_all_resid_train_membership",
    "study_gain_proxy_squared_error_all_resid_train_membership",
    "study_prediction_cov_all_oof_intersection",
    "study_prediction_var_all_oof_intersection",
    "study_prediction_rho_all_oof_intersection",
    "study_gain_proxy_prediction_all_oof_intersection",
    "study_error_cov_all_oof_intersection",
    "study_error_var_all_oof_intersection",
    "study_error_rho_all_oof_intersection",
    "study_gain_proxy_error_all_oof_intersection",
    "study_squared_error_cov_all_oof_intersection",
    "study_squared_error_var_all_oof_intersection",
    "study_squared_error_rho_all_oof_intersection",
    "study_gain_proxy_squared_error_all_oof_intersection",
    # PCam classification analogs: NLL and 0/1-accuracy OOF-intersection redundancy.
    # (The Brier version reuses the squared-error column above; NaN for regression.)
    "study_nll_rho_all_oof_intersection",
    "study_error01_rho_all_oof_intersection",
    "study_oof_intersection_pairs_all",
    "study_oof_intersection_mean_size_all",
    "split_index",
    "split_train_size",
    "split_test_size",
    "split_train_overlap_prev_frac",
    "split_test_overlap_prev_frac",
    "split_train_overlap_first_frac",
    "split_test_overlap_first_frac",
    "split_train_overlap_prev_count",
    "split_test_overlap_prev_count",
    "split_train_overlap_first_count",
    "split_test_overlap_first_count",
]

OPTIONAL_TAU_GAIN_COLUMNS = [
    "bench_error_cov",
    "bench_error_corr",
    "bench_error_var",
    "bench_error_rho",
    "bench_squared_error_cov",
    "bench_squared_error_corr",
    "bench_squared_error_var",
    "bench_squared_error_rho",
    "bench_squared_error_cov_all",
    "bench_squared_error_var_all",
    "bench_squared_error_rho_all",
    "bench_rho_delta_proxy_squared_error_all",
    "bench_gain_proxy_squared_error_all",
    "bench_prediction_cov_all",
    "bench_prediction_var_all",
    "bench_prediction_rho_all",
    "bench_gain_proxy_prediction_all",
    "bench_error_cov_all",
    "bench_error_var_all",
    "bench_error_rho_all",
    "bench_gain_proxy_error_all",
    "study_error_cov",
    "study_error_corr",
    "study_error_var",
    "study_error_rho",
    "study_squared_error_cov",
    "study_squared_error_corr",
    "study_squared_error_var",
    "study_squared_error_rho",
    "study_prediction_cov_all_full",
    "study_prediction_var_all_full",
    "study_prediction_rho_all_full",
    "study_gain_proxy_prediction_all_full",
    "study_error_cov_all_full",
    "study_error_var_all_full",
    "study_error_rho_all_full",
    "study_gain_proxy_error_all_full",
    "study_squared_error_cov_all_full",
    "study_squared_error_var_all_full",
    "study_squared_error_rho_all_full",
    "study_gain_proxy_squared_error_all_full",
    "study_prediction_cov_all_resid_train_membership",
    "study_prediction_var_all_resid_train_membership",
    "study_prediction_rho_all_resid_train_membership",
    "study_gain_proxy_prediction_all_resid_train_membership",
    "study_error_cov_all_resid_train_membership",
    "study_error_var_all_resid_train_membership",
    "study_error_rho_all_resid_train_membership",
    "study_gain_proxy_error_all_resid_train_membership",
    "study_squared_error_cov_all_resid_train_membership",
    "study_squared_error_var_all_resid_train_membership",
    "study_squared_error_rho_all_resid_train_membership",
    "study_gain_proxy_squared_error_all_resid_train_membership",
    "study_prediction_cov_all_oof_intersection",
    "study_prediction_var_all_oof_intersection",
    "study_prediction_rho_all_oof_intersection",
    "study_gain_proxy_prediction_all_oof_intersection",
    "study_error_cov_all_oof_intersection",
    "study_error_var_all_oof_intersection",
    "study_error_rho_all_oof_intersection",
    "study_gain_proxy_error_all_oof_intersection",
    "study_squared_error_cov_all_oof_intersection",
    "study_squared_error_var_all_oof_intersection",
    "study_squared_error_rho_all_oof_intersection",
    "study_gain_proxy_squared_error_all_oof_intersection",
    # PCam classification analogs: NLL and 0/1-accuracy OOF-intersection redundancy.
    # (The Brier version reuses the squared-error column above; NaN for regression.)
    "study_nll_rho_all_oof_intersection",
    "study_error01_rho_all_oof_intersection",
    "study_oof_intersection_pairs_all",
    "study_oof_intersection_mean_size_all",
    "split_index",
    "split_train_size",
    "split_test_size",
    "split_train_overlap_prev_frac",
    "split_test_overlap_prev_frac",
    "split_train_overlap_first_frac",
    "split_test_overlap_first_frac",
    "split_train_overlap_prev_count",
    "split_test_overlap_prev_count",
    "split_train_overlap_first_count",
    "split_test_overlap_first_count",
]


def build_delta_cv_table(
    data_cv: pd.DataFrame,
    cumulative_test_col: str,
    cumulative_bench_col: str,
    fold_test_col: str,
    fold_bench_col: str,
) -> pd.DataFrame:
    """Attach cumulative and per-fold delta columns to a CV table."""
    out = data_cv.copy()
    out["delta_score_test"] = out[cumulative_test_col] - out[cumulative_bench_col]
    out["delta_score_fold"] = out[fold_test_col] - out[fold_bench_col]
    return out


def coerce_scalar_columns(
    data: pd.DataFrame,
    columns: list[str],
    scalar_extractor: Callable[[object], float],
) -> pd.DataFrame:
    """Return a copy where selected columns are coerced to numeric scalars.

    This is used at the pipeline layer before cumulative sums, because some
    parquet sources store scalar scores with ``object`` dtype or as length-1
    array-like values.
    """
    out = data.copy()
    for col in columns:
        if col not in out.columns:
            continue
        series = out[col]
        if pd.api.types.is_numeric_dtype(series):
            out[col] = pd.to_numeric(series, errors="coerce")
        else:
            out[col] = pd.to_numeric(series.apply(scalar_extractor), errors="coerce")
    return out


def compute_standard_tau_tables(
    data_cv: pd.DataFrame,
    raw_fold_score_col: str,
    group_cols: list[str],
    model_filter_col: str,
    models_to_analyze: list[str] | None,
    seed_col: str = "seed",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute canonical raw-score and evaluation-error tau tables."""
    if "delta_score_fold" not in data_cv.columns:
        raise KeyError("data_cv must contain 'delta_score_fold'")

    work = data_cv.copy()
    if models_to_analyze is not None:
        work = work[work[model_filter_col].isin(models_to_analyze)].copy()

    tau_raw_df = compute.compute_tau(
        work,
        score_col=raw_fold_score_col,
        seed_col=seed_col,
        group_cols=group_cols,
    ).rename(columns={
        "sigma2": "sigma2_HO_K",
        "tau_lo": "tau_lo",
        "tau_hi": "tau_hi",
        "B_lo": "B_lo",
        "B_hi": "B_hi",
    })

    tau_te_df = compute.compute_tau(
        work,
        score_col="delta_score_fold",
        seed_col=seed_col,
        group_cols=group_cols,
    ).rename(columns={
        "tau": "tau_te",
        "sigma2": "sigma2_te_K",
        "tau_lo": "tau_te_lo",
        "tau_hi": "tau_te_hi",
    })
    return tau_raw_df, tau_te_df


def plot_sample_gain_by_train_size(
    results_delta: pd.DataFrame,
    study_sizes: list[int],
    models_to_analyze: list[str],
    model_display_names: dict[str, str],
    model_color_map: dict[str, tuple[float, float, float]] | None,
    title_prefix: str,
    plots_dir: str,
    xticks: list[int] | None = None,
    legend_loc: str = "lower right",
    hue_col: str = "model",
    style_col: str | None = None,
    style_order: list | None = None,
    dashes: bool | dict = False,
    direct_labels: bool = True,
    style_legend: bool = False,
    style_legend_loc: str = "lower right",
    figsize: tuple[float, float] | None = None,
    figsize_by_size: dict[int, tuple[float, float]] | None = None,
    axis_label_fontsize: float = 12,
    axis_label_fontsize_by_size: dict[int, float] | None = None,
    tick_label_fontsize: float | None = None,
    tick_label_fontsize_by_size: dict[int, float] | None = None,
    title_fontsize: float | None = None,
    title_fontsize_by_size: dict[int, float] | None = None,
    direct_label_fontsize: float | None = None,
    direct_label_fontsize_by_size: dict[int, float] | None = None,
    file_suffix: str = "",
) -> None:
    """Shared sample-gain plotting loop for NLP / PCam."""
    if xticks is None:
        xticks = [1, 5, 10, 15, 20]
    if results_delta.empty:
        return

    results_to_plot = results_delta[results_delta["N"].isin(study_sizes)]
    size_col = "train_size" if "train_size" in results_to_plot.columns else "N"
    unique_sizes = sorted(results_to_plot[size_col].unique())

    for size_value in unique_sizes:
        subset_df = results_to_plot[results_to_plot[size_col] == size_value]
        if subset_df.empty:
            continue

        fig_size = (
            figsize_by_size.get(size_value, figsize)
            if figsize_by_size is not None
            else figsize
        )
        axis_fs = (
            axis_label_fontsize_by_size.get(size_value, axis_label_fontsize)
            if axis_label_fontsize_by_size is not None
            else axis_label_fontsize
        )
        tick_fs = (
            tick_label_fontsize_by_size.get(size_value, tick_label_fontsize)
            if tick_label_fontsize_by_size is not None
            else tick_label_fontsize
        )
        title_fs = (
            title_fontsize_by_size.get(size_value, title_fontsize)
            if title_fontsize_by_size is not None
            else title_fontsize
        )
        direct_fs = (
            direct_label_fontsize_by_size.get(size_value, direct_label_fontsize)
            if direct_label_fontsize_by_size is not None
            else direct_label_fontsize
        )
        display_order = [
            model_display_names.get(model, model)
            for model in models_to_analyze
            if model in subset_df[hue_col].values
        ]
        plot_df = subset_df.copy()
        plot_df[hue_col] = plot_df[hue_col].map(model_display_names).fillna(plot_df[hue_col])
        palette = None
        if model_color_map is not None:
            palette = {
                model_display_names.get(model, model): color
                for model, color in model_color_map.items()
            }

        fig = P.plot_sample_gain_vs_splits(
            plot_df,
            y_col="sample_gain",
            hue_col=hue_col,
            hue_order=display_order,
            palette=palette,
            xticks=xticks,
            style_col=style_col,
            style_order=style_order,
            dashes=dashes,
            # Two-line, top-left title so long prefixes wrap and read from the
            # left edge instead of overflowing a centred anchor (which, combined
            # with clip_on in plot_sample_gain_vs_splits, would otherwise clip).
            title_text=f"{title_prefix}\n$n_{{\\mathrm{{tr}}}}={size_value}$",
            title_text_xy=(0.03, 0.97),
            title_text_ha="left",
            title_text_va="top",
            xlim=(0, max(xticks)),
            direct_labels=direct_labels,
            legend_loc=legend_loc,
            style_legend=style_legend,
            style_legend_loc=style_legend_loc,
            figsize=fig_size,
            axis_label_fontsize=axis_fs,
            tick_label_fontsize=tick_fs,
            title_fontsize=title_fs,
            direct_label_fontsize=direct_fs,
            output_path=(
                f"{plots_dir}/aggregated_test_gain_k_{size_value}"
                f"_delta_linear{file_suffix}.pdf"
            ),
        )
        plt.show()
        plt.close(fig)


def _clean_filename_component(value: object) -> str:
    """Return a filesystem-friendly tag for plot filenames."""
    return (
        str(value)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("=", "-")
    )


def _build_combined_nocv_scores(
    data_nocv_base: pd.DataFrame,
    df_nocv_k: pd.DataFrame,
    model_col: str,
    size_col: str,
    score_raw_col: str,
    bench_col: str,
    scalar_extractor: Callable[[object], float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build k=0 / k>=1 no-CV combined-score tables used by diagnostics."""
    base_scores = data_nocv_base[
        ["seed", model_col, size_col, score_raw_col, bench_col]
    ].copy()
    base_scores["base_score"] = base_scores[score_raw_col].apply(scalar_extractor)
    base_scores["bench_score"] = base_scores[bench_col].apply(scalar_extractor)

    df_hidden = df_nocv_k.merge(
        base_scores[["seed", model_col, size_col, "base_score", "bench_score"]],
        on=["seed", model_col, size_col],
        how="left",
    )
    df_hidden["combined_score"] = (
        df_hidden["base_score"] + df_hidden["k"] * df_hidden["mean_score_k"]
    ) / (df_hidden["k"] + 1)

    df_k0 = base_scores[["seed", model_col, size_col, "base_score", "bench_score"]].copy()
    df_k0["k"] = 0
    df_k0["mean_score_k"] = 0.0
    df_k0["combined_score"] = df_k0["base_score"]

    df_all_k = pd.concat(
        [
            df_hidden[["seed", model_col, size_col, "k", "mean_score_k", "combined_score", "bench_score"]],
            df_k0[["seed", model_col, size_col, "k", "mean_score_k", "combined_score", "bench_score"]],
        ],
        ignore_index=True,
    )
    df_all_k["delta_score"] = df_all_k["combined_score"] - df_all_k["bench_score"]
    df_all_k["multiplier"] = 1 + df_all_k["k"]
    return base_scores, df_all_k


def _bootstrap_binary_curve_ci(
    raw_correct: pd.DataFrame,
    *,
    x_col: str,
    correct_col: str = "correct_pick",
    seed_col: str = "seed",
    n_bootstraps: int = 100,
    random_state: int = 0,
) -> pd.DataFrame:
    """Bootstrap a binary retrieval curve by resampling seeds."""
    if raw_correct.empty:
        return pd.DataFrame(columns=[x_col, "ci_low", "ci_high"])

    matrix = raw_correct.pivot_table(
        index=seed_col,
        columns=x_col,
        values=correct_col,
        aggfunc="mean",
    ).sort_index(axis=1)
    if matrix.empty:
        return pd.DataFrame(columns=[x_col, "ci_low", "ci_high"])

    values = matrix.to_numpy(dtype=float)
    n_seeds = values.shape[0]
    rng = np.random.default_rng(random_state)
    boot_means = np.empty((n_bootstraps, values.shape[1]), dtype=float)
    for boot_idx in range(n_bootstraps):
        sample_idx = rng.integers(0, n_seeds, size=n_seeds)
        boot_means[boot_idx] = np.nanmean(values[sample_idx], axis=0)

    ci = pd.DataFrame({
        x_col: matrix.columns.to_numpy(),
        "ci_low": np.nanpercentile(boot_means, 2.5, axis=0),
        "ci_high": np.nanpercentile(boot_means, 97.5, axis=0),
    })
    return ci


def _bootstrap_binary_mean_ci(
    correct_values: pd.Series,
    *,
    n_bootstraps: int = 100,
    random_state: int = 0,
) -> tuple[float, float] | None:
    """Bootstrap a binary mean by resampling seeds."""
    values = correct_values.dropna().to_numpy(dtype=float)
    if values.size == 0:
        return None
    rng = np.random.default_rng(random_state)
    boot_means = np.empty(n_bootstraps, dtype=float)
    for boot_idx in range(n_bootstraps):
        boot_means[boot_idx] = values[rng.integers(0, values.size, size=values.size)].mean()
    return (
        float(np.nanpercentile(boot_means, 2.5)),
        float(np.nanpercentile(boot_means, 97.5)),
    )


def plot_standard_variance_diagnostics(
    data_nocv_base: pd.DataFrame,
    df_nocv_k: pd.DataFrame,
    data_cv: pd.DataFrame,
    models_to_analyze: list[str],
    target_splits_diag: list[int],
    test_size: float,
    plots_dir: str,
    scalar_extractor: Callable[[object], float],
    model_col: str = "model",
    size_col: str = "study_size",
    score_raw_col: str = "objective_score_test",
    bench_col: str = "objective_score_bench",
    cv_extra_group_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and plot the standard raw/delta variance diagnostics."""
    if cv_extra_group_cols is None:
        cv_extra_group_cols = []

    _, df_diag_all_k = _build_combined_nocv_scores(
        data_nocv_base=data_nocv_base[data_nocv_base[model_col].isin(models_to_analyze)].copy(),
        df_nocv_k=df_nocv_k[df_nocv_k[model_col].isin(models_to_analyze)].copy(),
        model_col=model_col,
        size_col=size_col,
        score_raw_col=score_raw_col,
        bench_col=bench_col,
        scalar_extractor=scalar_extractor,
    )

    cv_diag = data_cv[data_cv[model_col].isin(models_to_analyze)].copy()
    cv_diag["score_scalar"] = cv_diag[score_raw_col].apply(scalar_extractor)
    cv_diag["bench_scalar"] = cv_diag[bench_col].apply(scalar_extractor)
    cv_diag["delta_scalar"] = cv_diag["score_scalar"] - cv_diag["bench_scalar"]
    cv_diag = cv_diag[cv_diag["n_splits"].isin(target_splits_diag)]

    bright_colors = sns.color_palette("mako", n_colors=len(target_splits_diag))
    plot_configs = [
        {
            "nocv_col": "combined_score",
            "cv_col": "score_scalar",
            "file_tag": "acc",
            "x_transform": lambda k, ts: 1 + k * ts,
            "xlabel": "Effective Data Multiplier (Total / Study Size)",
        },
        {
            "nocv_col": "delta_score",
            "cv_col": "delta_scalar",
            "file_tag": "delta",
            "x_transform": lambda k, ts: 1 + k,
            "xlabel": "Effective Test Multiplier (Total Test / Original Test)",
        },
    ]

    for config in plot_configs:
        var_k_df = (
            df_diag_all_k.groupby([model_col, size_col, "k"])[config["nocv_col"]]
            .var()
            .reset_index(name="variance")
        )
        var_k_df["x_axis_val"] = var_k_df["k"].apply(
            lambda k: config["x_transform"](k, test_size)
        )

        group_cols = [model_col, size_col] + list(cv_extra_group_cols) + ["n_splits"]
        var_cv_df = (
            cv_diag.groupby(group_cols)[config["cv_col"]]
            .var()
            .reset_index(name="variance")
        )

        extra_group_cols = [col for col in cv_extra_group_cols if col in var_cv_df.columns]
        for model in models_to_analyze:
            for study_size in sorted(var_k_df[size_col].unique()):
                subset_k = var_k_df[
                    (var_k_df[model_col] == model) & (var_k_df[size_col] == study_size)
                ]
                subset_cv_all = var_cv_df[
                    (var_cv_df[model_col] == model) & (var_cv_df[size_col] == study_size)
                ]
                if subset_k.empty or subset_cv_all.empty:
                    continue

                cv_slices: list[tuple[str, pd.DataFrame]]
                if extra_group_cols:
                    cv_slices = []
                    extra_levels = subset_cv_all[extra_group_cols].drop_duplicates()
                    for _, extra_row in extra_levels.iterrows():
                        subset_cv = subset_cv_all.copy()
                        suffix_parts = []
                        for col in extra_group_cols:
                            subset_cv = subset_cv[subset_cv[col] == extra_row[col]]
                            suffix_parts.append(
                                f"{col}-{_clean_filename_component(extra_row[col])}"
                            )
                        if not subset_cv.empty:
                            cv_slices.append(("_" + "_".join(suffix_parts), subset_cv))
                else:
                    cv_slices = [("", subset_cv_all)]

                for suffix, subset_cv in cv_slices:
                    file_name = (
                        f"variance_diagnostic_{config['file_tag']}_{model}_N{study_size}{suffix}.pdf"
                    )
                    fig = P.plot_variance_diagnostic(
                        subset_k,
                        subset_cv,
                        target_splits=target_splits_diag,
                        x_col="x_axis_val",
                        var_col="variance",
                        xlabel=config["xlabel"],
                        cv_colors=bright_colors,
                        yscale="log",
                        output_path=f"{plots_dir}/{file_name}",
                    )
                    plt.close(fig)

    return df_diag_all_k, cv_diag


def run_retrieval_analysis(
    data_nocv_base: pd.DataFrame,
    df_nocv_k: pd.DataFrame,
    data_cv: pd.DataFrame,
    gt_source_df: pd.DataFrame,
    models_to_analyze: list[str],
    study_sizes: list[int],
    target_splits: list[int],
    model_display_names: dict[str, str],
    plots_dir: str,
    scalar_extractor: Callable[[object], float],
    model_col: str = "model",
    size_col: str = "study_size",
    score_raw_col: str = "objective_score_test",
    bench_col: str = "objective_score_bench",
    gt_score_col: str = "objective_score_bench",
    cv_group_col: str | None = None,
    cv_colors_retrieval: list | None = None,
    y_as_percent: bool = True,
    test_size: float = 0.2,
    retrieval_n_bootstraps: int = 100,
    retrieval_bootstrap_seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the shared retrieval analysis and emit retrieval plots."""
    if cv_colors_retrieval is None:
        cv_colors_retrieval = sns.color_palette("mako", n_colors=len(target_splits))

    _, df_retrieval_all = _build_combined_nocv_scores(
        data_nocv_base=data_nocv_base,
        df_nocv_k=df_nocv_k,
        model_col=model_col,
        size_col=size_col,
        score_raw_col=score_raw_col,
        bench_col=bench_col,
        scalar_extractor=scalar_extractor,
    )
    df_retrieval_all = df_retrieval_all[df_retrieval_all[model_col].isin(models_to_analyze)].copy()

    cv_data_retrieval = data_cv[data_cv[model_col].isin(models_to_analyze)].copy()
    cv_data_retrieval["score_scalar"] = cv_data_retrieval[score_raw_col].apply(scalar_extractor)

    gt_ranking = (
        gt_source_df.groupby([size_col, model_col])[gt_score_col]
        .mean()
        .reset_index()
    )

    curves_df, table_df = compute.compute_retrieval(
        df_nocv_all=df_retrieval_all.rename(columns={size_col: "study_size", model_col: "model"}),
        df_cv=cv_data_retrieval.rename(columns={size_col: "study_size", model_col: "model"}),
        gt_ranking=gt_ranking.rename(columns={size_col: "study_size", model_col: "model"}),
        models_list=models_to_analyze,
        study_sizes=study_sizes,
        target_splits=target_splits,
        model_col="model",
        score_col_nocv="combined_score",
        score_col_cv="score_scalar",
        test_size=test_size,
    )

    print("\nPairwise ranking summary:")
    print(table_df.to_string(index=False))

    if not curves_df.empty:
        for (study_size, best_model, worst_model), curve in curves_df.groupby(
            ["study_size", "best_model", "worst_model"]
        ):
            subset_cv_all = cv_data_retrieval[
                (cv_data_retrieval[size_col] == study_size)
                & (cv_data_retrieval[model_col].isin([best_model, worst_model]))
            ]
            if cv_group_col is not None and cv_group_col in subset_cv_all.columns:
                cv_group_values = list(subset_cv_all[cv_group_col].dropna().unique())
            else:
                cv_group_values = [None]

            for group_value in cv_group_values:
                if group_value is None:
                    subset_cv = subset_cv_all
                    file_suffix = ""
                else:
                    subset_cv = subset_cv_all[subset_cv_all[cv_group_col] == group_value]
                    file_suffix = f"_{cv_group_col}-{_clean_filename_component(group_value)}"

                cv_lines = {}
                cv_ci_lines = {}
                for split in target_splits:
                    split_data = subset_cv[subset_cv["n_splits"] == split]
                    pivot = split_data.pivot_table(
                        index="seed",
                        columns=model_col,
                        values="score_scalar",
                    ).dropna()
                    if not pivot.empty and best_model in pivot.columns and worst_model in pivot.columns:
                        correct = pivot[best_model] > pivot[worst_model]
                        cv_lines[split] = float(correct.mean())
                        ci = _bootstrap_binary_mean_ci(
                            correct,
                            n_bootstraps=retrieval_n_bootstraps,
                            random_state=retrieval_bootstrap_seed + split,
                        )
                        if ci is not None:
                            cv_ci_lines[split] = ci

                subset_nocv = df_retrieval_all[
                    (df_retrieval_all[size_col] == study_size)
                    & (df_retrieval_all[model_col].isin([best_model, worst_model]))
                ]
                raw_curve_ci = pd.DataFrame(columns=["multiplier", "ci_low", "ci_high"])
                if not subset_nocv.empty:
                    pivot_nocv = subset_nocv.pivot_table(
                        index=["seed", "multiplier"],
                        columns=model_col,
                        values="combined_score",
                    ).dropna()
                    if best_model in pivot_nocv.columns and worst_model in pivot_nocv.columns:
                        raw_correct = (
                            (pivot_nocv[best_model] > pivot_nocv[worst_model])
                            .rename("correct_pick")
                            .reset_index()
                        )
                        raw_curve_ci = _bootstrap_binary_curve_ci(
                            raw_correct,
                            x_col="multiplier",
                            n_bootstraps=retrieval_n_bootstraps,
                            random_state=retrieval_bootstrap_seed,
                        )

                fig = P.plot_retrieval_curve(
                    curve_data=curve,
                    cv_lines=cv_lines,
                    best_model=model_display_names.get(best_model, best_model),
                    worst_model=model_display_names.get(worst_model, worst_model),
                    target_splits=target_splits,
                    cv_colors=cv_colors_retrieval,
                    curve_ci_data=raw_curve_ci,
                    cv_ci_lines=cv_ci_lines,
                    y_as_percent=y_as_percent,
                    output_path=(
                        f"{plots_dir}/retrieval_best_{best_model}_vs_{worst_model}"
                        f"_N{study_size}{file_suffix}.pdf"
                    ),
                )
                plt.close(fig)

    return curves_df, table_df


def _merge_optional_frame(
    merged: pd.DataFrame,
    frame: pd.DataFrame | None,
    merge_keys: list[str],
) -> pd.DataFrame:
    """Merge *frame* on the subset of keys it actually carries."""
    if frame is None or frame.empty:
        return merged

    on = [col for col in merge_keys if col in merged.columns and col in frame.columns]
    if not on:
        raise ValueError("No shared merge keys between export frames.")
    return merged.merge(frame, on=on, how="left")


def build_tau_gain_export(
    *,
    tau_df: pd.DataFrame,
    merge_keys: list[str],
    pipeline: str,
    gain20_df: pd.DataFrame | None = None,
    gain20_log1mr2_df: pd.DataFrame | None = None,
    tau_te_df: pd.DataFrame | None = None,
    bench_means_df: pd.DataFrame | None = None,
    sigma2_HO_1_df: pd.DataFrame | None = None,
    sigma2_te_1_df: pd.DataFrame | None = None,
    sigma2_HO_trxi_df: pd.DataFrame | None = None,
    bench_cov_df: pd.DataFrame | None = None,
    study_cov_df: pd.DataFrame | None = None,
    fill_values: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Assemble the canonical cross-pipeline tau/gain export table."""
    merged = tau_df.copy()
    for frame in (
        gain20_df,
        gain20_log1mr2_df,
        tau_te_df,
        bench_means_df,
        sigma2_HO_1_df,
        sigma2_te_1_df,
        sigma2_HO_trxi_df,
        bench_cov_df,
        study_cov_df,
    ):
        merged = _merge_optional_frame(merged, frame, merge_keys)

    merged["pipeline"] = pipeline
    if fill_values:
        for col, value in fill_values.items():
            if col in merged.columns:
                merged[col] = merged[col].fillna(value)
            else:
                merged[col] = value

    merged["tau_over_sigma2_HO_K"] = merged["tau"] / merged["sigma2_HO_K"]
    merged["W_over_B"] = merged["W"] / merged["B"]

    defaults = {
        "algorithm": pd.NA,
        "dataset": pd.NA,
        "study_size": pd.NA,
        "procedure": pd.NA,
        "fixed_split": pd.NA,
        "train_size": pd.NA,
        "sigma2_HO_1": np.nan,
        "sigma2_HO_trxi": np.nan,
        "sigma2_te_1": np.nan,
        "tau_te": np.nan,
        "sigma2_te_K": np.nan,
        "tau_te_lo": np.nan,
        "tau_te_hi": np.nan,
        "sample_gain_K20": np.nan,
        "sample_gain_K20_log1mr2": np.nan,
        "mean_bench_score": np.nan,
        "bench_cov": np.nan,
        "study_cov": np.nan,
        "bench_error_cov": np.nan,
        "bench_error_corr": np.nan,
        "bench_error_var": np.nan,
        "bench_error_rho": np.nan,
        "bench_squared_error_cov": np.nan,
        "bench_squared_error_corr": np.nan,
        "bench_squared_error_var": np.nan,
        "bench_squared_error_rho": np.nan,
        "bench_squared_error_cov_all": np.nan,
        "bench_squared_error_var_all": np.nan,
        "bench_squared_error_rho_all": np.nan,
        "bench_rho_delta_proxy_squared_error_all": np.nan,
        "bench_gain_proxy_squared_error_all": np.nan,
        "bench_prediction_cov_all": np.nan,
        "bench_prediction_var_all": np.nan,
        "bench_prediction_rho_all": np.nan,
        "bench_gain_proxy_prediction_all": np.nan,
        "bench_error_cov_all": np.nan,
        "bench_error_var_all": np.nan,
        "bench_error_rho_all": np.nan,
        "bench_gain_proxy_error_all": np.nan,
        "study_error_cov": np.nan,
        "study_error_corr": np.nan,
        "study_error_var": np.nan,
        "study_error_rho": np.nan,
        "study_squared_error_cov": np.nan,
        "study_squared_error_corr": np.nan,
        "study_squared_error_var": np.nan,
        "study_squared_error_rho": np.nan,
        "study_prediction_cov_all_full": np.nan,
        "study_prediction_var_all_full": np.nan,
        "study_prediction_rho_all_full": np.nan,
        "study_gain_proxy_prediction_all_full": np.nan,
        "study_error_cov_all_full": np.nan,
        "study_error_var_all_full": np.nan,
        "study_error_rho_all_full": np.nan,
        "study_gain_proxy_error_all_full": np.nan,
        "study_squared_error_cov_all_full": np.nan,
        "study_squared_error_var_all_full": np.nan,
        "study_squared_error_rho_all_full": np.nan,
        "study_gain_proxy_squared_error_all_full": np.nan,
        "study_prediction_cov_all_resid_train_membership": np.nan,
        "study_prediction_var_all_resid_train_membership": np.nan,
        "study_prediction_rho_all_resid_train_membership": np.nan,
        "study_gain_proxy_prediction_all_resid_train_membership": np.nan,
        "study_error_cov_all_resid_train_membership": np.nan,
        "study_error_var_all_resid_train_membership": np.nan,
        "study_error_rho_all_resid_train_membership": np.nan,
        "study_gain_proxy_error_all_resid_train_membership": np.nan,
        "study_squared_error_cov_all_resid_train_membership": np.nan,
        "study_squared_error_var_all_resid_train_membership": np.nan,
        "study_squared_error_rho_all_resid_train_membership": np.nan,
        "study_gain_proxy_squared_error_all_resid_train_membership": np.nan,
        "study_prediction_cov_all_oof_intersection": np.nan,
        "study_prediction_var_all_oof_intersection": np.nan,
        "study_prediction_rho_all_oof_intersection": np.nan,
        "study_gain_proxy_prediction_all_oof_intersection": np.nan,
        "study_error_cov_all_oof_intersection": np.nan,
        "study_error_var_all_oof_intersection": np.nan,
        "study_error_rho_all_oof_intersection": np.nan,
        "study_gain_proxy_error_all_oof_intersection": np.nan,
        "study_squared_error_cov_all_oof_intersection": np.nan,
        "study_squared_error_var_all_oof_intersection": np.nan,
        "study_squared_error_rho_all_oof_intersection": np.nan,
        "study_gain_proxy_squared_error_all_oof_intersection": np.nan,
        "study_nll_rho_all_oof_intersection": np.nan,
        "study_error01_rho_all_oof_intersection": np.nan,
        "study_oof_intersection_pairs_all": np.nan,
        "study_oof_intersection_mean_size_all": np.nan,
        "split_index": np.nan,
        "split_train_size": np.nan,
        "split_test_size": np.nan,
        "split_train_overlap_prev_frac": np.nan,
        "split_test_overlap_prev_frac": np.nan,
        "split_train_overlap_first_frac": np.nan,
        "split_test_overlap_first_frac": np.nan,
        "split_train_overlap_prev_count": np.nan,
        "split_test_overlap_prev_count": np.nan,
        "split_train_overlap_first_count": np.nan,
        "split_test_overlap_first_count": np.nan,
    }
    for col in CANONICAL_TAU_GAIN_COLUMNS:
        if col in merged.columns:
            continue
        merged[col] = defaults.get(col, pd.NA)

    return merged[CANONICAL_TAU_GAIN_COLUMNS]
