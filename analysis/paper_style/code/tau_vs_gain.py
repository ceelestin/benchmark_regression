"""
tau_vs_gain.py — pooled cross-pipeline scatter gallery for sample gain @ K=20.

Reads the canonical per-pipeline parquet files written by ``simulated.py``,
``nlp.py`` and ``pcam.py`` (schema: ``pipeline_common.CANONICAL_TAU_GAIN_COLUMNS``)
and produces a suite of scatter plots. Every scatter uses the same visual
encoding via ``plot.plot_tau_vs_gain_encoded`` (called with ``sat_col=None``
throughout):

  - marker → algorithm family (hyperparameter variants share a marker)
  - marker size → ``size_for_plot`` = train_size ?? study_size (log-scaled)
  - colour → algorithm (consistent palette across pipelines)

Points produced with a RepeatedKFold CV design are excluded at load time.

The gallery groups roughly as (paper notation: $\\hat\\sigma^2_{HO|K}$ from
ANOVA on a $K$-fold CV run, $\\hat\\sigma^2_{HO|1}$ from across-seed HO
replicates, $\\hat\\sigma^2_{HO|tr,xi}$ from within-seed HO splits; ``te``
variants are the bench-adjusted analogues):

- **Raw-score reference scatters**: $\\hat\\tau$ vs sample gain at $K=20$ in
  both the native metric and $\\log(1-R^2)$ (simulated only).
- **Raw ↔ δ shrinkage pair**: $\\hat\\tau$ vs $\\hat\\tau^{te}$,
  $\\hat\\sigma^2_{HO|K}$ vs $\\hat\\sigma^{2,te}_{|K}$, and the
  variance-vs-covariance shrinkage ratio scatter.
- **Non-benchmark $\\tau^{te}$ proxies**: exploratory log-linear fits that
  reconstruct $\\tau^{te}$ from raw-only observables ($\\sigma^2_{HO|tr,xi}$
  and $\\hat\\tau$, $\\hat W/\\hat B$).
- **Core predictor scatters**: δ-based predicted vs observed (Plot F),
  raw-score predicted (Plot F-raw), within-seed raw-score predicted (Plot J),
  and the fully non-benchmark predictor (Plot P) that combines
  $\\hat\\sigma^2_{HO|tr,xi}$ with the fitted $\\tau^{te}$ proxy.
- **Single-knob decision diagnostic**: $\\rho_\\delta = \\tau^{te}/\\sigma^{2,te}_{|K}$
  vs observed sample gain with the $K/(1+(K-1)\\rho_\\delta)$ reference curve
  overlaid (Plot N).
- **Outlier diagnostic**: $\\sigma^2_{HO|1}/\\sigma^2_{HO|tr,xi}$ vs Plot-N
  residual (Plot O), testing whether seed-level training noise explains
  the scatter around the reference curve.
- **Exploratory A–M**: raw-score ANOVA quantities vs observed sample gain.

Which plots display inline is controlled by ``PLOT_DISPLAY`` below. Plots with
``False`` entries are still saved, but do not call ``plt.show()``.
"""

# %%
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gain_relationships as G
import pipeline_common as C
import plot as P
import proxy_tau_te as T

DEFAULT_RESULTS_DIR = "results"
DEFAULT_PLOTS_DIR = "plots/cross_pipeline"

PIPELINES = ["simulated", "nlp", "pcam"]
K_DEFAULT = 20

# Manual display switchboard. Set a plot id to True to show it inline in
# VSCode / Jupyter; False means "save only".
PLOT_DISPLAY = {
    # Requested primary figures.
    "tau_vs_tau_te": True,
    "sigma_HO_K_vs_sigma_te_K": True,
    "tau_te_proxy_baseline_trxi": True,
    "tau_te_proxy_baseline_tau_wb": True,
    "tau_te_proxy_combined_raw": True,
    "tau_te_proxy_shrinkage": True,
    "predicted_nonbench_best": True,
    "sample_gain_proxy_quantification": True,
    "predicted_delta": True,
    "rho_delta_vs_gain": True,
    "bench_cov_vs_gain": True,
    "log10_bench_cov_vs_gain": True,
    "bench_cov_loglog_fit": True,
    "bench_cov_vs_gain_stratified": True,
    "study_cov_vs_gain": True,
    "log10_study_cov_vs_gain": True,
    "study_cov_loglog_fit": True,
    "study_cov_vs_gain_stratified": True,
    "bench_error_cov_loglog_fit": False,
    "bench_error_cov_vs_gain_stratified": False,
    "bench_error_corr_loglog_fit": False,
    "bench_error_corr_vs_gain_stratified": False,
    "bench_error_var_loglog_fit": False,
    "bench_error_var_vs_gain_stratified": False,
    "bench_error_rho_loglog_fit": False,
    "bench_error_rho_vs_gain_stratified": False,
    "study_error_cov_loglog_fit": False,
    "study_error_cov_vs_gain_stratified": False,
    "study_error_corr_loglog_fit": False,
    "study_error_corr_vs_gain_stratified": False,
    "study_error_var_loglog_fit": False,
    "study_error_var_vs_gain_stratified": False,
    "study_error_rho_loglog_fit": False,
    "study_error_rho_vs_gain_stratified": False,
    "bench_squared_error_cov_loglog_fit": False,
    "bench_squared_error_cov_vs_gain_stratified": False,
    "bench_squared_error_corr_loglog_fit": False,
    "bench_squared_error_corr_vs_gain_stratified": False,
    "bench_squared_error_var_loglog_fit": False,
    "bench_squared_error_var_vs_gain_stratified": False,
    "bench_squared_error_rho_loglog_fit": False,
    "bench_squared_error_rho_vs_gain_stratified": False,
    "bench_squared_error_rho_all_loglog_fit": True,
    "bench_squared_error_rho_all_vs_gain_stratified": False,
    "bench_rho_delta_proxy_squared_error_all_loglog_fit": True,
    "bench_rho_delta_proxy_squared_error_all_vs_gain_stratified": False,
    "bench_gain_proxy_squared_error_all_loglog_fit": True,
    "bench_gain_proxy_squared_error_all_vs_gain_stratified": False,
    "study_squared_error_cov_loglog_fit": False,
    "study_squared_error_cov_vs_gain_stratified": False,
    "study_squared_error_corr_loglog_fit": False,
    "study_squared_error_corr_vs_gain_stratified": False,
    "study_squared_error_var_loglog_fit": False,
    "study_squared_error_var_vs_gain_stratified": False,
    "study_squared_error_rho_loglog_fit": False,
    "study_squared_error_rho_vs_gain_stratified": False,
    "bench_cov_times_error_rho_loglog_fit": False,
    "bench_cov_times_error_rho_vs_gain_stratified": False,
    "bench_cov_times_error_rho_icc_curve": True,
    "predicted_gain_bench_cov_error_rho_icc": True,
    "study_cov_times_error_rho_loglog_fit": False,
    "study_cov_times_error_rho_vs_gain_stratified": False,
    "bench_cov_times_squared_error_rho_loglog_fit": False,
    "bench_cov_times_squared_error_rho_vs_gain_stratified": False,
    "bench_cov_times_squared_error_rho_icc_curve": True,
    "predicted_gain_bench_cov_squared_error_rho_icc": True,
    "study_cov_times_squared_error_rho_loglog_fit": False,
    "study_cov_times_squared_error_rho_vs_gain_stratified": False,
    # Extra diagnostics, saved silently by default.
    "tau_vs_gain_raw": False,
    "tau_vs_gain_log1mr2": False,
    "tau_ratio_vs_sigma_ratio": False,
    "predicted_raw": False,
    "predicted_trxi": False,
    "tau_over_sigma2_HO_K_vs_gain": False,
    "W_over_B_vs_gain": False,
    "pairplot_anova_vs_gain": False,
    "B_vs_gain": False,
    "tau_over_sigma2_HO_K_vs_gain_by_N": False,
    "tau_over_sigma2_HO_K_vs_gain_by_N_no_sim": False,
    "seed_noise_inflation_vs_resid": False,
    "sigma2_HO_1_over_sigma2_HO_K_vs_gain": False,
    "sigma2_HO_1_vs_gain": False,
    "bench_score_vs_gain": True,
    "sigma2_HO_trxi_vs_gain": False,
    "sigma2_HO_trxi_over_sigma2_HO_K_vs_gain": False,
    "sigma2_HO_1_across_vs_within": False,
    "relationship_live_prefix_curve": True,
    "relationship_predicted_benchmark": True,
    "relationship_predicted_study": True,
    "relationship_predicted_nonbench_live": True,
    "relationship_family_benchmark": True,
    "relationship_family_study": True,
    "relationship_family_nonbench_live": True,
    "relationship_mlp_benchmark": True,
    "relationship_mlp_study": True,
    "relationship_mlp_nonbench_live": True,
    "followup_exact_predicted_linear1000": True,
    "followup_exact_predicted_refit": True,
    "followup_exact_family_correlations": True,
    "followup_exact_fixed_split_difference": True,
    "study_only_redundancy_rule": True,
    "study_redundancy_high_gain_probability": True,
    "study_score_transform_comparison": True,
    "study_only_search_frontier": True,
    "study_only_metric_comparison": True,
    "study_only_predictions_loo": True,
}

ERROR_COVARIANCE_SIGNAL_SPECS = [
    (
        "bench_error_cov",
        r"$\widehat{\mathrm{Cov}}_{X_{\mathrm{bench}}}(e_{f_1},e_{f_2})$",
        "bench error covariance",
    ),
    (
        "bench_error_corr",
        r"$\widehat{\mathrm{Corr}}_{X_{\mathrm{bench}}}(e_{f_1},e_{f_2})$",
        "bench error correlation",
    ),
    (
        "bench_error_var",
        r"$\widehat{\mathrm{Var}}_{X_{\mathrm{bench}}}(e_f)$",
        "bench error variance",
    ),
    (
        "bench_error_rho",
        r"$\widehat\rho_{X_{\mathrm{bench}},\mathrm{err}}$",
        "bench error ICC-style ratio",
    ),
    (
        "bench_squared_error_cov",
        r"$\widehat{\mathrm{Cov}}_{X_{\mathrm{bench}}}(e_{f_1}^2,e_{f_2}^2)$",
        "bench squared-error covariance",
    ),
    (
        "bench_squared_error_corr",
        r"$\widehat{\mathrm{Corr}}_{X_{\mathrm{bench}}}(e_{f_1}^2,e_{f_2}^2)$",
        "bench squared-error correlation",
    ),
    (
        "bench_squared_error_var",
        r"$\widehat{\mathrm{Var}}_{X_{\mathrm{bench}}}(e_f^2)$",
        "bench squared-error variance",
    ),
    (
        "bench_squared_error_rho",
        r"$\widehat\rho_{X_{\mathrm{bench}},e^2}$",
        "bench squared-error ICC-style ratio",
    ),
    (
        "study_error_cov",
        r"$\widehat{\mathrm{Cov}}_{X_{\mathrm{study}}}(e_{f_1},e_{f_2})$",
        "study error covariance",
    ),
    (
        "study_error_corr",
        r"$\widehat{\mathrm{Corr}}_{X_{\mathrm{study}}}(e_{f_1},e_{f_2})$",
        "study error correlation",
    ),
    (
        "study_error_var",
        r"$\widehat{\mathrm{Var}}_{X_{\mathrm{study}}}(e_f)$",
        "study error variance",
    ),
    (
        "study_error_rho",
        r"$\widehat\rho_{X_{\mathrm{study}},\mathrm{err}}$",
        "study error ICC-style ratio",
    ),
    (
        "study_squared_error_cov",
        r"$\widehat{\mathrm{Cov}}_{X_{\mathrm{study}}}(e_{f_1}^2,e_{f_2}^2)$",
        "study squared-error covariance",
    ),
    (
        "study_squared_error_corr",
        r"$\widehat{\mathrm{Corr}}_{X_{\mathrm{study}}}(e_{f_1}^2,e_{f_2}^2)$",
        "study squared-error correlation",
    ),
    (
        "study_squared_error_var",
        r"$\widehat{\mathrm{Var}}_{X_{\mathrm{study}}}(e_f^2)$",
        "study squared-error variance",
    ),
    (
        "study_squared_error_rho",
        r"$\widehat\rho_{X_{\mathrm{study}},e^2}$",
        "study squared-error ICC-style ratio",
    ),
]

DERIVED_COVARIANCE_SIGNAL_SPECS = [
    (
        "bench_cov_times_error_rho",
        (
            r"$\widehat{\mathrm{Cov}}_{X_{\mathrm{bench}}}"
            r"(\hat y_{f_1},\hat y_{f_2})"
            r"\,\widehat\rho_{X_{\mathrm{bench}},\mathrm{err}}$"
        ),
        "bench prediction covariance times bench error redundancy",
    ),
    (
        "study_cov_times_error_rho",
        (
            r"$\widehat{\mathrm{Cov}}_{X_{\mathrm{study}}}"
            r"(\hat y_{f_1},\hat y_{f_2})"
            r"\,\widehat\rho_{X_{\mathrm{study}},\mathrm{err}}$"
        ),
        "study prediction covariance times study error redundancy",
    ),
    (
        "bench_cov_times_squared_error_rho",
        (
            r"$\widehat{\mathrm{Cov}}_{X_{\mathrm{bench}}}"
            r"(\hat y_{f_1},\hat y_{f_2})"
            r"\,\widehat\rho_{X_{\mathrm{bench}},e^2}$"
        ),
        "bench prediction covariance times bench squared-error redundancy",
    ),
    (
        "study_cov_times_squared_error_rho",
        (
            r"$\widehat{\mathrm{Cov}}_{X_{\mathrm{study}}}"
            r"(\hat y_{f_1},\hat y_{f_2})"
            r"\,\widehat\rho_{X_{\mathrm{study}},e^2}$"
        ),
        "study prediction covariance times study squared-error redundancy",
    ),
]

BENCHMARK_ONLY_SIGNAL_SPECS = [
    (
        "bench_squared_error_rho_all",
        r"$\widehat\rho_{X_{\mathrm{bench}},e^2}^{\mathrm{all\ folds}}$",
        "all-fold bench squared-error ICC",
    ),
    (
        "bench_rho_delta_proxy_squared_error_all",
        r"$\widehat\rho_{\delta,B}^{\mathrm{proxy}}$",
        "benchmark-only delta-ICC proxy from all folds",
    ),
    (
        "bench_gain_proxy_squared_error_all",
        r"$\widehat g_B(K)$",
        "benchmark-only ICC gain proxy from all folds",
    ),
]

LEGACY_TAU_GAIN_RENAMES = {
    # Split-count legacy (before the "K" schema refactor).
    "s": "K",
    "sample_gain_s20_delta": "sample_gain_K20",
    "sample_gain_s20_log1mr2_delta": "sample_gain_K20_log1mr2",
    # Estimator-notation legacy (before the paper-aligned rename).
    "tau_raw": "tau",
    "W_raw": "W",
    "B_raw": "B",
    "sigma2_raw": "sigma2_HO_K",
    "sigma2_nocv_raw": "sigma2_HO_1",
    "sigma2_nocv_within_raw": "sigma2_HO_trxi",
    "sigma2_delta": "sigma2_te_K",
    "sigma2_nocv_delta": "sigma2_te_1",
    "tau_raw_over_sigma2_raw": "tau_over_sigma2_HO_K",
    "W_raw_over_B_raw": "W_over_B",
    "sample_gain_K20_delta": "sample_gain_K20",
    "sample_gain_K20_log1mr2_delta": "sample_gain_K20_log1mr2",
}


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.astype(float)
    num = numerator.astype(float)
    return pd.Series(
        np.where(np.isfinite(denom) & (denom != 0), num / denom, np.nan),
        index=numerator.index,
    )


def _add_identity_line(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    linestyle: str = ":",
) -> None:
    """Overlay the y=x reference line over the shared numeric support."""
    valid = data[[x_col, y_col]].dropna()
    if valid.empty:
        return

    lo = float(min(valid[x_col].min(), valid[y_col].min()))
    hi = float(max(valid[x_col].max(), valid[y_col].max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return

    if lo <= 0 and hi > 0:
        lo = min(
            float(valid.loc[valid[x_col] > 0, x_col].min()) if (valid[x_col] > 0).any() else hi,
            float(valid.loc[valid[y_col] > 0, y_col].min()) if (valid[y_col] > 0).any() else hi,
        )
    ax.plot([lo, hi], [lo, hi], color="black", linestyle=linestyle, linewidth=1.0, alpha=0.6)


def _reserve_right_legend_space(fig: plt.Figure) -> None:
    """Widen figures that carry legends anchored outside the axes."""
    width, height = fig.get_size_inches()
    if width < 9.25:
        fig.set_size_inches(9.25, height, forward=True)


def _finish_plot(
    fig: plt.Figure,
    plot_id: str,
    output_path: str | None = None,
) -> None:
    """Save, optionally show, then close a figure according to PLOT_DISPLAY."""
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved {output_path}")
    if PLOT_DISPLAY.get(plot_id, False):
        try:
            from IPython import get_ipython
            from IPython.display import display
        except ImportError:
            get_ipython = None
            display = None
        if get_ipython is not None and get_ipython() is not None:
            display(fig)
        else:
            plt.show()
    plt.close(fig)


def _plot_tau_te_proxy_scatter(
    df: pd.DataFrame,
    proxy_col: str,
    plot_id: str,
    xlabel: str,
    title: str,
    output_path: str,
) -> None:
    """Plot one tau_te proxy against observed tau_te."""
    valid = df.dropna(subset=[proxy_col, "tau_te"])
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    P.plot_tau_vs_gain_encoded(
        df,
        x_col=proxy_col,
        y_col="tau_te",
        sat_col=None,
        size_col="size_for_plot",
        size_label="data size",
        xlabel=xlabel,
        ylabel=r"Observed $\hat\tau^{te}$",
        ax=ax,
        title=title,
    )
    _add_identity_line(ax, valid, proxy_col, "tau_te")
    if (
        not valid.empty
        and (valid[proxy_col] > 0).all()
        and (valid["tau_te"] > 0).all()
    ):
        ax.set_xscale("log")
        ax.set_yscale("log")
    _reserve_right_legend_space(fig)
    _finish_plot(fig, plot_id, output_path)


def _plot_sample_gain_proxy_quantification(
    metrics: pd.DataFrame,
    output_path: str,
    top_n: int | None = None,
) -> None:
    """Bar plot of sample-gain prediction error, including benchmark baseline."""
    if metrics.empty:
        return

    work = metrics.copy()
    work["display_error"] = work["rmse_log_gain"]
    work = work.dropna(subset=["display_error"])
    if work.empty:
        return

    work = work.sort_values(["display_error", "proxy"]).copy()
    if top_n is not None:
        work = work.head(top_n).copy()
    work["label"] = work["proxy"].replace({
        "benchmark_delta_formula": "benchmark delta formula",
        "tau_formula_baseline_trxi": "tau formula: trxi",
        "tau_formula_baseline_tau_wb": "tau formula: tau + W/B",
        "tau_formula_combined_raw": "tau formula: combined raw",
        "tau_formula_shrinkage": "tau formula: shrinkage",
        "local_raw_HO1_over_raw_cv": "rowwise: HO1 / raw CV",
        "local_trxi_over_raw_cv": "rowwise: tr,xi / raw CV",
        "local_cv_variance_curve": "rowwise: CV curve",
        "local_HO1_denominator": "rowwise: HO1 denominator",
        "local_trxi_denominator": "rowwise: tr,xi denominator",
        "local_HO1_denominator_tau_pos": "rowwise: HO1 denom, tau+",
        "local_trxi_denominator_tau_pos": "rowwise: tr,xi denom, tau+",
        "local_trxi_tau_shrink_trxi_over_HOK": "rowwise: tr,xi shrink HO|K",
        "local_trxi_tau_shrink_trxi_over_HO1": "rowwise: tr,xi shrink HO|1",
        "local_trxi_tau_shrink_inv_WB": "rowwise: tr,xi tau/(W/B)",
        "local_trxi_tau_shrink_one_plus_WB": "rowwise: tr,xi tau/(1+W/B)",
        "local_trxi_tau_minus_seed_noise": "rowwise: tau minus seed noise",
        "local_trxi_tau_minus_seed_noise_over_K": "rowwise: tau minus seed/K",
        "local_trxi_tau_025_trxi_over_HOK": "rowwise: 0.25 tau tr,xi/HO|K",
        "local_geo_HOK_trxi_blend25_HO1_75_trxi_tau_060_trxi_over_HOK": (
            "rowwise: geo var + blend denom"
        ),
    })
    colors = np.where(
        work["family"].eq("benchmark_baseline"),
        "#4C72B0",
        "#55A868",
    )

    fig_h = max(4.2, 0.36 * len(work) + 1.4)
    fig, ax = plt.subplots(figsize=(8.2, fig_h))
    y_pos = np.arange(len(work))
    bars = ax.barh(y_pos, work["display_error"], color=colors, alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(work["label"])
    ax.invert_yaxis()
    ax.set_xlabel(r"RMSE of $\log_{10}$ predicted sample gain")
    ax.set_title("Pipeline-weighted rowwise sample-gain formula error")
    ax.grid(axis="x", alpha=0.25)

    for bar, (_, row) in zip(bars, work.iterrows()):
        source = "benchmark" if row["family"] == "benchmark_baseline" else "rowwise"
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {bar.get_width():.3f} ({source})",
            va="center",
            ha="left",
            fontsize=8,
        )
    ax.margins(x=0.12)
    _finish_plot(fig, "sample_gain_proxy_quantification", output_path)


def _pearson_corr(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    x_arr = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    y_arr = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    if len(x_arr) < 3 or np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _spearman_corr(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    x_rank = pd.Series(x).rank(method="average")
    y_rank = pd.Series(y).rank(method="average")
    return _pearson_corr(x_rank, y_rank)


def _relationship_subset_for_scope(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    scope: str,
    K: int,
) -> pd.DataFrame:
    """Return selected prediction rows for one simulated relationship scope."""
    sub = predictions[predictions["scope"] == scope].copy()
    if sub.empty:
        return sub
    if scope == G.SCOPE_NONBENCH_LIVE and "prefix_j" in sub.columns:
        prefix_values = pd.to_numeric(sub["prefix_j"], errors="coerce")
        if (prefix_values == K).any():
            sub = sub.loc[prefix_values == K].copy()
        else:
            selected_metrics = metrics[
                (metrics["scope"] == scope)
                & (metrics["is_selected"].astype(bool))
            ].copy()
            if not selected_metrics.empty:
                family_col = (
                    "min_model_family_signed_pearson_log_gain"
                    if "min_model_family_signed_pearson_log_gain" in selected_metrics.columns
                    else "min_model_family_abs_pearson_log_gain"
                )
                selected_metrics = selected_metrics.sort_values(
                    [
                        "passes_acceptance_gate",
                        family_col,
                        "abs_corr_log_gain",
                    ],
                    ascending=[False, False, False],
                )
                best_prefix = selected_metrics["prefix_j"].iloc[0]
                sub = sub.loc[prefix_values == best_prefix].copy()
    return sub


def _plot_relationship_family_correlations(
    pred_df: pd.DataFrame,
    *,
    title: str,
    output_path: str,
    plot_id: str,
) -> None:
    """Forest-style plot of prediction robustness by model family."""
    valid = pred_df.dropna(
        subset=["predicted_sample_gain_K20", "sample_gain_K20", "model_family"]
    ).copy()
    valid = valid[
        (valid["predicted_sample_gain_K20"] > 0)
        & (valid["sample_gain_K20"] > 0)
    ]
    if len(valid) < 3:
        return

    rows = []
    for label, sub in [("overall", valid)] + list(valid.groupby("model_family")):
        if len(sub) < 3:
            continue
        log_pred = np.log10(sub["predicted_sample_gain_K20"])
        log_obs = np.log10(sub["sample_gain_K20"])
        rows.append({
            "label": str(label),
            "n": len(sub),
            "pearson_log": _pearson_corr(log_pred, log_obs),
            "spearman": _spearman_corr(
                sub["predicted_sample_gain_K20"],
                sub["sample_gain_K20"],
            ),
        })
    if not rows:
        return

    family_df = pd.DataFrame(rows)
    family_df["display"] = family_df.apply(
        lambda row: f"{row['label']}  (n={int(row['n'])})", axis=1
    )
    y_pos = np.arange(len(family_df))[::-1]

    fig, ax = plt.subplots(figsize=(8.0, max(3.2, 0.45 * len(family_df) + 1.4)))
    ax.scatter(
        family_df["pearson_log"],
        y_pos,
        s=72,
        marker="o",
        color="#4C72B0",
        edgecolor="black",
        linewidth=0.5,
        label="Pearson r (log-log)",
    )
    ax.scatter(
        family_df["spearman"],
        y_pos,
        s=72,
        marker="s",
        color="#DD8452",
        edgecolor="black",
        linewidth=0.5,
        label="Spearman rho",
    )
    ax.axvline(0.65, color="black", linestyle="--", linewidth=1.0, alpha=0.55)
    ax.axvline(-0.65, color="black", linestyle="--", linewidth=1.0, alpha=0.35)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.45)
    ax.set_xlim(-1.05, 1.05)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(family_df["display"])
    ax.set_xlabel("correlation with observed sample_gain_K20")
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, axis="x", linestyle=":", alpha=0.35)
    _finish_plot(fig, plot_id, output_path)


def _plot_relationship_mlp_highlight(
    pred_df: pd.DataFrame,
    *,
    title: str,
    output_path: str,
    plot_id: str,
) -> None:
    """Predicted-vs-observed scatter with MLP emphasized."""
    valid = pred_df.dropna(
        subset=["predicted_sample_gain_K20", "sample_gain_K20"]
    ).copy()
    valid = valid[
        (valid["predicted_sample_gain_K20"] > 0)
        & (valid["sample_gain_K20"] > 0)
    ]
    if len(valid) < 3:
        return
    is_mlp = valid.get("model_family", pd.Series("", index=valid.index)).astype(str).str.contains(
        "MLP", na=False, regex=False
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.scatter(
        valid.loc[~is_mlp, "predicted_sample_gain_K20"],
        valid.loc[~is_mlp, "sample_gain_K20"],
        s=34,
        color="#4C72B0",
        alpha=0.75,
        edgecolor="black",
        linewidth=0.4,
        label="non-MLP",
    )
    if is_mlp.any():
        ax.scatter(
            valid.loc[is_mlp, "predicted_sample_gain_K20"],
            valid.loc[is_mlp, "sample_gain_K20"],
            s=54,
            color="#C44E52",
            marker="D",
            alpha=0.9,
            edgecolor="black",
            linewidth=0.5,
            label="MLP",
        )
    lo = min(valid["predicted_sample_gain_K20"].min(), valid["sample_gain_K20"].min())
    hi = max(valid["predicted_sample_gain_K20"].max(), valid["sample_gain_K20"].max())
    grid = np.linspace(lo, hi, 100)
    ax.plot(grid, grid, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Predicted sample gain at $K=20$")
    ax.set_ylabel(r"Observed sample gain at $K=20$")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", linestyle=":", alpha=0.3)
    _finish_plot(fig, plot_id, output_path)


def _plot_simulated_relationship_outputs(
    results_dir: str,
    plots_dir: str,
    K: int,
) -> None:
    """Plot simulated relationship-search artifacts when they are available."""
    metrics_path = os.path.join(results_dir, "simulated_gain_relationship_metrics.csv")
    predictions_path = os.path.join(
        results_dir,
        "simulated_gain_relationship_predictions.parquet",
    )
    if not os.path.exists(metrics_path):
        print(
            "Skipping simulated gain-relationship plots: "
            f"{metrics_path} not found."
        )
        return

    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        print("Skipping simulated gain-relationship plots: metrics table is empty.")
        return
    if "is_selected" in metrics.columns:
        if pd.api.types.is_bool_dtype(metrics["is_selected"]):
            metrics["is_selected"] = metrics["is_selected"].astype(bool)
        else:
            metrics["is_selected"] = (
                metrics["is_selected"].astype(str).str.lower().isin(["true", "1", "yes"])
            )
    selected = metrics[metrics["is_selected"]].copy()

    live = selected[selected["scope"] == G.SCOPE_NONBENCH_LIVE].copy()
    if not live.empty:
        live = live.sort_values("prefix_j")
        family_col = (
            "min_model_family_signed_pearson_log_gain"
            if "min_model_family_signed_pearson_log_gain" in live.columns
            else "min_model_family_abs_pearson_log_gain"
        )
        mlp_col = (
            "mlp_signed_pearson_log_gain"
            if "mlp_signed_pearson_log_gain" in live.columns
            else "mlp_abs_pearson_log_gain"
        )
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.plot(
            live["prefix_j"],
            live["abs_corr_log_gain"],
            marker="o",
            label="overall |Pearson r| (log-log)",
        )
        ax.plot(
            live["prefix_j"],
            live[family_col],
            marker="s",
            label="minimum signed model-family r",
        )
        if live[mlp_col].notna().any():
            ax.plot(
                live["prefix_j"],
                live[mlp_col],
                marker="D",
                label="MLP signed r",
            )
        ax.axhline(0.85, color="black", linestyle="--", linewidth=1.0, alpha=0.45)
        ax.axhline(0.65, color="black", linestyle=":", linewidth=1.0, alpha=0.65)
        ax.set_xlabel("Prefix split count j")
        ax.set_ylabel("Selected relationship correlation")
        ax.set_ylim(-1.02, 1.02)
        ax.set_title(
            r"Live non-benchmark prediction of final $K=20$ sample gain"
        )
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.35)
        _finish_plot(
            fig,
            "relationship_live_prefix_curve",
            os.path.join(plots_dir, "relationship_live_prefix_curve.pdf"),
        )

    if not os.path.exists(predictions_path):
        print(
            "Skipping predicted-vs-observed relationship plots: "
            f"{predictions_path} not found."
        )
        return

    predictions = pd.read_parquet(predictions_path)
    if predictions.empty:
        return

    scope_titles = {
        G.SCOPE_BENCHMARK: "Benchmark-aware selected formula",
        G.SCOPE_STUDY: "Study-only selected formula",
        G.SCOPE_NONBENCH_LIVE: f"Live non-benchmark selected formula (j={K})",
    }
    for scope, title in scope_titles.items():
        sub = _relationship_subset_for_scope(predictions, metrics, scope, K)
        if sub.empty:
            continue
        safe_scope = scope.replace("_", "-")
        fig_pred = P.plot_predicted_vs_observed_gain(
            sub,
            x_col="predicted_sample_gain_K20",
            y_col="sample_gain_K20",
            sat_col=None,
            size_col="size_for_plot",
            title=rf"{title}: predicted vs observed sample gain at $K=20$",
            output_path=os.path.join(
                plots_dir,
                f"relationship_predicted_{safe_scope}.pdf",
            ),
        )
        _reserve_right_legend_space(fig_pred)
        _finish_plot(fig_pred, f"relationship_predicted_{scope}")

        _plot_relationship_family_correlations(
            sub,
            title=f"{title}: robustness by model family",
            output_path=os.path.join(
                plots_dir,
                f"relationship_family_correlations_{safe_scope}.pdf",
            ),
            plot_id=f"relationship_family_{scope}",
        )
        _plot_relationship_mlp_highlight(
            sub,
            title=f"{title}: MLP stress diagnostic",
            output_path=os.path.join(
                plots_dir,
                f"relationship_mlp_highlight_{safe_scope}.pdf",
            ),
            plot_id=f"relationship_mlp_{scope}",
        )


def _followup_exact_metrics(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    """Score one follow-up exact-probe prediction column."""
    valid = df.dropna(subset=[pred_col, "sample_gain_K20"]).copy()
    valid = valid[
        (valid[pred_col] > 0)
        & (valid["sample_gain_K20"] > 0)
    ]
    if len(valid) < 3:
        return {
            "n": int(len(valid)),
            "pearson": np.nan,
            "spearman": np.nan,
            "rmse": np.nan,
            "min_family": np.nan,
        }

    log_pred = np.log10(valid[pred_col].to_numpy(dtype=float))
    log_obs = np.log10(valid["sample_gain_K20"].to_numpy(dtype=float))
    resid = log_pred - log_obs
    family_rs = []
    family_col = "model_family" if "model_family" in valid.columns else "algorithm"
    for _, sub in valid.groupby(family_col):
        if len(sub) < 3:
            continue
        family_rs.append(
            _pearson_corr(
                np.log10(sub[pred_col].to_numpy(dtype=float)),
                np.log10(sub["sample_gain_K20"].to_numpy(dtype=float)),
            )
        )
    return {
        "n": int(len(valid)),
        "pearson": _pearson_corr(log_pred, log_obs),
        "spearman": _spearman_corr(valid[pred_col], valid["sample_gain_K20"]),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "min_family": float(np.nanmin(family_rs)) if family_rs else np.nan,
    }


def _plot_followup_exact_prediction(
    df: pd.DataFrame,
    pred_col: str,
    *,
    title: str,
    xlabel: str,
    output_path: str,
    plot_id: str,
) -> None:
    """Predicted-vs-observed scatter for the exact follow-up K=20 probe."""
    valid = df.dropna(subset=[pred_col, "sample_gain_K20"]).copy()
    valid = valid[
        (valid[pred_col] > 0)
        & (valid["sample_gain_K20"] > 0)
    ]
    if len(valid) < 3:
        print(f"Skipping {plot_id}: not enough valid rows.")
        return

    metrics = _followup_exact_metrics(valid, pred_col)
    families = sorted(valid.get("model_family", valid["algorithm"]).astype(str).unique())
    markers = ["o", "s", "D", "^", "P", "X", "v", "<", ">"]
    marker_map = {
        family: markers[i % len(markers)]
        for i, family in enumerate(families)
    }
    color_map = {False: "#4C72B0", True: "#C44E52"}

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for (fixed_split, family), sub in valid.groupby(["fixed_split", "model_family"]):
        ax.scatter(
            sub[pred_col],
            sub["sample_gain_K20"],
            s=34 + 0.05 * pd.to_numeric(sub["train_size"], errors="coerce").fillna(0),
            marker=marker_map.get(str(family), "o"),
            color=color_map.get(bool(fixed_split), "#777777"),
            alpha=0.72,
            edgecolor="black",
            linewidth=0.35,
            label=f"{family}, fixed={bool(fixed_split)}",
        )

    lo = min(valid[pred_col].min(), valid["sample_gain_K20"].min())
    hi = max(valid[pred_col].max(), valid["sample_gain_K20"].max())
    pad_lo = max(lo * 0.85, 1e-12)
    pad_hi = hi * 1.15
    ax.plot([pad_lo, pad_hi], [pad_lo, pad_hi], "k--", linewidth=1, alpha=0.65)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(pad_lo, pad_hi)
    ax.set_ylim(pad_lo, pad_hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Observed exact sample gain at $K=20$")
    ax.set_title(title)
    ax.text(
        0.03,
        0.97,
        (
            f"n={metrics['n']}\n"
            f"$r_\\log$={metrics['pearson']:.3f}\n"
            f"Spearman={metrics['spearman']:.3f}\n"
            f"RMSE={metrics['rmse']:.3f}\n"
            f"min family r={metrics['min_family']:.3f}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[:12],
        labels[:12],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=7,
        title="family / split",
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.3)
    _reserve_right_legend_space(fig)
    _finish_plot(fig, plot_id, output_path)


def _plot_followup_exact_family_correlations(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """Bar chart of follow-up exact-probe family correlations."""
    pred_specs = [
        ("linear1000_formula_gain", "linear-1000 coefficients"),
        ("refit_benchmark_gain", "follow-up refit"),
    ]
    rows = []
    for pred_col, label in pred_specs:
        if pred_col not in df.columns:
            continue
        for family, sub in df.groupby("model_family"):
            valid = sub.dropna(subset=[pred_col, "sample_gain_K20"])
            valid = valid[(valid[pred_col] > 0) & (valid["sample_gain_K20"] > 0)]
            if len(valid) < 3:
                continue
            rows.append({
                "predictor": label,
                "model_family": family,
                "pearson_log_gain": _pearson_corr(
                    np.log10(valid[pred_col].to_numpy(dtype=float)),
                    np.log10(valid["sample_gain_K20"].to_numpy(dtype=float)),
                ),
            })
    corr = pd.DataFrame(rows)
    if corr.empty:
        print("Skipping follow-up family correlations: no valid rows.")
        return

    families = sorted(corr["model_family"].unique())
    x = np.arange(len(families))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for i, (predictor, sub) in enumerate(corr.groupby("predictor")):
        aligned = (
            sub.set_index("model_family")
            .reindex(families)["pearson_log_gain"]
            .to_numpy(dtype=float)
        )
        ax.bar(
            x + (i - 0.5) * width,
            aligned,
            width=width,
            label=predictor,
            alpha=0.9,
        )
    ax.axhline(0.65, color="black", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=20, ha="right")
    ax.set_ylim(-1, 1)
    ax.set_ylabel(r"Within-family Pearson $r$ on $\log_{10}$ gain")
    ax.set_title("Follow-up exact K=20 benchmark-aware family robustness")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    _finish_plot(fig, "followup_exact_family_correlations", output_path)


def _plot_followup_exact_fixed_split_difference(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """Plot fixed-vs-random split sample-gain differences."""
    keys = ["dataset", "algorithm", "train_size"]
    pivot = df.pivot_table(
        index=keys,
        columns="fixed_split",
        values="sample_gain_K20",
        aggfunc="first",
    )
    if True not in pivot.columns or False not in pivot.columns:
        print("Skipping fixed-split difference plot: missing paired split modes.")
        return
    pivot_cols = list(pivot.columns)
    false_pos = next(i for i, col in enumerate(pivot_cols) if col == False)
    true_pos = next(i for i, col in enumerate(pivot_cols) if col == True)
    diff = pd.DataFrame(
        {
            "random_split": pivot.iloc[:, false_pos].to_numpy(dtype=float),
            "fixed_split": pivot.iloc[:, true_pos].to_numpy(dtype=float),
        },
        index=pivot.index,
    )
    diff["true_minus_false"] = diff["fixed_split"] - diff["random_split"]
    diff = diff.reset_index()
    diff["model_family"] = diff["algorithm"].map(G.model_family)
    if diff.empty:
        return

    families = sorted(diff["model_family"].unique())
    x = np.arange(len(families))
    values_by_family = [
        diff.loc[diff["model_family"].eq(family), "true_minus_false"].to_numpy(dtype=float)
        for family in families
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.boxplot(
        values_by_family,
        positions=x,
        widths=0.5,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.4},
    )
    for i, values in enumerate(values_by_family):
        if len(values) == 0:
            continue
        jitter = np.linspace(-0.13, 0.13, len(values))
        ax.scatter(
            np.full(len(values), x[i]) + jitter,
            values,
            s=18,
            alpha=0.65,
            edgecolor="black",
            linewidth=0.25,
        )
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=20, ha="right")
    ax.set_ylabel("Exact sample gain difference: fixed_split=True minus False")
    ax.set_title("Follow-up exact K=20 fixed-vs-random split effect")
    ax.grid(axis="y", alpha=0.25)
    _finish_plot(fig, "followup_exact_fixed_split_difference", output_path)


def _plot_followup_exact_probe_outputs(
    results_dir: str,
    plots_dir: str,
) -> None:
    """Plot the lightweight follow-up exact K=20 probe, if present."""
    path = os.path.join(results_dir, "followup_exact_k20_probe.csv")
    if not os.path.exists(path):
        print(f"Skipping follow-up exact-probe plots: {path} not found.")
        return

    df = pd.read_csv(path)
    if df.empty:
        print("Skipping follow-up exact-probe plots: table is empty.")
        return
    if "model_family" not in df.columns and "algorithm" in df.columns:
        df["model_family"] = df["algorithm"].map(G.model_family)

    if "linear1000_formula_gain" in df.columns:
        _plot_followup_exact_prediction(
            df,
            "linear1000_formula_gain",
            title="Follow-up exact K=20: linear-1000 benchmark formula transfer",
            xlabel=r"Predicted sample gain from linear-1000 benchmark formula",
            output_path=os.path.join(
                plots_dir,
                "followup_exact_predicted_linear1000.pdf",
            ),
            plot_id="followup_exact_predicted_linear1000",
        )
    if "refit_benchmark_gain" in df.columns:
        _plot_followup_exact_prediction(
            df,
            "refit_benchmark_gain",
            title="Follow-up exact K=20: benchmark formula refit",
            xlabel=r"Predicted sample gain from follow-up refit",
            output_path=os.path.join(
                plots_dir,
                "followup_exact_predicted_refit.pdf",
            ),
            plot_id="followup_exact_predicted_refit",
        )

    _plot_followup_exact_family_correlations(
        df,
        os.path.join(plots_dir, "followup_exact_family_correlations.pdf"),
    )
    _plot_followup_exact_fixed_split_difference(
        df,
        os.path.join(plots_dir, "followup_exact_fixed_split_difference.pdf"),
    )


def _split_signal_text(signal_text: object) -> tuple[str, ...]:
    """Split the serialized relationship signal list."""
    if pd.isna(signal_text):
        return tuple()
    return tuple(part.strip() for part in str(signal_text).split(" + ") if part.strip())


def _study_only_selected_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    """Pick representative study-only search rows for compact plots."""
    min_col = "loo_min_model_family_signed_pearson_log_gain"
    previous_signals = (
        "study_cov_times_error_cov_over_var_all_oof_intersection "
        "+ study_prediction_cov_all_full"
    )
    rows: list[pd.Series] = []
    labels: list[str] = []

    previous = metrics[
        (metrics["scope"] == "strict_study")
        & (metrics["mode"] == "gain_power")
        & (metrics["signals"] == previous_signals)
    ]
    if not previous.empty:
        rows.append(previous.iloc[0])
        labels.append("previous selected")

    calibrated = metrics[
        ~metrics["mode"].astype(str).str.contains("family", na=False)
        & (metrics["mode"] != "direct_signal")
        & (metrics["loo_rmse_log_gain"] < 0.2)
    ].drop_duplicates(["mode", "signals"])
    if not calibrated.empty:
        rows.append(calibrated.sort_values("loo_abs_corr_log_gain", ascending=False).iloc[0])
        labels.append("best global")
        rows.append(calibrated.sort_values(min_col, ascending=False).iloc[0])
        labels.append("best family floor")

    family = metrics[
        metrics["mode"].astype(str).str.contains("family", na=False)
        & (metrics["loo_rmse_log_gain"] < 0.2)
    ].drop_duplicates(["mode", "signals"])
    if not family.empty:
        rows.append(family.sort_values("loo_abs_corr_log_gain", ascending=False).iloc[0])
        labels.append("best family-aware")

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).reset_index(drop=True)
    out.insert(0, "plot_label", labels)
    return out.drop_duplicates(["plot_label", "mode", "signals"])


def _plot_study_only_search_frontier(
    metrics: pd.DataFrame,
    selected: pd.DataFrame,
    output_path: str,
) -> None:
    """Plot global study-only accuracy against the within-family floor."""
    valid = metrics[
        metrics["loo_abs_corr_log_gain"].notna()
        & metrics["loo_min_model_family_signed_pearson_log_gain"].notna()
        & (metrics["loo_rmse_log_gain"] < 0.2)
    ].copy()
    if valid.empty:
        return
    valid = valid.drop_duplicates(["scope", "mode", "signals"])
    colors = {
        "gain_power": "#4C72B0",
        "effective_icc": "#55A868",
        "gain_power_family_intercept": "#C44E52",
        "gain_power_family_slope": "#8172B3",
    }
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for mode, sub in valid.groupby("mode"):
        ax.scatter(
            sub["loo_abs_corr_log_gain"],
            sub["loo_min_model_family_signed_pearson_log_gain"],
            s=28,
            alpha=0.58,
            color=colors.get(mode, "#777777"),
            label=str(mode).replace("gain_power_", "").replace("_", " "),
            edgecolor="none",
        )
    ax.axvline(0.85, color="black", linestyle="--", linewidth=1.0, alpha=0.55)
    ax.axhline(0.65, color="black", linestyle=":", linewidth=1.2, alpha=0.75)
    ax.set_xlabel("LOOCV overall |Pearson r| on log gain")
    ax.set_ylabel("LOOCV minimum signed family r")
    ax.set_title("Study-only formulas: high global fit, weak family robustness")
    ax.set_xlim(max(0.75, valid["loo_abs_corr_log_gain"].min() - 0.03), 1.0)
    y_min = min(
        -0.7,
        float(valid["loo_min_model_family_signed_pearson_log_gain"].min()) - 0.05,
    )
    ax.set_ylim(y_min, 0.72)
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(loc="lower left", fontsize=8)

    for _, row in selected.iterrows():
        x = row.get("loo_abs_corr_log_gain")
        y = row.get("loo_min_model_family_signed_pearson_log_gain")
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        ax.scatter(
            [x],
            [y],
            s=92,
            facecolor="none",
            edgecolor="black",
            linewidth=1.2,
            zorder=5,
        )
        ax.annotate(
            str(row["plot_label"]),
            xy=(x, y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )
    _finish_plot(fig, "study_only_search_frontier", output_path)


def _plot_study_only_metric_comparison(
    selected: pd.DataFrame,
    output_path: str,
) -> None:
    """Compare validation metrics for representative study-only formulas."""
    if selected.empty:
        return
    labels = selected["plot_label"].astype(str).tolist()
    x = np.arange(len(selected))
    width = 0.22
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.4, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    ax = axes[0]
    ax.bar(
        x - width,
        selected["loo_abs_corr_log_gain"],
        width,
        label="overall |r|",
        color="#4C72B0",
    )
    ax.bar(
        x,
        selected["loo_min_model_family_signed_pearson_log_gain"],
        width,
        label="min family r",
        color="#C44E52",
    )
    ax.bar(
        x + width,
        selected["loo_mlp_signed_pearson_log_gain"],
        width,
        label="MLP r",
        color="#55A868",
    )
    ax.axhline(0.85, color="black", linestyle="--", linewidth=1.0, alpha=0.45)
    ax.axhline(0.65, color="black", linestyle=":", linewidth=1.2, alpha=0.65)
    ax.set_ylabel("LOOCV correlation")
    ax.set_title("Representative study-only approximations")
    ax.set_ylim(
        min(
            -0.7,
            float(selected["loo_min_model_family_signed_pearson_log_gain"].min()) - 0.05,
        ),
        1.02,
    )
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)

    ax_rmse = axes[1]
    ax_rmse.bar(x, selected["loo_rmse_log_gain"], color="#8172B3", width=0.45)
    ax_rmse.set_ylabel("LOOCV RMSE")
    ax_rmse.set_xticks(x)
    ax_rmse.set_xticklabels(labels, rotation=16, ha="right")
    ax_rmse.grid(True, axis="y", linestyle=":", alpha=0.35)

    _finish_plot(fig, "study_only_metric_comparison", output_path)


def _plot_study_only_prediction_panels(
    selected: pd.DataFrame,
    results_dir: str,
    output_path: str,
) -> None:
    """Plot LOOCV predicted-vs-observed gain for representative formulas."""
    if selected.empty:
        return
    summary_path = os.path.join(results_dir, "tau_gain_simulated.parquet")
    if not os.path.exists(summary_path):
        return
    try:
        import scripts.search_study_only_relationships as study_search
    except ImportError:
        print(
            "Skipping study-only prediction panels: "
            "scripts.search_study_only_relationships is not importable."
        )
        return

    df = study_search.add_study_extras(pd.read_parquet(summary_path))
    plot_rows = selected.head(4).copy()
    fig, axes = plt.subplots(
        1,
        len(plot_rows),
        figsize=(4.2 * len(plot_rows), 4.2),
        squeeze=False,
    )
    families = sorted(df["model_family"].astype(str).unique())
    color_values = plt.get_cmap("tab10").colors
    palette = {family: color_values[i % len(color_values)] for i, family in enumerate(families)}
    for ax, (_, row) in zip(axes[0], plot_rows.iterrows()):
        signals = _split_signal_text(row["signals"])
        spec = study_search.Spec(
            str(row["scope"]),
            str(row["mode"]),
            signals,
            str(row["formula"]),
            int(row["n_params"]),
        )
        pred = study_search.predict_cv(df, spec)
        valid = df.assign(predicted_gain=pred).dropna(
            subset=["predicted_gain", "sample_gain_K20", "model_family"]
        )
        valid = valid[(valid["predicted_gain"] > 0) & (valid["sample_gain_K20"] > 0)]
        for family, sub in valid.groupby("model_family"):
            ax.scatter(
                sub["predicted_gain"],
                sub["sample_gain_K20"],
                s=38,
                alpha=0.78,
                color=palette.get(str(family), "#777777"),
                edgecolor="black",
                linewidth=0.35,
                label=str(family),
            )
        if not valid.empty:
            lo = float(min(valid["predicted_gain"].min(), valid["sample_gain_K20"].min()))
            hi = float(max(valid["predicted_gain"].max(), valid["sample_gain_K20"].max()))
            ax.plot(
                [lo, hi],
                [lo, hi],
                color="black",
                linestyle="--",
                linewidth=1.0,
                alpha=0.55,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("LOOCV predicted gain")
        ax.set_title(str(row["plot_label"]))
        ax.grid(True, which="both", linestyle=":", alpha=0.32)
        ax.text(
            0.04,
            0.96,
            (
                f"r={row['loo_abs_corr_log_gain']:.3f}\n"
                f"min fam={row['loo_min_model_family_signed_pearson_log_gain']:.3f}\n"
                f"RMSE={row['loo_rmse_log_gain']:.3f}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )
    axes[0, 0].set_ylabel("Observed gain at K=20")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(handles)), fontsize=8)
        fig.subplots_adjust(bottom=0.18)
    _finish_plot(fig, "study_only_predictions_loo", output_path)


def _plot_study_only_redundancy_rule(results_dir: str, plots_dir: str) -> None:
    """Plot the simple study-only redundancy rule."""
    summary_path = os.path.join(results_dir, "tau_gain_simulated.parquet")
    if not os.path.exists(summary_path):
        print(
            "Skipping study-only redundancy rule plot: "
            f"{summary_path} not found."
        )
        return

    feature = "study_cov_times_squared_error_cov_over_var_all_oof_intersection"
    df = G.add_relationship_derived_columns(pd.read_parquet(summary_path), K=K_DEFAULT)
    if feature not in df.columns or "sample_gain_K20" not in df.columns:
        print(
            "Skipping study-only redundancy rule plot: "
            f"{feature} is not available."
        )
        return
    if "model_family" not in df.columns and "algorithm" in df.columns:
        df["model_family"] = df["algorithm"].map(G.model_family)

    data = df.dropna(subset=[feature, "sample_gain_K20", "model_family"]).copy()
    data = data[(data[feature] > 0) & (data["sample_gain_K20"] > 0)]
    if len(data) < 6:
        return

    q25 = float(data[feature].quantile(0.25))
    q75 = float(data[feature].quantile(0.75))
    data["redundancy_bin"] = pd.cut(
        data[feature],
        [-np.inf, q25, q75, np.inf],
        labels=["Low", "Middle", "High"],
    )
    bin_order = ["Low", "Middle", "High"]
    bin_stats = (
        data.groupby("redundancy_bin", observed=False)["sample_gain_K20"]
        .agg(["count", "median", "min", "max"])
        .reindex(bin_order)
    )

    families = sorted(data["model_family"].astype(str).unique())
    color_values = plt.get_cmap("tab10").colors
    family_colors = {
        family: color_values[i % len(color_values)]
        for i, family in enumerate(families)
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.8),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )
    ax = axes[0]
    x_min = float(data[feature].min())
    x_max = float(data[feature].max())
    x_left = x_min / 1.25
    x_right = x_max * 1.25
    ax.axvspan(x_left, q25, color="#55A868", alpha=0.13, label="low redundancy")
    ax.axvspan(q25, q75, color="#999999", alpha=0.10, label="middle")
    ax.axvspan(q75, x_right, color="#C44E52", alpha=0.12, label="high redundancy")
    ax.axvline(q25, color="#333333", linestyle="--", linewidth=1.0)
    ax.axvline(q75, color="#333333", linestyle="--", linewidth=1.0)

    for family, sub in data.groupby("model_family"):
        ax.scatter(
            sub[feature],
            sub["sample_gain_K20"],
            s=64,
            alpha=0.82,
            color=family_colors.get(str(family), "#777777"),
            edgecolor="black",
            linewidth=0.45,
            label=str(family),
        )
    ax.set_xscale("log")
    ax.set_xlim(x_left, x_right)
    ax.set_xlabel("Study-only redundancy score R_study")
    ax.set_ylabel("Observed sample gain at K=20")
    ax.set_title("High study redundancy means low sample gain")
    ax.grid(True, which="both", linestyle=":", alpha=0.35)
    ax.annotate(
        "higher redundancy -> lower gain",
        xy=(q75 * 1.25, bin_stats.loc["High", "median"]),
        xytext=(q25 / 1.25, max(12.5, float(data["sample_gain_K20"].quantile(0.85)))),
        arrowprops={"arrowstyle": "->", "linewidth": 1.4, "color": "black"},
        fontsize=10,
        ha="left",
    )
    ax.text(
        q25 / 1.1,
        0.98,
        f"R < {q25:.2f}\nmedian gain {bin_stats.loc['Low', 'median']:.1f}",
        transform=ax.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=9,
        color="#1B6E3A",
    )
    ax.text(
        q75 * 1.1,
        0.98,
        f"R > {q75:.2f}\nmedian gain {bin_stats.loc['High', 'median']:.1f}",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=9,
        color="#8C2D2D",
    )

    ax_bin = axes[1]
    y_groups = [
        data.loc[data["redundancy_bin"] == label, "sample_gain_K20"].to_numpy(dtype=float)
        for label in bin_order
    ]
    box = ax_bin.boxplot(
        y_groups,
        positions=np.arange(len(bin_order)),
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.4},
    )
    box_colors = ["#55A868", "#999999", "#C44E52"]
    for patch, color in zip(box["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.24)
        patch.set_edgecolor(color)
    rng = np.random.default_rng(0)
    for pos, label in enumerate(bin_order):
        sub = data[data["redundancy_bin"] == label]
        jitter = rng.uniform(-0.13, 0.13, size=len(sub))
        for offset, (_, row) in zip(jitter, sub.iterrows()):
            family = str(row["model_family"])
            ax_bin.scatter(
                pos + offset,
                row["sample_gain_K20"],
                s=46,
                color=family_colors.get(family, "#777777"),
                alpha=0.82,
                edgecolor="black",
                linewidth=0.35,
            )
        median = bin_stats.loc[label, "median"]
        ax_bin.text(
            pos,
            median + 0.35,
            f"median {median:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax_bin.set_xticks(np.arange(len(bin_order)))
    ax_bin.set_xticklabels([
        f"Low\nR < {q25:.2f}",
        f"Middle\n{q25:.2f}-{q75:.2f}",
        f"High\nR > {q75:.2f}",
    ])
    ax_bin.set_ylabel("Observed sample gain at K=20")
    ax_bin.set_title("Binned rule of thumb")
    ax_bin.grid(True, axis="y", linestyle=":", alpha=0.35)

    handles, labels = ax.get_legend_handles_labels()
    family_handles = [
        handle for handle, label in zip(handles, labels)
        if label in family_colors
    ]
    family_labels = [
        label for label in labels
        if label in family_colors
    ]
    if family_handles:
        fig.legend(
            family_handles,
            family_labels,
            loc="lower center",
            ncol=min(4, len(family_handles)),
            fontsize=8,
        )
        fig.subplots_adjust(bottom=0.19)

    output_path = os.path.join(plots_dir, "study_only_redundancy_rule.pdf")
    _finish_plot(fig, "study_only_redundancy_rule", output_path)


def _plot_study_redundancy_high_gain_probability(
    results_dir: str,
    plots_dir: str,
) -> None:
    """Plot large-gain probabilities across study-redundancy tertiles."""
    input_path = os.path.join(results_dir, "study_redundancy_high_gain_probability.csv")
    if not os.path.exists(input_path):
        print(
            "Skipping study redundancy high-gain probability plot: "
            f"{input_path} not found."
        )
        return

    data = pd.read_csv(input_path)
    required = {"tertile", "threshold", "label", "p", "lo", "hi"}
    if not required.issubset(data.columns) or data.empty:
        print(
            "Skipping study redundancy high-gain probability plot: "
            f"{input_path} does not contain the expected columns."
        )
        return

    tertile_order = ["Low", "Middle", "High"]
    thresholds = sorted(data["threshold"].dropna().unique())
    if not thresholds:
        return

    fig, ax = plt.subplots(figsize=(3.2, 3.0), constrained_layout=False)
    x = np.arange(len(tertile_order), dtype=float)
    width = min(0.22, 0.78 / max(len(thresholds), 1))
    offsets = (np.arange(len(thresholds)) - (len(thresholds) - 1) / 2) * width
    colors = ["#4C72B0", "#F28E2B", "#C44E52"]

    for index, threshold in enumerate(thresholds):
        sub = (
            data[data["threshold"] == threshold]
            .set_index("tertile")
            .reindex(tertile_order)
        )
        y = sub["p"].to_numpy(dtype=float)
        lo = sub["lo"].to_numpy(dtype=float)
        hi = sub["hi"].to_numpy(dtype=float)
        yerr = np.vstack([np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)])
        labels = sub["label"].dropna().astype(str)
        label = labels.iloc[0] if not labels.empty else rf"$G^{{\mathrm{{test}}}}_{{20}}\geq {threshold:g}$"
        ax.bar(
            x + offsets[index],
            y,
            width=width,
            color=colors[index % len(colors)],
            edgecolor="white",
            linewidth=0.45,
            label=label,
            zorder=3,
        )
        ax.errorbar(
            x + offsets[index],
            y,
            yerr=yerr,
            fmt="none",
            ecolor="0.25",
            elinewidth=0.75,
            capsize=1.8,
            capthick=0.75,
            zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["Low\nredundancy", "Middle", "High\nredundancy"])
    ax.set_ylabel(
        r"Probability of large sample gain $G^{\mathrm{test}}_{20}$",
        fontsize=9,
    )
    ax.set_xlabel(
        "Tertile of study-only redundancy\n"
        r"computed from $\widehat{\omega}^{\mathrm{study}}_{20}$ tertiles",
        labelpad=5,
    )
    y_top = max(0.5, float(np.nanmax(data["hi"])) + 0.045)
    ax.set_ylim(0, min(1.0, y_top))
    ax.grid(True, axis="y", linestyle=":", alpha=0.35, zorder=0)
    ax.legend(
        title="Large Sample Gains",
        loc="upper right",
        frameon=False,
        fontsize=7.0,
        title_fontsize=7.4,
        handlelength=1.2,
        handletextpad=0.45,
        labelspacing=0.25,
    )
    fig.subplots_adjust(left=0.24, bottom=0.31)

    pdf_path = os.path.join(plots_dir, "study_redundancy_high_gain_probability.pdf")
    png_path = os.path.join(plots_dir, "study_redundancy_high_gain_probability.png")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"Saved {png_path}")
    _finish_plot(fig, "study_redundancy_high_gain_probability", pdf_path)


def _score_transform_panel_metrics(
    data: pd.DataFrame,
    x_col: str,
) -> dict[str, float]:
    """Compute compact monotone-rule metrics for one score transform."""
    valid = data[
        data[x_col].gt(0)
        & data["sample_gain_K20"].gt(0)
        & np.isfinite(data[x_col])
        & np.isfinite(data["sample_gain_K20"])
    ].copy()
    if len(valid) < 3:
        return {
            "n": len(valid),
            "pearson": np.nan,
            "spearman": np.nan,
            "low_median": np.nan,
            "high_median": np.nan,
        }
    log_x = np.log10(valid[x_col].to_numpy(dtype=float))
    log_y = np.log10(valid["sample_gain_K20"].to_numpy(dtype=float))
    q25 = valid[x_col].quantile(0.25)
    q75 = valid[x_col].quantile(0.75)
    low = valid.loc[valid[x_col] <= q25, "sample_gain_K20"]
    high = valid.loc[valid[x_col] >= q75, "sample_gain_K20"]
    return {
        "n": int(len(valid)),
        "pearson": float(np.corrcoef(log_x, log_y)[0, 1]),
        "spearman": float(valid[x_col].rank().corr(valid["sample_gain_K20"].rank())),
        "low_median": float(low.median()) if len(low) else np.nan,
        "high_median": float(high.median()) if len(high) else np.nan,
    }


def _prepare_study_score_comparison_tables(
    results_dir: str,
) -> list[tuple[str, pd.DataFrame]]:
    """Return linear and follow-up tables with comparable score transforms."""
    tables: list[tuple[str, pd.DataFrame]] = []
    try:
        import scripts.search_study_score_relationships as score_search
    except ImportError:
        score_search = None

    if score_search is not None:
        linear = score_search.load_table().copy()
        linear["study_redundancy_score"] = (
            linear["study_cov"]
            * linear["study_squared_error_cov_all_oof_intersection"]
            / linear["study_squared_error_var_all_oof_intersection"]
        )
        linear["score_augmented_redundancy"] = (
            linear["study_redundancy_score"]
            * np.sqrt(linear["cv_test_score_pos"] * linear["cv_test_score_loss"])
        )
        tables.append(("Linear 1000-seed", linear))

    followup_path = os.path.join(results_dir, "followup_study_redundancy_probe.csv")
    if os.path.exists(followup_path):
        followup = pd.read_csv(followup_path)
        if "model_family" not in followup.columns and "algorithm" in followup.columns:
            followup["model_family"] = followup["algorithm"].map(G.model_family)
        followup["score_augmented_redundancy"] = (
            followup["study_redundancy_score"]
            * np.sqrt(followup["cv_test_score_pos"] * followup["cv_test_score_loss"])
        )
        tables.append(("Follow-up", followup))
    return tables


def _plot_study_score_transform_comparison(results_dir: str, plots_dir: str) -> None:
    """Compare the simple R_study rule to a score-augmented transform."""
    tables = _prepare_study_score_comparison_tables(results_dir)
    if not tables:
        print("Skipping study-score transform comparison: no input tables found.")
        return

    columns = [
        (
            "study_redundancy_score",
            r"$R_{\mathrm{study}}$",
            "simple redundancy",
        ),
        (
            "score_augmented_redundancy",
            r"$R_{\mathrm{study}}\sqrt{\mathrm{score}\,(1-\mathrm{score})}$",
            "score-augmented",
        ),
    ]
    families = sorted(
        {
            str(family)
            for _, table in tables
            for family in table.get("model_family", pd.Series(dtype=str)).dropna().unique()
        }
    )
    color_values = plt.get_cmap("tab10").colors
    family_colors = {
        family: color_values[i % len(color_values)]
        for i, family in enumerate(families)
    }
    fig, axes = plt.subplots(
        len(tables),
        len(columns),
        figsize=(11.5, 4.2 * len(tables)),
        squeeze=False,
    )

    for row_index, (table_name, table) in enumerate(tables):
        for col_index, (x_col, xlabel, title_suffix) in enumerate(columns):
            ax = axes[row_index, col_index]
            valid = table[
                table[x_col].gt(0)
                & table["sample_gain_K20"].gt(0)
                & np.isfinite(table[x_col])
                & np.isfinite(table["sample_gain_K20"])
            ].copy()
            for family, sub in valid.groupby("model_family"):
                ax.scatter(
                    sub[x_col],
                    sub["sample_gain_K20"],
                    s=48,
                    alpha=0.8,
                    color=family_colors.get(str(family), "#777777"),
                    edgecolor="black",
                    linewidth=0.35,
                    label=str(family),
                )
            if not valid.empty:
                q25 = float(valid[x_col].quantile(0.25))
                q75 = float(valid[x_col].quantile(0.75))
                x_min = float(valid[x_col].min())
                x_max = float(valid[x_col].max())
                ax.axvline(q25, color="#333333", linestyle="--", linewidth=0.9, alpha=0.65)
                ax.axvline(q75, color="#333333", linestyle="--", linewidth=0.9, alpha=0.65)
                ax.set_xlim(x_min / 1.25, x_max * 1.25)
            metrics = _score_transform_panel_metrics(table, x_col)
            ax.text(
                0.04,
                0.96,
                (
                    f"n={metrics['n']}\n"
                    f"log r={metrics['pearson']:.3f}\n"
                    f"rank r={metrics['spearman']:.3f}\n"
                    f"low/high med={metrics['low_median']:.1f}/{metrics['high_median']:.1f}"
                ),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
            )
            ax.set_xscale("log")
            ax.set_xlabel(xlabel)
            if col_index == 0:
                ax.set_ylabel("Observed sample gain at K=20")
            ax.set_title(f"{table_name}: {title_suffix}")
            ax.grid(True, which="both", linestyle=":", alpha=0.35)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(handles)), fontsize=8)
        fig.subplots_adjust(bottom=0.12)
    output_path = os.path.join(plots_dir, "study_score_transform_comparison.pdf")
    _finish_plot(fig, "study_score_transform_comparison", output_path)


def _plot_study_only_search_outputs(results_dir: str, plots_dir: str) -> None:
    """Plot study-only search artifacts when the compact CSV is available."""
    metrics_path = os.path.join(results_dir, "study_only_relationship_search.csv")
    if not os.path.exists(metrics_path):
        print(
            "Skipping study-only search plots: "
            f"{metrics_path} not found."
        )
        return
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        return
    selected = _study_only_selected_rows(metrics)
    _plot_study_only_search_frontier(
        metrics,
        selected,
        os.path.join(plots_dir, "study_only_search_frontier.pdf"),
    )
    _plot_study_only_metric_comparison(
        selected,
        os.path.join(plots_dir, "study_only_metric_comparison.pdf"),
    )
    _plot_study_only_prediction_panels(
        selected,
        results_dir,
        os.path.join(plots_dir, "study_only_predictions_loo.pdf"),
    )


def _summarise_cov_vs_gain(
    df: pd.DataFrame,
    cov_col: str,
    plots_dir: str,
    *,
    x_math_symbol: str,
    pretty_name: str,
    highlight_mlp: bool = False,
) -> dict:
    """Plot the relationship between a covariance-style column and sample gain.

    Generic over which column carries the covariance signal (e.g. ``bench_cov``
    or ``study_cov``). Produces two figures:
      * ``{cov_col}_loglog_fit.pdf`` -- log-log scatter with the OLS fit line
        annotated (intercept, slope, $R^2$, Spearman ρ, Pearson r).
      * ``{cov_col}_vs_gain_stratified.pdf`` -- horizontal forest plot of
        Spearman ρ (linear) and Pearson r (log--log) overall and stratified
        by train_size and by dataset.
    Returns the fit parameters as a dict.
    """
    valid = df.dropna(subset=[cov_col, "sample_gain_K20"]).copy()
    valid = valid[valid[cov_col] > 0]
    if len(valid) < 3:
        print(
            f"Skipping {cov_col} summary: only {len(valid)} valid points "
            "(need >= 3 for correlations)."
        )
        return {}

    log_cov_col = f"log10_{cov_col}"
    valid[log_cov_col] = np.log10(valid[cov_col])
    valid["log10_sample_gain_K20"] = np.log10(
        valid["sample_gain_K20"].clip(lower=1e-12)
    )
    n = int(len(valid))

    rho_s_lin = _spearman_corr(valid[cov_col], valid["sample_gain_K20"])
    rho_p_lin = _pearson_corr(valid[cov_col], valid["sample_gain_K20"])
    rho_s_log = _spearman_corr(valid[log_cov_col], valid["log10_sample_gain_K20"])
    rho_p_log = _pearson_corr(valid[log_cov_col], valid["log10_sample_gain_K20"])

    # log-log OLS fit: log10(g) = a + b * log10(cov)
    X = np.column_stack([np.ones(n), valid[log_cov_col].to_numpy()])
    y = valid["log10_sample_gain_K20"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(beta[0]), float(beta[1])
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    loglog_rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
    loglog_mae = float(np.mean(np.abs(yhat - y)))

    rho_delta_metrics = {
        "rho_delta_n": 0,
        "rho_delta_pearson_loglog": np.nan,
        "rho_delta_loglog_intercept": np.nan,
        "rho_delta_loglog_slope": np.nan,
        "rho_delta_loglog_R2": np.nan,
        "gain_via_rho_delta_log_RMSE": np.nan,
        "gain_via_rho_delta_log_R2": np.nan,
    }
    if {"tau_te", "sigma2_te_K"}.issubset(valid.columns):
        valid_rho = valid.copy()
        valid_rho["rho_delta"] = valid_rho["tau_te"] / valid_rho["sigma2_te_K"]
        valid_rho = valid_rho[
            (valid_rho[cov_col] > 0)
            & (valid_rho["rho_delta"] > 0)
            & (valid_rho["sample_gain_K20"] > 0)
        ].copy()
        if len(valid_rho) >= 3:
            log_signal = np.log10(valid_rho[cov_col].to_numpy(dtype=float))
            log_rho = np.log10(valid_rho["rho_delta"].to_numpy(dtype=float))
            X_rho = np.column_stack([np.ones(len(valid_rho)), log_signal])
            beta_rho, *_ = np.linalg.lstsq(X_rho, log_rho, rcond=None)
            ar, br = float(beta_rho[0]), float(beta_rho[1])
            log_rho_hat = X_rho @ beta_rho
            rho_ss_res = float(np.sum((log_rho - log_rho_hat) ** 2))
            rho_ss_tot = float(np.sum((log_rho - log_rho.mean()) ** 2))
            rho_r2 = 1.0 - rho_ss_res / rho_ss_tot if rho_ss_tot > 0 else float("nan")
            rho_hat = 10.0 ** log_rho_hat
            gain_hat = K_DEFAULT / (1.0 + (K_DEFAULT - 1.0) * rho_hat)
            log_gain = np.log10(valid_rho["sample_gain_K20"].to_numpy(dtype=float))
            log_gain_hat = np.log10(np.clip(gain_hat, 1e-12, None))
            gain_resid = log_gain_hat - log_gain
            gain_ss_res = float(np.sum(gain_resid ** 2))
            gain_ss_tot = float(np.sum((log_gain - log_gain.mean()) ** 2))
            rho_delta_metrics = {
                "rho_delta_n": int(len(valid_rho)),
                "rho_delta_pearson_loglog": _pearson_corr(log_signal, log_rho),
                "rho_delta_loglog_intercept": ar,
                "rho_delta_loglog_slope": br,
                "rho_delta_loglog_R2": rho_r2,
                "gain_via_rho_delta_log_RMSE": float(np.sqrt(np.mean(gain_resid ** 2))),
                "gain_via_rho_delta_log_R2": (
                    1.0 - gain_ss_res / gain_ss_tot if gain_ss_tot > 0 else float("nan")
                ),
            }

    # ── Plot 1: log-log scatter with OLS fit line ──────────────────────────
    fig_fit, ax_fit = plt.subplots(figsize=(7.5, 5.0))
    is_mlp = pd.Series(False, index=valid.index)
    if highlight_mlp and "algorithm" in valid.columns:
        is_mlp = valid["algorithm"].astype(str).str.contains(
            "MLP", na=False, regex=False
        )

    if highlight_mlp and is_mlp.any():
        ax_fit.scatter(
            valid.loc[~is_mlp, cov_col],
            valid.loc[~is_mlp, "sample_gain_K20"],
            s=28,
            alpha=0.75,
            color="#1f77b4",
            edgecolor="black",
            linewidth=0.4,
            label="non-MLP",
        )
        ax_fit.scatter(
            valid.loc[is_mlp, cov_col],
            valid.loc[is_mlp, "sample_gain_K20"],
            s=48,
            alpha=0.9,
            color="#d62728",
            marker="D",
            edgecolor="black",
            linewidth=0.5,
            label="MLP",
        )
    else:
        ax_fit.scatter(
            valid[cov_col], valid["sample_gain_K20"],
            s=24, alpha=0.7, edgecolor="black", linewidth=0.4, color="#1f77b4",
        )
    grid = np.geomspace(float(valid[cov_col].min()), float(valid[cov_col].max()), 200)
    fit_line = 10.0 ** (a + b * np.log10(grid))
    cov_label = cov_col.replace("_", r"\_")
    ax_fit.plot(
        grid, fit_line, "k--", linewidth=1.4,
        label=(
            fr"OLS fit: $\log_{{10}} g = {a:+.3f} "
            fr"{b:+.3f}\,\log_{{10}}\mathrm{{{cov_label}}}$"
            f"\nRMSE={loglog_rmse:.3f}, $R^2={r2:.3f}$"
        ),
    )
    ax_fit.set_xscale("log")
    ax_fit.set_yscale("log")
    ax_fit.set_xlabel(x_math_symbol)
    ax_fit.set_ylabel(r"Observed sample gain at $K=20$")
    ax_fit.set_title(
        f"Power-law fit: sample gain vs {pretty_name}"
        f"\nSpearman ρ={rho_s_log:+.3f}, "
        f"Pearson r (log-log)={rho_p_log:+.3f}, n={n}"
    )
    ax_fit.legend(loc="upper right", fontsize=8)
    ax_fit.grid(True, which="both", linestyle=":", alpha=0.3)
    fit_id = f"{cov_col}_loglog_fit"
    path_fit = os.path.join(plots_dir, f"{fit_id}.pdf")
    fig_fit.savefig(path_fit, bbox_inches="tight")
    print(f"Saved {path_fit}")
    _finish_plot(fig_fit, fit_id)

    # ── Plot 2: stratified correlations forest plot ────────────────────────
    rows = [("overall", "all", n, float(rho_s_log), float(rho_p_log))]
    for size, sub in valid.groupby("train_size"):
        if len(sub) < 5: continue
        s = _spearman_corr(sub[cov_col], sub["sample_gain_K20"])
        p = _pearson_corr(sub[log_cov_col], sub["log10_sample_gain_K20"])
        rows.append(("train_size", f"N={int(size)}", len(sub), float(s), float(p)))
    for ds, sub in valid.groupby("dataset"):
        if len(sub) < 5: continue
        s = _spearman_corr(sub[cov_col], sub["sample_gain_K20"])
        p = _pearson_corr(sub[log_cov_col], sub["log10_sample_gain_K20"])
        rows.append(("dataset", str(ds), len(sub), float(s), float(p)))

    labels = [f"{lvl}  (n={ni})" for _, lvl, ni, _, _ in rows]
    spearman_vals = [r for _, _, _, r, _ in rows]
    pearson_vals = [r for _, _, _, _, r in rows]
    y_pos = np.arange(len(rows))[::-1]

    fig_strat, ax_strat = plt.subplots(figsize=(8.0, 0.42 * len(rows) + 1.6))
    ax_strat.scatter(spearman_vals, y_pos, s=70, marker="o",
                     color="#1f77b4", edgecolor="black", linewidth=0.5,
                     label="Spearman ρ (linear)")
    ax_strat.scatter(pearson_vals, y_pos, s=70, marker="s",
                     color="#d62728", edgecolor="black", linewidth=0.5,
                     label="Pearson r (log-log)")
    for yi, rs, rp in zip(y_pos, spearman_vals, pearson_vals):
        ax_strat.text(rs, yi + 0.15, f"{rs:+.2f}", fontsize=8,
                      ha="center", va="bottom", color="#1f77b4")
        ax_strat.text(rp, yi - 0.15, f"{rp:+.2f}", fontsize=8,
                      ha="center", va="top", color="#d62728")
    ax_strat.axvline(0, color="black", linewidth=0.8, alpha=0.6)
    ax_strat.axvline(-1, color="grey", linestyle=":", linewidth=0.6, alpha=0.4)
    ax_strat.axvline(+1, color="grey", linestyle=":", linewidth=0.6, alpha=0.4)
    ax_strat.set_yticks(y_pos)
    ax_strat.set_yticklabels(labels, fontsize=9)
    ax_strat.set_xlim(-1.05, 1.05)
    ax_strat.set_xlabel("correlation with sample_gain_K20")
    ax_strat.set_title(
        f"{cov_col} ↔ sample_gain_K20: stratified correlations (simulated only)"
    )
    ax_strat.legend(loc="lower left", fontsize=9)
    ax_strat.grid(True, axis="x", linestyle=":", alpha=0.4)
    strat_id = f"{cov_col}_vs_gain_stratified"
    path_strat = os.path.join(plots_dir, f"{strat_id}.pdf")
    fig_strat.savefig(path_strat, bbox_inches="tight")
    print(f"Saved {path_strat}")
    _finish_plot(fig_strat, strat_id)

    return {
        "n": n, "spearman_linear": float(rho_s_lin), "pearson_linear": float(rho_p_lin),
        "spearman_loglog": float(rho_s_log), "pearson_loglog": float(rho_p_log),
        "loglog_intercept": a,
        "loglog_slope": b,
        "loglog_RMSE": loglog_rmse,
        "loglog_MAE": loglog_mae,
        "loglog_R2": r2,
        **rho_delta_metrics,
    }


def _fit_icc_gain_proxy_from_signal(
    df: pd.DataFrame,
    signal_col: str,
    *,
    K: int = K_DEFAULT,
    pred_rho_col: str,
    pred_gain_col: str,
) -> dict[str, float]:
    """Fit log10(rho_delta) from one positive signal, then plug into ICC gain."""
    required = [signal_col, "tau_te", "sigma2_te_K", "sample_gain_K20"]
    if any(col not in df.columns for col in required):
        return {}

    valid = df.dropna(subset=required).copy()
    valid["rho_delta"] = valid["tau_te"] / valid["sigma2_te_K"]
    valid = valid[
        (valid[signal_col] > 0)
        & (valid["rho_delta"] > 0)
        & (valid["sample_gain_K20"] > 0)
    ]
    if len(valid) < 3:
        return {}

    log_signal = np.log10(valid[signal_col].to_numpy(dtype=float))
    log_rho = np.log10(valid["rho_delta"].to_numpy(dtype=float))
    X = np.column_stack([np.ones(len(valid)), log_signal])
    beta, *_ = np.linalg.lstsq(X, log_rho, rcond=None)
    intercept, slope = float(beta[0]), float(beta[1])
    log_rho_hat = X @ beta
    rho_resid = log_rho_hat - log_rho
    rho_ss_res = float(np.sum(rho_resid ** 2))
    rho_ss_tot = float(np.sum((log_rho - log_rho.mean()) ** 2))

    signal_values = pd.to_numeric(df[signal_col], errors="coerce")
    positive_signal = signal_values > 0
    df[pred_rho_col] = np.nan
    df[pred_gain_col] = np.nan
    pred_rho = 10.0 ** (
        intercept + slope * np.log10(signal_values.loc[positive_signal])
    )
    pred_gain = K / (1.0 + (K - 1.0) * pred_rho)
    df.loc[positive_signal, pred_rho_col] = pred_rho
    df.loc[positive_signal, pred_gain_col] = pred_gain

    valid_pred = df.dropna(subset=[pred_gain_col, "sample_gain_K20"]).copy()
    valid_pred = valid_pred[
        (valid_pred[pred_gain_col] > 0) & (valid_pred["sample_gain_K20"] > 0)
    ]
    log_gain_pred = np.log10(valid_pred[pred_gain_col].to_numpy(dtype=float))
    log_gain_obs = np.log10(valid_pred["sample_gain_K20"].to_numpy(dtype=float))
    gain_resid = log_gain_pred - log_gain_obs
    gain_ss_res = float(np.sum(gain_resid ** 2))
    gain_ss_tot = float(np.sum((log_gain_obs - log_gain_obs.mean()) ** 2))

    return {
        "signal": signal_col,
        "n_fit": int(len(valid)),
        "rho_delta_intercept": intercept,
        "rho_delta_slope": slope,
        "rho_delta_log_R2": (
            1.0 - rho_ss_res / rho_ss_tot if rho_ss_tot > 0 else float("nan")
        ),
        "rho_delta_log_RMSE": float(np.sqrt(np.mean(rho_resid ** 2))),
        "gain_n_scored": int(len(valid_pred)),
        "gain_log_RMSE": float(np.sqrt(np.mean(gain_resid ** 2))),
        "gain_log_MAE": float(np.mean(np.abs(gain_resid))),
        "gain_log_bias": float(np.mean(gain_resid)),
        "gain_log_R2": (
            1.0 - gain_ss_res / gain_ss_tot if gain_ss_tot > 0 else float("nan")
        ),
    }


def _plot_icc_gain_curve_from_signal(
    df: pd.DataFrame,
    signal_col: str,
    metrics: dict[str, float],
    *,
    xlabel: str,
    title: str,
    output_path: str,
    plot_id: str,
    K: int = K_DEFAULT,
) -> None:
    """Plot observed gain against a signal with the fitted ICC plug-in curve."""
    valid = df.dropna(subset=[signal_col, "sample_gain_K20"]).copy()
    valid = valid[(valid[signal_col] > 0) & (valid["sample_gain_K20"] > 0)]
    if len(valid) < 3 or not metrics:
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    if "algorithm" in valid.columns:
        is_mlp = valid["algorithm"].astype(str).str.contains("MLP", na=False, regex=False)
        ax.scatter(
            valid.loc[~is_mlp, signal_col],
            valid.loc[~is_mlp, "sample_gain_K20"],
            s=28,
            alpha=0.75,
            color="#1f77b4",
            edgecolor="black",
            linewidth=0.4,
            label="non-MLP",
        )
        if is_mlp.any():
            ax.scatter(
                valid.loc[is_mlp, signal_col],
                valid.loc[is_mlp, "sample_gain_K20"],
                s=48,
                alpha=0.9,
                color="#d62728",
                marker="D",
                edgecolor="black",
                linewidth=0.5,
                label="MLP",
            )
    else:
        ax.scatter(
            valid[signal_col],
            valid["sample_gain_K20"],
            s=28,
            alpha=0.75,
            color="#1f77b4",
            edgecolor="black",
            linewidth=0.4,
        )

    grid = np.geomspace(float(valid[signal_col].min()), float(valid[signal_col].max()), 250)
    rho_hat = 10.0 ** (
        metrics["rho_delta_intercept"]
        + metrics["rho_delta_slope"] * np.log10(grid)
    )
    gain_hat = K / (1.0 + (K - 1.0) * rho_hat)
    ax.plot(
        grid,
        gain_hat,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=(
            r"$\hat g=K/(1+(K-1)\hat\rho)$"
            f"\nlog-RMSE={metrics['gain_log_RMSE']:.3f}, "
            f"$R^2$={metrics['gain_log_R2']:.3f}"
        ),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Observed sample gain at $K=20$")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    _finish_plot(fig, plot_id, output_path)


def _normalize_legacy_tau_gain_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Promote legacy split/gain column names to the canonical K-based schema.

    Intended only for parquets produced before the ``s → K`` schema refactor.
    Fires a :class:`DeprecationWarning` whenever it does any work so the shim
    can be removed once no caller still loads legacy files.
    """
    rename_map = {
        old: new
        for old, new in LEGACY_TAU_GAIN_RENAMES.items()
        if old in df.columns and new not in df.columns
    }
    if not rename_map:
        return df
    import warnings
    warnings.warn(
        "tau_gain parquet uses legacy split/gain column names "
        f"({sorted(rename_map)}); re-run the pipeline to regenerate with the "
        "canonical K-based schema. The legacy rename shim will be removed.",
        DeprecationWarning,
        stacklevel=3,
    )
    return df.rename(columns=rename_map)


def load_pipeline_tables(results_dir: str) -> pd.DataFrame:
    dfs = []
    for name in PIPELINES:
        path = os.path.join(results_dir, f"tau_gain_{name}.parquet")
        if not os.path.exists(path):
            print(f"skipping {name}: {path} not found (run the pipeline first)")
            continue
        df = _normalize_legacy_tau_gain_columns(pd.read_parquet(path))
        for col in getattr(C, "OPTIONAL_TAU_GAIN_COLUMNS", []):
            if col not in df.columns:
                df[col] = np.nan
        missing = sorted(set(C.CANONICAL_TAU_GAIN_COLUMNS) - set(df.columns))
        if missing:
            raise ValueError(
                f"{path} is missing canonical columns: {missing}. "
                "Re-run the pipeline after the schema refactor."
            )
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def _drop_repeated_kfold(df: pd.DataFrame) -> pd.DataFrame:
    """Remove every row whose CV procedure is (or maps to) RepeatedKFold."""
    if "procedure" not in df.columns:
        return df
    mask = df["procedure"].astype(str).str.contains("RepeatedKFold", na=False)
    return df.loc[~mask].copy()


def add_semantic_predictors(df: pd.DataFrame, K: int = K_DEFAULT) -> pd.DataFrame:
    """Add derived predictor columns aligned with the paper's estimator notation.

    Column-name conventions match Sec.~\\ref{sec:sample-gain-estimation}:
    ``sigma2_HO_K`` is $\\hat\\sigma^2_{HO\\mid K}$, ``sigma2_HO_1`` is
    $\\hat\\sigma^2_{HO\\mid 1}$, ``sigma2_HO_trxi`` is
    $\\hat\\sigma^2_{HO\\mid tr,xi}$; the bench-adjusted variants use ``te``
    in place of ``HO``.
    """
    out = df.copy()
    out["size_for_plot"] = out["train_size"].combine_first(out["study_size"])

    out["tau_over_sigma2_HO_K"] = _safe_ratio(out["tau"], out["sigma2_HO_K"])
    out["W_over_B"] = _safe_ratio(out["W"], out["B"])
    # rho_delta = tau^te / sigma^{2,te}_{|K}: intraclass correlation of fold
    # delta-scores; Eq.~\ref{eq:sample-gain-icc}.
    out["rho_delta"] = _safe_ratio(out["tau_te"], out["sigma2_te_K"])
    out["sigma2_te_K_over_sigma2_HO_K"] = _safe_ratio(out["sigma2_te_K"], out["sigma2_HO_K"])
    out["tau_te_over_tau"] = _safe_ratio(out["tau_te"], out["tau"])

    # predicted gain (delta form): K * sigma^{2,te}_{|1} / (sigma^{2,te}_{|K} + (K-1) tau^te)
    denom_te = out["sigma2_te_K"] + (K - 1) * out["tau_te"]
    out["predicted_sample_gain_K20"] = np.where(
        denom_te > 0,
        K * out["sigma2_te_1"] / denom_te,
        np.nan,
    )

    # predicted gain (raw form): K * sigma^2_{HO|1} / (sigma^2_{HO|K} + (K-1) tau)
    denom_HO = out["sigma2_HO_K"] + (K - 1) * out["tau"]
    out["predicted_sample_gain_K20_raw"] = np.where(
        denom_HO > 0,
        K * out["sigma2_HO_1"] / denom_HO,
        np.nan,
    )

    out["sigma2_HO_1_over_sigma2_HO_K"] = _safe_ratio(
        out["sigma2_HO_1"], out["sigma2_HO_K"]
    )

    if "sigma2_HO_trxi" in out.columns:
        # predicted gain (within-seed raw form): K * sigma^2_{HO|tr,xi} / (sigma^2_{HO|K} + (K-1) tau)
        out["predicted_sample_gain_K20_trxi"] = np.where(
            denom_HO > 0,
            K * out["sigma2_HO_trxi"] / denom_HO,
            np.nan,
        )
        out["sigma2_HO_trxi_over_sigma2_HO_K"] = _safe_ratio(
            out["sigma2_HO_trxi"], out["sigma2_HO_K"]
        )
        # Seed-noise inflation factor: sigma^2_{HO|1} / sigma^2_{HO|tr,xi}
        # is ~1 when training is stable and >>1 when training-seed noise
        # dominates.
        out["sigma2_HO_1_over_sigma2_HO_trxi"] = _safe_ratio(
            out["sigma2_HO_1"], out["sigma2_HO_trxi"]
        )

    return out


def main(
    results_dir: str = DEFAULT_RESULTS_DIR,
    plots_dir: str = DEFAULT_PLOTS_DIR,
    K: int = K_DEFAULT,
) -> pd.DataFrame:
    os.makedirs(plots_dir, exist_ok=True)
    P.set_paper_style()

    df = load_pipeline_tables(results_dir)
    if df.empty:
        raise RuntimeError(
            f"No per-pipeline parquet found in {results_dir}/. "
            "Run simulated.py, nlp.py and pcam.py first."
        )

    df = _drop_repeated_kfold(df)
    df = add_semantic_predictors(df, K=K)
    print(
        f"Loaded {len(df)} points from pipelines: "
        f"{sorted(df['pipeline'].unique())} (after dropping RepeatedKFold)"
    )

    # Inline display is controlled by PLOT_DISPLAY. All figures are still saved.
    fig_delta, ax_delta = plt.subplots(figsize=(7.5, 5.0))
    P.plot_tau_vs_gain_encoded(
        df,
        x_col="tau",
        y_col="sample_gain_K20",
        sat_col=None,
        size_col="size_for_plot",
        size_label="data size",
        xlabel=r"$\hat\tau$ (raw-score CV inter-fold covariance)",
        ax=ax_delta,
        title=r"$\hat\tau$ vs observed sample gain at $K=20$ ($\delta$-based)",
    )
    _reserve_right_legend_space(fig_delta)
    delta_path = os.path.join(plots_dir, "tau_vs_gain_scatter.pdf")
    _finish_plot(fig_delta, "tau_vs_gain_raw", delta_path)

    sim_df = df[df["pipeline"] == "simulated"]
    fig_log, ax_log = plt.subplots(figsize=(7.5, 5.0))
    P.plot_tau_vs_gain_encoded(
        sim_df,
        x_col="tau",
        y_col="sample_gain_K20_log1mr2",
        sat_col=None,
        size_col="size_for_plot",
        size_label="data size",
        xlabel=r"$\hat\tau$ (raw-score CV inter-fold covariance)",
        ax=ax_log,
        title=r"$\hat\tau$ vs observed sample gain at $K=20$ ($\log(1-R^2)$, simulated only)",
    )
    _reserve_right_legend_space(fig_log)
    log_path = os.path.join(plots_dir, "tau_vs_gain_log1mr2_scatter.pdf")
    _finish_plot(fig_log, "tau_vs_gain_log1mr2", log_path)

    valid_tau_pair = df.dropna(subset=["tau", "tau_te"])
    print(f"Raw/delta tau pairs available for {len(valid_tau_pair)}/{len(df)} points")

    fig_tau_pair, ax_tau_pair = plt.subplots(figsize=(7.5, 5.0))
    P.plot_tau_vs_gain_encoded(
        df,
        x_col="tau",
        y_col="tau_te",
        sat_col=None,
        size_col="size_for_plot",
        size_label="data size",
        xlabel=r"$\hat\tau$ (raw-score CV inter-fold covariance)",
        ylabel=r"$\hat\tau^{te}$ ($\delta$-based CV inter-fold covariance)",
        ax=ax_tau_pair,
        title=r"$\hat\tau$ vs $\hat\tau^{te}$ across pipelines",
    )
    _add_identity_line(ax_tau_pair, valid_tau_pair, "tau", "tau_te")
    if not valid_tau_pair.empty and (valid_tau_pair["tau"] > 0).all() and (valid_tau_pair["tau_te"] > 0).all():
        ax_tau_pair.set_xscale("log")
        ax_tau_pair.set_yscale("log")
    _reserve_right_legend_space(fig_tau_pair)
    tau_pair_path = os.path.join(plots_dir, "tau_vs_tau_te_scatter.pdf")
    _finish_plot(fig_tau_pair, "tau_vs_tau_te", tau_pair_path)

    valid_sigma_pair = df.dropna(subset=["sigma2_HO_K", "sigma2_te_K"])
    print(
        f"Raw/delta variance pairs available for {len(valid_sigma_pair)}/{len(df)} points"
    )

    fig_sigma_pair, ax_sigma_pair = plt.subplots(figsize=(7.5, 5.0))
    P.plot_tau_vs_gain_encoded(
        df,
        x_col="sigma2_HO_K",
        y_col="sigma2_te_K",
        sat_col=None,
        size_col="size_for_plot",
        size_label="data size",
        xlabel=r"$\hat\sigma^2_{\mathrm{HO}\mid K}$ (raw-score CV per-fold variance)",
        ylabel=r"$\hat\sigma^{2,\mathrm{te}}_{\mid K}$ ($\delta$-based CV per-fold variance)",
        ax=ax_sigma_pair,
        title=r"$\hat\sigma^2_{\mathrm{HO}\mid K}$ vs $\hat\sigma^{2,\mathrm{te}}_{\mid K}$ across pipelines",
    )
    _add_identity_line(ax_sigma_pair, valid_sigma_pair, "sigma2_HO_K", "sigma2_te_K")
    if not valid_sigma_pair.empty and (valid_sigma_pair["sigma2_HO_K"] > 0).all() and (valid_sigma_pair["sigma2_te_K"] > 0).all():
        ax_sigma_pair.set_xscale("log")
        ax_sigma_pair.set_yscale("log")
    _reserve_right_legend_space(fig_sigma_pair)
    sigma_pair_path = os.path.join(plots_dir, "sigma2_HO_K_vs_sigma2_te_K_scatter.pdf")
    _finish_plot(fig_sigma_pair, "sigma_HO_K_vs_sigma_te_K", sigma_pair_path)

    valid_ratio_pair = df.dropna(subset=["sigma2_te_K_over_sigma2_HO_K", "tau_te_over_tau"])
    print(
        "Variance/tau shrinkage ratios available for "
        f"{len(valid_ratio_pair)}/{len(df)} points"
    )

    fig_ratio_pair, ax_ratio_pair = plt.subplots(figsize=(7.5, 5.0))
    P.plot_tau_vs_gain_encoded(
        df,
        x_col="sigma2_te_K_over_sigma2_HO_K",
        y_col="tau_te_over_tau",
        sat_col=None,
        size_col="size_for_plot",
        size_label="data size",
        xlabel=r"$\hat\sigma^{2,\mathrm{te}}_{\mid K} / \hat\sigma^2_{\mathrm{HO}\mid K}$",
        ylabel=r"$\hat\tau^{te} / \hat\tau$",
        ax=ax_ratio_pair,
        title=(
            r"Variance shrinkage vs covariance shrinkage: "
            r"$\hat\tau^{te}/\hat\tau$ vs "
            r"$\hat\sigma^{2,\mathrm{te}}_{\mid K}/\hat\sigma^2_{\mathrm{HO}\mid K}$"
        ),
    )
    _add_identity_line(
        ax_ratio_pair,
        valid_ratio_pair,
        "sigma2_te_K_over_sigma2_HO_K",
        "tau_te_over_tau",
    )
    _reserve_right_legend_space(fig_ratio_pair)
    ratio_pair_path = os.path.join(plots_dir, "tau_ratio_vs_sigma_ratio_scatter.pdf")
    _finish_plot(fig_ratio_pair, "tau_ratio_vs_sigma_ratio", ratio_pair_path)

    proxy_fit = T.fit_tau_te_proxy_specs(df, K=K)
    df = proxy_fit["data"]
    print("\nNon-benchmark tau_te proxy fit diagnostics:")
    print(proxy_fit["metrics"].to_string(index=False))
    print(
        "Diagnostic proxy retained: "
        f"{proxy_fit['selected_spec']} "
        f"(fixed alpha={proxy_fit['selected_alpha']}, "
        f"rule={proxy_fit['selection_reason']})"
    )
    for spec in T.get_tau_te_proxy_specs():
        print(
            f"  {spec.name}: "
            f"log target = {T.format_proxy_formula(proxy_fit['coefficients'], spec.name)}"
        )

    # Backward-compatible aliases for older plot names / notebooks.
    df["tau_te_proxy_from_sigma2_HO_trxi"] = df["tau_te_proxy_baseline_trxi"]
    df["tau_te_proxy_from_tau_and_W_over_B"] = df["tau_te_proxy_baseline_tau_wb"]
    df["predicted_sample_gain_K20_nonbench"] = df["predicted_sample_gain_K20_proxy"]

    proxy_plot_configs = [
        {
            "col": "tau_te_proxy_baseline_trxi",
            "legacy_col": "tau_te_proxy_from_sigma2_HO_trxi",
            "plot_id": "tau_te_proxy_baseline_trxi",
            "xlabel": (
                r"Fitted raw-only proxy from "
                r"$\hat\sigma^2_{\mathrm{HO}\mid \mathrm{tr},\xi}$"
            ),
            "title": (
                r"Non-benchmark proxy: "
                r"$\hat\tau^{te}_{\mathrm{proxy}}(\hat\sigma^2_{\mathrm{HO}\mid \mathrm{tr},\xi})$ "
                r"vs observed $\hat\tau^{te}$"
            ),
            "filename": "tau_te_proxy_from_sigma2_HO_trxi_scatter.pdf",
        },
        {
            "col": "tau_te_proxy_baseline_tau_wb",
            "legacy_col": "tau_te_proxy_from_tau_and_W_over_B",
            "plot_id": "tau_te_proxy_baseline_tau_wb",
            "xlabel": (
                r"Fitted raw-only proxy from "
                r"$\hat\tau$ and "
                r"$\hat{W}/\hat{B}$"
            ),
            "title": (
                r"Non-benchmark proxy: "
                r"$\hat\tau^{te}_{\mathrm{proxy}}(\hat\tau, "
                r"\hat{W}/\hat{B})$ "
                r"vs observed $\hat\tau^{te}$"
            ),
            "filename": "tau_te_proxy_from_tau_and_W_over_B_scatter.pdf",
        },
        {
            "col": "tau_te_proxy_combined_raw",
            "plot_id": "tau_te_proxy_combined_raw",
            "xlabel": (
                r"Fitted raw-only proxy from "
                r"$\hat\tau$, "
                r"$\hat\sigma^2_{\mathrm{HO}\mid \mathrm{tr},\xi}$, "
                r"$\hat{W}/\hat{B}$"
            ),
            "title": (
                r"Non-benchmark proxy: combined raw observables "
                r"vs observed $\hat\tau^{te}$"
            ),
            "filename": "tau_te_proxy_combined_raw_scatter.pdf",
        },
        {
            "col": "tau_te_proxy_shrinkage",
            "plot_id": "tau_te_proxy_shrinkage",
            "xlabel": (
                r"Fitted shrinkage proxy reconstructing "
                r"$\hat\tau^{te}/\hat\tau$"
            ),
            "title": (
                r"Non-benchmark proxy: raw shrinkage reconstruction "
                r"vs observed $\hat\tau^{te}$"
            ),
            "filename": "tau_te_proxy_shrinkage_scatter.pdf",
        },
    ]
    for config in proxy_plot_configs:
        plot_col = config.get("legacy_col", config["col"])
        valid_proxy = df.dropna(subset=[plot_col, "tau_te"])
        print(
            f"{config['col']} pairs available for "
            f"{len(valid_proxy)}/{len(df)} points"
        )
        _plot_tau_te_proxy_scatter(
            df,
            proxy_col=plot_col,
            plot_id=config["plot_id"],
            xlabel=config["xlabel"],
            title=config["title"],
            output_path=os.path.join(plots_dir, config["filename"]),
        )

    # ── Rowwise non-benchmark predicted sample gain ───────────────────────
    # These formulas use only quantities computed on the same setup as the
    # sample gain being predicted. There is no cross-pipeline training.
    baseline_scores = T.score_sample_gain_prediction(
        df,
        df["predicted_sample_gain_K20"],
    )
    target_rmse_log_gain = 1.10 * baseline_scores["rmse_log_gain"]
    local_gain_fit = T.compute_local_sample_gain_proxies(
        df,
        K=K,
        target_rmse_log_gain=target_rmse_log_gain,
    )
    df = local_gain_fit["data"]
    nonbenchmark_gain_metrics = local_gain_fit["metrics"].copy()
    nonbenchmark_gain_metrics["display_error"] = nonbenchmark_gain_metrics[
        "rmse_log_gain"
    ]
    best_nonbench = nonbenchmark_gain_metrics.dropna(subset=["display_error"]).sort_values(
        ["display_error", "proxy"]
    ).iloc[0]
    df["predicted_sample_gain_K20_best_nonbench"] = df[str(best_nonbench["output_col"])]
    df["predicted_sample_gain_K20_nonbench"] = df["predicted_sample_gain_K20_best_nonbench"]
    gain_metrics = nonbenchmark_gain_metrics.sort_values(
        ["display_error", "proxy"],
        na_position="last",
    )
    baseline_row = {
        "proxy": "benchmark_delta_formula",
        "family": "benchmark_baseline",
        "spec": "delta_formula",
        "output_col": "predicted_sample_gain_K20",
        "baseline": True,
        "n_scored": baseline_scores["n_scored"],
        "n_score_groups": baseline_scores["n_score_groups"],
        "rmse_log_gain": baseline_scores["rmse_log_gain"],
        "mae_log_gain": baseline_scores["mae_log_gain"],
        "bias_log_gain": baseline_scores["bias_log_gain"],
        "mae_gain": baseline_scores["mae_gain"],
        "bias_gain": baseline_scores["bias_gain"],
        "median_pred_obs_ratio": baseline_scores["median_pred_obs_ratio"],
        "corr_log_gain": baseline_scores["corr_log_gain"],
        "display_error": baseline_scores["rmse_log_gain"],
        "is_selected": False,
        "metric_source": "benchmark-score formula, pipeline-weighted evaluation",
    }
    gain_metrics = nonbenchmark_gain_metrics.copy()
    missing_source = (
        gain_metrics["metric_source"].isna()
        if "metric_source" in gain_metrics.columns
        else pd.Series(True, index=gain_metrics.index)
    )
    gain_metrics.loc[missing_source, "metric_source"] = (
        "rowwise formula, no training; pipeline-weighted evaluation"
    )
    gain_metrics_with_baseline = pd.concat(
        [pd.DataFrame([baseline_row]), gain_metrics],
        ignore_index=True,
        sort=False,
    )
    print("\nSample-gain prediction quantification:")
    print(gain_metrics_with_baseline.to_string(index=False))
    print(
        "Best rowwise non-benchmark formula in this dataset: "
        f"{best_nonbench['proxy']} "
        f"(source={best_nonbench.get('metric_source', best_nonbench['family'])}, "
        f"error={best_nonbench['display_error']:.4f})"
    )
    print(
        "Rowwise formula search stopped after "
        f"{local_gain_fit['n_candidates_tried']} candidates "
        f"({local_gain_fit['stop_reason']}; "
        f"target={target_rmse_log_gain:.4f}, "
        f"baseline={baseline_scores['rmse_log_gain']:.4f})."
    )
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "nonbenchmark_sample_gain_proxy_metrics.csv")
    metrics_with_baseline_path = os.path.join(
        results_dir,
        "sample_gain_proxy_metrics_with_benchmark_baseline.csv",
    )
    best_metrics_path = os.path.join(results_dir, "best_nonbenchmark_sample_gain_proxies.csv")
    gain_metrics.to_csv(metrics_path, index=False)
    gain_metrics_with_baseline.to_csv(metrics_with_baseline_path, index=False)
    gain_metrics.head(12).to_csv(best_metrics_path, index=False)
    print(f"Wrote {metrics_path}")
    print(f"Wrote {metrics_with_baseline_path}")
    print(f"Wrote {best_metrics_path}")
    _plot_sample_gain_proxy_quantification(
        gain_metrics_with_baseline,
        output_path=os.path.join(plots_dir, "sample_gain_proxy_quantification.pdf"),
    )
    best_output_col = str(best_nonbench["output_col"])
    valid_nonbench = df.dropna(subset=[best_output_col, "sample_gain_K20"])
    print(f"{best_output_col} available for {len(valid_nonbench)}/{len(df)} points")
    fig_nonbench_best = P.plot_predicted_vs_observed_gain(
        df,
        x_col="predicted_sample_gain_K20_best_nonbench",
        y_col="sample_gain_K20",
        sat_col=None,
        size_col="size_for_plot",
        title=r"Best non-benchmark predicted sample gain vs observed at $K=20$",
        output_path=os.path.join(plots_dir, "predicted_nonbench_best_vs_observed_gain.pdf"),
    )
    _reserve_right_legend_space(fig_nonbench_best)
    _finish_plot(fig_nonbench_best, "predicted_nonbench_best")

    covariance_gain_summaries: list[dict[str, object]] = []

    # ── bench_cov: model-prediction covariance on bench inputs ──────────────
    # Direct measure of training-side fold redundancy: covariance of two
    # folds' predictions on X_bench (no bench labels needed). Strong negative
    # rank correlation with sample_gain_K20; computed only by the simulated
    # pipeline so the scatter shows simulated rows only.
    if "bench_cov" in df.columns and df["bench_cov"].notna().any():
        valid_bc = df.dropna(subset=["bench_cov", "sample_gain_K20"])
        print(
            f"bench_cov available for {len(valid_bc)}/{len(df)} points "
            "(simulated only)"
        )

        fig_bc, ax_bc = plt.subplots(figsize=(7.5, 5.0))
        P.plot_tau_vs_gain_encoded(
            df,
            x_col="bench_cov",
            y_col="sample_gain_K20",
            sat_col=None,
            size_col="size_for_plot",
            size_label="data size",
            xlabel=(
                r"$\widehat{\mathrm{Cov}}_{X_{\mathrm{bench}}}"
                r"(\hat y_{f_1},\hat y_{f_2})$ "
                r"(prediction covariance across folds, on bench inputs)"
            ),
            ax=ax_bc,
            title=(
                r"Bench-prediction covariance vs observed sample gain "
                r"at $K=20$ (simulated only)"
            ),
        )
        _reserve_right_legend_space(fig_bc)
        path_bc = os.path.join(plots_dir, "bench_cov_vs_gain.pdf")
        fig_bc.savefig(path_bc, bbox_inches="tight")
        print(f"Saved {path_bc}")
        _finish_plot(fig_bc, "bench_cov_vs_gain")

        # log10 version: Pearson r ~ -0.94 on simulated, exposing the
        # near-power-law relationship sample_gain ~ bench_cov^{beta}.
        df["log10_bench_cov"] = np.log10(df["bench_cov"].clip(lower=1e-12))
        fig_bc_log, ax_bc_log = plt.subplots(figsize=(7.5, 5.0))
        P.plot_tau_vs_gain_encoded(
            df,
            x_col="log10_bench_cov",
            y_col="sample_gain_K20",
            sat_col=None,
            size_col="size_for_plot",
            size_label="data size",
            xlabel=(
                r"$\log_{10}\widehat{\mathrm{Cov}}_{X_{\mathrm{bench}}}"
                r"(\hat y_{f_1},\hat y_{f_2})$"
            ),
            ax=ax_bc_log,
            title=(
                r"$\log_{10}$(bench-prediction covariance) vs observed "
                r"sample gain at $K=20$ (simulated only)"
            ),
        )
        _reserve_right_legend_space(fig_bc_log)
        path_bc_log = os.path.join(plots_dir, "log10_bench_cov_vs_gain.pdf")
        fig_bc_log.savefig(path_bc_log, bbox_inches="tight")
        print(f"Saved {path_bc_log}")
        _finish_plot(fig_bc_log, "log10_bench_cov_vs_gain")

        summary = _summarise_cov_vs_gain(
            df,
            cov_col="bench_cov",
            plots_dir=plots_dir,
            x_math_symbol=(
                r"$\widehat{\mathrm{Cov}}_{X_{\mathrm{bench}}}"
                r"(\hat y_{f_1},\hat y_{f_2})$"
            ),
            pretty_name="bench-prediction covariance",
            highlight_mlp=True,
        )
        if summary:
            covariance_gain_summaries.append({"signal": "bench_cov", **summary})

    # ── study_cov: model-prediction covariance on the study test set ────────
    # Same structure as bench_cov but on X_study instead of X_bench. Computed
    # only by the simulated pipeline.
    if "study_cov" in df.columns and df["study_cov"].notna().any():
        valid_sc = df.dropna(subset=["study_cov", "sample_gain_K20"])
        print(
            f"study_cov available for {len(valid_sc)}/{len(df)} points "
            "(simulated only)"
        )

        fig_sc, ax_sc = plt.subplots(figsize=(7.5, 5.0))
        P.plot_tau_vs_gain_encoded(
            df,
            x_col="study_cov",
            y_col="sample_gain_K20",
            sat_col=None,
            size_col="size_for_plot",
            size_label="data size",
            xlabel=(
                r"$\widehat{\mathrm{Cov}}_{X_{\mathrm{study}}}"
                r"(\hat y_{f_1},\hat y_{f_2})$ "
                r"(prediction covariance across folds, on study inputs)"
            ),
            ax=ax_sc,
            title=(
                r"Study-prediction covariance vs observed sample gain "
                r"at $K=20$ (simulated only)"
            ),
        )
        _reserve_right_legend_space(fig_sc)
        path_sc = os.path.join(plots_dir, "study_cov_vs_gain.pdf")
        fig_sc.savefig(path_sc, bbox_inches="tight")
        print(f"Saved {path_sc}")
        _finish_plot(fig_sc, "study_cov_vs_gain")

        df["log10_study_cov"] = np.log10(df["study_cov"].clip(lower=1e-12))
        fig_sc_log, ax_sc_log = plt.subplots(figsize=(7.5, 5.0))
        P.plot_tau_vs_gain_encoded(
            df,
            x_col="log10_study_cov",
            y_col="sample_gain_K20",
            sat_col=None,
            size_col="size_for_plot",
            size_label="data size",
            xlabel=(
                r"$\log_{10}\widehat{\mathrm{Cov}}_{X_{\mathrm{study}}}"
                r"(\hat y_{f_1},\hat y_{f_2})$"
            ),
            ax=ax_sc_log,
            title=(
                r"$\log_{10}$(study-prediction covariance) vs observed "
                r"sample gain at $K=20$ (simulated only)"
            ),
        )
        _reserve_right_legend_space(fig_sc_log)
        path_sc_log = os.path.join(plots_dir, "log10_study_cov_vs_gain.pdf")
        fig_sc_log.savefig(path_sc_log, bbox_inches="tight")
        print(f"Saved {path_sc_log}")
        _finish_plot(fig_sc_log, "log10_study_cov_vs_gain")

        summary = _summarise_cov_vs_gain(
            df,
            cov_col="study_cov",
            plots_dir=plots_dir,
            x_math_symbol=(
                r"$\widehat{\mathrm{Cov}}_{X_{\mathrm{study}}}"
                r"(\hat y_{f_1},\hat y_{f_2})$"
            ),
            pretty_name="study-prediction covariance",
            highlight_mlp=True,
        )
        if summary:
            covariance_gain_summaries.append({"signal": "study_cov", **summary})

    # ── Error covariance diagnostics vs observed sample gain ───────────────
    if {"bench_cov", "bench_error_rho"}.issubset(df.columns):
        df["bench_cov_times_error_rho"] = df["bench_cov"] * df["bench_error_rho"]
    if {"study_cov", "study_error_rho"}.issubset(df.columns):
        df["study_cov_times_error_rho"] = df["study_cov"] * df["study_error_rho"]
    if {"bench_cov", "bench_squared_error_rho"}.issubset(df.columns):
        df["bench_cov_times_squared_error_rho"] = (
            df["bench_cov"] * df["bench_squared_error_rho"]
        )
    if {"study_cov", "study_squared_error_rho"}.issubset(df.columns):
        df["study_cov_times_squared_error_rho"] = (
            df["study_cov"] * df["study_squared_error_rho"]
        )

    for cov_col, x_math_symbol, pretty_name in (
        ERROR_COVARIANCE_SIGNAL_SPECS
        + DERIVED_COVARIANCE_SIGNAL_SPECS
        + BENCHMARK_ONLY_SIGNAL_SPECS
    ):
        if cov_col not in df.columns or not df[cov_col].notna().any():
            continue
        valid_cov = df.dropna(subset=[cov_col, "sample_gain_K20"])
        valid_positive = valid_cov[valid_cov[cov_col] > 0]
        print(
            f"{cov_col} available for {len(valid_cov)}/{len(df)} points; "
            f"{len(valid_positive)} positive points used for log-log summaries."
        )
        summary = _summarise_cov_vs_gain(
            df,
            cov_col=cov_col,
            plots_dir=plots_dir,
            x_math_symbol=x_math_symbol,
            pretty_name=pretty_name,
        )
        if summary:
            covariance_gain_summaries.append({"signal": cov_col, **summary})

    icc_gain_proxy_metrics: list[dict[str, float]] = []
    icc_signal_specs = [
        (
            "bench_cov_times_error_rho",
            "rho_delta_proxy_bench_cov_error_rho",
            "predicted_sample_gain_K20_bench_cov_error_rho_icc",
            (
                r"$\widehat{\mathrm{Cov}}_{X_{\mathrm{bench}}}"
                r"(\hat y_{f_1},\hat y_{f_2})"
                r"\,\widehat\rho_{X_{\mathrm{bench}},\mathrm{err}}$"
            ),
            (
                r"ICC plug-in gain curve from "
                r"$\mathrm{bench\_cov}\times\mathrm{bench\_error\_rho}$"
            ),
            "bench_cov_times_error_rho_icc_gain_curve.pdf",
            "bench_cov_times_error_rho_icc_curve",
            (
                r"ICC plug-in predicted sample gain "
                r"from $\mathrm{bench\_cov}\times\mathrm{bench\_error\_rho}$"
            ),
            "predicted_gain_bench_cov_error_rho_icc_vs_observed.pdf",
            "predicted_gain_bench_cov_error_rho_icc",
        ),
        (
            "bench_cov_times_squared_error_rho",
            "rho_delta_proxy_bench_cov_squared_error_rho",
            "predicted_sample_gain_K20_bench_cov_squared_error_rho_icc",
            (
                r"$\widehat{\mathrm{Cov}}_{X_{\mathrm{bench}}}"
                r"(\hat y_{f_1},\hat y_{f_2})"
                r"\,\widehat\rho_{X_{\mathrm{bench}},e^2}$"
            ),
            (
                r"ICC plug-in gain curve from "
                r"$\mathrm{bench\_cov}\times\mathrm{bench\_squared\_error\_rho}$"
            ),
            "bench_cov_times_squared_error_rho_icc_gain_curve.pdf",
            "bench_cov_times_squared_error_rho_icc_curve",
            (
                r"ICC plug-in predicted sample gain from "
                r"$\mathrm{bench\_cov}\times\mathrm{bench\_squared\_error\_rho}$"
            ),
            "predicted_gain_bench_cov_squared_error_rho_icc_vs_observed.pdf",
            "predicted_gain_bench_cov_squared_error_rho_icc",
        ),
    ]
    for (
        signal_col,
        pred_rho_col,
        pred_gain_col,
        x_label,
        curve_title,
        curve_filename,
        curve_plot_id,
        pred_title,
        pred_filename,
        pred_plot_id,
    ) in icc_signal_specs:
        if signal_col not in df.columns or not df[signal_col].notna().any():
            continue
        icc_metrics = _fit_icc_gain_proxy_from_signal(
            df,
            signal_col,
            pred_rho_col=pred_rho_col,
            pred_gain_col=pred_gain_col,
        )
        if icc_metrics:
            icc_gain_proxy_metrics.append(icc_metrics)
            _plot_icc_gain_curve_from_signal(
                df,
                signal_col,
                icc_metrics,
                xlabel=x_label,
                title=curve_title,
                output_path=os.path.join(plots_dir, curve_filename),
                plot_id=curve_plot_id,
            )
            fig_icc = P.plot_predicted_vs_observed_gain(
                df,
                x_col=pred_gain_col,
                y_col="sample_gain_K20",
                sat_col=None,
                size_col="size_for_plot",
                title=pred_title,
                xlabel=r"Predicted gain from fitted $\hat\rho_\delta$ and ICC formula",
                output_path=os.path.join(plots_dir, pred_filename),
            )
            _reserve_right_legend_space(fig_icc)
            _finish_plot(fig_icc, pred_plot_id)

    if icc_gain_proxy_metrics:
        icc_metrics_path = os.path.join(results_dir, "icc_gain_proxy_metrics.csv")
        pd.DataFrame(icc_gain_proxy_metrics).to_csv(icc_metrics_path, index=False)
        print("\nICC plug-in gain proxy metrics:")
        print(pd.DataFrame(icc_gain_proxy_metrics).to_string(index=False))
        print(f"Wrote {icc_metrics_path}")

    if covariance_gain_summaries:
        covariance_gain_metrics = (
            pd.DataFrame(covariance_gain_summaries)
            .sort_values(["loglog_R2", "signal"], ascending=[False, True])
        )
        covariance_gain_metrics_path = os.path.join(
            results_dir,
            "covariance_signal_gain_metrics.csv",
        )
        covariance_gain_metrics.to_csv(covariance_gain_metrics_path, index=False)
        print("\nCovariance/error signal vs sample-gain summaries:")
        print(covariance_gain_metrics.to_string(index=False))
        print(f"Wrote {covariance_gain_metrics_path}")

    # ── Exploratory raw-score predictors vs observed delta gain (A–M) ──────

    fig_a, ax_a = plt.subplots(figsize=(7.5, 5.0))
    P.plot_tau_vs_gain_encoded(
        df,
        x_col="tau_over_sigma2_HO_K",
        y_col="sample_gain_K20",
        sat_col=None,
        size_col="size_for_plot",
        size_label="data size",
        xlabel=r"$\hat\tau / \hat\sigma^2_{\mathrm{HO}\mid K}$ (raw-score inter-fold cov. / CV per-fold var.)",
        ax=ax_a,
        title=r"Exploratory raw-score ratio $\hat\tau/\hat\sigma^2_{\mathrm{HO}\mid K}$ vs observed $\delta$-gain at $K=20$",
    )
    _reserve_right_legend_space(fig_a)
    path_a = os.path.join(plots_dir, "tau_over_sigma2_HO_K_vs_gain.pdf")
    _finish_plot(fig_a, "tau_over_sigma2_HO_K_vs_gain", path_a)

    fig_b, ax_b = plt.subplots(figsize=(7.5, 5.0))
    P.plot_tau_vs_gain_encoded(
        df,
        x_col="W_over_B",
        y_col="sample_gain_K20",
        sat_col=None,
        size_col="size_for_plot",
        size_label="data size",
        xlabel=r"$\hat{W} / \hat{B}$ (raw-score within-fold var. / between-fold var.)",
        ax=ax_b,
        title=r"Exploratory raw-score ANOVA ratio $\hat{W}/\hat{B}$ vs observed $\delta$-gain at $K=20$",
    )
    _reserve_right_legend_space(fig_b)
    path_b = os.path.join(plots_dir, "W_over_B_vs_gain.pdf")
    _finish_plot(fig_b, "W_over_B_vs_gain", path_b)

    pairplot_vars = [
        "tau",
        "sigma2_HO_K",
        "sigma2_HO_1",
        "W",
        "B",
        "tau_over_sigma2_HO_K",
        "predicted_sample_gain_K20",
        "sample_gain_K20",
    ]
    g = P.plot_pairplot_tau_gain(
        df,
        vars_=pairplot_vars,
        hue_col="pipeline",
        output_path=os.path.join(plots_dir, "pairplot_anova_vs_gain.pdf"),
    )
    _finish_plot(g.figure, "pairplot_anova_vs_gain")

    fig_d, ax_d = plt.subplots(figsize=(7.5, 5.0))
    P.plot_tau_vs_gain_encoded(
        df,
        x_col="B",
        y_col="sample_gain_K20",
        size_col="size_for_plot",
        size_label="data size",
        xlabel=r"$\hat{B} = \mathrm{Var}_s(\bar{X}_s)$ (raw-score between-fold variance)",
        ax=ax_d,
        title=r"Exploratory raw-score between-fold variance $\hat{B}$ vs observed $\delta$-gain at $K=20$",
    )
    _reserve_right_legend_space(fig_d)
    path_d = os.path.join(plots_dir, "B_vs_gain.pdf")
    _finish_plot(fig_d, "B_vs_gain", path_d)

    fig_e = P.plot_tau_vs_gain_studysize(
        df,
        x_col="tau_over_sigma2_HO_K",
        y_col="sample_gain_K20",
        size_col="size_for_plot",
        size_label="data size",
        xlabel=r"$\hat\tau / \hat\sigma^2_{\mathrm{HO}\mid K}$ (raw-score inter-fold cov. / CV per-fold var.)",
        title=r"Exploratory raw-score ratio $\hat\tau/\hat\sigma^2_{\mathrm{HO}\mid K}$ vs observed $\delta$-gain, coloured by study size",
        output_path=os.path.join(plots_dir, "tau_over_sigma2_HO_K_vs_gain_by_N.pdf"),
    )
    _finish_plot(fig_e, "tau_over_sigma2_HO_K_vs_gain_by_N")

    valid_pred = df.dropna(subset=["predicted_sample_gain_K20", "sample_gain_K20"])
    print(f"Predicted gain (delta-based) available for {len(valid_pred)}/{len(df)} points")

    fig_f = P.plot_predicted_vs_observed_gain(
        df,
        x_col="predicted_sample_gain_K20",
        y_col="sample_gain_K20",
        sat_col=None,
        size_col="size_for_plot",
        title=r"$\delta$-based predicted vs observed sample gain at $K=20$",
        output_path=os.path.join(plots_dir, "predicted_vs_observed_gain.pdf"),
    )
    _reserve_right_legend_space(fig_f)
    _finish_plot(fig_f, "predicted_delta")

    valid_pred_raw = df.dropna(subset=["predicted_sample_gain_K20_raw", "sample_gain_K20"])
    print(f"Predicted gain (raw exploratory) available for {len(valid_pred_raw)}/{len(df)} points")

    fig_f_raw = P.plot_predicted_vs_observed_gain(
        df,
        x_col="predicted_sample_gain_K20_raw",
        y_col="sample_gain_K20",
        sat_col=None,
        size_col="size_for_plot",
        title=r"Exploratory raw-score predicted vs observed $\delta$-gain at $K=20$",
        output_path=os.path.join(plots_dir, "predicted_raw_vs_observed_gain.pdf"),
    )
    _reserve_right_legend_space(fig_f_raw)
    _finish_plot(fig_f_raw, "predicted_raw")

    fig_n, ax_n = plt.subplots(figsize=(7.5, 5.0))
    P.plot_tau_vs_gain_encoded(
        df,
        x_col="rho_delta",
        y_col="sample_gain_K20",
        sat_col=None,
        size_col="size_for_plot",
        size_label="data size",
        xlabel=r"$\hat\tau^{te}/\hat\sigma^{2,\mathrm{te}}_{\mid K}$ ($\delta$-based inter-fold cov. / CV per-fold var.)",
        ax=ax_n,
        title=r"$\rho_\delta = \hat\tau^{te}/\hat\sigma^{2,\mathrm{te}}_{\mid K}$ vs observed $\delta$-gain at $K=20$",
        ols_line=False,
    )
    valid_n = df.dropna(subset=["rho_delta", "sample_gain_K20"])
    if not valid_n.empty:
        xv = valid_n["rho_delta"]
        xv_pos = xv[xv > 0]
        if not xv_pos.empty:
            ax_n.set_xscale("symlog", linthresh=max(xv_pos.min(), 1e-6))
        x_ref = np.linspace(max(xv.min(), 0), xv.max(), 200)
        ax_n.plot(
            x_ref,
            K / (1 + (K - 1) * x_ref),
            "k--",
            linewidth=1.2,
            alpha=0.7,
            label=rf"$K/(1+(K-1)\rho_\delta)$, $K={K}$",
        )
        ax_n.legend(loc="best", fontsize=8)
    _reserve_right_legend_space(fig_n)
    path_n = os.path.join(plots_dir, "rho_delta_vs_gain.pdf")
    _finish_plot(fig_n, "rho_delta_vs_gain", path_n)

    if "sigma2_HO_1_over_sigma2_HO_trxi" in df.columns:
        df["resid_curve"] = (
            df["sample_gain_K20"]
            - K / (1 + (K - 1) * df["rho_delta"])
        )
        fig_o, ax_o = plt.subplots(figsize=(7.5, 5.0))
        P.plot_tau_vs_gain_encoded(
            df,
            x_col="sigma2_HO_1_over_sigma2_HO_trxi",
            y_col="resid_curve",
            sat_col=None,
            size_col="size_for_plot",
            size_label="data size",
            xlabel=r"Seed-noise inflation $\hat\sigma^2_{\mathrm{HO}\mid 1} / \hat\sigma^2_{\mathrm{HO}\mid \mathrm{tr},\xi}$",
            ylabel=r"Residual (observed $\delta$-gain) $-\ K/(1+(K-1)\rho_\delta)$",
            ax=ax_o,
            title=r"Outlier diagnostic: raw-score seed-noise inflation vs $\tau^{te}/\sigma^{2,\mathrm{te}}_{\mid K}$ residual",
            ols_line=True,
        )
        valid_o = df.dropna(subset=["sigma2_HO_1_over_sigma2_HO_trxi", "resid_curve"])
        if not valid_o.empty:
            xv = valid_o["sigma2_HO_1_over_sigma2_HO_trxi"]
            xv_pos = xv[xv > 0]
            if not xv_pos.empty:
                ax_o.set_xscale("log")
            ax_o.axhline(0, color="k", linestyle="--", linewidth=1, alpha=0.5)
            ax_o.axvline(1, color="k", linestyle=":", linewidth=1, alpha=0.4)
        _reserve_right_legend_space(fig_o)
        path_o = os.path.join(plots_dir, "sigma2_HO_1_over_sigma2_HO_trxi_vs_resid.pdf")
        _finish_plot(fig_o, "seed_noise_inflation_vs_resid", path_o)

    fig_g, ax_g = plt.subplots(figsize=(7.5, 5.0))
    P.plot_tau_vs_gain_encoded(
        df,
        x_col="sigma2_HO_1_over_sigma2_HO_K",
        y_col="sample_gain_K20",
        sat_col=None,
        size_col="size_for_plot",
        size_label="data size",
        xlabel=r"$\hat\sigma^2_{\mathrm{HO}\mid 1} / \hat\sigma^2_{\mathrm{HO}\mid K}$ (single-split across-seed var. / CV per-fold var.)",
        ax=ax_g,
        title=r"Exploratory raw-score variance ratio $\hat\sigma^2_{\mathrm{HO}\mid 1}/\hat\sigma^2_{\mathrm{HO}\mid K}$ vs observed $\delta$-gain at $K=20$",
    )
    _reserve_right_legend_space(fig_g)
    path_g = os.path.join(plots_dir, "sigma2_HO_1_over_sigma2_HO_K_vs_gain.pdf")
    _finish_plot(fig_g, "sigma2_HO_1_over_sigma2_HO_K_vs_gain", path_g)

    fig_h, ax_h = plt.subplots(figsize=(7.5, 5.0))
    P.plot_tau_vs_gain_encoded(
        df,
        x_col="sigma2_HO_1",
        y_col="sample_gain_K20",
        sat_col=None,
        size_col="size_for_plot",
        size_label="data size",
        xlabel=r"$\hat\sigma^2_{\mathrm{HO}\mid 1}$ (raw-score single-split variance across seeds)",
        ax=ax_h,
        title=r"Exploratory raw-score variance $\hat\sigma^2_{\mathrm{HO}\mid 1}$ vs observed $\delta$-gain at $K=20$",
    )
    _reserve_right_legend_space(fig_h)
    path_h = os.path.join(plots_dir, "sigma2_HO_1_vs_gain.pdf")
    _finish_plot(fig_h, "sigma2_HO_1_vs_gain", path_h)

    fig_i, ax_i = plt.subplots(figsize=(7.5, 5.0))
    P.plot_tau_vs_gain_encoded(
        df,
        x_col="mean_bench_score",
        y_col="sample_gain_K20",
        sat_col=None,
        size_col="size_for_plot",
        size_label="data size",
        xlabel=r"Mean benchmark score (raw-score algorithm performance)",
        ax=ax_i,
        title=r"Mean benchmark score (raw) vs observed $\delta$-gain at $K=20$",
    )
    _reserve_right_legend_space(fig_i)
    path_i = os.path.join(plots_dir, "bench_score_vs_gain.pdf")
    _finish_plot(fig_i, "bench_score_vs_gain", path_i)

    real_df = df[df["pipeline"] != "simulated"]
    if not real_df.empty:
        fig_e2 = P.plot_tau_vs_gain_studysize(
            real_df,
            x_col="tau_over_sigma2_HO_K",
            y_col="sample_gain_K20",
            size_col="size_for_plot",
            size_label="data size",
            xlabel=r"$\hat\tau / \hat\sigma^2_{\mathrm{HO}\mid K}$ (raw-score inter-fold cov. / CV per-fold var.)",
            title=r"Exploratory raw-score ratio $\hat\tau/\hat\sigma^2_{\mathrm{HO}\mid K}$ vs observed $\delta$-gain, coloured by study size (no simulated)",
            output_path=os.path.join(plots_dir, "tau_over_sigma2_HO_K_vs_gain_by_N_no_sim.pdf"),
        )
        _finish_plot(fig_e2, "tau_over_sigma2_HO_K_vs_gain_by_N_no_sim")

    if "sigma2_HO_trxi" in df.columns:
        fig_j = P.plot_predicted_vs_observed_gain(
            df,
            x_col="predicted_sample_gain_K20_trxi",
            y_col="sample_gain_K20",
            sat_col=None,
            size_col="size_for_plot",
            title=r"Exploratory within-seed raw-score predicted vs observed $\delta$-gain at $K=20$",
            output_path=os.path.join(plots_dir, "predicted_trxi_vs_observed_gain.pdf"),
        )
        _reserve_right_legend_space(fig_j)
        _finish_plot(fig_j, "predicted_trxi")

        fig_k, ax_k = plt.subplots(figsize=(7.5, 5.0))
        P.plot_tau_vs_gain_encoded(
            df,
            x_col="sigma2_HO_trxi",
            y_col="sample_gain_K20",
            sat_col=None,
            size_col="size_for_plot",
            size_label="data size",
            xlabel=r"$\hat\sigma^2_{\mathrm{HO}\mid \mathrm{tr},\xi}$ (raw-score within-seed single-split variance)",
            ax=ax_k,
            title=r"Exploratory raw-score within-seed variance $\hat\sigma^2_{\mathrm{HO}\mid \mathrm{tr},\xi}$ vs observed $\delta$-gain at $K=20$",
        )
        _reserve_right_legend_space(fig_k)
        path_k = os.path.join(plots_dir, "sigma2_HO_trxi_vs_gain.pdf")
        _finish_plot(fig_k, "sigma2_HO_trxi_vs_gain", path_k)

        fig_l, ax_l = plt.subplots(figsize=(7.5, 5.0))
        P.plot_tau_vs_gain_encoded(
            df,
            x_col="sigma2_HO_trxi_over_sigma2_HO_K",
            y_col="sample_gain_K20",
            sat_col=None,
            size_col="size_for_plot",
            size_label="data size",
            xlabel=r"$\hat\sigma^2_{\mathrm{HO}\mid \mathrm{tr},\xi} / \hat\sigma^2_{\mathrm{HO}\mid K}$",
            ax=ax_l,
            title=r"Exploratory raw-score variance ratio $\hat\sigma^2_{\mathrm{HO}\mid \mathrm{tr},\xi}/\hat\sigma^2_{\mathrm{HO}\mid K}$ vs observed $\delta$-gain at $K=20$",
        )
        _reserve_right_legend_space(fig_l)
        path_l = os.path.join(plots_dir, "sigma2_HO_trxi_over_sigma2_HO_K_vs_gain.pdf")
        _finish_plot(fig_l, "sigma2_HO_trxi_over_sigma2_HO_K_vs_gain", path_l)

        fig_m, ax_m = plt.subplots(figsize=(6.0, 6.0))
        valid = df.dropna(subset=["sigma2_HO_1", "sigma2_HO_trxi"])
        for pipeline_name, sub in valid.groupby("pipeline"):
            ax_m.scatter(
                sub["sigma2_HO_1"],
                sub["sigma2_HO_trxi"],
                label=pipeline_name,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.4,
            )
        if not valid.empty:
            lim_lo = min(valid["sigma2_HO_1"].min(), valid["sigma2_HO_trxi"].min())
            lim_hi = max(valid["sigma2_HO_1"].max(), valid["sigma2_HO_trxi"].max())
            ax_m.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", linewidth=1, alpha=0.6)
            ax_m.set_xscale("log")
            ax_m.set_yscale("log")
        ax_m.set_xlabel(r"$\hat\sigma^2_{\mathrm{HO}\mid 1}$ (across seeds)")
        ax_m.set_ylabel(r"$\hat\sigma^2_{\mathrm{HO}\mid \mathrm{tr},\xi}$ (within seed)")
        ax_m.legend(title="pipeline", loc="best", fontsize=8)
        ax_m.set_title(r"$\hat\sigma^2_{\mathrm{HO}\mid 1}$ across seeds vs $\hat\sigma^2_{\mathrm{HO}\mid \mathrm{tr},\xi}$ within seed")
        path_m = os.path.join(plots_dir, "sigma2_HO_1_across_vs_within.pdf")
        _finish_plot(fig_m, "sigma2_HO_1_across_vs_within", path_m)

    _plot_simulated_relationship_outputs(results_dir, plots_dir, K)
    _plot_study_only_redundancy_rule(results_dir, plots_dir)
    _plot_study_redundancy_high_gain_probability(results_dir, plots_dir)
    _plot_study_score_transform_comparison(results_dir, plots_dir)
    _plot_study_only_search_outputs(results_dir, plots_dir)
    _plot_followup_exact_probe_outputs(results_dir, plots_dir)

    return df


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Pooled raw-τ vs sample gain at K=20 across datasets."
    )
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--plots-dir", default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--K", type=int, default=K_DEFAULT)
    return parser.parse_args(argv)


def _running_in_jupyter():
    try:
        get_ipython()  # noqa: F821
        return True
    except NameError:
        return False


if __name__ == "__main__":
    if _running_in_jupyter():
        main()
    else:
        args = parse_args()
        main(results_dir=args.results_dir, plots_dir=args.plots_dir, K=args.K)

# %%
