"""
nlp.py — Validation Crisis analysis on Yelp (NLP / text classification).

Data pipeline:
  1. Load parquet files from multiple directories; rename columns to canonical names.
  2. Build data_nlp with expanding-mean n_splits column (procedure always ShuffleSplit).
  3. Separate CV and no-CV data.
  4. Expand hidden test accuracies; build combined no-CV DataFrames.
  5. Run bootstrapped sample gain analysis (delta-bench score).
  6. Build variance diagnostic DataFrames.
  7. Generate all plots.
"""

# %%
import argparse
import gc
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

import compute
import pipeline_common as C
import plot as P

# %% ──────────────────────────────────────────────────────────────────────────
# Default configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DATA_DIRS = [
    '/home/ceve/Documents/Run outputs/NLP/data/nlp(1)/results_output/parquet',
    '/home/ceve/Documents/Run outputs/NLP/data/nlp/results_output/parquet',

    '/home/ceve/Documents/Run outputs/NLP/data/Qwen3-0.6B',
]

DEFAULT_PLOTS_DIR = "plots/nlp"
DEFAULT_MODELS = ['bert', 'roberta', 'qwen3-0.6b']
DEFAULT_STUDY_SIZES = [1250, 3750, 12500]
THREE_PANEL_SAMPLE_GAIN_AXIS_FONTSIZE = 17
THREE_PANEL_SAMPLE_GAIN_TICK_FONTSIZE = 14
THREE_PANEL_SAMPLE_GAIN_LABEL_FONTSIZE = 13
THREE_PANEL_SAMPLE_GAIN_TITLE_FONTSIZE = 13

MODEL_DISPLAY_NAMES = {'bert': 'BERT', 'roberta': 'XLM-RoBERTa', 'qwen3-0.6b': 'Qwen3-0.6B'}

# %% ──────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _find_parquet_files(directories: list[str]) -> list[str]:
    paths = []
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Warning: directory not found: {directory}")
            continue
        for root, _, files in os.walk(directory):
            for f in files:
                if f.endswith('.parquet'):
                    paths.append(os.path.join(root, f))
    return paths


def load_nlp_data(directories: list[str]) -> pd.DataFrame:
    """Load NLP parquet files from multiple directories and rename columns."""
    file_paths = _find_parquet_files(directories)
    if not file_paths:
        print("No parquet files found.")
        return pd.DataFrame()

    print(f"Loading {len(file_paths)} parquet files...")
    dfs = []
    for i, fp in enumerate(file_paths):
        try:
            dfs.append(pd.read_parquet(fp))
        except Exception as e:
            print(f"Error reading {fp}: {e}")
        if (i + 1) % 20 == 0:
            gc.collect()

    if not dfs:
        return pd.DataFrame()

    data = pd.concat(dfs, ignore_index=True, sort=True)
    gc.collect()

    # Rename columns to canonical names used throughout the pipeline
    rename_map = {
        'model_arg_choice':      'model',
        'study_set_size':        'study_size',
        'fold_test_eval_accuracy': 'test_accuracy',
        'benchmark_eval_accuracy': 'benchmark_accuracy',
        'fold_index':            'fold',
    }
    data.rename(columns={k: v for k, v in rename_map.items()
                          if k in data.columns and v not in data.columns},
                inplace=True)

    print(f"Loaded {len(data)} rows  columns: {data.columns.tolist()}")
    return data


def _extract_scalar(x):
    if isinstance(x, (list, np.ndarray)) and len(x) > 0:
        return x[-1]
    return x if np.isscalar(x) else np.nan


def _infer_cross_validation(data: pd.DataFrame) -> pd.Series:
    """Infer the CV/no-CV flag for older parquet schemas."""
    if "cross_validation" in data.columns:
        return data["cross_validation"].astype(bool)

    if "n_splits_cv" in data.columns:
        n_splits = pd.to_numeric(data["n_splits_cv"], errors="coerce")
        if n_splits.notna().any():
            return n_splits.fillna(1).astype(int) > 1

    if "hidden_test_accuracies" in data.columns:
        has_hidden = data["hidden_test_accuracies"].apply(
            lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0
        )
        if has_hidden.any():
            return ~has_hidden

    raise KeyError(
        "Cannot infer 'cross_validation': expected either "
        "'cross_validation', 'n_splits_cv', or informative "
        "'hidden_test_accuracies'."
    )


def _plot_yelp_pairwise_score_distributions(
    data_nocv_base: pd.DataFrame,
    data_cv: pd.DataFrame,
    *,
    study_size: int,
    target_splits: list[int],
    plots_dir: str,
) -> None:
    """Create the score-distribution panel used in the main pairwise figure."""
    # This is a 2-model split-violin (BERT vs XLM-RoBERTa); the palette and
    # oracle annotations below are defined for exactly these two. Other models
    # (e.g. Qwen3-0.6B) are handled in the bar/retrieval panels, not here.
    pair_models = ["bert", "roberta"]
    rows = []
    for model in pair_models:
        subset = data_nocv_base[
            (data_nocv_base["study_size"] == study_size)
            & (data_nocv_base["model"] == model)
        ]
        for score in subset["objective_score_test"].dropna():
            rows.append({
                "K": "1",
                "model": MODEL_DISPLAY_NAMES.get(model, model),
                "score": float(score),
            })

    for split in target_splits:
        subset = data_cv[
            (data_cv["study_size"] == study_size)
            & (data_cv["n_splits"] == split)
            & (data_cv["model"].isin(pair_models))
        ].copy()
        subset["score_scalar"] = subset["objective_score_test"].apply(_extract_scalar)
        for _, row in subset.iterrows():
            rows.append({
                "K": str(split),
                "model": MODEL_DISPLAY_NAMES.get(row["model"], row["model"]),
                "score": float(row["score_scalar"]),
            })

    dist_df = pd.DataFrame(rows)
    if dist_df.empty:
        return

    order = ["1"] + [str(split) for split in target_splits]
    hue_order = [MODEL_DISPLAY_NAMES.get(model, model) for model in pair_models]
    palette = {
        MODEL_DISPLAY_NAMES["bert"]: "#4C78A8",
        MODEL_DISPLAY_NAMES["roberta"]: "#F58518",
    }

    fig, ax = plt.subplots(figsize=(3.2, 2.55))
    sns.violinplot(
        data=dist_df, x="K", y="score", hue="model",
        order=order, hue_order=hue_order, palette=palette,
        split=True, inner="quartile", bw_adjust=0.9, linewidth=1.0,
        saturation=0.95, ax=ax,
    )
    sns.stripplot(
        data=dist_df, x="K", y="score", hue="model",
        order=order, hue_order=hue_order, palette=palette,
        dodge=True, jitter=0.10, size=1.9, alpha=0.24,
        linewidth=0, ax=ax, legend=False,
    )

    oracle = (
        data_nocv_base[data_nocv_base["study_size"] == study_size]
        .groupby("model")["benchmark_accuracy"]
        .mean()
    )
    oracle_positions = []
    for model in pair_models:
        if model not in oracle:
            continue
        display = MODEL_DISPLAY_NAMES.get(model, model)
        value = float(oracle.loc[model])
        oracle_positions.append([display, value, palette[display]])
        ax.axhline(
            value, color=palette[display], linestyle="--",
            linewidth=1.35, alpha=0.95, zorder=0,
        )

    y_axis_min = max(0.0, dist_df["score"].min() - 0.08)
    y_axis_max = min(1.0, dist_df["score"].max() + 0.05)
    multi_split_scores = dist_df[dist_df["K"] != "1"]["score"]
    legend_center_y = 0.5 * (y_axis_min + multi_split_scores.min())

    oracle_positions.sort(key=lambda item: item[1])
    ax.plot(
        [1.10, 1.46], [legend_center_y + 0.042, legend_center_y + 0.042],
        color="0.25", linestyle="--", linewidth=1.35,
        clip_on=True,
    )
    ax.text(
        1.54, legend_center_y + 0.042, "Quasi-oracle score",
        color="0.25", ha="left", va="center", fontsize=8.1,
        path_effects=P._white_text_halo(),
    )
    annotation_y = {
        MODEL_DISPLAY_NAMES["bert"]: legend_center_y,
        MODEL_DISPLAY_NAMES["roberta"]: legend_center_y - 0.042,
    }
    for display, value, color in oracle_positions:
        ax.text(
            1.54, annotation_y.get(display, 0.315),
            f"{display} {value * 100:.1f}%",
            color=color, ha="left", va="center", fontsize=8.1,
            path_effects=P._white_text_halo(),
        )

    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    ax.set_xlim(-0.50, 3.50)
    ax.set_ylim(y_axis_min, y_axis_max)
    ax.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Number of splits (K)")
    ax.set_ylabel("Test accuracy")
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    sns.despine(ax=ax)

    train_size = int(study_size * 0.8)
    for stem in [f"violin_plot_study_size_{study_size}",
                 f"violin_plot_study_size_N{train_size}"]:
        P.save_figure(fig, f"{plots_dir}/{stem}.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


# %% ──────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validation Crisis analysis on NLP (Yelp) data."
    )
    parser.add_argument(
        "--data-dirs", nargs="+", default=DEFAULT_DATA_DIRS,
        help="Directories to search recursively for parquet files.",
    )
    parser.add_argument("--plots-dir", default=DEFAULT_PLOTS_DIR,
                        help="Output directory for plots.")
    parser.add_argument("--n-bootstraps", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--hairlines", type=int, default=None)
    parser.add_argument("--save-gains", default=None, help="directory for the gain tables (estimate + draws)")
    parser.add_argument("--plugin-estimate", action="store_true",
                        help="also compute the estimate on the original seeds; the figures draw it as the thick line")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--study-sizes", nargs="+", type=int,
                        default=DEFAULT_STUDY_SIZES)
    return parser.parse_args(argv)


# %% ──────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _save_gains(res, save_dir, tag):
    """Write a gain table (estimate rows bootstrap_iter = -1, draws >= 0) for the CI appendix (CE, 2026-10-31)."""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        res.to_parquet(os.path.join(save_dir, re.sub(r"[^A-Za-z0-9_.-]+", "_", tag) + ".parquet"), index=False)


def main(
    data_dirs=None,
    plots_dir=DEFAULT_PLOTS_DIR,
    n_bootstraps=100,
    test_size=0.2,
    models_to_analyze=None,
    study_sizes=None,
    random_state=None,
    hairlines=None,
    plugin_estimate=False,
    save_gains=None,
):
    if hairlines:
        P.GAIN_DISPLAY["hairlines"] = hairlines
    if data_dirs is None:
        data_dirs = DEFAULT_DATA_DIRS
    if models_to_analyze is None:
        models_to_analyze = DEFAULT_MODELS
    if study_sizes is None:
        study_sizes = DEFAULT_STUDY_SIZES

    os.makedirs(plots_dir, exist_ok=True)
    P.set_paper_style()

    target_splits_diag = [5, 10, 20]

    # ── 1. Load data ────────────────────────────────────────────────────────
    data = load_nlp_data(data_dirs)

    if data.empty:
        raise RuntimeError("No data loaded — check data_dirs.")

    print(f"Seeds: {len(data['seed'].unique())}")
    print(f"Models: {sorted(data['model'].unique()) if 'model' in data.columns else 'no model column'}")

    # ── 2. Build data_nlp with expanding-mean n_splits ──────────────────────
    group_cols = ['seed', 'model', 'study_size', 'cross_validation']
    cols_to_copy = group_cols + [
        'hidden_test_accuracies', 'fold',
        'study_test_eval_accuracy', 'validation_eval_accuracy', 'benchmark_accuracy',
    ]
    cols_to_copy = [c for c in cols_to_copy if c in data.columns]

    if 'cross_validation' not in data.columns:
        print(
            "Warning: 'cross_validation' column missing — "
            "inferring from n_splits_cv / hidden-score presence."
        )
        data['cross_validation'] = _infer_cross_validation(data)

    data_nlp = (
        data[cols_to_copy]
        .drop_duplicates(subset=group_cols + ['fold'])
        .sort_values(group_cols + ['fold'])
        .copy()
    )
    data_nlp = C.coerce_scalar_columns(
        data_nlp,
        columns=[
            'study_test_eval_accuracy',
            'validation_eval_accuracy',
            'benchmark_accuracy',
        ],
        scalar_extractor=_extract_scalar,
    )
    grouped_nlp = data_nlp.groupby(group_cols, sort=False)
    data_nlp['n_splits'] = grouped_nlp.cumcount() + 1
    data_nlp['objective_score_test'] = (
        grouped_nlp['study_test_eval_accuracy'].cumsum() / data_nlp['n_splits']
    )
    if 'validation_eval_accuracy' in data_nlp.columns:
        data_nlp['objective_score_val'] = (
            grouped_nlp['validation_eval_accuracy'].cumsum() / data_nlp['n_splits']
        )
    else:
        data_nlp['objective_score_val'] = np.nan
    data_nlp['objective_score_bench'] = (
        grouped_nlp['benchmark_accuracy'].cumsum() / data_nlp['n_splits']
    )
    data_nlp['procedure'] = 'ShuffleSplit'
    data_nlp['n_repeats'] = 1
    data_nlp = data_nlp.loc[:, [
        'seed', 'model', 'study_size', 'n_splits',
        'objective_score_test', 'objective_score_val', 'objective_score_bench',
        'procedure', 'n_repeats', 'hidden_test_accuracies', 'cross_validation',
        'study_test_eval_accuracy', 'benchmark_accuracy',
    ]]

    # ── 3. Separate CV and no-CV; filter to target study sizes ──────────────
    data_nlp_cv   = data_nlp[data_nlp['cross_validation'] == True].copy()
    data_nlp_nocv = data_nlp[data_nlp['cross_validation'] == False].copy()

    study_sizes_available = sorted(data_nlp['study_size'].unique())
    print(f"Study sizes: {study_sizes_available}")

    data_nlp_cv   = data_nlp_cv[data_nlp_cv['study_size'].isin(study_sizes)]
    data_nlp_nocv = data_nlp_nocv[data_nlp_nocv['study_size'].isin(study_sizes)]

    # ── 4. Expand hidden scores and build combined DataFrames ───────────────
    data_nlp_nocv_base = data_nlp_nocv[data_nlp_nocv['n_splits'] == 1].copy()

    df_nocv_k = compute.expand_hidden_scores(
        data_nlp_nocv_base,
        scores_col='hidden_test_accuracies',
        id_cols=['seed', 'model', 'study_size'],
    )
    print(f"Expanded no-CV rows: {len(df_nocv_k)}")

    _df_combined_acc, df_combined_delta = compute.build_nocv_combined_df(
        df_nocv_k,
        data_nlp_nocv_base,
        model_col='model',
        score_raw_col='objective_score_test',
        bench_col='objective_score_bench',
        include_k0=True,
    )

    # ── 5. Pre-compute delta scores for CV ──────────────────────────────────
    # - ``delta_score_test``: *cummean* delta, used by sample_gain_bootstrap
    #   which expects scores aligned with the n_splits=1..K expanding axis.
    # - ``delta_score_fold``: *per-fold* raw delta, used for τ^te where
    #   compute_tau needs genuine per-fold realisations (cummeans violate
    #   the ANOVA iid-across-folds assumption).
    data_nlp_cv_delta = C.build_delta_cv_table(
        data_nlp_cv,
        cumulative_test_col='objective_score_test',
        cumulative_bench_col='objective_score_bench',
        fold_test_col='study_test_eval_accuracy',
        fold_bench_col='benchmark_accuracy',
    )

    # ── 5b. τ estimates: raw-score and evaluation-error ─────────────────────
    tau_raw_df, tau_te_df = C.compute_standard_tau_tables(
        data_nlp_cv_delta,
        raw_fold_score_col='study_test_eval_accuracy',
        group_cols=['model', 'study_size'],
        model_filter_col='model',
        models_to_analyze=models_to_analyze,
    )
    if not tau_raw_df.empty:
        tau_raw_disp = tau_raw_df.copy()
        tau_raw_disp['model'] = tau_raw_disp['model'].map(MODEL_DISPLAY_NAMES).fillna(tau_raw_disp['model'])
        print("\nτ_raw per algorithm (Yelp accuracy):")
        print(tau_raw_disp.to_string(index=False))
    if not tau_te_df.empty:
        tau_te_disp = tau_te_df.copy()
        tau_te_disp['model'] = tau_te_disp['model'].map(MODEL_DISPLAY_NAMES).fillna(tau_te_disp['model'])
        print("\nτ^te per algorithm (Yelp delta-bench accuracy):")
        print(tau_te_disp[['model', 'study_size', 'tau_te', 'sigma2_te_K',
                           'tau_te_lo', 'tau_te_hi']].to_string(index=False))
        fig = P.plot_tau_bar(
            tau_te_disp,
            x_col="model",
            hue_col="study_size",
            value_col="tau_te",
            title=r"$\hat\tau^{te}$ — Yelp (delta-bench accuracy)",
            ylabel=r"$\hat\tau^{te}$",
            output_path=f"{plots_dir}/tau_te_nlp.pdf",
            ci_cols=("tau_te_lo", "tau_te_hi"),
            overlay_B=False,
        )
        plt.show()
        plt.close(fig)

    # ── 6. Bootstrap — delta-bench score ────────────────────────────────────
    results_delta = compute.sample_gain_bootstrap(
        data_cv=data_nlp_cv_delta,
        score_col_cv='delta_score_test',
        model_col='model',
        n_bootstraps=n_bootstraps,
        random_state=random_state, plugin_estimate=plugin_estimate,
        method='absolute',
        test_size=test_size,
        df_nocv_combined=df_combined_delta,
        score_col_nocv='delta_score_combined',
        models_to_analyze=models_to_analyze,
        study_sizes=study_sizes,
    )
    print(f"Bootstrap results (delta): {len(results_delta)} rows")
    _save_gains(results_delta, save_gains, "nlp_delta")

    # ── 7. Sample gain plots ────────────────────────────────────────────────
    _tab10 = sns.color_palette('tab10', n_colors=10)
    _model_color_map = {
        'bert':       _tab10[0],
        'roberta':    _tab10[1],
        'qwen3-0.6b': _tab10[2],
    }

    C.plot_sample_gain_by_train_size(
        results_delta,
        study_sizes=study_sizes,
        models_to_analyze=models_to_analyze,
        model_display_names=MODEL_DISPLAY_NAMES,
        model_color_map=_model_color_map,
        title_prefix="Yelp",
        plots_dir=plots_dir,
        # Every per-size Yelp figure shares one style: no size gets enlarged
        # fonts. A previous 1000-only font override (for a shrunk main-figure
        # panel) inflated that figure's tight-bbox relative to the other sizes;
        # dropping it keeps all training sizes at the same figsize.
    )

    # ── 8. Variance diagnostic plots ────────────────────────────────────────
    C.plot_standard_variance_diagnostics(
        data_nocv_base=data_nlp_nocv_base,
        df_nocv_k=df_nocv_k,
        data_cv=data_nlp_cv,
        models_to_analyze=models_to_analyze,
        target_splits_diag=target_splits_diag,
        test_size=test_size,
        plots_dir=plots_dir,
        scalar_extractor=_extract_scalar,
        model_col='model',
        size_col='study_size',
        score_raw_col='objective_score_test',
        bench_col='objective_score_bench',
    )

    # ── 9. Retrieval of the best model ──────────────────────────────────────
    C.run_retrieval_analysis(
        data_nocv_base=data_nlp_nocv_base,
        df_nocv_k=df_nocv_k,
        data_cv=data_nlp_cv,
        gt_source_df=data_nlp,
        models_to_analyze=models_to_analyze,
        study_sizes=study_sizes,
        target_splits=[5, 10, 20],
        model_display_names=MODEL_DISPLAY_NAMES,
        plots_dir=plots_dir,
        scalar_extractor=_extract_scalar,
        model_col='model',
        size_col='study_size',
        score_raw_col='objective_score_test',
        bench_col='objective_score_bench',
        gt_score_col='objective_score_bench',
        cv_colors_retrieval=["#4C78A8", "#59A14F", "#E15759"],
        retrieval_n_bootstraps=100,
        test_size=test_size,
        y_as_percent=True,
    )
    _plot_yelp_pairwise_score_distributions(
        data_nocv_base=data_nlp_nocv_base,
        data_cv=data_nlp_cv,
        study_size=3750,
        target_splits=[5, 10, 20],
        plots_dir=plots_dir,
    )

    # ── Export τ̂ + sample-gain@K=20 table for cross-pipeline scatter ─────
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    if not tau_raw_df.empty and not results_delta.empty:
        gain20_df = compute.sample_gain_at_splits(
            results_delta,
            group_cols=['model', 'N'],
            target_splits=[20],
        ).rename(columns={
            'model': 'algorithm',
            'N': 'study_size',
            'sample_gain': 'sample_gain_K20',
        }).drop(columns=['K'], errors='ignore')

        tau_raw_export = tau_raw_df.rename(columns={'model': 'algorithm'})[[
            'algorithm', 'study_size', 'tau', 'sigma2_HO_K', 'W', 'B', 'K', 'n_seeds',
        ]]

        if not tau_te_df.empty:
            tau_te_export = tau_te_df.rename(columns={'model': 'algorithm'})[[
                'algorithm', 'study_size',
                'tau_te', 'sigma2_te_K', 'tau_te_lo', 'tau_te_hi',
            ]]
        else:
            tau_te_export = None

        bench_means = (
            data_nlp_cv
            .groupby(['model', 'study_size'])['benchmark_accuracy']
            .mean()
            .reset_index()
            .rename(columns={
                'model':              'algorithm',
                'benchmark_accuracy': 'mean_bench_score',
            })
        )

        # sigma^2_HO,|1: across-seed variance of single-split HO scores.
        sigma2_HO_1 = (
            data_nlp_nocv_base
            .groupby(['model', 'study_size'])['study_test_eval_accuracy']
            .var()
            .reset_index()
            .rename(columns={
                'model':                     'algorithm',
                'study_test_eval_accuracy':  'sigma2_HO_1',
            })
        )

        # sigma^{2,te},|1: across-seed variance of single-split delta-bench
        # scores. Numerator of the delta-based predicted gain; matches units
        # with the delta-based denominator (sigma^{2,te},|K + (K-1) tau^te).
        _nocv_delta = data_nlp_nocv_base.copy()
        _nocv_delta['_delta'] = (
            _nocv_delta['study_test_eval_accuracy']
            - _nocv_delta['benchmark_accuracy']
        )
        sigma2_te_1 = (
            _nocv_delta.groupby(['model', 'study_size'])['_delta']
            .var().reset_index()
            .rename(columns={'model': 'algorithm', '_delta': 'sigma2_te_1'})
        )

        # sigma^2_HO,|tr,xi: within-seed variance of hidden test splits for a
        # fixed trained model, averaged across seeds. Drops the seed-level
        # training component that inflates the across-seeds estimator.
        sigma2_HO_trxi = compute.sigma2_nocv_within_seed(
            data_nlp_nocv_base,
            scores_col='hidden_test_accuracies',
            group_cols=['model', 'study_size'],
        ).rename(columns={
            'model': 'algorithm',
            'sigma2_nocv_within': 'sigma2_HO_trxi',
        })

        merged = C.build_tau_gain_export(
            tau_df=tau_raw_export,
            merge_keys=['algorithm', 'study_size'],
            pipeline='nlp',
            gain20_df=gain20_df,
            tau_te_df=tau_te_export,
            bench_means_df=bench_means,
            sigma2_HO_1_df=sigma2_HO_1,
            sigma2_te_1_df=sigma2_te_1,
            sigma2_HO_trxi_df=sigma2_HO_trxi,
            fill_values={
                'dataset': 'yelp',
                'procedure': 'ShuffleSplit',
                'fixed_split': pd.NA,
            },
        )
        merged.to_parquet(f"{results_dir}/tau_gain_nlp.parquet", index=False)
        print(f"Wrote {results_dir}/tau_gain_nlp.parquet ({len(merged)} rows)")

    print("\nDone.")
    return results_delta


def _running_in_jupyter():
    try:
        get_ipython()  # noqa: F821
        return True
    except NameError:
        return False


# %% ──────────────────────────────────────────────────────────────────────────
# Run the pipeline.
# - VSCode cells / Jupyter: calls main() with defaults.
# - CLI (python nlp.py --data-dirs ...): parses arguments first.
# - import nlp: does nothing.

if __name__ == '__main__':
    if _running_in_jupyter():
        main()
    else:
        args = parse_args()
        main(
            data_dirs=args.data_dirs,
            plots_dir=args.plots_dir,
            n_bootstraps=args.n_bootstraps,
            test_size=args.test_size,
            models_to_analyze=args.models,
            study_sizes=args.study_sizes,
            random_state=args.random_state,
            hairlines=args.hairlines,
            plugin_estimate=args.plugin_estimate,
            save_gains=args.save_gains,
        )

# %%
