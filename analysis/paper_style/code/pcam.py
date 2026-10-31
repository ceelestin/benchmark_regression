"""
pcam.py — Validation Crisis analysis on PatchCamelyon (medical imaging).

Data pipeline:
  1. Load CV parquet files; extract procedure from filename.
  2. Build data_pcam with expanding-mean n_splits column.
  3. Separate CV and no-CV data.
  4. Expand hidden test accuracies; build combined no-CV DataFrames.
  5. Run bootstrapped sample gain analysis (delta-bench score).
  6. Build variance diagnostic DataFrames.
  7. Generate all plots.
"""

# %%
import argparse
import os
import re
import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FixedLocator

import compute
import pipeline_common as C
import plot as P

# %% ──────────────────────────────────────────────────────────────────────────
# Default configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DATA_DIR = '/home/ceve/Documents/Run outputs/PCam/results_output_pcam'
# DEFAULT_DATA_DIR = '/home/ceve/Documents/Run outputs/PCam/stratandnewmetrics'
# DEFAULT_DATA_DIR = '/home/ceve/Documents/Run outputs/PCam/redundancy'
DEFAULT_PLOTS_DIR = "plots/pcam/rebuttal"
DEFAULT_MODELS = ['densenet121', 'mobilenet_v2', 'wide_resnet101_2', 'vit_b_16']
DEFAULT_STUDY_SIZES = [30, 100, 300, 500, 800]

# Metrics recorded per fold (each has test / benchmark / hidden-test columns).
# All are instance-mean metrics, so the no-CV growth curve uses the standard
# averaging combination (build_nocv_combined_df). Direction is irrelevant here:
# sample gain is variance-based, so lower-is-better metrics (nll, brier) need no
# sign flip. ``accuracy`` reproduces the original single-metric behaviour.
METRICS = [
    {
        'name': 'accuracy',
        'display': 'Accuracy',
        'test_col': 'test_accuracy',
        'bench_col': 'benchmark_accuracy',
        'hidden_col': 'hidden_test_accuracies',
    },
    {
        'name': 'nll',
        'display': 'NLL',
        'test_col': 'test_nll',
        'bench_col': 'benchmark_nll',
        'hidden_col': 'hidden_test_nlls',
    },
    {
        'name': 'brier',
        'display': 'Brier',
        'test_col': 'test_brier',
        'bench_col': 'benchmark_brier',
        'hidden_col': 'hidden_test_briers',
    },
]

# The metric whose secondary analyses (tau bars, variance diagnostics,
# retrieval, export) are produced — kept to accuracy, as before.
PRIMARY_METRIC = 'accuracy'

# Study-only OOF-intersection redundancy columns emitted per fold by the updated
# PCam objective (`_pairwise_oof_icc`), mapped to their canonical tau/gain export
# names. The Brier version fills the regression squared-error column so PCam and
# simulated share one redundancy column; NLL and 0/1-accuracy get their own.
REDUNDANCY_SOURCE_TO_EXPORT = [
    ('study_squared_error_rho_all_oof_intersection',
     'study_squared_error_rho_all_oof_intersection'),   # Brier == squared error
    ('study_nll_rho_all_oof_intersection',
     'study_nll_rho_all_oof_intersection'),
    ('study_error01_rho_all_oof_intersection',
     'study_error01_rho_all_oof_intersection'),
    ('oof_intersection_pairs', 'study_oof_intersection_pairs_all'),
    ('oof_intersection_mean_size', 'study_oof_intersection_mean_size_all'),
]

# Right-censoring guard. A (model, study_size, stratified, K) sample-gain cell is
# unreliable when the no-CV baseline fails to cross the CV variance within the
# hidden-chunk budget for many bootstrap resamples: both variances have then
# collapsed to the ~noise floor (e.g. ViT stratified NLL/Brier, where the
# test−benchmark delta cancels almost perfectly), and the reported gain is a
# meaningless, censored number (~150), not a real test-data multiplier. Such
# cells are dropped from the plots and the export rather than shown.
# NEAR_CAP is deliberately well below 1.0: a crossing deep in the growth-curve
# tail (not only at the very cap) is already noise-floor dominated and unreliable
# — e.g. ViT stratified Brier crosses ~150 of 200 against the unstratified
# baseline. All legitimate gains here are k_equiv < 15, so 0.5 leaves them intact.
CENSOR_NEAR_CAP_FRAC = 0.5     # k_equiv >= this * max_k  ⇒ this draw "did not cross"
CENSOR_CELL_THRESHOLD = 0.10   # drop the cell if > this fraction of draws are censored
THREE_PANEL_SAMPLE_GAIN_AXIS_FONTSIZE = 17
THREE_PANEL_SAMPLE_GAIN_TICK_FONTSIZE = 14
THREE_PANEL_SAMPLE_GAIN_LABEL_FONTSIZE = 13
# 11pt (not 13) so the verbose two-line "PCam (Metric, Stratify) / N=…" title
# fits inside the compact paper-style axes without being clipped.
THREE_PANEL_SAMPLE_GAIN_TITLE_FONTSIZE = 11

MODEL_DISPLAY_NAMES = {
    'densenet121':      'DenseNet121',
    'mobilenet_v2':     'MobileNetV2',
    'wide_resnet101_2': 'WideResNet101_2',
    'vit_b_16':         'ViT-B/16',
}

_PROC_MAP = {'shuffle_split': 'MCCV', 'repeated_kfold': 'RepeatedKFold'}

# %% ──────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def load_cv_data(directory: str) -> pd.DataFrame:
    """Load CV parquet files, extracting procedure name from filename."""
    if not os.path.isdir(directory):
        print(f"Warning: directory not found: {directory}")
        return pd.DataFrame()

    dfs = []
    files = [f for f in os.listdir(directory) if f.endswith('.parquet')]
    print(f"Found {len(files)} parquet files in {directory}")

    for fname in files:
        procedure = None
        for token, canonical in _PROC_MAP.items():
            if f'CV_{token}' in fname:
                procedure = canonical
                break
        if procedure is None:
            continue

        df = pd.read_parquet(os.path.join(directory, fname))
        df['procedure'] = procedure
        df['cross_validation'] = True
        dfs.append(df)

    if not dfs:
        print("No CV parquet files found.")
        return pd.DataFrame()

    data = pd.concat(dfs, ignore_index=True, sort=True)
    gc.collect()
    print(f"Loaded {len(data)} rows  procedures: {data['procedure'].value_counts().to_dict()}")
    return data


def _extract_scalar(x):
    if isinstance(x, (list, np.ndarray)) and len(x) > 0:
        return x[-1]
    return x if np.isscalar(x) else np.nan


def aggregate_redundancy(data_stratified):
    """Per-(algorithm, study_size, procedure) study-only OOF-intersection redundancy.

    The objective computes one rho-hat per seed (across that seed's CV folds) and
    broadcasts it onto the seed's fold rows, so we take one value per seed and then
    average over seeds — matching the simulated pipeline's covariance-diagnostic
    aggregation. Returns a frame keyed like the tau/gain export (or ``None`` when the
    redundancy columns are absent, e.g. data predating the updated objective).
    """
    pairs = [(s, o) for s, o in REDUNDANCY_SOURCE_TO_EXPORT
             if s in data_stratified.columns]
    if not pairs:
        return None
    src = [s for s, _ in pairs]
    d = data_stratified[['model', 'study_size', 'procedure', 'seed'] + src].copy()
    for c in src:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    seed_level = (
        d.dropna(subset=src, how='all')
         .groupby(['model', 'study_size', 'procedure', 'seed'], dropna=False)[src]
         .first().reset_index()
    )
    if seed_level.empty:
        return None
    return (
        seed_level
        .groupby(['model', 'study_size', 'procedure'], dropna=False)[src]
        .mean().reset_index()
        .rename(columns={'model': 'algorithm', **{s: o for s, o in pairs}})
    )


def drop_censored_gains(res, max_k, model_display_names, metric_name=""):
    """Drop whole (model, N, stratified) sample-gain curves that are censored.

    A curve is censored when, at its largest K (where the CV variance is lowest
    and most likely to sink below the no-CV baseline floor), more than
    ``CENSOR_CELL_THRESHOLD`` of its bootstrap draws fail to cross within the
    hidden-chunk budget (``k_equiv >= CENSOR_NEAR_CAP_FRAC * max_k``). In that
    regime the CV and no-CV variances have both collapsed to the ~noise floor
    (e.g. ViT stratified NLL/Brier, where the balanced-study-test vs unbalanced-
    hidden-chunk mismatch also makes the growth curve non-monotonic), so the gain
    is a censored artifact — spuriously high at *every* K, not just the tail.
    The whole curve is dropped rather than plotted/exported.

    Returns ``(clean_res, n_rows_dropped)``.
    """
    if res is None or res.empty or 'k_equiv' not in res.columns or max_k <= 0:
        return res, 0
    grp_cols = [c for c in ['model', 'N', 'stratified'] if c in res.columns]
    near_cap = res['k_equiv'] >= CENSOR_NEAR_CAP_FRAC * max_k
    # Evaluate censoring at each curve's largest available K.
    kmax = res.groupby(grp_cols)['K'].transform('max')
    at_kmax = res[res['K'] == kmax]
    frac = near_cap[res['K'] == kmax].groupby(
        [at_kmax[c] for c in grp_cols]).mean()
    censored_curves = frac.index[frac > CENSOR_CELL_THRESHOLD]
    if len(censored_curves) == 0:
        return res, 0
    mask = pd.MultiIndex.from_frame(res[grp_cols]).isin(censored_curves)
    for keyvals in censored_curves:
        info = dict(zip(grp_cols, keyvals if isinstance(keyvals, tuple) else (keyvals,)))
        md = model_display_names.get(info.get('model'), info.get('model'))
        st = f", stratified={info['stratified']}" if 'stratified' in info else ""
        print(f"  [censored] {metric_name}: {md}{st} — gain is a right-censored "
              f"noise-floor artifact; whole curve dropped")
    return res[~mask].copy(), int(mask.sum())


def plot_sample_gain_by_metric(all_results, plots_dir, model_display_names,
                               xticks=None):
    """One sample-gain figure per (model, study size, stratify); curves = metric.

    The transpose of the per-metric aggregated plots: hue is the metric for a
    fixed learning algorithm. Sample gain is a unitless multiplier, so every
    metric shares one y-axis. Consumes ``all_results`` (metric name → bootstrap
    results, each carrying a ``stratified`` column and a ``metric`` display tag).
    """
    if xticks is None:
        xticks = [1, 5, 10, 15, 20]
    frames = [res for res in all_results.values() if res is not None and not res.empty]
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)

    metric_order = [m['display'] for m in METRICS
                    if m['display'] in set(combined['metric'])]
    _tab10 = sns.color_palette('tab10', n_colors=max(10, len(metric_order)))
    metric_palette = {name: _tab10[i % 10] for i, name in enumerate(metric_order)}

    has_strat = 'stratified' in combined.columns
    group_cols = ['model', 'N'] + (['stratified'] if has_strat else [])
    for keys, grp in combined.groupby(group_cols, dropna=False):
        if has_strat:
            model, N, sval = keys
            strat_tag = f"_strat{sval}"
        else:
            model, N = keys
            strat_tag = ""
        model_disp = model_display_names.get(model, model)
        model_clean = str(model).replace(' ', '_')
        present = [name for name in metric_order if name in set(grp['metric'])]
        sub = grp[grp['K'] <= max(xticks)]
        if sub.empty:
            continue
        fig = P.plot_sample_gain_vs_splits(
            sub, y_col='sample_gain', hue_col='metric',
            hue_order=present, palette=metric_palette,
            xticks=xticks,
            title_text=f"PCam, {model_disp}, $n_{{\\mathrm{{tr}}}}={N}$",
            xlim=(0, max(xticks)),
            direct_labels=True,
            output_path=(f"{plots_dir}/sample_gain_by_metric_{model_clean}"
                         f"_N{N}{strat_tag}.pdf"),
        )
        plt.close(fig)


# %% ──────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validation Crisis analysis on PatchCamelyon (medical imaging) data."
    )
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR,
        help="Directory containing CV parquet files.",
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
    data_dir=None,
    plots_dir=DEFAULT_PLOTS_DIR,
    n_bootstraps=100,
    test_size=1/5,
    models_to_analyze=None,
    study_sizes=None,
    random_state=None,
    hairlines=None,
    plugin_estimate=False,
    save_gains=None,
):
    if hairlines:
        P.GAIN_DISPLAY["hairlines"] = hairlines
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    if models_to_analyze is None:
        models_to_analyze = DEFAULT_MODELS
    if study_sizes is None:
        study_sizes = DEFAULT_STUDY_SIZES

    os.makedirs(plots_dir, exist_ok=True)
    P.set_paper_style()

    target_splits_diag = [5, 10, 20]

    # ── 1. Load data ────────────────────────────────────────────────────────
    data = load_cv_data(data_dir)

    if data.empty:
        raise RuntimeError("No data loaded — check data_dir.")

    print(f"Seeds: {len(data['seed'].unique())}")

    # Old parquets predate the stratified flag (all-unstratified runs) and the
    # nll/brier columns (accuracy-only runs). Backfill the flag and restrict to
    # the metrics actually present so old and new data share one pipeline.
    if 'stratified' not in data.columns:
        print("No 'stratified' column found — treating all rows as unstratified.")
        data['stratified'] = False
    else:
        data['stratified'] = data['stratified'].fillna(False)

    available_metrics = [
        m for m in METRICS
        if all(c in data.columns
               for c in (m['test_col'], m['bench_col'], m['hidden_col']))
    ]
    _absent = [m['name'] for m in METRICS if m not in available_metrics]
    if _absent:
        print(f"Metrics absent from this data, skipped: {_absent}")
    if not any(m['name'] == PRIMARY_METRIC for m in available_metrics):
        raise RuntimeError(
            f"Primary metric '{PRIMARY_METRIC}' columns missing from data.")

    # ── 2. Build data_pcam with expanding-mean n_splits per metric ──────────
    # ``stratified`` joins the grouping so cumulative means never interleave a
    # seed's stratified and unstratified folds. All metric fold columns are
    # carried through; cumulative-mean CV scores are computed for every metric.
    group_cols = ['seed', 'model', 'study_size', 'cross_validation',
                  'procedure', 'stratified']
    metric_fold_cols = []
    for m in available_metrics:
        metric_fold_cols += [m['test_col'], m['bench_col'], m['hidden_col']]
    cols_to_copy = group_cols + ['fold', 'val_accuracy'] + metric_fold_cols
    cols_to_copy = [c for c in cols_to_copy if c in data.columns]

    data_pcam = (
        data[cols_to_copy]
        .drop_duplicates(subset=group_cols + ['fold'])
        .sort_values(group_cols + ['fold'])
        .copy()
    )
    scalar_cols = ['val_accuracy'] + [m['test_col'] for m in available_metrics] \
        + [m['bench_col'] for m in available_metrics]
    data_pcam = C.coerce_scalar_columns(
        data_pcam,
        columns=[c for c in scalar_cols if c in data_pcam.columns],
        scalar_extractor=_extract_scalar,
    )
    grouped_pcam = data_pcam.groupby(group_cols, sort=False)
    data_pcam['n_splits'] = grouped_pcam.cumcount() + 1
    for m in available_metrics:
        data_pcam[f"cummean_test_{m['name']}"] = (
            grouped_pcam[m['test_col']].cumsum() / data_pcam['n_splits']
        )
        data_pcam[f"cummean_bench_{m['name']}"] = (
            grouped_pcam[m['bench_col']].cumsum() / data_pcam['n_splits']
        )
    data_pcam['n_repeats'] = 1
    # Expose the primary metric under the canonical column names the secondary
    # (accuracy-based) analyses expect.
    _prim = next(m for m in available_metrics if m['name'] == PRIMARY_METRIC)
    data_pcam['objective_score_test'] = data_pcam[f"cummean_test_{_prim['name']}"]
    data_pcam['objective_score_bench'] = data_pcam[f"cummean_bench_{_prim['name']}"]

    # ── 3. Separate CV and no-CV ────────────────────────────────────────────
    data_pcam_cv = data_pcam[data_pcam['cross_validation']].copy()

    data_pcam_nocv = data_pcam_cv[
        (data_pcam_cv['n_splits'] == 1) & (data_pcam_cv['procedure'] == 'MCCV')
    ].copy()

    study_sizes_available = sorted(data_pcam['study_size'].unique())
    print(f"Study sizes: {study_sizes_available}")
    print(f"Stratified values: {sorted(data_pcam['stratified'].unique())}")

    data_pcam_cv = data_pcam_cv[data_pcam_cv['study_size'].isin(study_sizes)].copy()
    data_pcam_nocv = data_pcam_nocv[data_pcam_nocv['study_size'].isin(study_sizes)].copy()
    study_sizes_to_use = sorted(data_pcam_cv['study_size'].unique())
    strat_values = sorted(data_pcam_cv['stratified'].unique())

    # ── 3b. Per-metric × per-stratify sample gain ───────────────────────────
    # For each metric, point the working score columns at that metric, build the
    # CV delta table and the no-CV growth curve (per stratify, since
    # build_nocv_combined_df merges on seed/model/size), then bootstrap with
    # ``stratified`` as an extra matching dimension.
    all_results: dict[str, pd.DataFrame] = {}
    # Cross-baseline: gains for both stratify modes matched against a single
    # common single-split baseline — the unstratified one, and the stratified
    # one — see the two plots after the loop.
    all_results_cross: dict[str, pd.DataFrame] = {}        # vs UNSTRATIFIED baseline
    all_results_cross_strat: dict[str, pd.DataFrame] = {}  # vs STRATIFIED baseline
    for m in available_metrics:
        name = m['name']
        cvm = data_pcam_cv.copy()
        cvm['objective_score_test'] = cvm[f'cummean_test_{name}']
        cvm['objective_score_bench'] = cvm[f'cummean_bench_{name}']
        cvm_delta = C.build_delta_cv_table(
            cvm,
            cumulative_test_col='objective_score_test',
            cumulative_bench_col='objective_score_bench',
            fold_test_col=m['test_col'],
            fold_bench_col=m['bench_col'],
        )

        nocv_frames = []
        for sval in strat_values:
            nocv_s = data_pcam_nocv[data_pcam_nocv['stratified'] == sval].copy()
            if nocv_s.empty:
                continue
            nocv_s['objective_score_test'] = nocv_s[f'cummean_test_{name}']
            nocv_s['objective_score_bench'] = nocv_s[f'cummean_bench_{name}']
            df_nocv_k_m = compute.expand_hidden_scores(
                nocv_s,
                scores_col=m['hidden_col'],
                id_cols=['seed', 'model', 'study_size'],
            )
            _, comb_delta = compute.build_nocv_combined_df(
                df_nocv_k_m,
                nocv_s,
                model_col='model',
                score_raw_col='objective_score_test',
                bench_col='objective_score_bench',
                include_k0=True,
            )
            comb_delta['stratified'] = sval
            nocv_frames.append(comb_delta)
        if not nocv_frames:
            continue
        df_comb_delta = pd.concat(nocv_frames, ignore_index=True)

        res = compute.sample_gain_bootstrap(
            data_cv=cvm_delta,
            score_col_cv='delta_score_test',
            model_col='model',
            n_bootstraps=n_bootstraps,
            random_state=random_state, plugin_estimate=plugin_estimate,
            method='absolute',
            test_size=test_size,
            df_nocv_combined=df_comb_delta,
            score_col_nocv='delta_score_combined',
            models_to_analyze=models_to_analyze,
            study_sizes=study_sizes,
            extra_cv_cols=['stratified'],
            compute_train_size=False,
        )
        res = res.copy()
        res['metric'] = m['display']
        # Drop right-censored cells so noise-floor artifacts (e.g. ViT stratified
        # NLL/Brier ~150) are not reported as high gains.
        max_k = int(df_comb_delta['k'].max())
        res, n_censored = drop_censored_gains(res, max_k, MODEL_DISPLAY_NAMES, name)
        all_results[name] = res
        _save_gains(res, save_gains, f"pcam_{name}")
        _sv = sorted(res['stratified'].unique()) if 'stratified' in res.columns else 'N/A'
        print(f"[{name}] bootstrap rows: {len(res)} (dropped {n_censored} censored) "
              f"| stratified: {_sv}")

        # Cross-baseline: match BOTH stratify modes' CV variance against ONE
        # common single-split growth curve — once the unstratified baseline, once
        # the stratified baseline — by relabeling that baseline's no-CV table onto
        # every stratify slot. Each stratify curve is then measured against the
        # same reference, so the two are directly comparable.
        fs_false = next((v for v in strat_values if bool(v) is False), None)
        fs_true = next((v for v in strat_values if bool(v) is True), None)
        if fs_false is not None and fs_true is not None:
            for base_val, store, base_tag in (
                (fs_false, all_results_cross, 'unstrat'),
                (fs_true, all_results_cross_strat, 'strat'),
            ):
                base_comb = df_comb_delta[df_comb_delta['stratified'] == base_val].drop(
                    columns=['stratified'])
                df_comb_cross = pd.concat(
                    [base_comb.assign(stratified=sv) for sv in strat_values],
                    ignore_index=True,
                )
                res_cross = compute.sample_gain_bootstrap(
                    data_cv=cvm_delta,
                    score_col_cv='delta_score_test',
                    model_col='model',
                    n_bootstraps=n_bootstraps,
                    random_state=random_state, plugin_estimate=plugin_estimate,
                    method='absolute',
                    test_size=test_size,
                    df_nocv_combined=df_comb_cross,
                    score_col_nocv='delta_score_combined',
                    models_to_analyze=models_to_analyze,
                    study_sizes=study_sizes,
                    extra_cv_cols=['stratified'],
                    compute_train_size=False,
                )
                res_cross = res_cross.copy()
                res_cross['metric'] = m['display']
                res_cross, _ = drop_censored_gains(
                    res_cross, max_k, MODEL_DISPLAY_NAMES,
                    f"{name} (vs {base_tag} baseline)")
                store[name] = res_cross
                _save_gains(res_cross, save_gains, f"pcam_{name}_vs_{base_tag}_baseline")

    # ── 4. Plot 1 (per study size, stratify) + Plot 3 (stratify overlay) ─────
    specific_xticks = [1, 5, 10, 15, 20]
    _strat_label = {False: 'Unstratified', True: 'Stratified'}
    _strat_dashes = {'Unstratified': (), 'Stratified': (4, 2)}
    # Every per-size sample-gain figure shares one style: no size gets enlarged
    # fonts. A previous 500-only font override (for a shrunk main-figure panel)
    # inflated that figure's tight-bbox height (~244 pt vs ~224 pt) relative to
    # the other sizes; dropping it keeps all study sizes at the same figsize.
    _fontsizes = {}
    # When the data carries a single metric and a single split mode (the
    # accuracy-only, unstratified main-text setting), the "(Accuracy,
    # Unstratified)" qualifier is redundant — no other metric or stratification
    # is shown to contrast against — so the title collapses to just "PCam". With
    # several metrics or both split modes present (the appendix runs), the
    # qualifier is kept to tell the figures apart.
    _single_panel = len(available_metrics) == 1 and len(strat_values) == 1
    for m in available_metrics:
        res = all_results.get(m['name'])
        if res is None or res.empty:
            continue
        # Plot 1: one figure per (study size, stratify); each model a curve.
        for sval in strat_values:
            res_s = res[res['stratified'] == sval]
            if 'cv_procedure' in res_s.columns:
                # One procedure per curve (CE, 2026-10-29): Repeated K-Fold is compared with MCCV in its
                # own figure, not pooled with MCCV draws here.
                res_s = res_s[res_s['cv_procedure'] == 'MCCV']
            if res_s.empty:
                continue
            C.plot_sample_gain_by_train_size(
                res_s, study_sizes=study_sizes_to_use,
                models_to_analyze=models_to_analyze,
                model_display_names=MODEL_DISPLAY_NAMES, model_color_map=None,
                title_prefix=("PCam" if _single_panel
                              else f"PCam ({m['display']}, "
                                   f"{_strat_label.get(sval, sval)})"),
                plots_dir=plots_dir, xticks=specific_xticks,
                file_suffix=f"_{m['name']}_strat{sval}", **_fontsizes,
            )
        # Plot 3: one figure per study size; each model a curve; solid =
        # unstratified, dotted = stratified.
        if res['stratified'].nunique() >= 2:
            res_ov = res.copy()
            res_ov['stratified_display'] = res_ov['stratified'].map(_strat_label)
            C.plot_sample_gain_by_train_size(
                res_ov, study_sizes=study_sizes_to_use,
                models_to_analyze=models_to_analyze,
                model_display_names=MODEL_DISPLAY_NAMES, model_color_map=None,
                title_prefix=f"PCam ({m['display']})",
                plots_dir=plots_dir, xticks=specific_xticks,
                style_col='stratified_display',
                style_order=['Unstratified', 'Stratified'],
                dashes=_strat_dashes, style_legend=True,
                file_suffix=f"_{m['name']}_stratify_overlay", **_fontsizes,
            )

    # ── 5. Plot 2: per (model, study size, stratify), each curve a metric ────
    plot_sample_gain_by_metric(all_results, plots_dir, MODEL_DISPLAY_NAMES,
                               xticks=specific_xticks)

    # ── 5b. Gains measured against a single common baseline ─────────────────
    # One figure per (metric, study size); each model a curve, solid =
    # unstratified / dotted = stratified — but BOTH curves matched against the
    # same single-split baseline, so the two stratify modes are directly
    # comparable. Produced twice: against the unstratified baseline and against
    # the stratified baseline.
    for store, base_tag, base_name in (
        (all_results_cross, 'unstrat', 'unstrat. baseline'),
        (all_results_cross_strat, 'strat', 'strat. baseline'),
    ):
        for m in available_metrics:
            rc = store.get(m['name'])
            if rc is None or rc.empty or 'stratified' not in rc.columns \
                    or rc['stratified'].nunique() < 2:
                continue
            rc_ov = rc.copy()
            rc_ov['stratified_display'] = rc_ov['stratified'].map(_strat_label)
            C.plot_sample_gain_by_train_size(
                rc_ov, study_sizes=study_sizes_to_use,
                models_to_analyze=models_to_analyze,
                model_display_names=MODEL_DISPLAY_NAMES, model_color_map=None,
                title_prefix=f"PCam ({m['display']}) — {base_name}",
                plots_dir=plots_dir, xticks=specific_xticks,
                style_col='stratified_display',
                style_order=['Unstratified', 'Stratified'],
                dashes=_strat_dashes, style_legend=True,
                file_suffix=f"_{m['name']}_vs_{base_tag}_baseline", **_fontsizes,
            )

    # ── 6. Secondary analyses per stratify mode (accuracy metric) ───────────
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    bright_colors = sns.color_palette("mako", n_colors=len(target_splits_diag))
    tau_gain_exports = []
    for sval in strat_values:
        strat_dir = f"{plots_dir}/strat_{sval}"
        os.makedirs(strat_dir, exist_ok=True)
        strat_name = _strat_label.get(sval, str(sval))
        cv_s = data_pcam_cv[data_pcam_cv['stratified'] == sval].copy()
        nocv_s = data_pcam_nocv[data_pcam_nocv['stratified'] == sval].copy()
        pcam_s = data_pcam[data_pcam['stratified'] == sval].copy()
        if cv_s.empty or nocv_s.empty:
            continue
        print(f"\n=== Secondary analyses — stratified={sval} ({strat_name}) ===")

        # τ tables + bars (accuracy)
        cv_s_delta = C.build_delta_cv_table(
            cv_s,
            cumulative_test_col='objective_score_test',
            cumulative_bench_col='objective_score_bench',
            fold_test_col='test_accuracy',
            fold_bench_col='benchmark_accuracy',
        )
        tau_raw_df, tau_te_df = C.compute_standard_tau_tables(
            cv_s_delta,
            raw_fold_score_col='test_accuracy',
            group_cols=['model', 'study_size', 'procedure'],
            model_filter_col='model',
            models_to_analyze=models_to_analyze,
        )
        if not tau_te_df.empty:
            tau_te_disp = tau_te_df.copy()
            tau_te_disp['model'] = tau_te_disp['model'].map(MODEL_DISPLAY_NAMES).fillna(tau_te_disp['model'])
            for proc, sub in tau_te_disp.groupby("procedure"):
                fig = P.plot_tau_bar(
                    sub, x_col="model", hue_col="study_size", value_col="tau_te",
                    title=rf"$\hat\tau^{{te}}$ — PCam, {proc} ({strat_name})",
                    ylabel=rf"$\hat\tau^{{te}}$",
                    output_path=f"{strat_dir}/tau_te_pcam_{proc}.pdf",
                    ci_cols=("tau_te_lo", "tau_te_hi"), overlay_B=False,
                )
                plt.close(fig)

        # No-CV accuracy expansion (shared by variance diagnostics + retrieval)
        df_nocv_k_s = compute.expand_hidden_scores(
            nocv_s,
            scores_col='hidden_test_accuracies',
            id_cols=['seed', 'model', 'study_size'],
        )

        # Variance diagnostics
        df_diag_all_k, cv_diag = C.plot_standard_variance_diagnostics(
            data_nocv_base=nocv_s,
            df_nocv_k=df_nocv_k_s,
            data_cv=cv_s,
            models_to_analyze=models_to_analyze,
            target_splits_diag=target_splits_diag,
            test_size=test_size,
            plots_dir=strat_dir,
            scalar_extractor=_extract_scalar,
            model_col='model',
            size_col='study_size',
            score_raw_col='objective_score_test',
            bench_col='objective_score_bench',
            cv_extra_group_cols=['procedure'],
        )

        # Pedagogical figure (delta, mobilenet_v2 only)
        ped_config = {
            'nocv_col': 'delta_score',
            'cv_col': 'delta_scalar',
            'x_transform': lambda k, ts: 1 + k,
            'xlabel': (
                'How many times more test data a\n'
                'single split needs to match CV variance'
            ),
        }
        var_k_ped = (
            df_diag_all_k.groupby(['model', 'study_size', 'k'])[ped_config['nocv_col']]
            .var().reset_index(name='variance')
        )
        var_k_ped['x_axis_val'] = var_k_ped['k'].apply(lambda k: ped_config['x_transform'](k, test_size))
        var_cv_ped = (
            cv_diag.groupby(['model', 'study_size', 'procedure', 'n_splits'])[ped_config['cv_col']]
            .var().reset_index(name='variance')
        )
        for N in sorted(var_k_ped['study_size'].unique()):
            subset_k = var_k_ped[(var_k_ped['model'] == 'mobilenet_v2') & (var_k_ped['study_size'] == N)]
            subset_cv_all = var_cv_ped[
                (var_cv_ped['model'] == 'mobilenet_v2') & (var_cv_ped['study_size'] == N)
            ]
            if subset_k.empty or subset_cv_all.empty:
                continue
            for proc, subset_cv in subset_cv_all.groupby('procedure'):
                P.set_paper_style(legend_frameon=True)
                fig = P.plot_pedagogical_variance_diagnostic(
                    subset_k, subset_cv, target_splits=target_splits_diag,
                    x_col='x_axis_val', var_col='variance', xlabel=ped_config['xlabel'],
                    cv_colors=bright_colors,
                    output_path=f"{strat_dir}/pedagogical_variance_diagnostic_delta_mobilenet_v2_N{N}_{proc}.pdf",
                )
                plt.close(fig)
                fig = P.plot_pedagogical_variance_diagnostic_paper_style(
                    subset_k, subset_cv, target_splits=target_splits_diag,
                    x_col='x_axis_val', var_col='variance', xlabel=ped_config['xlabel'],
                    cv_colors=["#4C78A8", "#59A14F", "#E15759"],
                    output_path=f"{strat_dir}/pedagogical_variance_diagnostic_delta_mobilenet_v2_N{N}_{proc}_paper_style.pdf",
                )
                plt.close(fig)
                P.set_paper_style()

        # Retrieval of the best model
        C.run_retrieval_analysis(
            data_nocv_base=nocv_s,
            df_nocv_k=df_nocv_k_s,
            data_cv=cv_s,
            gt_source_df=pcam_s,
            models_to_analyze=models_to_analyze,
            study_sizes=study_sizes_to_use,
            target_splits=[5, 10, 20],
            model_display_names=MODEL_DISPLAY_NAMES,
            plots_dir=strat_dir,
            scalar_extractor=_extract_scalar,
            model_col='model',
            size_col='study_size',
            score_raw_col='objective_score_test',
            bench_col='objective_score_bench',
            gt_score_col='objective_score_bench',
            cv_group_col='procedure',
            cv_colors_retrieval=['C0', 'C1', 'C2'],
            y_as_percent=False,
            test_size=test_size,
        )

        # ── Export τ̂ + sample-gain@K=20 (accuracy) for this stratify mode ──
        acc_res = all_results.get('accuracy')
        if not tau_raw_df.empty and acc_res is not None and not acc_res.empty:
            acc_res_s = acc_res[acc_res['stratified'] == sval] if 'stratified' in acc_res.columns else acc_res
            group_cols_g = ['model', 'N']
            if 'cv_procedure' in acc_res_s.columns:
                group_cols_g.append('cv_procedure')
            gain20_df = compute.sample_gain_at_splits(
                acc_res_s, group_cols=group_cols_g, target_splits=[20],
            ).rename(columns={
                'model': 'algorithm', 'N': 'study_size',
                'cv_procedure': 'procedure', 'sample_gain': 'sample_gain_K20',
            }).drop(columns=['K'], errors='ignore')

            tau_raw_export = tau_raw_df.rename(columns={'model': 'algorithm'})[[
                'algorithm', 'study_size', 'procedure',
                'tau', 'sigma2_HO_K', 'W', 'B', 'K', 'n_seeds',
            ]]
            tau_te_export = tau_te_df.rename(columns={'model': 'algorithm'})[[
                'algorithm', 'study_size', 'procedure',
                'tau_te', 'sigma2_te_K', 'tau_te_lo', 'tau_te_hi',
            ]] if not tau_te_df.empty else None

            bench_means = (
                cv_s.groupby(['model', 'study_size', 'procedure'])['benchmark_accuracy']
                .mean().reset_index()
                .rename(columns={'model': 'algorithm', 'benchmark_accuracy': 'mean_bench_score'})
            )
            sigma2_HO_1 = (
                nocv_s.groupby(['model', 'study_size'])['test_accuracy']
                .var().reset_index()
                .rename(columns={'model': 'algorithm', 'test_accuracy': 'sigma2_HO_1'})
            )
            _nocv_delta = nocv_s.copy()
            _nocv_delta['_delta'] = _nocv_delta['test_accuracy'] - _nocv_delta['benchmark_accuracy']
            sigma2_te_1 = (
                _nocv_delta.groupby(['model', 'study_size'])['_delta']
                .var().reset_index()
                .rename(columns={'model': 'algorithm', '_delta': 'sigma2_te_1'})
            )
            sigma2_HO_trxi = compute.sigma2_nocv_within_seed(
                nocv_s, scores_col='hidden_test_accuracies',
                group_cols=['model', 'study_size'],
            ).rename(columns={'model': 'algorithm', 'sigma2_nocv_within': 'sigma2_HO_trxi'})

            # Study-only OOF-intersection redundancy (rho-hat) for this stratify
            # mode, aggregated per (algorithm, study_size, procedure) from the raw
            # per-fold rows. Carried through build_tau_gain_export as bench_cov_df
            # (None for data predating the redundancy objective).
            redund_df = aggregate_redundancy(data[data['stratified'] == sval])

            merged = C.build_tau_gain_export(
                tau_df=tau_raw_export,
                merge_keys=['algorithm', 'study_size', 'procedure'],
                pipeline='pcam',
                gain20_df=gain20_df,
                tau_te_df=tau_te_export,
                bench_means_df=bench_means,
                sigma2_HO_1_df=sigma2_HO_1,
                sigma2_te_1_df=sigma2_te_1,
                sigma2_HO_trxi_df=sigma2_HO_trxi,
                bench_cov_df=redund_df,
                fill_values={'dataset': 'pcam', 'fixed_split': pd.NA},
            )
            merged['stratified'] = sval
            tau_gain_exports.append(merged)

    if tau_gain_exports:
        merged_all = pd.concat(tau_gain_exports, ignore_index=True)
        merged_all.to_parquet(f"{results_dir}/tau_gain_pcam.parquet", index=False)
        print(f"Wrote {results_dir}/tau_gain_pcam.parquet ({len(merged_all)} rows)")

    print("\nDone.")
    return all_results


def _running_in_jupyter():
    try:
        get_ipython()  # noqa: F821
        return True
    except NameError:
        return False


# %% ──────────────────────────────────────────────────────────────────────────
# Run the pipeline.
# - VSCode cells / Jupyter: calls main() with defaults.
# - CLI (python pcam.py --data-dir ...): parses arguments first.
# - import pcam: does nothing.

if __name__ == '__main__':
    if _running_in_jupyter():
        main()
    else:
        args = parse_args()
        main(
            data_dir=args.data_dir,
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
