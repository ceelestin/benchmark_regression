"""
simulated.py — Validation Crisis analysis on synthetic (simulated) data.

Data pipeline:
  1. Load parquet files via Dask (benchopt output format).
  2. Extract metadata directly from p_obj_* / p_dataset_* columns.
  3. Separate CV data (ShuffleSplit / RepeatedKFold); build cumulative-mean records.
  4. Extract no-CV baseline from idx_rep==0 rows; build no-CV cache.
  5. Run bootstrapped sample gain analysis.
  6. Generate all plots.
"""

# %% ──────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import gc
import os
import re as _re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import compute
import gain_relationships as G
import pipeline_common as C
import plot as P

# %% ──────────────────────────────────────────────────────────────────────────
# Default configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DATA_PATHS = [
    # # ## '/home/ceve/Documents/Run outputs/Regression/data/benchopt_linear_100seeds_run_2026-04-04_11h03m49.parquet',
    # # '/home/ceve/Documents/Run outputs/Regression/data/benchopt_linear_1000seeds_run_2026-04-15_00h41m43.parquet',
    # # '/home/ceve/Documents/Run outputs/Regression/data/benchopt_linear-noisy_100seeds_run_2026-04-21_13h34m09.parquet',
    # # ## '/home/ceve/Documents/Run outputs/Regression/data/benchopt_linear-noisy_1000seeds_run_2026-04-21_19h04m07.parquet',
    # # ## '/home/ceve/Documents/Run outputs/Regression/data/benchopt_linear-full_100seeds_run_2026-04-27_17h43m22.parquet',
    # # # '/home/ceve/Documents/Run outputs/Regression/data/benchopt_linear_100seeds_MLP_run_2026-04-28_12h36m54.parquet',
    # # # '/home/ceve/Documents/Run outputs/Regression/data/benchopt_linear-noisy_100seeds_MLP_run_2026-04-28_14h53m46.parquet',
    # # '/home/ceve/Documents/Run outputs/Regression/data/benchopt_linear-full_1000seeds_MLP_run_2026-04-28_19h04m46.parquet',
    
    # # '/home/ceve/Documents/Run outputs/Regression/data/benchopt_linear-full_100seeds_run_2026-04-29_12h25m18.parquet',
    # # '/home/ceve/Documents/Run outputs/Regression/data/benchopt_linear-full_100seeds_K20_run_2026-04-29_15h16m17.parquet',
    # # '/home/ceve/Documents/Run outputs/Regression/data/benchopt_linear-full_1000seeds_K20_run_2026-04-30_01h00m50.parquet',
    # # '/home/ceve/Documents/Run outputs/Regression/data/benchopt_linear-full_100seeds_K20_newdiags_run_2026-04-30_14h29m40.parquet',

    # # '/home/ceve/Documents/Run outputs/Regression/data/benchopt_run_2026-05-01_13h01m00.parquet', # same as below but with 100 seeds
    # # '/home/ceve/Documents/Run outputs/Regression/data/1000seeds_K20_benchopt_run_2026-05-01_18h34m53.parquet',
    # # '/home/ceve/Documents/Run outputs/Regression/data/followup_benchopt_run_2026-05-02_07h15m13.parquet',
    # # '/home/ceve/Documents/Run outputs/Regression/data/100seeds_K200_benchopt_run_2026-05-02_12h44m20.parquet',
    # # '/home/ceve/Documents/Run outputs/Regression/data/newsolvers_benchopt_run_2026-05-02_18h56m38.parquet',

    # # '/home/ceve/Documents/Run outputs/Regression/data/testredundancy_benchopt_run_2026-05-04_12h45m52.parquet',

    # # '/home/ceve/Documents/Run outputs/Regression/data/benchopt_run_2026-06-05_03h27m19.parquet', # linear_1000_full_N1000.yml
    # # '/home/ceve/Documents/Run outputs/Regression/data/benchopt_run_2026-06-06_02h59m46.parquet', # linear_1000_full_N300.yml
    # # '/home/ceve/Documents/Run outputs/Regression/data/benchopt_run_2026-06-06_01h38m42.parquet', # linear_1000_full_N100.yml
    '/home/ceve/Documents/Run outputs/Regression/data/benchopt_run_2026-06-05_03h53m16.parquet', # linear_1000_full_N1000_noise0.yml

    # # '/home/ceve/Documents/Run outputs/Regression/data/hpo/benchopt_run_2026-07-23_20h11m43.parquet', # ExtraTrees, 50 seeds
    # # '/home/ceve/Documents/Run outputs/Regression/data/hpo/benchopt_run_2026-07-23_20h29m43.parquet',
    # # '/home/ceve/Documents/Run outputs/Regression/data/hpo/benchopt_run_2026-07-23_21h21m19.parquet', # MLP, 50 seeds
    # # '/home/ceve/Documents/Run outputs/Regression/data/hpo/benchopt_run_2026-07-23_21h22m28.parquet'
    # '/home/ceve/Documents/Run outputs/Regression/data/hpo/benchopt_run_2026-07-23_21h44m24.parquet', # MLP, 1000 seeds, HPO
    # '/home/ceve/Documents/Run outputs/Regression/data/hpo/benchopt_run_2026-07-23_21h50m22.parquet',
    # '/home/ceve/Documents/Run outputs/Regression/data/hpo/benchopt_run_2026-07-23_22h24m34.parquet', # ExtraTrees, 1000 seeds, HPO
    # '/home/ceve/Documents/Run outputs/Regression/data/hpo/benchopt_run_2026-07-23_22h27m00.parquet',
    # '/home/ceve/Documents/Run outputs/Regression/data/hpo/benchopt_run_2026-07-27_12h40m37.parquet', # ExtraTrees, 1000 seeds, HPO, n_estimators=200
    # '/home/ceve/Documents/Run outputs/Regression/data/hpo/benchopt_run_2026-07-27_12h47m49.parquet',
    
    # # '/home/ceve/Documents/Run outputs/Regression/data/newmetrics/benchopt_run_2026-07-24_10h02m05.parquet', # newmetrics, 100 seeds, half1
    # # '/home/ceve/Documents/Run outputs/Regression/data/newmetrics/benchopt_run_2026-07-24_10h01m27.parquet', # newmetrics, 100 seeds, half2 
    # '/home/ceve/Documents/Run outputs/Regression/data/newmetrics/benchopt_run_2026-07-24_10h46m24.parquet', # newmetrics, 1000 seeds, half1
    # '/home/ceve/Documents/Run outputs/Regression/data/newmetrics/benchopt_run_2026-07-24_10h42m52.parquet', # newmetrics, 1000 seeds, half2 
    
    # '/home/ceve/Documents/Run outputs/Regression/data/fixed/benchopt_run_2026-07-24_13h28m44.parquet', # WRONG fixed, 1000 seeds, half1 
    # '/home/ceve/Documents/Run outputs/Regression/data/fixed/benchopt_run_2026-07-24_13h25m39.parquet', # WRONG fixed, 1000 seeds, half2 
    # '/home/ceve/Documents/Run outputs/Regression/data/fixed/benchopt_run_2026-07-27_19h28m24.parquet', # fixed, N=1000, half1 
    # '/home/ceve/Documents/Run outputs/Regression/data/fixed/benchopt_run_2026-07-27_19h31m36.parquet', # fixed, N=1000, half2 
    # '/home/ceve/Documents/Run outputs/Regression/data/fixed/benchopt_run_2026-07-27_19h48m06.parquet', # fixed, N=100, half1 
    # '/home/ceve/Documents/Run outputs/Regression/data/fixed/benchopt_run_2026-07-27_19h46m34.parquet', # fixed, N=100, half2 
    # '/home/ceve/Documents/Run outputs/Regression/data/fixed/benchopt_run_2026-07-27_19h57m34.parquet', # not fixed, N=100, half1 
    # '/home/ceve/Documents/Run outputs/Regression/data/fixed/benchopt_run_2026-07-27_19h55m50.parquet', # not fixed, N=100, half2 
        
    # '/home/ceve/Documents/Run outputs/Regression/data/fixed/benchopt_run_2026-07-27_21h00m32.parquet', # new methodo, fixed, N=100
    # '/home/ceve/Documents/Run outputs/Regression/data/fixed/benchopt_run_2026-07-27_21h01m05.parquet', # new methodo, not fixed, N=100
    # '/home/ceve/Documents/Run outputs/Regression/data/fixed/benchopt_run_2026-07-27_21h17m23.parquet', # new methodo, fixed, N=1000
    # '/home/ceve/Documents/Run outputs/Regression/data/fixed/benchopt_run_2026-07-27_21h17m36.parquet', # new methodo, not fixed, N=1000

    # '/home/ceve/Documents/Run outputs/Regression/data/smalldata/benchopt_run_2026-07-31_19h07m06.parquet', # smalldata, half1
    # '/home/ceve/Documents/Run outputs/Regression/data/smalldata/benchopt_run_2026-07-31_19h02m17.parquet', # smalldata, half2
    # '/home/ceve/Documents/Run outputs/Regression/data/smalldata/benchopt_run_2026-07-31_21h05m17.parquet', # smalldata2, half1
    # '/home/ceve/Documents/Run outputs/Regression/data/smalldata/benchopt_run_2026-07-31_21h07m56.parquet', # smalldata2, half2
        
]

DEFAULT_PLOTS_DIR = "plots/simulated"
# THREE_PANEL_SAMPLE_GAIN_FIGSIZE = (2.15, 2.05)
THREE_PANEL_SAMPLE_GAIN_FIGSIZE = (3.2, 3.0)

# If True, display solver hyperparameters as a subscript in plots/tables, e.g.
# ExtraTrees$_{n = 200}$.  If False, display only the solver family name, e.g.
# ExtraTrees.
print_HPC = False

# Master switch for the multi-metric outputs.
#   False → only the R² metric is analysed and plotted (the original behaviour,
#           unchanged: aggregated test-gain, retrieval, variance diagnostics, τ).
#   True  → every metric in ``METRICS`` is analysed, adding the per-metric
#           sample-gain curves and the per-algorithm "sample gain by metric"
#           figures.
PLOT_ALL_METRICS = False

# New regression metrics recorded by the updated objective.py, carried through
# process_parquet so the per-metric sample-gain loop can reach them. Guarded by
# the ``if c in df.columns`` filter there, so older parquets (which lack these)
# are unaffected — only R² runs on them.
METRIC_SCALAR_COLUMNS = [
    f'objective_{m}_{split}'
    for m in ('neg_mse', 'neg_mae', 'neg_median_ae', 'spearman', 'pearson')
    for split in ('train', 'test', 'bench')
]
# Per-chunk lists combine via the running-mean (base + k*mean_k)/(k+1) path;
# *_cum lists are ALREADY pooled on test+chunks 0..k -> used directly ('pooled').
METRIC_LIST_COLUMNS = [
    'objective_outer_neg_mse',
    'objective_outer_neg_mae',
    'objective_outer_r2_cum',
    'objective_outer_neg_median_ae_cum',
    'objective_outer_spearman_cum',
    'objective_outer_pearson_cum',
]

# Each metric drives one pass of the sample-gain loop. ``test_col``/``bench_col``
# are the per-fold CV scores; ``outer_col`` is the no-CV hidden-score list;
# ``combination`` selects how build_nocv_cache reconstructs the growth curve
# ('average' for per-chunk lists, 'pooled' for *_cum). ``full_suite`` gates the
# secondary plots (retrieval, variance diagnostics, tau) to R² only, so those
# outputs stay byte-for-byte unchanged.
METRICS = [
    {
        'name': 'r2',
        'transform': None,
        'file_tag': 'r2',
        'score_label': r'$R^2$',
        'test_col': 'objective_score_test',
        'bench_col': 'objective_score_bench',
        'outer_col': 'objective_outer_scores',
        'combination': 'average',
        'full_suite': True,
    },
    {
        'name': 'neg_mse',
        'transform': None,
        'file_tag': 'neg_mse',
        'score_label': r'$-\mathrm{MSE}$',
        'test_col': 'objective_neg_mse_test',
        'bench_col': 'objective_neg_mse_bench',
        'outer_col': 'objective_outer_neg_mse',
        'combination': 'average',
        'full_suite': False,
    },
    {
        'name': 'neg_mae',
        'transform': None,
        'file_tag': 'neg_mae',
        'score_label': r'$-\mathrm{MAE}$',
        'test_col': 'objective_neg_mae_test',
        'bench_col': 'objective_neg_mae_bench',
        'outer_col': 'objective_outer_neg_mae',
        'combination': 'average',
        'full_suite': False,
    },
    {
        'name': 'neg_median_ae',
        'transform': None,
        'file_tag': 'neg_median_ae',
        'score_label': r'$-\mathrm{MedAE}$',
        'test_col': 'objective_neg_median_ae_test',
        'bench_col': 'objective_neg_median_ae_bench',
        'outer_col': 'objective_outer_neg_median_ae_cum',
        'combination': 'pooled',
        'full_suite': False,
    },
    {
        'name': 'spearman',
        'transform': None,
        'file_tag': 'spearman',
        'score_label': r'Spearman $\rho$',
        'test_col': 'objective_spearman_test',
        'bench_col': 'objective_spearman_bench',
        'outer_col': 'objective_outer_spearman_cum',
        'combination': 'pooled',
        'full_suite': False,
    },
    {
        'name': 'pearson',
        'transform': None,
        'file_tag': 'pearson',
        'score_label': r'Pearson $r$',
        'test_col': 'objective_pearson_test',
        'bench_col': 'objective_pearson_bench',
        'outer_col': 'objective_outer_pearson_cum',
        'combination': 'pooled',
        'full_suite': False,
    },
    # {
    #     'name': 'log_residual',
    #     'transform': lambda x: np.log(1 - x),
    #     'file_tag': 'log1mr2',
    #     'score_label': r'$\log(1 - R^2)$',
    #     'test_col': 'objective_score_test',
    #     'bench_col': 'objective_score_bench',
    #     'outer_col': 'objective_outer_scores',
    #     'combination': 'average',
    #     'full_suite': False,
    # },
]

COVARIANCE_DIAGNOSTIC_COLUMNS = [
    ('objective_bench_covariance', 'bench_cov'),
    ('objective_study_covariance', 'study_cov'),
    ('objective_bench_error_cov', 'bench_error_cov'),
    ('objective_bench_squared_error_cov', 'bench_squared_error_cov'),
    ('objective_bench_error_corr', 'bench_error_corr'),
    ('objective_bench_squared_error_corr', 'bench_squared_error_corr'),
    ('objective_bench_error_var', 'bench_error_var'),
    ('objective_bench_squared_error_var', 'bench_squared_error_var'),
    ('objective_bench_error_rho', 'bench_error_rho'),
    ('objective_bench_squared_error_rho', 'bench_squared_error_rho'),
    ('objective_bench_squared_error_cov_all', 'bench_squared_error_cov_all'),
    ('objective_bench_squared_error_var_all', 'bench_squared_error_var_all'),
    ('objective_bench_squared_error_rho_all', 'bench_squared_error_rho_all'),
    (
        'objective_bench_rho_delta_proxy_squared_error_all',
        'bench_rho_delta_proxy_squared_error_all',
    ),
    (
        'objective_bench_gain_proxy_squared_error_all',
        'bench_gain_proxy_squared_error_all',
    ),
    ('objective_bench_prediction_cov_all', 'bench_prediction_cov_all'),
    ('objective_bench_prediction_var_all', 'bench_prediction_var_all'),
    ('objective_bench_prediction_rho_all', 'bench_prediction_rho_all'),
    ('objective_bench_gain_proxy_prediction_all', 'bench_gain_proxy_prediction_all'),
    ('objective_bench_error_cov_all', 'bench_error_cov_all'),
    ('objective_bench_error_var_all', 'bench_error_var_all'),
    ('objective_bench_error_rho_all', 'bench_error_rho_all'),
    ('objective_bench_gain_proxy_error_all', 'bench_gain_proxy_error_all'),
    ('objective_study_error_cov', 'study_error_cov'),
    ('objective_study_squared_error_cov', 'study_squared_error_cov'),
    ('objective_study_error_corr', 'study_error_corr'),
    ('objective_study_squared_error_corr', 'study_squared_error_corr'),
    ('objective_study_error_var', 'study_error_var'),
    ('objective_study_squared_error_var', 'study_squared_error_var'),
    ('objective_study_error_rho', 'study_error_rho'),
    ('objective_study_squared_error_rho', 'study_squared_error_rho'),
    ('objective_study_prediction_cov_all_full', 'study_prediction_cov_all_full'),
    ('objective_study_prediction_var_all_full', 'study_prediction_var_all_full'),
    ('objective_study_prediction_rho_all_full', 'study_prediction_rho_all_full'),
    (
        'objective_study_gain_proxy_prediction_all_full',
        'study_gain_proxy_prediction_all_full',
    ),
    ('objective_study_error_cov_all_full', 'study_error_cov_all_full'),
    ('objective_study_error_var_all_full', 'study_error_var_all_full'),
    ('objective_study_error_rho_all_full', 'study_error_rho_all_full'),
    ('objective_study_gain_proxy_error_all_full', 'study_gain_proxy_error_all_full'),
    (
        'objective_study_squared_error_cov_all_full',
        'study_squared_error_cov_all_full',
    ),
    (
        'objective_study_squared_error_var_all_full',
        'study_squared_error_var_all_full',
    ),
    (
        'objective_study_squared_error_rho_all_full',
        'study_squared_error_rho_all_full',
    ),
    (
        'objective_study_gain_proxy_squared_error_all_full',
        'study_gain_proxy_squared_error_all_full',
    ),
    (
        'objective_study_prediction_cov_all_resid_train_membership',
        'study_prediction_cov_all_resid_train_membership',
    ),
    (
        'objective_study_prediction_var_all_resid_train_membership',
        'study_prediction_var_all_resid_train_membership',
    ),
    (
        'objective_study_prediction_rho_all_resid_train_membership',
        'study_prediction_rho_all_resid_train_membership',
    ),
    (
        'objective_study_gain_proxy_prediction_all_resid_train_membership',
        'study_gain_proxy_prediction_all_resid_train_membership',
    ),
    (
        'objective_study_error_cov_all_resid_train_membership',
        'study_error_cov_all_resid_train_membership',
    ),
    (
        'objective_study_error_var_all_resid_train_membership',
        'study_error_var_all_resid_train_membership',
    ),
    (
        'objective_study_error_rho_all_resid_train_membership',
        'study_error_rho_all_resid_train_membership',
    ),
    (
        'objective_study_gain_proxy_error_all_resid_train_membership',
        'study_gain_proxy_error_all_resid_train_membership',
    ),
    (
        'objective_study_squared_error_cov_all_resid_train_membership',
        'study_squared_error_cov_all_resid_train_membership',
    ),
    (
        'objective_study_squared_error_var_all_resid_train_membership',
        'study_squared_error_var_all_resid_train_membership',
    ),
    (
        'objective_study_squared_error_rho_all_resid_train_membership',
        'study_squared_error_rho_all_resid_train_membership',
    ),
    (
        'objective_study_gain_proxy_squared_error_all_resid_train_membership',
        'study_gain_proxy_squared_error_all_resid_train_membership',
    ),
    (
        'objective_study_prediction_cov_all_oof_intersection',
        'study_prediction_cov_all_oof_intersection',
    ),
    (
        'objective_study_prediction_var_all_oof_intersection',
        'study_prediction_var_all_oof_intersection',
    ),
    (
        'objective_study_prediction_rho_all_oof_intersection',
        'study_prediction_rho_all_oof_intersection',
    ),
    (
        'objective_study_gain_proxy_prediction_all_oof_intersection',
        'study_gain_proxy_prediction_all_oof_intersection',
    ),
    (
        'objective_study_error_cov_all_oof_intersection',
        'study_error_cov_all_oof_intersection',
    ),
    (
        'objective_study_error_var_all_oof_intersection',
        'study_error_var_all_oof_intersection',
    ),
    (
        'objective_study_error_rho_all_oof_intersection',
        'study_error_rho_all_oof_intersection',
    ),
    (
        'objective_study_gain_proxy_error_all_oof_intersection',
        'study_gain_proxy_error_all_oof_intersection',
    ),
    (
        'objective_study_squared_error_cov_all_oof_intersection',
        'study_squared_error_cov_all_oof_intersection',
    ),
    (
        'objective_study_squared_error_var_all_oof_intersection',
        'study_squared_error_var_all_oof_intersection',
    ),
    (
        'objective_study_squared_error_rho_all_oof_intersection',
        'study_squared_error_rho_all_oof_intersection',
    ),
    (
        'objective_study_gain_proxy_squared_error_all_oof_intersection',
        'study_gain_proxy_squared_error_all_oof_intersection',
    ),
    ('objective_study_oof_intersection_pairs_all', 'study_oof_intersection_pairs_all'),
    (
        'objective_study_oof_intersection_mean_size_all',
        'study_oof_intersection_mean_size_all',
    ),
    ('objective_split_index', 'split_index'),
    ('objective_split_train_size', 'split_train_size'),
    ('objective_split_test_size', 'split_test_size'),
    ('objective_split_train_overlap_prev_frac', 'split_train_overlap_prev_frac'),
    ('objective_split_test_overlap_prev_frac', 'split_test_overlap_prev_frac'),
    ('objective_split_train_overlap_first_frac', 'split_train_overlap_first_frac'),
    ('objective_split_test_overlap_first_frac', 'split_test_overlap_first_frac'),
    ('objective_split_train_overlap_prev_count', 'split_train_overlap_prev_count'),
    ('objective_split_test_overlap_prev_count', 'split_test_overlap_prev_count'),
    ('objective_split_train_overlap_first_count', 'split_train_overlap_first_count'),
    ('objective_split_test_overlap_first_count', 'split_test_overlap_first_count'),
]

DATASET_FORMULA_LABELS = {
    'sim_linear':       r'$y = X\beta$',
    'sim_interactions': r'$y = X\beta + \sum_{i<j} \gamma_{ij} X_i X_j$',
    'sim_nonlinear':    r'$y = \sin(4X)\beta$',
    'sim_classification': r'Classification',
}


def dataset_formula_label(dataset_type: str) -> str:
    """Return a readable LaTeX label for a dataset_type, including noise suffix.

    e.g. ``sim_linear_noise0.5`` → ``$y = X\beta$ (noise=0.5)``.
    """
    if dataset_type in DATASET_FORMULA_LABELS:
        return DATASET_FORMULA_LABELS[dataset_type]
    m = _re.match(r'^(.*)_noise([0-9.]+)$', dataset_type)
    if m:
        base, noise = m.group(1), m.group(2)
        base_label = DATASET_FORMULA_LABELS.get(base, base)
        return f"{base_label} (noise={noise})"
    return dataset_type

# %% ──────────────────────────────────────────────────────────────────────────
# Helpers specific to the simulated data format
# ─────────────────────────────────────────────────────────────────────────────

def process_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a single Dask chunk to one row per (objective, solver, dataset, idx_rep)."""
    agg_dict = {
        'objective_score_test':  'mean',
        'objective_score_train': 'mean',
        'objective_score_bench': 'mean',
        'objective_value':       'mean',
        'time':                  'sum',
    }
    if 'objective_outer_scores' in df.columns:
        agg_dict['objective_outer_scores'] = 'first'
    # New per-metric scores: scalars averaged like the R² score, list columns
    # taken as-is ('first' — averaging would corrupt the list). Absent columns
    # are dropped by the ``existing`` filter below, so old parquets are unaffected.
    for col in METRIC_SCALAR_COLUMNS:
        agg_dict[col] = 'mean'
    for col in METRIC_LIST_COLUMNS:
        agg_dict[col] = 'first'
    # Covariance diagnostics are scalar objective outputs. Within each
    # (seed, idx_rep) group there is at most one non-null value, so 'first'
    # preserves it.
    for col, _ in COVARIANCE_DIAGNOSTIC_COLUMNS:
        if col in df.columns:
            agg_dict[col] = 'first'
    # Carry through parameter columns (all identical within a group)
    param_cols = [c for c in df.columns if c.startswith('p_obj_') or
                  c.startswith('p_dataset_') or c.startswith('p_solver_')]
    for c in param_cols:
        if c not in agg_dict:
            agg_dict[c] = 'first'

    existing = {c: agg_dict[c] for c in agg_dict if c in df.columns}
    return (
        df.groupby(['objective_name', 'solver_name', 'dataset_name', 'idx_rep'])
        .agg(existing)
        .reset_index()
    )


def _split_top_level_commas(s: str) -> list[str]:
    """Split *s* on commas that sit at bracket depth 0.

    Lets us parse hyperparameter lists like ``hidden_layer_sizes=[32, 32]``
    without the inner comma being treated as a kv separator.
    """
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in s:
        if ch in '([{':
            depth += 1
            buf.append(ch)
        elif ch in ')]}':
            depth -= 1
            buf.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append(''.join(buf).strip())
    return parts


def format_solver_label(solver_name: str) -> str:
    """Examples:
        ``ExtraTrees_noHPO[n_estimators=10]`` → ``ExtraTrees$_{n = 10}$``
        ``MLP[hidden_layer_sizes=[32]]``      → ``MLP$_{[32]}$``
        ``MLP[hidden_layer_sizes=[32, 32]]``  → ``MLP$_{[32, 32]}$``
    """
    base = _re.sub(r'_noHPO', '', solver_name.split('[')[0])
    if not print_HPC:
        return base
    m = _re.search(r'\[(.+)\]', solver_name)
    if not m:
        return base
    parts = []
    for kv in _split_top_level_commas(m.group(1)):
        if '=' not in kv:
            continue
        k, v = kv.split('=', 1)
        k, v = k.strip(), v.strip()
        if k == 'n_estimators':
            parts.append(f'n = {v}')
        elif k == 'alpha':
            parts.append(f'\\alpha = {v}')
        elif k == 'max_depth':
            parts.append(f'd = {v}')
        elif k == 'hidden_layer_sizes':
            # Compact: just the architecture list, no key prefix.
            parts.append(v)
        else:
            parts.append(f'{k} = {v}')
    return f'{base}$_{{{",".join(parts)}}}$' if parts else base


def _running_in_jupyter() -> bool:
    """True when invoked from an IPython kernel (Jupyter / VSCode cells)."""
    try:
        get_ipython()  # noqa: F821
        return True
    except NameError:
        return False


def _show_if_interactive() -> None:
    """``plt.show()`` only when an IPython kernel is driving us.

    A plain ``python simulated.py`` shell has no GUI event loop, so
    ``plt.show()`` blocks indefinitely on the first figure. Gating it
    on the IPython detection keeps interactive cells working while
    making headless runs portable.
    """
    if _running_in_jupyter():
        plt.show()


def _extract_scalar(x):
    if isinstance(x, (list, np.ndarray)) and len(x) > 0:
        return x[-1]
    return x if np.isscalar(x) else np.nan


def _extract_scalar_series(s: pd.Series) -> pd.Series:
    """Vectorized scalar extraction — skips apply when column is numeric."""
    if pd.api.types.is_numeric_dtype(s):
        return s
    return s.apply(_extract_scalar)


def plot_sample_gain_by_metric(all_results, plots_dir, n_splits_max):
    """One sample-gain figure per algorithm; each curve is a different metric.

    The transpose of the per-metric ``aggregated_test_gain`` plots: there the
    hue is the solver for a fixed metric, here the hue is the metric for a fixed
    solver. Sample gain is a unitless multiplier (k_equiv + 1), so every metric
    shares one y-axis. Consumes ``all_results`` (file_tag → bootstrap results),
    grouping by (dataset, solver, N, fixed_split) and emitting a full-sweep and a
    K≤20 figure per group.
    """
    # Long frame tagged by metric name, kept in METRICS order for stable colours.
    frames = []
    for m in METRICS:
        res = all_results.get(m['file_tag'])
        if res is None or res.empty:
            continue
        r = res.copy()
        r['metric'] = m['name']
        frames.append(r)
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)

    present_metrics = set(combined['metric'])
    metric_order = [m['name'] for m in METRICS if m['name'] in present_metrics]
    _tab10 = sns.color_palette('tab10', n_colors=max(10, len(metric_order)))
    metric_palette = {name: _tab10[i % 10] for i, name in enumerate(metric_order)}

    # X-axis ticks: mirror the per-metric loop (full sweep + K≤20 zoom).
    _step = max(1, int(round(n_splits_max / 5 / 10)) * 10) or max(1, n_splits_max // 5)
    ticks = list(range(0, n_splits_max + 1, _step))
    if ticks[0] == 0:
        ticks[0] = 1
    if ticks[-1] != n_splits_max:
        ticks.append(n_splits_max)
    k20_xticks = [1, 5, 10, 15, 20]

    has_fs = 'fixed_split' in combined.columns
    group_cols = ['dataset_type', 'solver_name', 'N'] + (['fixed_split'] if has_fs else [])

    for keys, grp in combined.groupby(group_cols, dropna=False):
        if has_fs:
            dataset_type, solver_name, N, fixed_split_val = keys
            fs_tag = f"_fixed{fixed_split_val}"
        else:
            dataset_type, solver_name, N = keys
            fs_tag = ""
        ds_tag = str(dataset_type).strip().replace(' ', '_')
        solver_clean = (solver_name.replace('=', '_').replace(',', '_')
                                   .replace('[', '_').replace(']', ''))
        formula = dataset_formula_label(dataset_type)
        solver_label = format_solver_label(solver_name)
        present = [name for name in metric_order if name in set(grp['metric'])]

        for range_tag, xtk, sub in [('full', ticks, grp),
                                    ('K20', k20_xticks, grp[grp['K'] <= 20])]:
            if sub.empty:
                continue
            fig = P.plot_sample_gain_vs_splits(
                sub, y_col='sample_gain', hue_col='metric',
                hue_order=present, palette=metric_palette,
                xticks=xtk,
                title_text=f"{formula}, {solver_label}, $n_{{\\mathrm{{tr}}}}={N}$",
                xlim=(0, max(xtk)),
                direct_labels=True,
                figsize=(THREE_PANEL_SAMPLE_GAIN_FIGSIZE
                         if range_tag == 'K20' else None),
                output_path=(f"{plots_dir}/sample_gain_by_metric_"
                             f"{solver_clean}_{ds_tag}{fs_tag}_N{N}_{range_tag}.pdf"),
            )
            _show_if_interactive()
            plt.close(fig)


# %% ──────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validation Crisis analysis on simulated (benchopt) data."
    )
    parser.add_argument(
        "--data-paths", nargs="+", default=DEFAULT_DATA_PATHS,
        help="Path(s) to benchopt parquet file(s).",
    )
    parser.add_argument("--plots-dir", default=DEFAULT_PLOTS_DIR,
                        help="Output directory for plots.")
    parser.add_argument("--n-bootstraps", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--n-splits-max", type=int, default=200)
    return parser.parse_args(argv)


# %% ──────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main(
    data_paths=None,
    plots_dir=DEFAULT_PLOTS_DIR,
    n_bootstraps=100,
    test_size=0.20,
    n_splits_max=200,
):
    if data_paths is None:
        data_paths = DEFAULT_DATA_PATHS

    os.makedirs(plots_dir, exist_ok=True)
    P.set_paper_style()

    # ── 1. Load data ────────────────────────────────────────────────────────
    processed_dataframes = []
    for file_path in data_paths:
        try:
            df_raw = pd.read_parquet(file_path)
            processed_dataframes.append(process_parquet(df_raw))
            del df_raw
            gc.collect()
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    data = pd.concat(processed_dataframes, ignore_index=True)
    del processed_dataframes
    gc.collect()

    # Keep only one solver instance per algorithm family (first alphabetically)
    # to avoid cluttered aggregated plots with many hyperparameter variants.
    _seen_families: dict[str, str] = {}
    for solver in sorted(data['solver_name'].unique()):
        fam = solver.split('[')[0]
        _seen_families.setdefault(fam, solver)
    _selected_solvers = set(_seen_families.values())
    print(f"Selected solvers (one per family): {sorted(_selected_solvers)}")
    data = data[data['solver_name'].isin(_selected_solvers)].copy()

    # ── 2. Metadata extraction ──────────────────────────────────────────────
    data['procedure']    = data['p_obj_procedure']
    data['n_splits']     = data['p_obj_n_splits'].astype(int)
    data['train_size']   = data['p_obj_train_size'].astype(int)
    data['seed']         = data['p_dataset_seed'].astype(int)
    data['fixed_split']  = data['p_obj_fixed_split']

    # Base family name (e.g. "sim_linear") from the "family[params]" name.
    _family = data['dataset_name'].str.extract(r'^([^\[]+)')[0]
    # Treat different noise levels as distinct datasets — same family, noise=0.2
    # and noise=0.5 have different signal-to-noise ratios and must not be pooled
    # into one tau / sigma² estimate. Suffix only appears when noise > 0, so the
    # existing noiseless runs keep their original dataset_type.
    if 'p_dataset_noise' in data.columns:
        _noise = data['p_dataset_noise'].astype(float)
        data['dataset_type'] = np.where(
            _noise > 0,
            _family + '_noise' + _noise.astype(str),
            _family,
        )
    else:
        data['dataset_type'] = _family

    nb_exp = len(data['seed'].unique())
    print(f"Seeds: {nb_exp}  |  Datasets: {sorted(data['dataset_type'].unique())}")

    train_sizes_all = sorted(data['train_size'].unique())
    n_values_to_plot = train_sizes_all
    dataset_names_all = sorted(data['dataset_type'].unique())

    # ── 3. Build CV data; sort by group (cumulative means computed per metric) ─
    data_cv_raw = data[data['procedure'].isin(['ShuffleSplit', 'RepeatedKFold'])].copy()

    data_cv_sorted = data_cv_raw.sort_values(
        ['objective_name', 'solver_name', 'dataset_name', 'idx_rep']
    ).copy()

    # Extract no-CV base (idx_rep == 0) before any transform
    data_nocv_base_raw = data_cv_sorted[data_cv_sorted['idx_rep'] == 0].copy()

    print(f"CV rows: {len(data_cv_sorted)}")
    print(f"Procedures: {data_cv_sorted['procedure'].value_counts().to_dict()}")
    print(f"fixed_split distribution: {data_cv_sorted['fixed_split'].value_counts().to_dict()}")
    print(f"No-CV base rows: {len(data_nocv_base_raw)}")
    _n_with_outer = data_nocv_base_raw['objective_outer_scores'].apply(
        lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0
    ).sum()
    print(f"Rows with non-empty outer_scores: {_n_with_outer}")

    # ── τ estimation (raw R² and per-fold δ, delegated to the shared helper) ─
    data_cv_sorted['_obj_scalar']   = data_cv_sorted['objective_score_test'].apply(compute.to_scalar)
    data_cv_sorted['_bench_scalar'] = data_cv_sorted['objective_score_bench'].apply(compute.to_scalar)
    data_cv_sorted['delta_score_fold'] = (
        data_cv_sorted['_obj_scalar'] - data_cv_sorted['_bench_scalar']
    )

    tau_raw_df, tau_te_df = C.compute_standard_tau_tables(
        data_cv=data_cv_sorted,
        raw_fold_score_col='objective_score_test',
        group_cols=['solver_name', 'dataset_type', 'train_size',
                    'fixed_split', 'procedure'],
        model_filter_col='solver_name',
        models_to_analyze=None,
        seed_col='seed',
    )
    if not tau_raw_df.empty:
        tau_raw_display = tau_raw_df.copy()
        tau_raw_display['solver_name'] = tau_raw_display['solver_name'].map(format_solver_label)
        print("\nτ_raw per algorithm (raw R²):")
        print(tau_raw_display.to_string(index=False))
    else:
        print("compute_tau returned no rows — check seed/fold availability.")
    if not tau_te_df.empty:
        tau_te_display = tau_te_df.copy()
        tau_te_display['solver_name'] = tau_te_display['solver_name'].map(format_solver_label)
        print("\nτ^te per algorithm (per-fold delta R²):")
        print(tau_te_display[['solver_name', 'dataset_type', 'train_size',
                              'fixed_split', 'procedure',
                              'tau_te', 'sigma2_te_K',
                              'tau_te_lo', 'tau_te_hi']].to_string(index=False))
        for (ds, fs, proc), sub in tau_te_display.groupby(
            ['dataset_type', 'fixed_split', 'procedure']
        ):
            fig = P.plot_tau_bar(
                sub,
                x_col='solver_name',
                hue_col='train_size',
                value_col='tau_te',
                title=rf"$\hat\tau^{{te}}$ — {dataset_formula_label(ds)}, "
                      rf"fixed_split={fs}, {proc}",
                ylabel=rf"$\hat\tau^{{te}}$",
                output_path=f"{plots_dir}/tau_te_{ds}_fixed{fs}_{proc}.pdf",
                ci_cols=('tau_te_lo', 'tau_te_hi'),
                overlay_B=False,
            )
            _show_if_interactive()
            plt.close(fig)

    # ── Loop over metrics (R² and log(1-R²)) ──────────────────────────────
    # Unless the multi-metric switch is on, restrict to R² so the run produces
    # exactly the original R²-only outputs.
    all_results = {}
    metrics_to_run = (
        METRICS if PLOT_ALL_METRICS
        else [m for m in METRICS if m['name'] == 'r2']
    )
    for metric in metrics_to_run:
        transform = metric['transform']
        file_tag = metric['file_tag']
        test_col = metric.get('test_col', 'objective_score_test')
        bench_col = metric.get('bench_col', 'objective_score_bench')
        outer_col = metric.get('outer_col', 'objective_outer_scores')
        combination = metric.get('combination', 'average')
        full_suite = metric.get('full_suite', False)
        print(f"\n{'='*60}")
        print(f"Metric: {metric['name']} (file_tag={file_tag})")
        print(f"{'='*60}")

        # Skip metrics whose columns are absent (e.g. older parquets that only
        # carry R²), so this loop degrades to the historical R²-only behaviour.
        _required = [test_col, bench_col, outer_col]
        _missing = [c for c in _required if c not in data_cv_sorted.columns]
        if _missing:
            print(f"Skipping metric {metric['name']}: missing columns {_missing}")
            continue

        # -- Point the working score columns at this metric, then transform &
        #    compute cumulative means. For R² the remap is an identity. --
        data_cv = data_cv_sorted.copy()
        if test_col != 'objective_score_test':
            data_cv['objective_score_test'] = data_cv[test_col]
            data_cv['objective_score_bench'] = data_cv[bench_col]
        if transform is not None:
            data_cv['objective_score_test'] = transform(data_cv['objective_score_test'])
            data_cv['objective_score_bench'] = transform(data_cv['objective_score_bench'])

        _grp = data_cv.groupby(['objective_name', 'solver_name', 'dataset_name'])
        data_cv['n_splits'] = _grp.cumcount() + 1
        data_cv['_cumsum_t'] = _grp['objective_score_test'].cumsum()
        data_cv['objective_score_test'] = data_cv['_cumsum_t'] / data_cv['n_splits']
        data_cv['_cumsum_b'] = _grp['objective_score_bench'].cumsum()
        data_cv['objective_score_bench'] = data_cv['_cumsum_b'] / data_cv['n_splits']
        data_cv.drop(columns=['_cumsum_t', '_cumsum_b'], inplace=True)
        data_cv['delta_score_test'] = (
            data_cv['objective_score_test'] - data_cv['objective_score_bench']
        )

        # -- Apply transform to no-CV data --
        data_nocv_base = data_nocv_base_raw.copy()
        if transform is not None:
            data_nocv_base['objective_score_test'] = transform(
                data_nocv_base['objective_score_test']
            )
            data_nocv_base['objective_score_bench'] = transform(
                data_nocv_base['objective_score_bench']
            )
            data_nocv_base['objective_outer_scores'] = data_nocv_base[
                'objective_outer_scores'
            ].apply(
                lambda scores: list(transform(np.array(scores)))
                if isinstance(scores, (list, np.ndarray)) and len(scores) > 0
                else scores
            )

        # ── 4. Build no-CV cache and run bootstrap ──────────────────────────
        # Feed the metric's own single-split score / hidden-list columns; the
        # combination mode tells build_nocv_cache whether the list is per-chunk
        # (averaged) or already pooled.
        nocv_cache = compute.build_nocv_cache(
            data_nocv_base,
            model_col='solver_name',
            size_col='train_size',
            dataset_col='dataset_type',
            hidden_col=outer_col,
            test_col=test_col,
            bench_col=bench_col,
            use_delta=True,
            extra_key_cols=['fixed_split'],
            combination=combination,
        )
        print(f"nocv_cache entries: {len(nocv_cache)}")

        results_df = compute.sample_gain_bootstrap(
            data_cv=data_cv,
            score_col_cv='delta_score_test',
            model_col='solver_name',
            n_bootstraps=n_bootstraps,
            method='absolute',
            test_size=test_size,
            nocv_cache=nocv_cache,
            models_to_analyze=None,
            study_sizes=n_values_to_plot,
            size_col='train_size',
            dataset_col='dataset_type',
            extra_cv_cols=['fixed_split'],
            compute_train_size=False,
        )

        if results_df.empty:
            print("No bootstrap results — check data and cache.")
        else:
            results_df = results_df[results_df['K'] <= n_splits_max]
            print(f"Bootstrap results: {len(results_df)} rows")
            print(f"fixed_split values in results: {results_df['fixed_split'].unique() if 'fixed_split' in results_df.columns else 'N/A'}")

        all_results[file_tag] = results_df

        # ── 5. Build diagnostic DataFrames ──────────────────────────────────
        target_splits_diag = [5, 10, 20]
        diag_id_cols = ['seed', 'solver_name', 'train_size', 'dataset_type', 'fixed_split']

        df_base_diag = data_nocv_base.copy()
        df_base_diag['base_score']  = df_base_diag['objective_score_test'].pipe(_extract_scalar_series)
        df_base_diag['bench_score'] = df_base_diag['objective_score_bench'].pipe(_extract_scalar_series)

        df_nocv_k = compute.expand_hidden_scores(
            df_base_diag,
            scores_col='objective_outer_scores',
            id_cols=diag_id_cols + ['base_score', 'bench_score'],
        )
        print(f"Expanded no-CV rows: {len(df_nocv_k)}")

        df_hidden_diag = df_nocv_k.copy()
        df_hidden_diag['combined_score'] = (
            df_hidden_diag['base_score'] + df_hidden_diag['k'] * df_hidden_diag['mean_score_k']
        ) / (df_hidden_diag['k'] + 1)

        df_k0 = df_base_diag[diag_id_cols + ['base_score', 'bench_score']].copy()
        df_k0['k'] = 0
        df_k0['combined_score'] = df_k0['base_score']

        df_diag_all_k = pd.concat([
            df_k0[diag_id_cols + ['k', 'combined_score', 'bench_score']],
            df_hidden_diag[diag_id_cols + ['k', 'combined_score', 'bench_score']],
        ], ignore_index=True)
        df_diag_all_k['delta_score'] = df_diag_all_k['combined_score'] - df_diag_all_k['bench_score']

        cv_diag = data_cv.copy()
        cv_diag['score_scalar'] = cv_diag['objective_score_test'].pipe(_extract_scalar_series)
        cv_diag['bench_scalar'] = cv_diag['objective_score_bench'].pipe(_extract_scalar_series)
        cv_diag['delta_scalar'] = cv_diag['score_scalar'] - cv_diag['bench_scalar']
        cv_diag_filt = cv_diag[cv_diag['n_splits'].isin(target_splits_diag)]

        # ── 6. Plots ────────────────────────────────────────────────────────
        if not results_df.empty:
            # Linearly-spaced x-axis ticks (roughly 5 evenly-spaced positions).
            _step = max(1, int(round(n_splits_max / 5 / 10)) * 10) or max(1, n_splits_max // 5)
            ticks = list(range(0, n_splits_max + 1, _step))
            if ticks[0] == 0:
                ticks[0] = 1
            if ticks[-1] != n_splits_max:
                ticks.append(n_splits_max)

            unique_train_sizes = sorted(results_df['N'].unique())
            unique_solvers     = sorted(results_df['solver_name'].unique())
            solver_display     = {
                solver_name: format_solver_label(solver_name)
                for solver_name in unique_solvers
            }

            fixed_split_values = (
                sorted(results_df['fixed_split'].unique())
                if 'fixed_split' in results_df.columns else [None]
            )

            _tab10         = sns.color_palette('tab10', n_colors=max(10, len(unique_solvers)))
            solver_palette = {
                solver_name: _tab10[i % 10]
                for i, solver_name in enumerate(unique_solvers)
            }
            bright_colors  = sns.color_palette("mako", n_colors=len(target_splits_diag))
            cv_colors_retrieval = list(bright_colors[:3])

            plot_configs_var = [
                {
                    'nocv_col':    'combined_score',
                    'cv_col':      'score_scalar',
                    'file_tag':    'acc',
                    'x_transform': lambda k, ts: 1 + np.asarray(k) * ts,
                    'xlabel':      'Effective Data Multiplier (Total / Study Size)',
                },
                {
                    'nocv_col':    'delta_score',
                    'cv_col':      'delta_scalar',
                    'file_tag':    'delta',
                    'x_transform': lambda k, ts: 1 + np.asarray(k),
                    'xlabel':      'Effective Test Multiplier (Total Test / Original Test)',
                },
            ]

            # ── Overlay: fixed_split=True vs False on one figure ───────────────
            # Both settings superposed per algorithm — solid for fixed_split=False,
            # dotted for fixed_split=True — so freezing the split is visible at a
            # glance. Only drawn when both settings are present in the results.
            if ('fixed_split' in results_df.columns
                    and results_df['fixed_split'].nunique() >= 2):
                _fs_label = {False: 'Resampled split', True: 'Fixed split'}
                _fs_dashes = {'Resampled split': (), 'Fixed split': (1, 1)}
                _fs_order = [lbl for val, lbl in _fs_label.items()
                             if val in set(results_df['fixed_split'])]
                for dataset_type in sorted(results_df['dataset_type'].unique()):
                    ov_ds = results_df[results_df['dataset_type'] == dataset_type]
                    ds_tag = dataset_type.strip().replace(' ', '_')
                    formula = dataset_formula_label(dataset_type)
                    ds_solvers = sorted(ov_ds['solver_name'].unique())
                    solver_palette_disp = {
                        solver_display[s]: solver_palette[s]
                        for s in ds_solvers if s in solver_palette
                    }
                    for t_size in unique_train_sizes:
                        for range_tag, xtk in [('full', ticks),
                                               ('K20', [1, 5, 10, 15, 20])]:
                            sub = ov_ds[(ov_ds['N'] == t_size)
                                        & (ov_ds['K'] <= max(xtk))]
                            if sub['fixed_split'].nunique() < 2:
                                continue
                            plot_df = sub.copy()
                            plot_df['solver_name'] = plot_df['solver_name'].map(solver_display)
                            plot_df['split_mode'] = plot_df['fixed_split'].map(_fs_label)
                            present_disp = [
                                solver_display[s] for s in ds_solvers
                                if s in sub['solver_name'].values
                            ]
                            fig = P.plot_sample_gain_vs_splits(
                                plot_df, y_col='sample_gain', hue_col='solver_name',
                                hue_order=present_disp, palette=solver_palette_disp,
                                style_col='split_mode', style_order=_fs_order,
                                dashes=_fs_dashes, style_legend=True,
                                style_legend_loc='upper left',
                                style_legend_bbox=(0.02, 0.86),
                                xticks=xtk,
                                title_text=f"{formula}, $n_{{\\mathrm{{tr}}}}={t_size}$",
                                title_text_xy=(0.02, 0.97),
                                title_text_ha="left",
                                title_text_va="top",
                                xlim=(0, max(xtk)),
                                direct_labels=True,
                                figsize=(THREE_PANEL_SAMPLE_GAIN_FIGSIZE
                                         if range_tag == 'K20' else None),
                                output_path=(
                                    f"{plots_dir}/aggregated_test_gain_k_{t_size}_"
                                    f"{ds_tag}_{file_tag}_fixedsplit_overlay_{range_tag}.pdf"
                                ),
                            )
                            _show_if_interactive()
                            plt.close(fig)

            for fixed_split_val in fixed_split_values:
                fs_tag = f"_fixed{fixed_split_val}" if fixed_split_val is not None else ""
                if fixed_split_val is not None:
                    fs_res  = results_df[results_df['fixed_split'] == fixed_split_val]
                    fs_diag = df_diag_all_k[df_diag_all_k['fixed_split'] == fixed_split_val]
                    fs_cv   = cv_diag_filt[cv_diag_filt['fixed_split'] == fixed_split_val]
                    fs_cv_all = cv_diag[cv_diag['fixed_split'] == fixed_split_val]
                    fs_nocv = data_nocv_base[data_nocv_base['fixed_split'] == fixed_split_val]
                else:
                    fs_res, fs_diag, fs_cv, fs_nocv = results_df, df_diag_all_k, cv_diag_filt, data_nocv_base
                    fs_cv_all = cv_diag

                for dataset_type in sorted(fs_res['dataset_type'].unique()):
                    ds_res  = fs_res[fs_res['dataset_type'] == dataset_type]
                    ds_diag = fs_diag[fs_diag['dataset_type'] == dataset_type]
                    ds_cv   = fs_cv[fs_cv['dataset_type'] == dataset_type]
                    # Retrieval needs every K (compute_retrieval reads n_splits==1
                    # for its single-split baseline), not just target_splits_diag.
                    ds_cv_all = fs_cv_all[fs_cv_all['dataset_type'] == dataset_type]
                    ds_nocv = fs_nocv[fs_nocv['dataset_type'] == dataset_type]
                    ds_tag  = dataset_type.strip().replace(' ', '_')
                    formula = dataset_formula_label(dataset_type)
                    ds_solvers = sorted(ds_res['solver_name'].unique())
                    print(f"\n=== {dataset_type}  fixed_split={fixed_split_val}  metric={metric['name']} ===")

                    # ── Aggregated per train size: sample_gain vs K, hue = solver ─
                    solver_palette_disp = {
                        solver_display[solver_name]: solver_palette[solver_name]
                        for solver_name in ds_solvers
                        if solver_name in solver_palette
                    }
                    k20_xticks = [1, 5, 10, 15, 20]
                    # Same K20 plot with the GradientBoosting and Ridge
                    # families dropped (matched on the solver-name prefix, so
                    # HPO variants like ``Ridge_GridSearch`` are excluded too).
                    _excluded_families = ('GradientBoosting', 'Ridge')
                    ds_res_k20 = ds_res[ds_res['K'] <= 20]
                    ds_res_k20_noGB_noRidge = ds_res_k20[
                        ~ds_res_k20['solver_name'].map(
                            lambda s: s.split('[')[0].startswith(_excluded_families)
                        )
                    ]
                    plot_ranges = [
                        ('full', ticks, ds_res),
                        ('K20',  k20_xticks, ds_res_k20),
                        ('K20_noGB_noRidge', k20_xticks, ds_res_k20_noGB_noRidge),
                    ]
                    for t_size in unique_train_sizes:
                        for range_tag, xtk, range_df in plot_ranges:
                            subset_df = range_df[range_df['N'] == t_size]
                            if subset_df.empty:
                                continue
                            present_disp = [
                                solver_display[solver_name]
                                for solver_name in ds_solvers
                                if solver_name in subset_df['solver_name'].values
                            ]
                            plot_df_disp = subset_df.copy()
                            plot_df_disp['solver_name'] = plot_df_disp['solver_name'].map(solver_display)
                            fig = P.plot_sample_gain_vs_splits(
                                plot_df_disp, y_col='sample_gain', hue_col='solver_name',
                                hue_order=present_disp, palette=solver_palette_disp,
                                xticks=xtk,
                                title_text=f"{formula}, $n_{{\\mathrm{{tr}}}}={t_size}$",
                                title_text_xy=(0.03, 0.97),
                                title_text_ha="left",
                                title_text_va="top",
                                xlim=(0, max(xtk)),
                                direct_labels=True,
                                figsize=(
                                    THREE_PANEL_SAMPLE_GAIN_FIGSIZE
                                    if range_tag.startswith('K20')
                                    else None
                                ),
                                output_path=f"{plots_dir}/aggregated_test_gain_k_{t_size}_{ds_tag}{fs_tag}_{file_tag}_{range_tag}.pdf",
                            )
                            _show_if_interactive()
                            plt.close(fig)

                    # Secondary plots (variance diagnostics, retrieval) are the
                    # validated R²-only outputs; the new metrics emit sample-gain
                    # curves only, so skip the rest of this dataset iteration.
                    if not full_suite:
                        continue

                    # ── Variance diagnostic ───────────────────────────────────
                    for config in plot_configs_var:
                        var_k_df = (
                            ds_diag.groupby(['solver_name', 'train_size', 'k'])[config['nocv_col']]
                            .var().reset_index(name='variance')
                        )
                        _xt = config['x_transform']
                        var_k_df['x_axis_val'] = _xt(var_k_df['k'].to_numpy(), test_size)
                        var_cv_df = (
                            ds_cv.groupby(['solver_name', 'train_size', 'procedure', 'n_splits'])[config['cv_col']]
                            .var().reset_index(name='variance')
                        )
                        # Pre-group to avoid repeated boolean indexing
                        _k_groups = dict(list(var_k_df.groupby(['solver_name', 'train_size'])))
                        _cv_groups = dict(list(var_cv_df.groupby(['solver_name', 'train_size', 'procedure'])))
                        for (solver, N), subset_k in _k_groups.items():
                                if solver not in ds_solvers:
                                    continue
                                clean = (solver.replace('=', '_').replace(',', '_')
                                               .replace('[', '_').replace(']', ''))
                                for proc in sorted(ds_cv['procedure'].unique()):
                                    subset_cv = _cv_groups.get((solver, N, proc))
                                    if subset_cv is None or subset_cv.empty:
                                        continue
                                    fig = P.plot_variance_diagnostic(
                                        subset_k, subset_cv,
                                        target_splits=target_splits_diag,
                                        x_col='x_axis_val', var_col='variance',
                                        xlabel=config['xlabel'],
                                        cv_colors=bright_colors, yscale='log',
                                        output_path=(
                                            f"{plots_dir}/variance_diagnostic_{config['file_tag']}"
                                            f"_{clean}_{ds_tag}{fs_tag}_{file_tag}_N{N}_{proc}.pdf"
                                        ),
                                    )
                                    plt.close(fig)

                    # ── Retrieval of best model ───────────────────────────────
                    _ret_all = ds_diag[['seed', 'solver_name', 'train_size', 'k', 'combined_score']].copy()
                    _ret_all['multiplier'] = 1 + _ret_all['k']
                    _ret_all = _ret_all.rename(columns={'train_size': 'study_size',
                                                         'solver_name': 'model'})

                    cv_ret_cols = ['seed', 'solver_name', 'train_size', 'n_splits', 'score_scalar']
                    if 'procedure' in ds_cv_all.columns:
                        cv_ret_cols.append('procedure')
                    _cv_ret = ds_cv_all[cv_ret_cols].copy()
                    _cv_ret = _cv_ret.rename(columns={'train_size': 'study_size',
                                                       'solver_name': 'model'})

                    _gt = (
                        ds_nocv.groupby(['solver_name', 'train_size'])['objective_score_bench']
                        .mean().reset_index()
                        .rename(columns={'train_size': 'study_size', 'solver_name': 'model'})
                    )

                    _study_sizes_ret = sorted(_ret_all['study_size'].unique())
                    _models_ret      = sorted(_ret_all['model'].unique())

                    if len(_models_ret) >= 2 and not _ret_all.empty:
                        curves_df_ret, table_df_ret = compute.compute_retrieval(
                            df_nocv_all=_ret_all,
                            df_cv=_cv_ret,
                            gt_ranking=_gt,
                            models_list=_models_ret,
                            study_sizes=_study_sizes_ret,
                            target_splits=[5, 10, 20],
                            model_col='model',
                            score_col_nocv='combined_score',
                            score_col_cv='score_scalar',
                            test_size=test_size,
                        )
                        print("\nPairwise ranking summary:")
                        print(table_df_ret.to_string(index=False))

                        if not curves_df_ret.empty:
                            for (n_size, best, worst), curve in curves_df_ret.groupby(
                                ['study_size', 'best_model', 'worst_model']
                            ):
                                pair_cv_all = _cv_ret[
                                    (_cv_ret['study_size'] == n_size) &
                                    (_cv_ret['model'].isin([best, worst]))
                                ]
                                clean_b = (best.replace('=', '_').replace(',', '_')
                                               .replace('[', '_').replace(']', ''))
                                clean_w = (worst.replace('=', '_').replace(',', '_')
                                                .replace('[', '_').replace(']', ''))
                                if 'procedure' in pair_cv_all.columns:
                                    procedures = sorted(pair_cv_all['procedure'].dropna().unique())
                                    if not procedures:
                                        procedures = [None]
                                else:
                                    procedures = [None]

                                for proc in procedures:
                                    if proc is None:
                                        pair_cv = pair_cv_all
                                        proc_suffix = ""
                                    else:
                                        pair_cv = pair_cv_all[pair_cv_all['procedure'] == proc]
                                        proc_suffix = f"_{proc}"
                                    cv_lines = {}
                                    for K_split in [5, 10, 20]:
                                        split_data = pair_cv[pair_cv['n_splits'] == K_split]
                                        pivot  = split_data.pivot_table(
                                            index='seed', columns='model', values='score_scalar'
                                        ).dropna()
                                        if not pivot.empty and best in pivot.columns and worst in pivot.columns:
                                            cv_lines[K_split] = float((pivot[best] > pivot[worst]).mean())

                                    fig = P.plot_retrieval_curve(
                                        curve_data=curve,
                                        cv_lines=cv_lines,
                                        best_model=format_solver_label(best),
                                        worst_model=format_solver_label(worst),
                                        target_splits=[5, 10, 20],
                                        cv_colors=cv_colors_retrieval,
                                        y_as_percent=True,
                                        output_path=(
                                            f"{plots_dir}/retrieval_best_{clean_b}_vs_{clean_w}"
                                            f"_{ds_tag}{fs_tag}_{file_tag}_N{n_size}{proc_suffix}.pdf"
                                        ),
                                    )
                                    plt.close(fig)

    # ── Per-algorithm sample-gain across metrics ─────────────────────────────
    # One figure per learning algorithm; each curve is that algorithm's sample
    # gain under a different metric. Only meaningful with the multi-metric switch.
    if PLOT_ALL_METRICS:
        plot_sample_gain_by_metric(all_results, plots_dir, n_splits_max)

    # ── Export τ̂ + sample-gain@K=20 table for cross-pipeline scatter ─────
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    if not tau_raw_df.empty and all_results:
        group_cols = ['solver_name', 'dataset_type', 'N', 'fixed_split', 'cv_procedure']

        def _gain20(file_tag):
            res = all_results.get(file_tag)
            if res is None or res.empty:
                return pd.DataFrame(
                    columns=group_cols + ['K', 'sample_gain']
                )
            return compute.sample_gain_at_splits(res, group_cols=group_cols, target_splits=[20])

        gain_r2 = _gain20('r2').rename(columns={'sample_gain': 'sample_gain_K20'})
        gain_log = _gain20('log1mr2').rename(columns={'sample_gain': 'sample_gain_K20_log1mr2'})

        # Normalize keys so the merge with the raw-τ table works.
        rename_gain = {
            'solver_name':  'algorithm',
            'dataset_type': 'dataset',
            'N':            'train_size',
            'cv_procedure': 'procedure',
        }
        gain_r2 = gain_r2.rename(columns=rename_gain).drop(columns=['K'], errors='ignore')
        gain_log = gain_log.rename(columns=rename_gain).drop(columns=['K'], errors='ignore')

        tau_raw_export = tau_raw_df.rename(columns={
            'solver_name':  'algorithm',
            'dataset_type': 'dataset',
            'train_size':   'train_size',
        })[[
            'algorithm', 'dataset', 'train_size', 'fixed_split', 'procedure',
            'tau', 'sigma2_HO_K', 'W', 'B', 'K', 'n_seeds',
        ]]

        merge_keys = ['algorithm', 'dataset', 'train_size', 'fixed_split', 'procedure']
        if not tau_te_df.empty:
            tau_te_export = tau_te_df.rename(columns={
                'solver_name':  'algorithm',
                'dataset_type': 'dataset',
                'train_size':   'train_size',
            })[[
                'algorithm', 'dataset', 'train_size', 'fixed_split', 'procedure',
                'tau_te', 'sigma2_te_K', 'tau_te_lo', 'tau_te_hi',
            ]]
        else:
            tau_te_export = None

        bench_means = (
            data_cv_sorted.groupby(
                ['solver_name', 'dataset_type', 'train_size',
                 'fixed_split', 'procedure'],
                dropna=False,
            )['objective_score_bench']
            .mean()
            .reset_index()
            .rename(columns={
                'solver_name':            'algorithm',
                'dataset_type':           'dataset',
                'train_size':             'train_size',
                'objective_score_bench':  'mean_bench_score',
            })
        )

        # sigma^2_HO,|1: across-seed variance of single-split (idx_rep=0) HO
        # score, on raw R^2 (matching tau_df's metric -- no transform applied).
        _nocv = data_nocv_base_raw.copy()
        _nocv['_score'] = _nocv['objective_score_test'].pipe(_extract_scalar_series)
        _nocv['_bench'] = _nocv['objective_score_bench'].pipe(_extract_scalar_series)
        _nocv['_delta'] = _nocv['_score'] - _nocv['_bench']
        sigma2_HO_1 = (
            _nocv.groupby(
                ['solver_name', 'dataset_type', 'train_size',
                 'fixed_split', 'procedure'],
                dropna=False,
            )['_score']
            .var()
            .reset_index()
            .rename(columns={
                'solver_name':  'algorithm',
                'dataset_type': 'dataset',
                'train_size':   'train_size',
                '_score':       'sigma2_HO_1',
            })
        )

        # sigma^{2,te},|1: across-seed variance of single-split delta-bench
        # scores. Numerator of the delta-based predicted gain; matches units
        # with the delta-based denominator (sigma^{2,te},|K + (K-1) tau^te).
        sigma2_te_1 = (
            _nocv.groupby(
                ['solver_name', 'dataset_type', 'train_size',
                 'fixed_split', 'procedure'],
                dropna=False,
            )['_delta']
            .var()
            .reset_index()
            .rename(columns={
                'solver_name':  'algorithm',
                'dataset_type': 'dataset',
                'train_size':   'train_size',
                '_delta':       'sigma2_te_1',
            })
        )

        # sigma^2_HO,|tr,xi: within-seed variance of outer-test splits for a
        # fixed trained model, averaged across seeds. Drops the seed-level
        # training component that inflates the across-seeds estimator.
        sigma2_HO_trxi = compute.sigma2_nocv_within_seed(
            data_nocv_base_raw,
            scores_col='objective_outer_scores',
            group_cols=['solver_name', 'dataset_type', 'train_size',
                        'fixed_split', 'procedure'],
        ).rename(columns={
            'solver_name':  'algorithm',
            'dataset_type': 'dataset',
            'train_size':   'train_size',
            'sigma2_nocv_within': 'sigma2_HO_trxi',
        })

        # Prediction- and error-covariance diagnostics on X_bench / X_study,
        # averaged across seeds. The raw prediction covariance is label-free;
        # the error diagnostics use labels and target the shared-error term
        # more directly.
        def _last_non_null(series):
            valid = series.dropna()
            if valid.empty:
                return np.nan
            return valid.iloc[-1]

        def _aggregate_cov_diagnostics():
            available = [
                (source_col, out_col)
                for source_col, out_col in COVARIANCE_DIAGNOSTIC_COLUMNS
                if source_col in data_cv_sorted.columns
            ]
            if not available:
                return None
            source_cols = [source_col for source_col, _ in available]
            out_cols = {source_col: out_col for source_col, out_col in available}
            seed_keys = [
                'objective_name', 'solver_name', 'dataset_name',
                'dataset_type', 'train_size', 'fixed_split', 'procedure',
                'seed',
            ]
            seed_level = (
                data_cv_sorted
                .dropna(subset=source_cols, how='all')
                .sort_values(seed_keys + ['idx_rep'])
                .groupby(seed_keys, dropna=False)[source_cols]
                .agg(_last_non_null)
                .reset_index()
            )
            grouped = (
                seed_level
                .dropna(subset=source_cols, how='all')
                .groupby(
                    ['solver_name', 'dataset_type', 'train_size',
                     'fixed_split', 'procedure'],
                    dropna=False,
                )[source_cols]
                .mean()
                .reset_index()
                .rename(columns={
                    'solver_name': 'algorithm',
                    'dataset_type': 'dataset',
                    'train_size':   'train_size',
                    **out_cols,
                })
            )
            return grouped

        cov_diagnostics_df = _aggregate_cov_diagnostics()

        merged = C.build_tau_gain_export(
            tau_df=tau_raw_export,
            merge_keys=merge_keys,
            pipeline='simulated',
            gain20_df=gain_r2,
            gain20_log1mr2_df=gain_log,
            tau_te_df=tau_te_export,
            bench_means_df=bench_means,
            sigma2_HO_1_df=sigma2_HO_1,
            sigma2_te_1_df=sigma2_te_1,
            sigma2_HO_trxi_df=sigma2_HO_trxi,
            bench_cov_df=cov_diagnostics_df,
        )
        merged.to_parquet(f"{results_dir}/tau_gain_simulated.parquet", index=False)
        print(f"Wrote {results_dir}/tau_gain_simulated.parquet ({len(merged)} rows)")

        # ── Simulated-only relationship search deliverables ───────────────
        # These outputs are the bridge between the Jean-Zay benchopt parquet
        # runs and tau_vs_gain.py. The live table is seed/prefix-level and uses
        # no benchmark/study diagnostics; benchmark/study searches consume only
        # scalar diagnostics already exported into the canonical table.
        live_features = G.build_single_seed_live_features(
            data_cv_sorted,
            gain_r2,
            K=20,
        )
        relationship_outputs = G.run_simulated_gain_relationships(
            config_df=merged,
            live_df=live_features,
            K=20,
        )
        relationship_paths = G.write_relationship_outputs(
            results_dir=results_dir,
            live_features=live_features,
            metrics=relationship_outputs["metrics"],
            predictions=relationship_outputs["predictions"],
        )
        for label, path in relationship_paths.items():
            print(f"Wrote {label}: {path}")

    print("\nDone.")
    return all_results


# %% ──────────────────────────────────────────────────────────────────────────
# Run the pipeline.
# - VSCode cells / Jupyter: calls main() with defaults.
# - CLI (python simulated.py --data-paths ...): parses arguments first.
# - import simulated: does nothing.

if __name__ == '__main__':
    if _running_in_jupyter():
        main()
    else:
        args = parse_args()
        main(
            data_paths=args.data_paths,
            plots_dir=args.plots_dir,
            n_bootstraps=args.n_bootstraps,
            test_size=args.test_size,
            n_splits_max=args.n_splits_max,
        )

# %%
    