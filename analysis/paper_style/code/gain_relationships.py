"""Relationship search for simulated K=20 sample-gain diagnostics.

The helpers in this module are intentionally scoped to the simulated pipeline.
They build single-seed live MCCV features, fit low-complexity gain formulas,
and quantify robustness by model family.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


K_DEFAULT = 20
EPS = 1e-12

SCOPE_NONBENCH_LIVE = "nonbench_live"
SCOPE_BENCHMARK = "benchmark"
SCOPE_STUDY = "study"
SCOPES = (SCOPE_NONBENCH_LIVE, SCOPE_BENCHMARK, SCOPE_STUDY)

OBSERVED_TARGET_COLUMNS = {
    "sample_gain_K20",
    "sample_gain_K20_log1mr2",
    "k_equiv",
}

NONBENCH_FORBIDDEN_FRAGMENTS = (
    "bench",
    "benchmark",
    "study",
    "tau_te",
    "sigma2_te",
    "delta_score",
    "objective_score_bench",
)

STUDY_FORBIDDEN_FRAGMENTS = (
    "bench",
    "benchmark",
    "tau_te",
    "sigma2_te",
    "objective_score_bench",
)

LIVE_FEATURE_COLUMNS = [
    "train_size_feature",
    "inv_train_size_feature",
    "prefix_j",
    "prefix_inv_j",
    "prefix_score_mean_abs",
    "prefix_score_loss",
    "prefix_score_var",
    "prefix_score_std",
    "prefix_score_sem",
    "prefix_score_range",
    "prefix_score_last_loss",
    "prefix_score_delta_abs",
    "prefix_score_slope_abs",
    "prefix_train_mean_abs",
    "prefix_train_loss",
    "prefix_train_var",
    "prefix_generalization_gap_abs_mean",
    "prefix_score_cv2",
    "prefix_split_test_overlap_prev_frac_mean",
    "prefix_split_train_overlap_prev_frac_mean",
    "prefix_split_test_overlap_first_frac_mean",
    "prefix_split_train_overlap_first_frac_mean",
    "prefix_split_test_overlap_prev_frac_last",
    "prefix_split_train_overlap_prev_frac_last",
    "prefix_split_test_overlap_first_frac_last",
    "prefix_split_train_overlap_first_frac_last",
    "prefix_train_score_ratio",
    "prefix_gap_over_score_loss",
    "prefix_gap_over_train_loss",
    "prefix_score_var_over_train_var",
]

BENCHMARK_BASE_FEATURE_COLUMNS = [
    "train_size_feature",
    "inv_train_size_feature",
    "rho_delta",
    "icc_gain_from_rho_delta",
    "predicted_sample_gain_K20",
    "tau_te",
    "sigma2_te_K",
    "sigma2_te_1",
    "mean_bench_score",
    "bench_cov",
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
    "bench_cov_times_error_rho",
    "bench_cov_times_error_corr",
    "bench_cov_times_squared_error_rho",
    "bench_cov_times_squared_error_corr",
    "bench_cov_times_sqrt_error_squared_rho",
    "bench_error_cov_over_error_var",
    "bench_squared_error_cov_over_squared_error_var",
    "bench_cov_times_error_cov_over_error_var",
    "bench_cov_times_sqerr_cov_over_sqerr_var",
    "bench_cov_times_prediction_rho_all",
    "bench_cov_times_error_rho_all",
    "bench_cov_times_squared_error_rho_all",
    "bench_prediction_cov_over_var_all",
    "bench_error_cov_over_var_all",
    "bench_squared_error_cov_over_var_all",
    "bench_cov_times_prediction_cov_over_var_all",
    "bench_cov_times_error_cov_over_var_all",
    "bench_cov_times_squared_error_cov_over_var_all",
]

STUDY_BASE_FEATURE_COLUMNS = [
    "train_size_feature",
    "inv_train_size_feature",
    "study_cov",
    "study_error_cov",
    "study_error_corr",
    "study_error_var",
    "study_error_rho",
    "study_squared_error_cov",
    "study_squared_error_corr",
    "study_squared_error_var",
    "study_squared_error_rho",
    "study_cov_times_error_rho",
    "study_cov_times_error_corr",
    "study_cov_times_squared_error_rho",
    "study_cov_times_squared_error_corr",
    "study_cov_times_sqrt_error_squared_rho",
    "study_error_cov_over_error_var",
    "study_squared_error_cov_over_squared_error_var",
    "study_cov_times_error_cov_over_error_var",
    "study_cov_times_sqerr_cov_over_sqerr_var",
    "study_cov_times_prediction_rho_all_full",
    "study_cov_times_error_rho_all_full",
    "study_cov_times_squared_error_rho_all_full",
    "study_prediction_cov_over_var_all_full",
    "study_error_cov_over_var_all_full",
    "study_squared_error_cov_over_var_all_full",
    "study_cov_times_prediction_cov_over_var_all_full",
    "study_cov_times_error_cov_over_var_all_full",
    "study_cov_times_squared_error_cov_over_var_all_full",
    "study_cov_times_prediction_rho_all_resid_train_membership",
    "study_cov_times_error_rho_all_resid_train_membership",
    "study_cov_times_squared_error_rho_all_resid_train_membership",
    "study_prediction_cov_over_var_all_resid_train_membership",
    "study_error_cov_over_var_all_resid_train_membership",
    "study_squared_error_cov_over_var_all_resid_train_membership",
    "study_cov_times_prediction_cov_over_var_all_resid_train_membership",
    "study_cov_times_error_cov_over_var_all_resid_train_membership",
    "study_cov_times_squared_error_cov_over_var_all_resid_train_membership",
    "study_cov_times_prediction_rho_all_oof_intersection",
    "study_cov_times_error_rho_all_oof_intersection",
    "study_cov_times_squared_error_rho_all_oof_intersection",
    "study_prediction_cov_over_var_all_oof_intersection",
    "study_error_cov_over_var_all_oof_intersection",
    "study_squared_error_cov_over_var_all_oof_intersection",
    "study_cov_times_prediction_cov_over_var_all_oof_intersection",
    "study_cov_times_error_cov_over_var_all_oof_intersection",
    "study_cov_times_squared_error_cov_over_var_all_oof_intersection",
]

PREDICTION_OUTPUT_COLUMNS = [
    "scope",
    "prefix_j",
    "pipeline",
    "algorithm",
    "model_family",
    "dataset",
    "train_size",
    "study_size",
    "size_for_plot",
    "fixed_split",
    "procedure",
    "seed",
    "sample_gain_K20",
    "predicted_sample_gain_K20",
    "candidate_mode",
    "signals",
    "formula",
]


@dataclass(frozen=True)
class Candidate:
    """A fitted or direct candidate relationship."""

    scope: str
    mode: str
    signals: tuple[str, ...]
    formula: str
    target_transformed: str


def model_family(algorithm: object) -> str:
    """Return a stable model-family label from a solver string."""
    text = str(algorithm)
    base = text.split("[", 1)[0]
    return base.replace("_noHPO", "")


def _to_scalar(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value[0] if value.size else np.nan
    if isinstance(value, list):
        return value[0] if value else np.nan
    return value


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    series = df[col]
    if not pd.api.types.is_numeric_dtype(series):
        series = series.apply(_to_scalar)
    return pd.to_numeric(series, errors="coerce").astype(float)


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    num_arr = pd.to_numeric(num, errors="coerce").to_numpy(dtype=float)
    den_arr = pd.to_numeric(den, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(num_arr), np.nan, dtype=float)
    mask = np.isfinite(num_arr) & np.isfinite(den_arr) & (den_arr != 0)
    out[mask] = num_arr[mask] / den_arr[mask]
    return pd.Series(out, index=num.index, dtype=float)


def _positive(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return pd.Series(np.isfinite(arr) & (arr > 0), index=values.index)


def _log10(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    mask = np.isfinite(arr) & (arr > 0)
    out[mask] = np.log10(arr[mask])
    return out


def _pearson(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    x_arr = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    y_arr = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    if len(x_arr) < 3 or np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    return _pearson(pd.Series(x).rank(method="average"), pd.Series(y).rank(method="average"))


def _fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    if len(y) < X.shape[1] + 2:
        return None
    X_design = np.column_stack([np.ones(len(X)), X])
    try:
        beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return beta


def _predict_ols(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X]) @ beta


def validate_scope_features(scope: str, feature_cols: Iterable[str]) -> None:
    """Raise if a feature list violates a practitioner-information scope."""
    if scope not in SCOPES:
        raise ValueError(f"Unknown relationship scope: {scope!r}")
    cols = [str(col) for col in feature_cols]
    target_leaks = sorted(set(cols) & OBSERVED_TARGET_COLUMNS)
    if target_leaks:
        raise ValueError(f"{scope} feature list uses observed target columns: {target_leaks}")

    fragments: tuple[str, ...]
    if scope == SCOPE_NONBENCH_LIVE:
        fragments = NONBENCH_FORBIDDEN_FRAGMENTS
    elif scope == SCOPE_STUDY:
        fragments = STUDY_FORBIDDEN_FRAGMENTS
    else:
        fragments = ()

    leaked = sorted(
        col for col in cols
        if any(fragment in col.lower() for fragment in fragments)
    )
    if leaked:
        raise ValueError(f"{scope} feature list uses forbidden columns: {leaked}")


def add_relationship_derived_columns(df: pd.DataFrame, K: int = K_DEFAULT) -> pd.DataFrame:
    """Add reusable benchmark/study/raw derived relationship features."""
    out = df.copy()
    if "algorithm" in out.columns:
        out["model_family"] = out["algorithm"].map(model_family)
    if "size_for_plot" not in out.columns:
        train = out["train_size"] if "train_size" in out.columns else pd.Series(np.nan, index=out.index)
        study = out["study_size"] if "study_size" in out.columns else pd.Series(np.nan, index=out.index)
        out["size_for_plot"] = train.combine_first(study)
    if "train_size" in out.columns:
        train_size = _numeric(out, "train_size")
        out["train_size_feature"] = np.where(train_size > 0, train_size, np.nan)
        out["inv_train_size_feature"] = np.where(train_size > 0, 1.0 / train_size, np.nan)

    tau_te = _numeric(out, "tau_te")
    sigma2_te_K = _numeric(out, "sigma2_te_K")
    sigma2_te_1 = _numeric(out, "sigma2_te_1")
    out["rho_delta"] = _safe_ratio(tau_te, sigma2_te_K)
    denom_delta = sigma2_te_K + (K - 1) * tau_te
    out["predicted_sample_gain_K20"] = np.where(
        denom_delta > 0,
        K * sigma2_te_1 / denom_delta,
        np.nan,
    )
    out["icc_gain_from_rho_delta"] = np.where(
        out["rho_delta"] > 0,
        K / (1.0 + (K - 1.0) * out["rho_delta"]),
        np.nan,
    )

    tau = _numeric(out, "tau")
    sigma2_HO_K = _numeric(out, "sigma2_HO_K")
    sigma2_HO_1 = _numeric(out, "sigma2_HO_1")
    sigma2_HO_trxi = _numeric(out, "sigma2_HO_trxi")
    W = _numeric(out, "W")
    B = _numeric(out, "B")
    out["rho_raw"] = _safe_ratio(tau, sigma2_HO_K)
    out["W_over_B"] = _safe_ratio(W, B)
    denom_raw = sigma2_HO_K + (K - 1) * tau
    out["predicted_sample_gain_K20_raw_HO1"] = np.where(
        denom_raw > 0,
        K * sigma2_HO_1 / denom_raw,
        np.nan,
    )
    out["predicted_sample_gain_K20_raw_trxi"] = np.where(
        denom_raw > 0,
        K * sigma2_HO_trxi / denom_raw,
        np.nan,
    )

    for prefix in ("bench", "study"):
        cov = _numeric(out, f"{prefix}_cov")
        err_cov = _numeric(out, f"{prefix}_error_cov")
        err_corr = _numeric(out, f"{prefix}_error_corr")
        err_var = _numeric(out, f"{prefix}_error_var")
        err_rho = _numeric(out, f"{prefix}_error_rho")
        sq_cov = _numeric(out, f"{prefix}_squared_error_cov")
        sq_corr = _numeric(out, f"{prefix}_squared_error_corr")
        sq_var = _numeric(out, f"{prefix}_squared_error_var")
        sq_rho = _numeric(out, f"{prefix}_squared_error_rho")

        out[f"{prefix}_cov_times_error_rho"] = cov * err_rho
        out[f"{prefix}_cov_times_error_corr"] = cov * err_corr
        out[f"{prefix}_cov_times_squared_error_rho"] = cov * sq_rho
        out[f"{prefix}_cov_times_squared_error_corr"] = cov * sq_corr
        out[f"{prefix}_cov_times_sqrt_error_squared_rho"] = cov * np.sqrt(
            np.where((err_rho > 0) & (sq_rho > 0), err_rho * sq_rho, np.nan)
        )
        out[f"{prefix}_error_cov_over_error_var"] = _safe_ratio(err_cov, err_var)
        out[f"{prefix}_squared_error_cov_over_squared_error_var"] = _safe_ratio(
            sq_cov, sq_var
        )
        out[f"{prefix}_cov_times_error_cov_over_error_var"] = (
            cov * out[f"{prefix}_error_cov_over_error_var"]
        )
        out[f"{prefix}_cov_times_sqerr_cov_over_sqerr_var"] = (
            cov * out[f"{prefix}_squared_error_cov_over_squared_error_var"]
        )

    bench_cov = _numeric(out, "bench_cov")
    for quantity in ("prediction", "error", "squared_error"):
        cov_col = f"bench_{quantity}_cov_all"
        var_col = f"bench_{quantity}_var_all"
        rho_col = f"bench_{quantity}_rho_all"
        if rho_col in out.columns:
            rho = _numeric(out, rho_col)
            out[f"bench_cov_times_{quantity}_rho_all"] = bench_cov * rho
        if cov_col in out.columns and var_col in out.columns:
            ratio = _safe_ratio(_numeric(out, cov_col), _numeric(out, var_col))
            out[f"bench_{quantity}_cov_over_var_all"] = ratio
            out[f"bench_cov_times_{quantity}_cov_over_var_all"] = bench_cov * ratio

    study_cov = _numeric(out, "study_cov")
    for suffix in ("full", "resid_train_membership", "oof_intersection"):
        for quantity in ("prediction", "error", "squared_error"):
            cov_col = f"study_{quantity}_cov_all_{suffix}"
            var_col = f"study_{quantity}_var_all_{suffix}"
            rho_col = f"study_{quantity}_rho_all_{suffix}"
            if rho_col in out.columns:
                rho = _numeric(out, rho_col)
                out[f"study_cov_times_{quantity}_rho_all_{suffix}"] = study_cov * rho
            if cov_col in out.columns and var_col in out.columns:
                ratio = _safe_ratio(_numeric(out, cov_col), _numeric(out, var_col))
                out[f"study_{quantity}_cov_over_var_all_{suffix}"] = ratio
                out[f"study_cov_times_{quantity}_cov_over_var_all_{suffix}"] = (
                    study_cov * ratio
                )

    if {"prefix_train_mean_abs", "prefix_score_mean_abs"}.issubset(out.columns):
        out["prefix_train_score_ratio"] = _safe_ratio(
            _numeric(out, "prefix_train_mean_abs"),
            _numeric(out, "prefix_score_mean_abs"),
        )
    if {"prefix_generalization_gap_abs_mean", "prefix_score_loss"}.issubset(out.columns):
        out["prefix_gap_over_score_loss"] = _safe_ratio(
            _numeric(out, "prefix_generalization_gap_abs_mean"),
            _numeric(out, "prefix_score_loss"),
        )
    if {"prefix_generalization_gap_abs_mean", "prefix_train_loss"}.issubset(out.columns):
        out["prefix_gap_over_train_loss"] = _safe_ratio(
            _numeric(out, "prefix_generalization_gap_abs_mean"),
            _numeric(out, "prefix_train_loss"),
        )
    if {"prefix_score_var", "prefix_train_var"}.issubset(out.columns):
        out["prefix_score_var_over_train_var"] = _safe_ratio(
            _numeric(out, "prefix_score_var"),
            _numeric(out, "prefix_train_var"),
        )

    return out


def scope_feature_columns(df: pd.DataFrame, scope: str) -> list[str]:
    """Return existing positive-candidate feature columns for *scope*."""
    if scope == SCOPE_NONBENCH_LIVE:
        candidates = LIVE_FEATURE_COLUMNS
    elif scope == SCOPE_BENCHMARK:
        candidates = BENCHMARK_BASE_FEATURE_COLUMNS
        candidates = candidates + sorted(
            col for col in df.columns
            if col.startswith("bench_")
            and col not in candidates
            and col not in OBSERVED_TARGET_COLUMNS
        )
    elif scope == SCOPE_STUDY:
        candidates = STUDY_BASE_FEATURE_COLUMNS
        candidates = candidates + sorted(
            col for col in df.columns
            if col.startswith("study_")
            and col not in candidates
            and col not in OBSERVED_TARGET_COLUMNS
        )
    else:
        raise ValueError(f"Unknown relationship scope: {scope!r}")

    existing = []
    for col in candidates:
        if col not in df.columns or col in existing:
            continue
        series = _numeric(df, col)
        if series.notna().any() and (series > 0).any():
            existing.append(col)
    validate_scope_features(scope, existing)
    return existing


def _prefix_rows_for_group(
    sub: pd.DataFrame,
    key_values: dict,
    K: int,
) -> list[dict]:
    scores = _numeric(sub, "objective_score_test").to_numpy(dtype=float)
    train_scores = _numeric(sub, "objective_score_train").to_numpy(dtype=float)
    overlap_cols = [
        "split_test_overlap_prev_frac",
        "split_train_overlap_prev_frac",
        "split_test_overlap_first_frac",
        "split_train_overlap_first_frac",
    ]
    overlap_values = {
        col: _numeric(sub, col).to_numpy(dtype=float)
        for col in overlap_cols
        if col in sub.columns
    }
    rows = []
    max_j = min(K, len(scores))
    for j in range(1, max_j + 1):
        prefix = scores[:j]
        train_prefix = train_scores[:j]
        finite = prefix[np.isfinite(prefix)]
        train_finite = train_prefix[np.isfinite(train_prefix)]
        if finite.size == 0:
            continue
        score_mean = float(np.mean(finite))
        score_var = float(np.var(finite, ddof=1)) if finite.size >= 2 else np.nan
        score_std = float(np.sqrt(score_var)) if np.isfinite(score_var) else np.nan
        score_range = float(np.nanmax(finite) - np.nanmin(finite)) if finite.size >= 2 else np.nan
        train_mean = float(np.mean(train_finite)) if train_finite.size else np.nan
        train_var = (
            float(np.var(train_finite, ddof=1)) if train_finite.size >= 2 else np.nan
        )
        x = np.arange(1, j + 1, dtype=float)
        slope = np.nan
        if finite.size >= 2 and np.isfinite(prefix).all():
            beta = _fit_ols(x.reshape(-1, 1), prefix)
            if beta is not None:
                slope = float(beta[1])

        row = dict(key_values)
        row.update({
            "prefix_j": j,
            "prefix_inv_j": 1.0 / j,
            "prefix_score_mean": score_mean,
            "prefix_score_mean_abs": abs(score_mean) + EPS,
            "prefix_score_loss": max(1.0 - score_mean, EPS),
            "prefix_score_var": score_var,
            "prefix_score_std": score_std,
            "prefix_score_sem": score_std / np.sqrt(j) if np.isfinite(score_std) else np.nan,
            "prefix_score_range": score_range,
            "prefix_score_last": float(prefix[-1]),
            "prefix_score_last_loss": max(1.0 - float(prefix[-1]), EPS),
            "prefix_score_delta_abs": abs(float(prefix[-1] - prefix[0])) + EPS,
            "prefix_score_slope": slope,
            "prefix_score_slope_abs": abs(slope) + EPS if np.isfinite(slope) else np.nan,
            "prefix_train_mean": train_mean,
            "prefix_train_mean_abs": abs(train_mean) + EPS if np.isfinite(train_mean) else np.nan,
            "prefix_train_loss": max(1.0 - train_mean, EPS) if np.isfinite(train_mean) else np.nan,
            "prefix_train_var": train_var,
            "prefix_generalization_gap_abs_mean": (
                abs(train_mean - score_mean) + EPS if np.isfinite(train_mean) else np.nan
            ),
            "prefix_score_cv2": (
                score_var / (abs(score_mean) + EPS) ** 2
                if np.isfinite(score_var) else np.nan
            ),
        })
        for col, values in overlap_values.items():
            prefix_values = values[:j]
            finite_values = prefix_values[np.isfinite(prefix_values)]
            mean = float(np.mean(finite_values)) if finite_values.size else np.nan
            last = (
                float(prefix_values[np.where(np.isfinite(prefix_values))[0][-1]])
                if np.isfinite(prefix_values).any()
                else np.nan
            )
            row[f"prefix_{col}_mean"] = max(mean, EPS) if np.isfinite(mean) else np.nan
            row[f"prefix_{col}_last"] = max(last, EPS) if np.isfinite(last) else np.nan
        rows.append(row)
    return rows


def build_single_seed_live_features(
    data_cv: pd.DataFrame,
    gain20_df: pd.DataFrame,
    *,
    K: int = K_DEFAULT,
) -> pd.DataFrame:
    """Build prefix-j features using only one seed's first j MCCV splits."""
    if data_cv.empty:
        return pd.DataFrame()

    work = data_cv.copy()
    rename = {
        "solver_name": "algorithm",
        "dataset_type": "dataset",
        "cv_procedure": "procedure",
        "N": "train_size",
    }
    work = work.rename(columns={old: new for old, new in rename.items() if old in work.columns})
    if "procedure" not in work.columns and "p_obj_procedure" in work.columns:
        work["procedure"] = work["p_obj_procedure"]
    if "train_size" not in work.columns and "p_obj_train_size" in work.columns:
        work["train_size"] = work["p_obj_train_size"]
    if "fixed_split" not in work.columns and "p_obj_fixed_split" in work.columns:
        work["fixed_split"] = work["p_obj_fixed_split"]
    if "seed" not in work.columns and "p_dataset_seed" in work.columns:
        work["seed"] = work["p_dataset_seed"]
    if "dataset" not in work.columns and "dataset_name" in work.columns:
        work["dataset"] = work["dataset_name"].astype(str).str.extract(r"^([^\[]+)")[0]

    required = [
        "seed", "algorithm", "dataset", "train_size", "fixed_split",
        "procedure", "idx_rep", "objective_score_test", "objective_score_train",
    ]
    missing = [col for col in required if col not in work.columns]
    if missing:
        raise KeyError(f"Cannot build live features; missing columns: {missing}")

    work = work[~work["procedure"].astype(str).str.contains("RepeatedKFold", na=False)].copy()
    group_cols = ["seed", "algorithm", "dataset", "train_size", "fixed_split", "procedure"]
    work = work.sort_values(group_cols + ["idx_rep"])

    rows: list[dict] = []
    for key, sub in work.groupby(group_cols, dropna=False):
        key_values = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        rows.extend(_prefix_rows_for_group(sub, key_values, K=K))

    features = pd.DataFrame(rows)
    if features.empty:
        return features

    gain = gain20_df.copy()
    gain = gain.rename(columns={old: new for old, new in rename.items() if old in gain.columns})
    if "cv_procedure" in gain.columns and "procedure" not in gain.columns:
        gain = gain.rename(columns={"cv_procedure": "procedure"})
    merge_keys = ["algorithm", "dataset", "train_size", "fixed_split", "procedure"]
    gain_cols = [col for col in merge_keys + ["sample_gain_K20"] if col in gain.columns]
    if set(merge_keys + ["sample_gain_K20"]).issubset(gain_cols):
        features = features.merge(
            gain[gain_cols].drop_duplicates(merge_keys),
            on=merge_keys,
            how="left",
        )

    features["pipeline"] = "simulated"
    features["model_family"] = features["algorithm"].map(model_family)
    features["study_size"] = np.nan
    features["size_for_plot"] = features["train_size"]
    return features


def _valid_prediction_rows(
    df: pd.DataFrame,
    feature_cols: tuple[str, ...],
    observed_col: str = "sample_gain_K20",
) -> pd.Series:
    valid = _positive(_numeric(df, observed_col))
    for col in feature_cols:
        valid &= _positive(_numeric(df, col))
    return valid


def _score_predictions(
    df: pd.DataFrame,
    pred: pd.Series,
    *,
    observed_col: str = "sample_gain_K20",
) -> dict[str, float]:
    observed = _numeric(df, observed_col)
    valid = pred.notna() & observed.notna() & (pred > 0) & (observed > 0)
    if valid.sum() < 3:
        return {
            "n": int(valid.sum()),
            "rmse_log_gain": np.nan,
            "mae_log_gain": np.nan,
            "bias_log_gain": np.nan,
            "pearson_log_gain": np.nan,
            "spearman_gain": np.nan,
            "abs_corr_log_gain": np.nan,
            "r2_log_gain": np.nan,
            "min_model_family_abs_pearson_log_gain": np.nan,
            "min_model_family_abs_spearman_gain": np.nan,
            "min_model_family_signed_pearson_log_gain": np.nan,
            "min_model_family_signed_spearman_gain": np.nan,
            "mlp_abs_pearson_log_gain": np.nan,
            "mlp_signed_pearson_log_gain": np.nan,
            "passes_acceptance_gate": False,
        }

    pred_v = pred.loc[valid].astype(float)
    obs_v = observed.loc[valid].astype(float)
    log_pred = np.log10(pred_v)
    log_obs = np.log10(obs_v)
    resid = log_pred - log_obs
    pearson = _pearson(log_pred, log_obs)
    spearman = _spearman(pred_v, obs_v)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((log_obs - log_obs.mean()) ** 2))

    family_rows = []
    if "model_family" in df.columns:
        fam_series = df.loc[valid, "model_family"].astype(str)
    elif "algorithm" in df.columns:
        fam_series = df.loc[valid, "algorithm"].map(model_family)
    else:
        fam_series = pd.Series("all", index=log_obs.index)
    for fam, idx in fam_series.groupby(fam_series).groups.items():
        if len(idx) < 3:
            continue
        fam_log_pred = log_pred.loc[idx]
        fam_log_obs = log_obs.loc[idx]
        fam_pred = pred_v.loc[idx]
        fam_obs = obs_v.loc[idx]
        family_rows.append({
            "family": fam,
            "pearson": _pearson(fam_log_pred, fam_log_obs),
            "spearman": _spearman(fam_pred, fam_obs),
        })

    family_pearson = [
        abs(row["pearson"]) for row in family_rows if np.isfinite(row["pearson"])
    ]
    family_spearman = [
        abs(row["spearman"]) for row in family_rows if np.isfinite(row["spearman"])
    ]
    overall_sign = np.sign(pearson) if np.isfinite(pearson) and pearson != 0 else 1.0
    family_signed_pearson = [
        overall_sign * row["pearson"]
        for row in family_rows if np.isfinite(row["pearson"])
    ]
    family_signed_spearman = [
        overall_sign * row["spearman"]
        for row in family_rows if np.isfinite(row["spearman"])
    ]
    mlp_abs = np.nan
    mlp_signed = np.nan
    for row in family_rows:
        if "MLP" in row["family"] or row["family"].lower() == "mlp":
            mlp_abs = abs(row["pearson"]) if np.isfinite(row["pearson"]) else np.nan
            mlp_signed = (
                overall_sign * row["pearson"]
                if np.isfinite(row["pearson"]) else np.nan
            )
            break

    min_family_pearson = min(family_pearson) if family_pearson else np.nan
    min_family_spearman = min(family_spearman) if family_spearman else np.nan
    min_family_signed_pearson = (
        min(family_signed_pearson) if family_signed_pearson else np.nan
    )
    min_family_signed_spearman = (
        min(family_signed_spearman) if family_signed_spearman else np.nan
    )
    abs_corr = abs(pearson) if np.isfinite(pearson) else np.nan
    passes = (
        np.isfinite(abs_corr)
        and abs_corr >= 0.85
        and np.isfinite(min_family_signed_pearson)
        and min_family_signed_pearson >= 0.65
        and (not np.isfinite(mlp_signed) or mlp_signed >= 0.65)
    )
    return {
        "n": int(valid.sum()),
        "rmse_log_gain": float(np.sqrt(np.mean(resid ** 2))),
        "mae_log_gain": float(np.mean(np.abs(resid))),
        "bias_log_gain": float(np.mean(resid)),
        "pearson_log_gain": float(pearson),
        "spearman_gain": float(spearman),
        "abs_corr_log_gain": float(abs_corr) if np.isfinite(abs_corr) else np.nan,
        "r2_log_gain": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "min_model_family_abs_pearson_log_gain": min_family_pearson,
        "min_model_family_abs_spearman_gain": min_family_spearman,
        "min_model_family_signed_pearson_log_gain": min_family_signed_pearson,
        "min_model_family_signed_spearman_gain": min_family_signed_spearman,
        "mlp_abs_pearson_log_gain": mlp_abs,
        "mlp_signed_pearson_log_gain": mlp_signed,
        "passes_acceptance_gate": bool(passes),
    }


def _candidate_prediction(
    df: pd.DataFrame,
    candidate: Candidate,
    *,
    K: int,
) -> pd.Series:
    valid = _valid_prediction_rows(df, candidate.signals)
    pred = pd.Series(np.nan, index=df.index, dtype=float)
    if valid.sum() < max(3, len(candidate.signals) + 2):
        return pred

    X = np.column_stack([_log10(df.loc[valid, col]) for col in candidate.signals])
    g = _numeric(df.loc[valid], "sample_gain_K20")
    if candidate.mode == "direct_signal":
        pred.loc[valid] = _numeric(df.loc[valid], candidate.signals[0])
        return pred

    if candidate.mode == "gain_power":
        y = np.log10(g.to_numpy(dtype=float))
        beta = _fit_ols(X, y)
        if beta is None:
            return pred
        X_all_valid = np.column_stack([
            _log10(df.loc[valid, col]) for col in candidate.signals
        ])
        pred.loc[valid] = np.power(10.0, _predict_ols(X_all_valid, beta))
        return pred

    if candidate.mode == "effective_icc":
        g_arr = g.to_numpy(dtype=float)
        rho = (K / g_arr - 1.0) / (K - 1.0)
        rho_valid = np.isfinite(rho) & (rho > 0)
        if rho_valid.sum() < max(3, len(candidate.signals) + 2):
            return pred
        X_rho = X[rho_valid]
        y = np.log10(rho[rho_valid])
        beta = _fit_ols(X_rho, y)
        if beta is None:
            return pred
        rho_hat = np.power(10.0, _predict_ols(X, beta))
        pred.loc[valid] = K / (1.0 + (K - 1.0) * rho_hat)
        return pred

    raise ValueError(f"Unknown candidate mode: {candidate.mode!r}")


def _candidate_formula(mode: str, signals: tuple[str, ...], K: int) -> tuple[str, str]:
    joined = ", ".join(signals)
    if mode == "direct_signal":
        return signals[0], "signal"
    if mode == "gain_power":
        return f"log10(g) ~ 1 + log10({joined})", "log10(g)"
    if mode == "effective_icc":
        return (
            f"log10((K/g - 1)/(K-1)) ~ 1 + log10({joined}); "
            f"g = K/(1+(K-1)*rho_hat), K={K}",
            "log10((K/g - 1)/(K-1))",
        )
    raise ValueError(f"Unknown candidate mode: {mode!r}")


def _candidate_feature_sets(features: list[str], max_combo_features: int) -> list[tuple[str, ...]]:
    feature_sets: list[tuple[str, ...]] = [(feature,) for feature in features]
    top = features[: min(len(features), 10)]
    for width in range(2, max_combo_features + 1):
        feature_sets.extend(combinations(top, width))
    return feature_sets


def _sort_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    out = metrics.copy()
    out["_mode_rank"] = out["mode"].map({
        "gain_power": 0,
        "effective_icc": 1,
        "direct_signal": 2,
    }).fillna(9)
    signed_col = "min_model_family_signed_pearson_log_gain"
    if signed_col not in out.columns:
        signed_col = "min_model_family_abs_pearson_log_gain"
    out["_min_family"] = out[signed_col].fillna(-np.inf)
    out["_abs_corr"] = out["abs_corr_log_gain"].fillna(-np.inf)
    out["_rmse"] = out["rmse_log_gain"].fillna(np.inf)
    out = out.sort_values(
        [
            "passes_acceptance_gate",
            "_mode_rank",
            "_min_family",
            "_abs_corr",
            "_rmse",
            "signals",
        ],
        ascending=[False, True, False, False, True, True],
    )
    return out.drop(columns=["_mode_rank", "_min_family", "_abs_corr", "_rmse"])


def search_scope_relationships(
    df: pd.DataFrame,
    scope: str,
    *,
    K: int = K_DEFAULT,
    prefix_j: int | None = None,
    max_candidates: int = 500,
    max_combo_features: int = 3,
) -> dict[str, pd.DataFrame | dict | None]:
    """Search low-complexity relationships for one feature scope."""
    work = add_relationship_derived_columns(df, K=K)
    if "procedure" in work.columns:
        work = work[
            ~work["procedure"].astype(str).str.contains("RepeatedKFold", na=False)
        ].copy()
    if prefix_j is not None:
        work = work[work["prefix_j"] == prefix_j].copy()

    if work.empty or "sample_gain_K20" not in work.columns:
        return {"metrics": pd.DataFrame(), "predictions": pd.DataFrame(), "selected": None}

    features = scope_feature_columns(work, scope)
    if not features:
        return {"metrics": pd.DataFrame(), "predictions": pd.DataFrame(), "selected": None}

    direct_features = []
    for feature in features:
        valid = _valid_prediction_rows(work, (feature,))
        if valid.sum() < 3:
            continue
        corr = abs(_pearson(_log10(work.loc[valid, feature]), _log10(work.loc[valid, "sample_gain_K20"])))
        direct_features.append((feature, corr if np.isfinite(corr) else -np.inf))
    ranked_features = [feature for feature, _ in sorted(direct_features, key=lambda item: item[1], reverse=True)]
    feature_sets = _candidate_feature_sets(ranked_features, max_combo_features=max_combo_features)

    candidate_specs: list[Candidate] = []
    for feature in ranked_features:
        formula, target = _candidate_formula("direct_signal", (feature,), K)
        candidate_specs.append(Candidate(scope, "direct_signal", (feature,), formula, target))
    for signals in feature_sets:
        for mode in ("gain_power", "effective_icc"):
            formula, target = _candidate_formula(mode, signals, K)
            candidate_specs.append(Candidate(scope, mode, signals, formula, target))

    rows = []
    predictions_by_key = {}
    for trial_index, candidate in enumerate(candidate_specs[:max_candidates], start=1):
        pred = _candidate_prediction(work, candidate, K=K)
        scores = _score_predictions(work, pred)
        signal_text = " + ".join(candidate.signals)
        output_col = (
            f"predicted_sample_gain_K20_{scope}_"
            f"{prefix_j:02d}_" if prefix_j is not None else
            f"predicted_sample_gain_K20_{scope}_"
        ) + f"{trial_index:03d}"
        rows.append({
            "scope": scope,
            "prefix_j": prefix_j,
            "trial_index": trial_index,
            "mode": candidate.mode,
            "signals": signal_text,
            "n_signals": len(candidate.signals),
            "out_col": output_col,
            "target_transformed": candidate.target_transformed,
            "formula": candidate.formula,
            "is_selected": False,
            **scores,
        })
        predictions_by_key[trial_index] = (candidate, pred, output_col)

    metrics = _sort_metrics(pd.DataFrame(rows))
    selected = None
    predictions = pd.DataFrame()
    if not metrics.empty:
        selected_row = metrics.iloc[0]
        metrics["is_selected"] = metrics["trial_index"] == selected_row["trial_index"]
        candidate, pred, _ = predictions_by_key[int(selected_row["trial_index"])]
        selected = selected_row.to_dict()
        predictions = _prediction_frame(
            work,
            pred,
            scope=scope,
            prefix_j=prefix_j,
            candidate=candidate,
        )
    return {"metrics": metrics, "predictions": predictions, "selected": selected}


def _prediction_frame(
    df: pd.DataFrame,
    pred: pd.Series,
    *,
    scope: str,
    prefix_j: int | None,
    candidate: Candidate,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in PREDICTION_OUTPUT_COLUMNS:
        if col in df.columns:
            out[col] = df[col]
    out["scope"] = scope
    out["prefix_j"] = prefix_j
    out["predicted_sample_gain_K20"] = pred
    out["candidate_mode"] = candidate.mode
    out["signals"] = " + ".join(candidate.signals)
    out["formula"] = candidate.formula
    if "pipeline" not in out.columns:
        out["pipeline"] = "simulated"
    if "study_size" not in out.columns:
        out["study_size"] = np.nan
    if "seed" not in out.columns:
        out["seed"] = np.nan
    if "model_family" not in out.columns and "algorithm" in out.columns:
        out["model_family"] = out["algorithm"].map(model_family)
    if "size_for_plot" not in out.columns:
        train = out["train_size"] if "train_size" in out.columns else pd.Series(np.nan, index=out.index)
        study = out["study_size"] if "study_size" in out.columns else pd.Series(np.nan, index=out.index)
        out["size_for_plot"] = train.combine_first(study)
    for col in PREDICTION_OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[PREDICTION_OUTPUT_COLUMNS]


def run_simulated_gain_relationships(
    *,
    config_df: pd.DataFrame,
    live_df: pd.DataFrame | None = None,
    K: int = K_DEFAULT,
    max_candidates: int = 500,
) -> dict[str, pd.DataFrame]:
    """Run the three simulated relationship searches and combine outputs."""
    metrics_frames = []
    prediction_frames = []

    config = add_relationship_derived_columns(config_df, K=K)
    for scope in (SCOPE_BENCHMARK, SCOPE_STUDY):
        result = search_scope_relationships(
            config,
            scope,
            K=K,
            max_candidates=max_candidates,
        )
        if not result["metrics"].empty:
            metrics_frames.append(result["metrics"])
        if not result["predictions"].empty:
            prediction_frames.append(result["predictions"])

    if live_df is not None and not live_df.empty:
        for prefix_j in sorted(pd.to_numeric(live_df["prefix_j"], errors="coerce").dropna().astype(int).unique()):
            result = search_scope_relationships(
                live_df,
                SCOPE_NONBENCH_LIVE,
                K=K,
                prefix_j=int(prefix_j),
                max_candidates=max_candidates,
            )
            if not result["metrics"].empty:
                metrics_frames.append(result["metrics"])
            if not result["predictions"].empty:
                prediction_frames.append(result["predictions"])

    metrics = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames else pd.DataFrame(columns=PREDICTION_OUTPUT_COLUMNS)
    )
    return {"metrics": metrics, "predictions": predictions}


def selected_formula_latex_table(metrics: pd.DataFrame) -> str:
    """Return a compact LaTeX table for selected formulas."""
    if metrics.empty:
        return ""
    selected = metrics[metrics["is_selected"]].copy()
    if selected.empty:
        return ""
    selected = selected.sort_values(["scope", "prefix_j"], na_position="last")
    lines = [
        r"\begin{tabular}{llllrrr}",
        r"\toprule",
        r"Scope & Prefix $j$ & Mode & Signals & $|r_{\log}|$ & min signed family $r_{\log}$ & RMSE \\",
        r"\midrule",
    ]
    for _, row in selected.iterrows():
        prefix = "" if pd.isna(row["prefix_j"]) else str(int(row["prefix_j"]))
        signals = str(row["signals"]).replace("_", r"\_")
        family_col = (
            "min_model_family_signed_pearson_log_gain"
            if "min_model_family_signed_pearson_log_gain" in row.index
            else "min_model_family_abs_pearson_log_gain"
        )
        lines.append(
            f"{row['scope']} & {prefix} & {row['mode']} & "
            f"{signals} & "
            f"{row['abs_corr_log_gain']:.3f} & "
            f"{row[family_col]:.3f} & "
            f"{row['rmse_log_gain']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def write_relationship_outputs(
    *,
    results_dir: str | Path,
    live_features: pd.DataFrame,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict[str, Path]:
    """Write the simulated relationship parquet/CSV/LaTeX deliverables."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "live_features": results_path / "simulated_single_seed_gain_features.parquet",
        "predictions": results_path / "simulated_gain_relationship_predictions.parquet",
        "metrics": results_path / "simulated_gain_relationship_metrics.csv",
        "latex": results_path / "simulated_gain_relationship_formulas.tex",
    }
    live_features.to_parquet(paths["live_features"], index=False)
    predictions.to_parquet(paths["predictions"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    paths["latex"].write_text(selected_formula_latex_table(metrics), encoding="utf-8")
    return paths
