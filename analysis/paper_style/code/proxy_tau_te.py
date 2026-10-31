"""Non-benchmark proxies for tau_te and rowwise sample-gain formulas.

The tau_te helpers fit exploratory log-space proxies. The sample-gain helpers
do not train models: they apply explicit formulas using only quantities from
the same setup being scored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_ALPHA_GRID = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
FORBIDDEN_PREDICTION_FEATURES = {
    "tau_te",
    "tau_te_lo",
    "tau_te_hi",
    "sigma2_te_1",
    "sigma2_te_K",
    "mean_bench_score",
    "benchmark_accuracy",
    "benchmark_eval_accuracy",
    "objective_score_bench",
}


@dataclass(frozen=True)
class TauTeProxySpec:
    """A log-space tau_te proxy specification."""

    name: str
    output_col: str
    feature_names: tuple[str, ...]
    prediction_input_cols: tuple[str, ...]
    baseline: bool = False


PROXY_SPECS = (
    TauTeProxySpec(
        name="baseline_trxi",
        output_col="tau_te_proxy_baseline_trxi",
        feature_names=("log10_sigma2_HO_trxi",),
        prediction_input_cols=("sigma2_HO_trxi",),
        baseline=True,
    ),
    TauTeProxySpec(
        name="baseline_tau_wb",
        output_col="tau_te_proxy_baseline_tau_wb",
        feature_names=("log10_tau", "log10_W_over_B"),
        prediction_input_cols=("tau", "W_over_B"),
        baseline=True,
    ),
    TauTeProxySpec(
        name="combined_raw",
        output_col="tau_te_proxy_combined_raw",
        feature_names=("log10_tau", "log10_sigma2_HO_trxi", "log10_W_over_B"),
        prediction_input_cols=("tau", "sigma2_HO_trxi", "W_over_B"),
    ),
    TauTeProxySpec(
        name="shrinkage",
        output_col="tau_te_proxy_shrinkage",
        feature_names=("log10_sigma2_HO_trxi_over_sigma2_HO_K", "log10_W_over_B"),
        prediction_input_cols=("tau", "sigma2_HO_trxi", "sigma2_HO_K", "W_over_B"),
    ),
)

LOCAL_SAMPLE_GAIN_PROXY_DESCRIPTIONS = {
    "local_raw_HO1_over_raw_cv": (
        "K*sigma2_HO_1 / (sigma2_HO_K + (K-1)*tau)"
    ),
    "local_trxi_over_raw_cv": (
        "K*sigma2_HO_trxi / (sigma2_HO_K + (K-1)*tau)"
    ),
    "local_cv_variance_curve": (
        "K*sigma2_HO_K / (sigma2_HO_K + (K-1)*tau)"
    ),
    "local_HO1_denominator": (
        "K*sigma2_HO_1 / (sigma2_HO_1 + (K-1)*tau)"
    ),
    "local_trxi_denominator": (
        "K*sigma2_HO_trxi / (sigma2_HO_trxi + (K-1)*tau)"
    ),
    "local_HO1_denominator_tau_pos": (
        "K*sigma2_HO_1 / (sigma2_HO_1 + (K-1)*max(tau,0))"
    ),
    "local_trxi_denominator_tau_pos": (
        "K*sigma2_HO_trxi / (sigma2_HO_trxi + (K-1)*max(tau,0))"
    ),
    "local_trxi_tau_shrink_trxi_over_HOK": (
        "K*sigma2_HO_trxi / (sigma2_HO_trxi + (K-1)*tau*sigma2_HO_trxi/sigma2_HO_K)"
    ),
    "local_trxi_tau_shrink_trxi_over_HO1": (
        "K*sigma2_HO_trxi / (sigma2_HO_trxi + (K-1)*tau*sigma2_HO_trxi/sigma2_HO_1)"
    ),
    "local_trxi_tau_shrink_inv_WB": (
        "K*sigma2_HO_trxi / (sigma2_HO_trxi + (K-1)*tau/(W/B))"
    ),
    "local_trxi_tau_shrink_one_plus_WB": (
        "K*sigma2_HO_trxi / (sigma2_HO_trxi + (K-1)*tau/(1+W/B))"
    ),
    "local_trxi_tau_minus_seed_noise": (
        "K*sigma2_HO_trxi / (sigma2_HO_trxi + (K-1)*max(tau-(sigma2_HO_1-sigma2_HO_trxi)_+,0))"
    ),
    "local_trxi_tau_minus_seed_noise_over_K": (
        "K*sigma2_HO_trxi / (sigma2_HO_trxi + (K-1)*max(tau-(sigma2_HO_1-sigma2_HO_trxi)_+/K,0))"
    ),
}


def get_tau_te_proxy_specs() -> tuple[TauTeProxySpec, ...]:
    """Return the registered proxy specifications."""
    return PROXY_SPECS


def validate_prediction_feature_hygiene(
    specs: Iterable[TauTeProxySpec] = PROXY_SPECS,
) -> None:
    """Raise if any proxy uses benchmark-adjusted values at prediction time."""
    for spec in specs:
        leaked = sorted(set(spec.prediction_input_cols) & FORBIDDEN_PREDICTION_FEATURES)
        leaked += sorted(
            col for col in spec.prediction_input_cols if col.startswith("sigma2_te")
        )
        if leaked:
            raise ValueError(
                f"Proxy spec {spec.name!r} uses benchmark-derived prediction "
                f"feature(s): {sorted(set(leaked))}"
            )


def _spec_by_name(spec_name: str) -> TauTeProxySpec:
    for spec in PROXY_SPECS:
        if spec.name == spec_name:
            return spec
    raise KeyError(f"Unknown tau_te proxy spec: {spec_name!r}")


def _positive_log10(values: pd.Series | np.ndarray) -> pd.Series:
    series = pd.Series(values)
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    arr = numeric.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    mask = np.isfinite(arr) & (arr > 0)
    out[mask] = np.log10(arr[mask])
    return pd.Series(out, index=series.index)


def _positive_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    num_arr = pd.to_numeric(num, errors="coerce").to_numpy(dtype=float)
    den_arr = pd.to_numeric(den, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(num_arr), np.nan, dtype=float)
    mask = (
        np.isfinite(num_arr)
        & np.isfinite(den_arr)
        & (num_arr > 0)
        & (den_arr > 0)
    )
    out[mask] = num_arr[mask] / den_arr[mask]
    return pd.Series(out, index=num.index)


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    num_arr = pd.to_numeric(num, errors="coerce").to_numpy(dtype=float)
    den_arr = pd.to_numeric(den, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(num_arr), np.nan, dtype=float)
    mask = np.isfinite(num_arr) & np.isfinite(den_arr) & (den_arr != 0)
    out[mask] = num_arr[mask] / den_arr[mask]
    return pd.Series(out, index=num.index)


def _column_or_nan(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def build_proxy_design(df: pd.DataFrame, spec_name: str) -> pd.DataFrame:
    """Build the log-space design frame for one proxy spec.

    The returned frame preserves the input index and contains feature columns,
    ``target_log10``, ``reconstruct_scale``, ``valid_prediction`` and
    ``valid_training``.
    """
    spec = _spec_by_name(spec_name)
    out = pd.DataFrame(index=df.index)

    tau = _column_or_nan(df, "tau")
    tau_te = _column_or_nan(df, "tau_te")
    sigma2_HO_trxi = _column_or_nan(df, "sigma2_HO_trxi")
    sigma2_HO_K = _column_or_nan(df, "sigma2_HO_K")
    W_over_B = _column_or_nan(df, "W_over_B")

    if spec.name == "baseline_trxi":
        out["log10_sigma2_HO_trxi"] = _positive_log10(sigma2_HO_trxi)
        out["target_log10"] = _positive_log10(tau_te)
        out["reconstruct_scale"] = 1.0
    elif spec.name == "baseline_tau_wb":
        out["log10_tau"] = _positive_log10(tau)
        out["log10_W_over_B"] = _positive_log10(W_over_B)
        out["target_log10"] = _positive_log10(tau_te)
        out["reconstruct_scale"] = 1.0
    elif spec.name == "combined_raw":
        out["log10_tau"] = _positive_log10(tau)
        out["log10_sigma2_HO_trxi"] = _positive_log10(sigma2_HO_trxi)
        out["log10_W_over_B"] = _positive_log10(W_over_B)
        out["target_log10"] = _positive_log10(tau_te)
        out["reconstruct_scale"] = 1.0
    elif spec.name == "shrinkage":
        out["log10_sigma2_HO_trxi_over_sigma2_HO_K"] = _positive_log10(
            _positive_ratio(sigma2_HO_trxi, sigma2_HO_K)
        )
        out["log10_W_over_B"] = _positive_log10(W_over_B)
        out["target_log10"] = _positive_log10(_positive_ratio(tau_te, tau))
        scale = tau.where(np.isfinite(tau) & (tau > 0), np.nan)
        out["reconstruct_scale"] = scale.astype(float)
    else:
        raise KeyError(f"Unknown tau_te proxy spec: {spec.name!r}")

    feature_ok = out[list(spec.feature_names)].notna().all(axis=1)
    scale_ok = pd.to_numeric(out["reconstruct_scale"], errors="coerce").notna()
    out["valid_prediction"] = feature_ok & scale_ok
    out["valid_training"] = out["valid_prediction"] & out["target_log10"].notna()
    return out


def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    X_design = np.column_stack([np.ones(len(X)), X])
    if alpha <= 0:
        beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
        return beta
    penalty = np.eye(X_design.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    lhs = X_design.T @ X_design + penalty
    rhs = X_design.T @ y
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        return beta


def _predict_log(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    X_design = np.column_stack([np.ones(len(X)), X])
    return X_design @ beta


def _minimum_train_rows(spec: TauTeProxySpec) -> int:
    return len(spec.feature_names) + 2


def _fit_spec_on_rows(
    design: pd.DataFrame,
    spec: TauTeProxySpec,
    alpha: float,
    train_mask: pd.Series,
) -> np.ndarray | None:
    train = design.loc[train_mask & design["valid_training"]]
    if len(train) < _minimum_train_rows(spec):
        return None
    X = train[list(spec.feature_names)].to_numpy(dtype=float)
    y = train["target_log10"].to_numpy(dtype=float)
    return _fit_ridge(X, y, alpha)


def _predict_spec(
    design: pd.DataFrame,
    spec: TauTeProxySpec,
    beta: np.ndarray | None,
) -> pd.Series:
    pred = pd.Series(np.nan, index=design.index, dtype=float)
    if beta is None:
        return pred
    pred_rows = design[design["valid_prediction"]]
    if pred_rows.empty:
        return pred
    X = pred_rows[list(spec.feature_names)].to_numpy(dtype=float)
    pred_log = _predict_log(X, beta)
    scale = pred_rows["reconstruct_scale"].to_numpy(dtype=float)
    pred.loc[pred_rows.index] = scale * np.power(10.0, pred_log)
    return pred


def _score_predictions(
    design: pd.DataFrame,
    spec: TauTeProxySpec,
    pred: pd.Series,
) -> dict[str, float]:
    valid = design["valid_training"] & pred.notna() & (pred > 0)
    if not valid.any():
        return {
            "n_scored": 0,
            "rmse_log": np.nan,
            "mae_log": np.nan,
            "bias_log": np.nan,
            "mae_raw": np.nan,
        }

    observed_target = design.loc[valid, "target_log10"].to_numpy(dtype=float)
    if spec.name == "shrinkage":
        scale = design.loc[valid, "reconstruct_scale"].to_numpy(dtype=float)
        pred_target = np.log10(pred.loc[valid].to_numpy(dtype=float) / scale)
    else:
        pred_target = np.log10(pred.loc[valid].to_numpy(dtype=float))

    residual = pred_target - observed_target
    observed_raw = (
        design.loc[valid, "reconstruct_scale"].to_numpy(dtype=float)
        * np.power(10.0, observed_target)
    )
    pred_raw = pred.loc[valid].to_numpy(dtype=float)
    return {
        "n_scored": int(valid.sum()),
        "rmse_log": float(np.sqrt(np.mean(residual ** 2))),
        "mae_log": float(np.mean(np.abs(residual))),
        "bias_log": float(np.mean(residual)),
        "mae_raw": float(np.mean(np.abs(pred_raw - observed_raw))),
    }


def _gain_formula(
    numerator: pd.Series,
    variance: pd.Series,
    tau_like: pd.Series,
    K: int,
) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    variance = pd.to_numeric(variance, errors="coerce")
    tau_like = pd.to_numeric(tau_like, errors="coerce")
    denom = variance + (K - 1) * tau_like
    return pd.Series(
        np.where(denom > 0, K * numerator / denom, np.nan),
        index=numerator.index,
        dtype=float,
    )


def _series_min(*values: pd.Series) -> pd.Series:
    return pd.concat(values, axis=1).min(axis=1)


def _series_max(*values: pd.Series) -> pd.Series:
    return pd.concat(values, axis=1).max(axis=1)


def _series_geo(a: pd.Series, b: pd.Series) -> pd.Series:
    product = pd.to_numeric(a, errors="coerce") * pd.to_numeric(b, errors="coerce")
    return pd.Series(np.where(product > 0, np.sqrt(product), np.nan), index=a.index)


def _series_harm(a: pd.Series, b: pd.Series) -> pd.Series:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    denom = a + b
    return pd.Series(
        np.where((a > 0) & (b > 0) & (denom > 0), 2.0 * a * b / denom, np.nan),
        index=a.index,
        dtype=float,
    )


def _score_weights(
    df: pd.DataFrame,
    valid: pd.Series,
    group_col: str = "pipeline",
) -> tuple[np.ndarray, int]:
    if group_col not in df.columns:
        weights = np.full(int(valid.sum()), 1.0 / int(valid.sum()), dtype=float)
        return weights, 0

    groups = df.loc[valid, group_col].astype(str).fillna("__missing__")
    counts = groups.value_counts()
    row_weights = groups.map(lambda value: 1.0 / counts.loc[value]).to_numpy(dtype=float)
    weight_sum = row_weights.sum()
    if weight_sum <= 0 or not np.isfinite(weight_sum):
        weights = np.full(int(valid.sum()), 1.0 / int(valid.sum()), dtype=float)
        return weights, 0
    return row_weights / weight_sum, int(len(counts))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    return float(sorted_values[np.searchsorted(cumulative, 0.5, side="left")])


def _rowwise_sample_gain_candidates(out: pd.DataFrame, K: int):
    tau = _column_or_nan(out, "tau")
    sigma2_HO_K = _column_or_nan(out, "sigma2_HO_K")
    sigma2_HO_1 = _column_or_nan(out, "sigma2_HO_1")
    sigma2_HO_trxi = _column_or_nan(out, "sigma2_HO_trxi")
    W_over_B = _column_or_nan(out, "W_over_B")

    tau_pos = tau.clip(lower=0)
    seed_noise = (sigma2_HO_1 - sigma2_HO_trxi).clip(lower=0)
    raw_gap = (sigma2_HO_K - sigma2_HO_trxi).clip(lower=0)
    trxi_over_HOK = _positive_ratio(sigma2_HO_trxi, sigma2_HO_K)
    trxi_over_HO1 = _positive_ratio(sigma2_HO_trxi, sigma2_HO_1)
    sqrt_trxi_over_HOK = pd.Series(
        np.sqrt(trxi_over_HOK),
        index=out.index,
        dtype=float,
    )
    tau_minus_seed_noise = (tau - seed_noise).clip(lower=0)
    tau_minus_seed_noise_over_K = (tau - seed_noise / K).clip(lower=0)
    tau_minus_raw_gap = (tau - raw_gap).clip(lower=0)
    tau_minus_raw_gap_over_K = (tau - raw_gap / K).clip(lower=0)
    tau_shrink_trxi_over_HOK = tau * trxi_over_HOK
    tau_shrink_trxi_over_HO1 = tau * trxi_over_HO1
    tau_shrink_sqrt_trxi_over_HOK = tau * sqrt_trxi_over_HOK
    tau_shrink_inv_WB = _safe_divide(tau, W_over_B)
    tau_shrink_inv_1p_WB = _safe_divide(tau, 1.0 + W_over_B)
    tau_shrink_WB_over_1p_WB = tau * _safe_divide(W_over_B, 1.0 + W_over_B)
    tau_shrink_inv_sqrt_WB = _safe_divide(tau, np.sqrt(W_over_B))
    sigma2_geo_HOK_trxi = _series_geo(sigma2_HO_K, sigma2_HO_trxi)
    sigma2_blend_25_HO1_75_trxi = 0.25 * sigma2_HO_1 + 0.75 * sigma2_HO_trxi

    def candidate(name: str, numerator, variance, tau_like, formula: str):
        return (
            name,
            _gain_formula(
                pd.Series(numerator, index=out.index),
                pd.Series(variance, index=out.index),
                pd.Series(tau_like, index=out.index),
                K,
            ),
            formula,
        )

    base = [
        candidate(
            "local_raw_HO1_over_raw_cv",
            sigma2_HO_1,
            sigma2_HO_K,
            tau,
            "K*sigma2_HO_1 / (sigma2_HO_K + (K-1)*tau)",
        ),
        candidate(
            "local_trxi_over_raw_cv",
            sigma2_HO_trxi,
            sigma2_HO_K,
            tau,
            "K*sigma2_HO_trxi / (sigma2_HO_K + (K-1)*tau)",
        ),
        candidate(
            "local_cv_variance_curve",
            sigma2_HO_K,
            sigma2_HO_K,
            tau,
            "K*sigma2_HO_K / (sigma2_HO_K + (K-1)*tau)",
        ),
        candidate(
            "local_HO1_denominator",
            sigma2_HO_1,
            sigma2_HO_1,
            tau,
            "K*sigma2_HO_1 / (sigma2_HO_1 + (K-1)*tau)",
        ),
        candidate(
            "local_trxi_denominator",
            sigma2_HO_trxi,
            sigma2_HO_trxi,
            tau,
            "K*sigma2_HO_trxi / (sigma2_HO_trxi + (K-1)*tau)",
        ),
        candidate(
            "local_HO1_denominator_tau_pos",
            sigma2_HO_1,
            sigma2_HO_1,
            tau_pos,
            "K*sigma2_HO_1 / (sigma2_HO_1 + (K-1)*max(tau,0))",
        ),
        candidate(
            "local_trxi_denominator_tau_pos",
            sigma2_HO_trxi,
            sigma2_HO_trxi,
            tau_pos,
            "K*sigma2_HO_trxi / (sigma2_HO_trxi + (K-1)*max(tau,0))",
        ),
        candidate(
            "local_trxi_tau_shrink_trxi_over_HOK",
            sigma2_HO_trxi,
            sigma2_HO_trxi,
            tau_shrink_trxi_over_HOK,
            "K*sigma2_HO_trxi / (sigma2_HO_trxi + (K-1)*tau*sigma2_HO_trxi/sigma2_HO_K)",
        ),
        candidate(
            "local_trxi_tau_shrink_trxi_over_HO1",
            sigma2_HO_trxi,
            sigma2_HO_trxi,
            tau_shrink_trxi_over_HO1,
            "K*sigma2_HO_trxi / (sigma2_HO_trxi + (K-1)*tau*sigma2_HO_trxi/sigma2_HO_1)",
        ),
        candidate(
            "local_trxi_tau_shrink_inv_WB",
            sigma2_HO_trxi,
            sigma2_HO_trxi,
            tau_shrink_inv_WB,
            "K*sigma2_HO_trxi / (sigma2_HO_trxi + (K-1)*tau/(W/B))",
        ),
        candidate(
            "local_trxi_tau_shrink_one_plus_WB",
            sigma2_HO_trxi,
            sigma2_HO_trxi,
            tau_shrink_inv_1p_WB,
            "K*sigma2_HO_trxi / (sigma2_HO_trxi + (K-1)*tau/(1+W/B))",
        ),
        candidate(
            "local_trxi_tau_minus_seed_noise",
            sigma2_HO_trxi,
            sigma2_HO_trxi,
            tau_minus_seed_noise,
            "K*sigma2_HO_trxi / (sigma2_HO_trxi + (K-1)*max(tau-(sigma2_HO_1-sigma2_HO_trxi)_+,0))",
        ),
        candidate(
            "local_trxi_tau_minus_seed_noise_over_K",
            sigma2_HO_trxi,
            sigma2_HO_trxi,
            tau_minus_seed_noise_over_K,
            "K*sigma2_HO_trxi / (sigma2_HO_trxi + (K-1)*max(tau-(sigma2_HO_1-sigma2_HO_trxi)_+/K,0))",
        ),
        candidate(
            "local_trxi_tau_025_trxi_over_HOK",
            sigma2_HO_trxi,
            sigma2_HO_trxi,
            0.25 * tau_shrink_trxi_over_HOK,
            "K*sigma2_HO_trxi / (sigma2_HO_trxi + (K-1)*0.25*tau*sigma2_HO_trxi/sigma2_HO_K)",
        ),
        candidate(
            "local_geo_HOK_trxi_blend25_HO1_75_trxi_tau_060_trxi_over_HOK",
            sigma2_geo_HOK_trxi,
            sigma2_blend_25_HO1_75_trxi,
            0.60 * tau_shrink_trxi_over_HOK,
            (
                "K*sqrt(sigma2_HO_K*sigma2_HO_trxi) / "
                "(0.25*sigma2_HO_1 + 0.75*sigma2_HO_trxi + "
                "(K-1)*0.60*tau*sigma2_HO_trxi/sigma2_HO_K)"
            ),
        ),
    ]
    for item in base:
        yield item

    numerators = [
        ("h1", sigma2_HO_1, "sigma2_HO_1"),
        ("trxi", sigma2_HO_trxi, "sigma2_HO_trxi"),
        ("HOK", sigma2_HO_K, "sigma2_HO_K"),
        ("min_h1_trxi", _series_min(sigma2_HO_1, sigma2_HO_trxi), "min(sigma2_HO_1,sigma2_HO_trxi)"),
        ("geo_h1_trxi", _series_geo(sigma2_HO_1, sigma2_HO_trxi), "sqrt(sigma2_HO_1*sigma2_HO_trxi)"),
        ("harm_h1_trxi", _series_harm(sigma2_HO_1, sigma2_HO_trxi), "harmonic_mean(sigma2_HO_1,sigma2_HO_trxi)"),
    ]
    variances = [
        ("HOK", sigma2_HO_K, "sigma2_HO_K"),
        ("h1", sigma2_HO_1, "sigma2_HO_1"),
        ("trxi", sigma2_HO_trxi, "sigma2_HO_trxi"),
        ("geo_HOK_trxi", _series_geo(sigma2_HO_K, sigma2_HO_trxi), "sqrt(sigma2_HO_K*sigma2_HO_trxi)"),
        ("harm_HOK_trxi", _series_harm(sigma2_HO_K, sigma2_HO_trxi), "harmonic_mean(sigma2_HO_K,sigma2_HO_trxi)"),
        ("max_HOK_trxi", _series_max(sigma2_HO_K, sigma2_HO_trxi), "max(sigma2_HO_K,sigma2_HO_trxi)"),
    ]
    tau_likes = [
        ("tau", tau, "tau"),
        ("tau_pos", tau_pos, "max(tau,0)"),
        ("tau_minus_seed", tau_minus_seed_noise, "max(tau-(sigma2_HO_1-sigma2_HO_trxi)_+,0)"),
        ("tau_minus_seed_over_K", tau_minus_seed_noise_over_K, "max(tau-(sigma2_HO_1-sigma2_HO_trxi)_+/K,0)"),
        ("tau_minus_raw_gap", tau_minus_raw_gap, "max(tau-(sigma2_HO_K-sigma2_HO_trxi)_+,0)"),
        ("tau_minus_raw_gap_over_K", tau_minus_raw_gap_over_K, "max(tau-(sigma2_HO_K-sigma2_HO_trxi)_+/K,0)"),
        ("tau_trxi_over_HOK", tau_shrink_trxi_over_HOK, "tau*sigma2_HO_trxi/sigma2_HO_K"),
        ("tau_trxi_over_HO1", tau_shrink_trxi_over_HO1, "tau*sigma2_HO_trxi/sigma2_HO_1"),
        ("tau_sqrt_trxi_over_HOK", tau_shrink_sqrt_trxi_over_HOK, "tau*sqrt(sigma2_HO_trxi/sigma2_HO_K)"),
        ("tau_over_WB", tau_shrink_inv_WB, "tau/(W/B)"),
        ("tau_over_1p_WB", tau_shrink_inv_1p_WB, "tau/(1+W/B)"),
        ("tau_WB_over_1p_WB", tau_shrink_WB_over_1p_WB, "tau*(W/B)/(1+W/B)"),
        ("tau_over_sqrt_WB", tau_shrink_inv_sqrt_WB, "tau/sqrt(W/B)"),
    ]
    for num_name, numerator, num_expr in numerators:
        for var_name, variance, var_expr in variances:
            for tau_name, tau_like, tau_expr in tau_likes:
                name = f"local_{num_name}_{var_name}_{tau_name}"
                formula = f"K*{num_expr} / ({var_expr} + (K-1)*{tau_expr})"
                yield candidate(name, numerator, variance, tau_like, formula)

    scale_factors = (0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0)
    scaled_tau_likes = [
        ("tau", tau, "tau"),
        ("tau_trxi_over_HOK", tau_shrink_trxi_over_HOK, "tau*sigma2_HO_trxi/sigma2_HO_K"),
        ("tau_sqrt_trxi_over_HOK", tau_shrink_sqrt_trxi_over_HOK, "tau*sqrt(sigma2_HO_trxi/sigma2_HO_K)"),
        ("tau_over_1p_WB", tau_shrink_inv_1p_WB, "tau/(1+W/B)"),
    ]
    for factor in scale_factors:
        factor_label = str(factor).replace(".", "p")
        for tau_name, tau_like, tau_expr in scaled_tau_likes:
            name = f"local_trxi_trxi_{tau_name}_x{factor_label}"
            formula = (
                "K*sigma2_HO_trxi / "
                f"(sigma2_HO_trxi + (K-1)*{factor:g}*{tau_expr})"
            )
            yield candidate(name, sigma2_HO_trxi, sigma2_HO_trxi, factor * tau_like, formula)


def _score_gain_predictions(
    df: pd.DataFrame,
    pred: pd.Series,
    observed_col: str = "sample_gain_K20",
    group_col: str = "pipeline",
) -> dict[str, float]:
    observed = _column_or_nan(df, observed_col)
    valid = pred.notna() & observed.notna() & (pred > 0) & (observed > 0)
    if not valid.any():
        return {
            "n_scored": 0,
            "n_score_groups": 0,
            "rmse_log_gain": np.nan,
            "mae_log_gain": np.nan,
            "bias_log_gain": np.nan,
            "mae_gain": np.nan,
            "bias_gain": np.nan,
            "median_pred_obs_ratio": np.nan,
            "corr_log_gain": np.nan,
        }
    pred_values = pred.loc[valid].to_numpy(dtype=float)
    observed_values = observed.loc[valid].to_numpy(dtype=float)
    log_pred = np.log10(pred_values)
    log_observed = np.log10(observed_values)
    residual = log_pred - log_observed
    weights, n_groups = _score_weights(df, valid, group_col=group_col)
    mean_log_pred = float(np.sum(weights * log_pred))
    mean_log_observed = float(np.sum(weights * log_observed))
    centered_pred = log_pred - mean_log_pred
    centered_observed = log_observed - mean_log_observed
    cov = float(np.sum(weights * centered_pred * centered_observed))
    var_pred = float(np.sum(weights * centered_pred ** 2))
    var_observed = float(np.sum(weights * centered_observed ** 2))
    corr = (
        cov / np.sqrt(var_pred * var_observed)
        if len(log_pred) > 1 and var_pred > 0 and var_observed > 0
        else np.nan
    )
    return {
        "n_scored": int(valid.sum()),
        "n_score_groups": n_groups,
        "rmse_log_gain": float(np.sqrt(np.sum(weights * residual ** 2))),
        "mae_log_gain": float(np.sum(weights * np.abs(residual))),
        "bias_log_gain": float(np.sum(weights * residual)),
        "mae_gain": float(np.sum(weights * np.abs(pred_values - observed_values))),
        "bias_gain": float(np.sum(weights * (pred_values - observed_values))),
        "median_pred_obs_ratio": _weighted_median(
            pred_values / observed_values,
            weights,
        ),
        "corr_log_gain": corr,
    }


def score_sample_gain_prediction(
    df: pd.DataFrame,
    pred: pd.Series,
    observed_col: str = "sample_gain_K20",
    group_col: str = "pipeline",
) -> dict[str, float]:
    """Quantify one sample-gain prediction column against observed gain."""
    return _score_gain_predictions(
        df,
        pred,
        observed_col=observed_col,
        group_col=group_col,
    )


def compute_local_sample_gain_proxies(
    df: pd.DataFrame,
    K: int = 20,
    max_candidates: int = 100,
    target_rmse_log_gain: float = 0.2,
) -> dict[str, object]:
    """Compute rowwise sample-gain formulas and all-row metrics.

    Each candidate uses only one row/configuration's own raw observables. There
    is no fitted coefficient and no cross-configuration training step. The
    search stops when a candidate reaches ``target_rmse_log_gain`` or after
    ``max_candidates`` formulas have been scored.
    """
    out = df.copy()

    rows = []
    prediction_cols = {}
    stop_reason = "candidate_library_exhausted"
    best_proxy = None
    best_rmse = np.inf
    for trial_index, (proxy, pred, formula) in enumerate(
        _rowwise_sample_gain_candidates(out, K),
        start=1,
    ):
        if trial_index > max_candidates:
            stop_reason = "max_candidates_reached"
            break
        output_col = f"predicted_sample_gain_K20_{proxy}"
        prediction_cols[output_col] = pred
        scores = _score_gain_predictions(out, pred)
        rmse = scores["rmse_log_gain"]
        if np.isfinite(rmse) and rmse < best_rmse:
            best_rmse = rmse
            best_proxy = proxy
        rows.append({
            "proxy": proxy,
            "family": "rowwise_formula",
            "spec": proxy,
            "output_col": output_col,
            "baseline": False,
            "trial_index": trial_index,
            "n_scored": scores["n_scored"],
            "n_score_groups": scores["n_score_groups"],
            "rmse_log_gain": scores["rmse_log_gain"],
            "mae_log_gain": scores["mae_log_gain"],
            "bias_log_gain": scores["bias_log_gain"],
            "mae_gain": scores["mae_gain"],
            "bias_gain": scores["bias_gain"],
            "median_pred_obs_ratio": scores["median_pred_obs_ratio"],
            "corr_log_gain": scores["corr_log_gain"],
            "is_selected": False,
            "metric_source": (
                "rowwise formula, no training; pipeline-weighted evaluation"
            ),
            "formula": formula,
        })
        if np.isfinite(best_rmse) and best_rmse <= target_rmse_log_gain:
            stop_reason = "target_rmse_reached"
            break

    if prediction_cols:
        out = pd.concat([out, pd.DataFrame(prediction_cols, index=out.index)], axis=1)

    metrics = pd.DataFrame(rows).sort_values(
        ["rmse_log_gain", "proxy"],
        na_position="last",
    )
    if not metrics.empty and best_proxy is not None:
        metrics["is_selected"] = metrics["proxy"] == best_proxy
    metrics["stop_reason"] = stop_reason
    metrics["n_candidates_tried"] = len(rows)
    metrics["target_rmse_log_gain"] = target_rmse_log_gain
    return {
        "data": out,
        "metrics": metrics,
        "selected_proxy": best_proxy,
        "selected_output_col": (
            f"predicted_sample_gain_K20_{best_proxy}" if best_proxy else None
        ),
        "selected_rmse_log_gain": best_rmse if np.isfinite(best_rmse) else np.nan,
        "stop_reason": stop_reason,
        "n_candidates_tried": len(rows),
    }


def _fit_tau_te_metric(
    spec: TauTeProxySpec,
    design: pd.DataFrame,
    alpha: float,
) -> tuple[dict[str, object], np.ndarray | None]:
    beta = _fit_spec_on_rows(
        design,
        spec,
        alpha,
        pd.Series(True, index=design.index),
    )
    pred = _predict_spec(design, spec, beta)
    scores = _score_predictions(design, spec, pred)
    metric = {
        "spec": spec.name,
        "output_col": spec.output_col,
        "baseline": bool(spec.baseline),
        "alpha": alpha,
        "n_fit": int(design["valid_training"].sum()),
        "n_scored": scores["n_scored"],
        "rmse_log": scores["rmse_log"],
        "mae_log": scores["mae_log"],
        "bias_log": scores["bias_log"],
        "mae_raw": scores["mae_raw"],
    }
    return metric, beta


def _finite_metric(metrics: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    if metrics.empty or metric_col not in metrics.columns:
        return pd.DataFrame(columns=metrics.columns)
    return metrics[np.isfinite(metrics[metric_col].astype(float))].copy()


def _select_spec(metrics: pd.DataFrame) -> tuple[str | None, str]:
    usable = _finite_metric(metrics, "rmse_log")
    metric_col = "rmse_log"
    if usable.empty:
        return None, "no_valid_proxy"

    baselines = usable[usable["baseline"]]
    best_overall = usable.sort_values([metric_col, "alpha", "spec"]).iloc[0]
    if baselines.empty:
        return str(best_overall["spec"]), "best_available_no_baseline"

    best_baseline = baselines.sort_values([metric_col, "alpha", "spec"]).iloc[0]
    if (
        not bool(best_overall["baseline"])
        and float(best_overall[metric_col]) < float(best_baseline[metric_col])
    ):
        return str(best_overall["spec"]), "improved_candidate_beats_baselines"
    return str(best_baseline["spec"]), "baseline_retained"


def _coefficient_table(
    betas: dict[str, np.ndarray | None],
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    metric_lookup = metrics.set_index("spec") if not metrics.empty else pd.DataFrame()
    for spec in PROXY_SPECS:
        beta = betas.get(spec.name)
        alpha = (
            float(metric_lookup.loc[spec.name, "alpha"])
            if spec.name in metric_lookup.index
            else np.nan
        )
        terms = ("intercept",) + spec.feature_names
        for i, term in enumerate(terms):
            rows.append({
                "spec": spec.name,
                "output_col": spec.output_col,
                "alpha": alpha,
                "term": term,
                "coefficient": float(beta[i]) if beta is not None else np.nan,
            })
    return pd.DataFrame(rows)


def format_proxy_formula(coefficients: pd.DataFrame, spec_name: str) -> str:
    """Format fitted coefficients for logs printed by analysis scripts."""
    sub = coefficients[coefficients["spec"] == spec_name]
    if sub.empty or sub["coefficient"].isna().all():
        return "unavailable"
    terms = []
    for _, row in sub.iterrows():
        coef = row["coefficient"]
        if pd.isna(coef):
            return "unavailable"
        if row["term"] == "intercept":
            terms.append(f"{coef:+.3f}")
        else:
            terms.append(f"{coef:+.3f}*{row['term']}")
    return " ".join(terms)


def fit_tau_te_proxy_specs(
    df: pd.DataFrame,
    K: int = 20,
    alpha_grid: Iterable[float] = DEFAULT_ALPHA_GRID,
) -> dict[str, object]:
    """Fit all tau_te proxy specs for diagnostic plots.

    Returns a dictionary containing:
    ``data`` (input plus proxy columns), ``metrics``, ``coefficients``, and
    selected-spec metadata. No cross-pipeline validation or refitting is done.
    """
    validate_prediction_feature_hygiene()
    alpha_grid = tuple(float(a) for a in alpha_grid)
    if not alpha_grid:
        raise ValueError("alpha_grid must contain at least one value.")
    alpha = float(alpha_grid[0])

    out = df.copy()
    designs = {spec.name: build_proxy_design(out, spec.name) for spec in PROXY_SPECS}

    metric_rows = []
    betas: dict[str, np.ndarray | None] = {}
    for spec in PROXY_SPECS:
        metric, beta = _fit_tau_te_metric(spec, designs[spec.name], alpha)
        metric_rows.append(metric)
        betas[spec.name] = beta
        out[spec.output_col] = _predict_spec(designs[spec.name], spec, beta)

    metrics = pd.DataFrame(metric_rows)
    selected_spec, selection_reason = _select_spec(metrics)
    metrics["is_selected"] = metrics["spec"] == selected_spec
    selected_output_col = None
    selected_alpha = np.nan
    if selected_spec is not None:
        selected = metrics[metrics["spec"] == selected_spec].iloc[0]
        selected_output_col = str(selected["output_col"])
        selected_alpha = float(selected["alpha"])
        denom = out["sigma2_HO_trxi"] + (K - 1) * out[selected_output_col]
        out["predicted_sample_gain_K20_proxy"] = np.where(
            denom > 0,
            K * out["sigma2_HO_trxi"] / denom,
            np.nan,
        )
    else:
        out["predicted_sample_gain_K20_proxy"] = np.nan

    coefficients = _coefficient_table(betas, metrics)
    return {
        "data": out,
        "metrics": metrics,
        "coefficients": coefficients,
        "selected_spec": selected_spec,
        "selected_output_col": selected_output_col,
        "selected_alpha": selected_alpha,
        "selection_reason": selection_reason,
    }
