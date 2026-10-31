#!/usr/bin/env python3
"""Paper-style figures for the study-only statistic x_k = t * rho_e,k.

x_k is the test fraction t times the loss-correlation factor rho_e of the paper's
redundancy score omega-hat (eq. study-loss-correlation), computed from the first k
splits of ONE CV run on the gain's own metric (squared error for the MSE gain,
0-1 error for the accuracy gain, NLL for the NLL gain), then averaged over the seeds
of a configuration.  The orchestrator's finding is that the sample gain follows the
ICC curve G_K ~ K / (1 + (K-1) x_k) on both benchmarks (simulated and PCam).

Inputs: the configuration-level tables written by benchmark_regression
analysis/omega_gain_relation.py (omega_<set>_<loss>.csv) and
analysis/pcam_consistency_check.py (pcam_<loss>_k<k>.csv), pulled from the NFS to
"Run outputs/Regression/orchestrator_analysis/analysis".  Their gain is the paper's
order-averaged G^test_20 on delta scores (fold score minus benchmark score).

Outputs (PDF + PNG), in the style of Images/rho_delta_vs_gain.pdf and
Images/study_redundancy_single_seed_early_stop_K20.pdf:
  study_only_statistic_vs_gain_k{k}_K20.pdf              scatter, three benchmarks, ICC curve
  study_only_statistic_vs_gain_k{k}_K20_by_benchmark.pdf  three panels (MSE / accuracy / NLL)
  study_only_statistic_vs_gain_k{k}_K20_curve.pdf         binned mean curve + 95% CI, no points
  study_only_statistic_vs_gain_k{k}_K20_by_benchmark_curve.pdf  same, three panels
  study_only_statistic_tertiles_k{k}_K20.pdf             P(large gain) by tertile of x_k
  study_only_statistic_tertiles_k{k}_K20_with_pcam.pdf   same, PCam configurations pooled in
  study_only_statistic_k{k}_K20_table.csv                 the pooled configuration table
  study_only_statistic_k{k}_K20_summary.txt               rank correlations and curve fit

Usage: python study_only_statistic_figures.py [--k 3] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot as P

ANALYSIS_DIR = os.path.expanduser(
    "~/Documents/Run outputs/Regression/orchestrator_analysis/analysis")
DEFAULT_OUT = os.path.expanduser(
    "~/Documents/Papiers/Miens/Crossing the Validation Crisis/NeurIPS 2026 - Rebuttal/"
    "proposed_figures_study_only_statistic")
T, K = 0.2, 20
THRESHOLDS = (7, 8, 10)
N_BOOT = 2000

# Benchmark groups: (label, marker, colour) -- same encoding family as rho_delta_vs_gain
GROUPS = {
    "sim_reg": ("Synthetic regression", "o", "#4C72B0"),
    "sim_clf": ("Synthetic classification", "^", "#55A868"),
    "pcam": ("PCam", "s", "#DD8452"),
}
# Which orchestrator tables feed which group (gain loss <-> redundancy loss are matched).
SIM_SETS = {
    "neg_mse": ["derivation", "holdout1", "holdout2", "holdout3", "coupling"],
    "accuracy": ["derivation_clf", "holdout2_clf", "holdout3_clf", "coupling_clf", "sweep_clf"],
    "neg_nll": ["derivation_clf", "holdout2_clf", "holdout3_clf", "coupling_clf", "sweep_clf"],
}


def load_sim(gain_loss: str, k: int) -> pd.DataFrame:
    frames = []
    for s in SIM_SETS[gain_loss]:
        for path in glob.glob(os.path.join(ANALYSIS_DIR, "out_omega*", f"omega_{s}_{gain_loss}.csv")):
            d = pd.read_csv(path)
            d["source_set"] = s
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df[(df.p_obj_procedure == "ShuffleSplit")
            & (df.p_obj_fixed_split.astype(str) != "True")
            & (df.p_obj_train_source.fillna("study") == "study")]
    out = pd.DataFrame({
        "benchmark": "sim_reg" if gain_loss == "neg_mse" else "sim_clf",
        "gain_loss": gain_loss,
        "source_set": df.source_set,
        "family": df.family,
        "algorithm": df.solver_name,
        "train_size": df.p_obj_train_size.astype(int),
        "noise": df.p_dataset_noise,
        "n_seeds": df.n_seeds,
        "rho_e_k": df[f"rho_e_mean_{k}"].astype(float),
        "G20": df.G_paper.astype(float),
        "G20_lo90": df.G_paper_lo.astype(float),
        "G20_hi90": df.G_paper_hi.astype(float),
    })
    out["x_k"] = T * out.rho_e_k
    return out.dropna(subset=["x_k", "G20"]).reset_index(drop=True)


def load_pcam(gain_loss: str, k: int) -> pd.DataFrame:
    name = {"accuracy": "accuracy", "neg_nll": "nll"}[gain_loss]
    d = pd.read_csv(os.path.join(ANALYSIS_DIR, "out_pcam_consistency", f"pcam_{name}_k{k}.csv"))
    out = pd.DataFrame({
        "benchmark": "pcam",
        "gain_loss": gain_loss,
        "source_set": d.set,
        "family": "PCam",
        "algorithm": d.model,
        "train_size": d.train_size.astype(int),
        "noise": np.nan,
        "n_seeds": d.n_seeds,
        "rho_e_k": d["mean"].astype(float),
        "G20": d.G_paper.astype(float),
        "G20_lo90": d.G_paper_lo.astype(float),
        "G20_hi90": d.G_paper_hi.astype(float),
    })
    out["x_k"] = T * out.rho_e_k
    return out.dropna(subset=["x_k", "G20"]).reset_index(drop=True)


Y_LOG = True  # set from --linear-y
BAND = "bootstrap"  # or "empirical" (--band empirical): 2.5-97.5 percentiles of the points in the bin


def _set_y_axis(ax) -> None:
    if Y_LOG:
        ax.set_yscale("log")
        ax.set_yticks([2, 5, 10, 20, 50])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    else:
        ax.set_ylim(bottom=0)


def _annotate(ax, rho: float, n: int) -> None:
    """Rank-correlation box: lower left on the log axis (empty corner), below the
    legend on the linear axis where the lower left is full of points."""
    txt = f"rank corr. = {rho:.2f}\n$n$ = {n}"
    if Y_LOG:
        ax.text(0.03, 0.04, txt, transform=ax.transAxes, ha="left", va="bottom", fontsize=8)
    else:
        ax.text(0.97, 0.60, txt, transform=ax.transAxes, ha="right", va="top", fontsize=8)


def icc_curve(x: np.ndarray) -> np.ndarray:
    return K / (1.0 + (K - 1.0) * x)


def _rank_corr(d: pd.DataFrame) -> tuple[float, int]:
    if len(d) < 3:
        return float("nan"), len(d)
    return float(P._spearmanr_with_fallback(d.x_k.to_numpy(float), d.G20.to_numpy(float))[0]), len(d)


def _curve_stats(d: pd.DataFrame) -> str:
    pos = d[d.x_k > 0]
    if pos.empty:
        return "n/a"
    lr = np.log(pos.G20 / icc_curve(pos.x_k.to_numpy()))
    within = float(np.mean(np.abs(lr) <= np.log(1.25)))
    return f"median log(G20/curve) = {np.median(lr):+.2f}, within +-25% of the curve: {within:.0%} (n={len(pos)})"


# --------------------------------------------------------------------------- figure 1

def scatter_main(data: pd.DataFrame, k: int, out_base: str, summary: list[str]) -> None:
    P.set_paper_style()
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    for key, (label, marker, color) in GROUPS.items():
        sub = data[data.benchmark == key]
        if sub.empty:
            continue
        ax.scatter(sub.x_k, sub.G20, s=14 if key != "pcam" else 22, marker=marker,
                   facecolor=color, edgecolor="black", linewidth=0.3, alpha=0.85,
                   label=label, zorder=3 if key == "pcam" else 2)
    x_ref = np.linspace(0.0, max(0.21, float(data.x_k.max())), 300)
    ax.plot(x_ref, icc_curve(x_ref), "k--", linewidth=1.2, alpha=0.8,
            label=r"$K/(1+(K-1)\,x)$", zorder=1)
    rho, n = _rank_corr(data)
    _annotate(ax, rho, n)
    _set_y_axis(ax)
    ax.set_xlim(min(-0.02, float(data.x_k.min()) - 0.005), x_ref.max() + 0.005)
    ax.set_xlabel(rf"Study-only statistic $t\,\widehat\rho_{{e,{k}}}$ after {k} splits")
    ax.set_ylabel(r"Test sample gain ($G^{\mathrm{test}}_{20}$)")
    ax.legend(loc="upper right", fontsize=7, handlelength=1.4, borderaxespad=0.2,
              labelspacing=0.3, handletextpad=0.5)
    for ext in ("pdf", "png"):
        P.save_figure(fig, f"{out_base}.{ext}", bbox_inches="tight")
    plt.close(fig)
    summary.append(f"[scatter, all benchmarks] Spearman(x_{k}, G20) = {rho:.3f}, n = {n}; {_curve_stats(data)}")
    for key, (label, _, _) in GROUPS.items():
        sub = data[data.benchmark == key]
        r, n_ = _rank_corr(sub)
        summary.append(f"  {label}: Spearman = {r:.3f}, n = {n_}; {_curve_stats(sub)}")


# --------------------------------------------------------------------------- figure 2

def scatter_by_benchmark(panels: dict[str, pd.DataFrame], k: int, out_base: str,
                         summary: list[str]) -> None:
    P.set_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), sharey=True)
    titles = {
        "neg_mse": "Synthetic regression\ngain on MSE, $\\widehat\\rho_e$ on squared error",
        "accuracy": "Classification\ngain on accuracy, $\\widehat\\rho_e$ on 0-1 error",
        "neg_nll": "Classification\ngain on NLL, $\\widehat\\rho_e$ on NLL",
    }
    x_max = max(0.21, max(float(d.x_k.max()) for d in panels.values()))
    x_ref = np.linspace(0.0, x_max, 300)
    for ax, (loss, d) in zip(axes, panels.items()):
        for key, (label, marker, color) in GROUPS.items():
            sub = d[d.benchmark == key]
            if sub.empty:
                continue
            ax.scatter(sub.x_k, sub.G20, s=14 if key != "pcam" else 24, marker=marker,
                       facecolor=color, edgecolor="black", linewidth=0.3, alpha=0.85,
                       label=label, zorder=3 if key == "pcam" else 2)
        ax.plot(x_ref, icc_curve(x_ref), "k--", linewidth=1.2, alpha=0.8,
                label=r"$K/(1+(K-1)\,x)$", zorder=1)
        rho, n = _rank_corr(d)
        _annotate(ax, rho, n)
        ax.set_title(titles[loss], fontsize=9)
        _set_y_axis(ax)
        ax.set_xlim(min(-0.02, float(d.x_k.min()) - 0.005), x_max + 0.005)
        ax.set_xlabel(rf"$t\,\widehat\rho_{{e,{k}}}$ after {k} splits")
        ax.legend(loc="upper right", fontsize=7, handlelength=1.4, borderaxespad=0.2,
                  labelspacing=0.3, handletextpad=0.5)
        summary.append(f"[panel {loss}] Spearman = {rho:.3f}, n = {n}; {_curve_stats(d)}")
        for key, (label, _, _) in GROUPS.items():
            sub = d[d.benchmark == key]
            if not sub.empty:
                r, n_ = _rank_corr(sub)
                lab = f"{label} ({loss})"
                summary.append(f"    {lab}: Spearman = {r:.3f}, n = {n_}; {_curve_stats(sub)}")
    axes[0].set_ylabel(r"Test sample gain ($G^{\mathrm{test}}_{20}$)")
    for ext in ("pdf", "png"):
        P.save_figure(fig, f"{out_base}.{ext}", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- figure 3

def _boot_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, values.size, size=(N_BOOT, values.size))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def tertile_bars(data: pd.DataFrame, k: int, out_base: str, summary: list[str],
                 tag: str) -> pd.DataFrame:
    d = data.dropna(subset=["x_k", "G20"]).copy()
    q1, q2 = d.x_k.quantile([1 / 3, 2 / 3]).to_numpy()
    d["tertile"] = pd.cut(d.x_k, [-np.inf, q1, q2, np.inf], labels=["Low", "Middle", "High"])
    rng = np.random.default_rng(0)
    rows = []
    for thr in THRESHOLDS:
        for tert in ["Low", "Middle", "High"]:
            vals = (d.loc[d.tertile == tert, "G20"] >= thr).to_numpy(dtype=float)
            lo, hi = _boot_ci(vals, rng)
            rows.append(dict(tertile=tert, threshold=thr, p=vals.mean() if vals.size else np.nan,
                             lo=lo, hi=hi, n=int(vals.size)))
    stats = pd.DataFrame(rows)

    P.set_paper_style()
    fig, ax = plt.subplots(figsize=(3.2, 3.0), constrained_layout=False)
    order = ["Low", "Middle", "High"]
    x = np.arange(3, dtype=float)
    width = 0.22
    offsets = (np.arange(len(THRESHOLDS)) - (len(THRESHOLDS) - 1) / 2) * width
    colors = ["#4C72B0", "#F28E2B", "#C44E52"]
    for i, thr in enumerate(THRESHOLDS):
        sub = stats[stats.threshold == thr].set_index("tertile").reindex(order)
        y = sub.p.to_numpy(float)
        yerr = np.vstack([np.clip(y - sub.lo.to_numpy(float), 0, None),
                          np.clip(sub.hi.to_numpy(float) - y, 0, None)])
        ax.bar(x + offsets[i], y, width=width, color=colors[i], edgecolor="white",
               linewidth=0.45, label=rf"$G^{{\mathrm{{test}}}}_{{20}} \geq {thr}$", zorder=3)
        ax.errorbar(x + offsets[i], y, yerr=yerr, fmt="none", ecolor="0.25",
                    elinewidth=0.75, capsize=1.8, capthick=0.75, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(["Low", "Middle", "High"])
    ax.set_ylabel("Probability", fontsize=10)
    ax.set_xlabel(rf"Tertile of $t\,\widehat\rho_{{e,{k}}}$ after {k} splits", labelpad=5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    y_top = max(0.5, float(np.nanmax(stats.hi)) + 0.06)
    ax.set_ylim(0, min(1.0, y_top))
    ax.grid(True, axis="y", linestyle=":", alpha=0.35, zorder=0)
    ax.legend(title="Large sample gain\nif pushing to 20 splits", loc="upper right", frameon=False,
              fontsize=7.0, title_fontsize=7.4, handlelength=1.2, handletextpad=0.45,
              labelspacing=0.25)
    fig.subplots_adjust(left=0.2, bottom=0.2)
    for ext in ("pdf", "png"):
        P.save_figure(fig, f"{out_base}.{ext}", bbox_inches="tight")
    plt.close(fig)

    summary.append(f"[tertiles {tag}] n = {len(d)}, cut points x_{k} = {q1:.4f} / {q2:.4f}; "
                   "95% bootstrap CIs over configurations")
    for thr in THRESHOLDS:
        s = stats[stats.threshold == thr].set_index("tertile").reindex(order)
        summary.append("    G20 >= %d: " % thr + ", ".join(
            f"{t} {s.loc[t, 'p']:.2f} [{s.loc[t, 'lo']:.2f}, {s.loc[t, 'hi']:.2f}] (n={int(s.loc[t, 'n'])})"
            for t in order))
    stats["cut_low_middle"], stats["cut_middle_high"] = q1, q2
    return stats


# --------------------------------------------------------------------------- binned curves

N_BINS = {"sim_reg": 10, "sim_clf": 8, "pcam": 4}


def binned_curve(d: pd.DataFrame, n_bins: int, rng: np.random.Generator) -> pd.DataFrame:
    """Mean G20 per quantile bin of x_k with a 95% bootstrap CI over configurations."""
    d = d.sort_values("x_k").reset_index(drop=True)
    if len(d) < 2 * n_bins:
        n_bins = max(2, len(d) // 4)
    edges = np.unique(np.quantile(d.x_k, np.linspace(0, 1, n_bins + 1)))
    d["bin"] = pd.cut(d.x_k, edges, include_lowest=True, labels=False)
    rows = []
    for b, g in d.groupby("bin"):
        vals = g.G20.to_numpy(float)
        if BAND == "empirical":
            # 95% of the configurations in the bin lie inside the band
            lo, hi = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
        else:
            lo, hi = _boot_ci(vals, rng)
        rows.append(dict(x=float(g.x_k.mean()), x_lo=float(g.x_k.min()), x_hi=float(g.x_k.max()),
                         mean=float(vals.mean()), lo=lo, hi=hi, n=int(vals.size)))
    return pd.DataFrame(rows)


def _draw_curves(ax, data: pd.DataFrame, rng: np.random.Generator, curves_out: list, tag: str) -> None:
    for key, (label, marker, color) in GROUPS.items():
        sub = data[data.benchmark == key]
        if sub.empty:
            continue
        c = binned_curve(sub, N_BINS[key], rng)
        ax.fill_between(c.x, c.lo, c.hi, color=color, alpha=0.22, linewidth=0, zorder=2)
        ax.plot(c.x, c["mean"], color=color, marker=marker, markersize=4 if key != "pcam" else 5,
                markeredgecolor="black", markeredgewidth=0.3, linewidth=1.4, label=label, zorder=3)
        c.insert(0, "benchmark", key)
        c.insert(0, "panel", tag)
        curves_out.append(c)
    x_ref = np.linspace(0.0, max(0.21, float(data.x_k.max())), 300)
    ax.plot(x_ref, icc_curve(x_ref), "k--", linewidth=1.2, alpha=0.8, label=r"$K/(1+(K-1)\,x)$", zorder=1)
    _set_y_axis(ax)
    ax.set_xlim(min(-0.02, float(data.x_k.min()) - 0.005), x_ref.max() + 0.005)


def curve_main(data: pd.DataFrame, k: int, out_base: str, curves_out: list) -> None:
    P.set_paper_style()
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    _draw_curves(ax, data, rng, curves_out, "all")
    rho, n = _rank_corr(data)
    _annotate(ax, rho, n)
    ax.set_xlabel(rf"Study-only statistic $t\,\widehat\rho_{{e,{k}}}$ after {k} splits")
    ax.set_ylabel(r"Test sample gain ($G^{\mathrm{test}}_{20}$)")
    ax.legend(loc="upper right", fontsize=7, handlelength=1.6, borderaxespad=0.2,
              labelspacing=0.3, handletextpad=0.5)
    for ext in ("pdf", "png"):
        P.save_figure(fig, f"{out_base}.{ext}", bbox_inches="tight")
    plt.close(fig)


def curve_by_benchmark(panels: dict[str, pd.DataFrame], k: int, out_base: str, curves_out: list) -> None:
    P.set_paper_style()
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), sharey=True)
    titles = {
        "neg_mse": "Synthetic regression\ngain on MSE, $\\widehat\\rho_e$ on squared error",
        "accuracy": "Classification\ngain on accuracy, $\\widehat\\rho_e$ on 0-1 error",
        "neg_nll": "Classification\ngain on NLL, $\\widehat\\rho_e$ on NLL",
    }
    for ax, (loss, d) in zip(axes, panels.items()):
        _draw_curves(ax, d, rng, curves_out, loss)
        rho, n = _rank_corr(d)
        _annotate(ax, rho, n)
        ax.set_title(titles[loss], fontsize=9)
        ax.set_xlabel(rf"$t\,\widehat\rho_{{e,{k}}}$ after {k} splits")
        ax.legend(loc="upper right", fontsize=7, handlelength=1.6, borderaxespad=0.2,
                  labelspacing=0.3, handletextpad=0.5)
    axes[0].set_ylabel(r"Test sample gain ($G^{\mathrm{test}}_{20}$)")
    for ext in ("pdf", "png"):
        P.save_figure(fig, f"{out_base}.{ext}", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--linear-y", action="store_true", help="linear gain axis (files get the suffix _liny)")
    ap.add_argument("--band", choices=["bootstrap", "empirical"], default="bootstrap",
                    help="curve band: 95%% bootstrap CI of the bin mean, or the empirical 2.5-97.5 "
                         "percentile range of the configurations in the bin (suffix _empband)")
    a = ap.parse_args()
    k = a.k
    global Y_LOG, BAND
    Y_LOG = not a.linear_y
    BAND = a.band
    sfx = ("_liny" if a.linear_y else "") + ("_empband" if a.band == "empirical" else "")
    os.makedirs(a.out_dir, exist_ok=True)
    summary: list[str] = [
        f"Study-only statistic x_{k} = t * rho_e,{k} (t = {T}), gain = paper G^test_20 on delta scores, K = {K}.",
        "Sources: benchmark_regression analysis/out_omega*/omega_<set>_<loss>.csv and "
        "out_pcam_consistency/pcam_<loss>_k<k>.csv (orchestrator, NFS snapshot).",
        "Per-configuration values are means over seeds; the gain CIs in those tables are 90% and are not drawn.",
    ]

    sim_reg = load_sim("neg_mse", k)
    sim_acc = load_sim("accuracy", k)
    sim_nll = load_sim("neg_nll", k)
    pcam_acc = load_pcam("accuracy", k)
    pcam_nll = load_pcam("neg_nll", k)

    main_df = pd.concat([sim_reg, sim_acc, pcam_acc], ignore_index=True)
    main_df.to_csv(os.path.join(a.out_dir, f"study_only_statistic_k{k}_K20_table.csv"), index=False)

    scatter_main(main_df, k, os.path.join(a.out_dir, f"study_only_statistic_vs_gain_k{k}_K20{sfx}"), summary)
    scatter_by_benchmark(
        {"neg_mse": sim_reg,
         "accuracy": pd.concat([sim_acc, pcam_acc], ignore_index=True),
         "neg_nll": pd.concat([sim_nll, pcam_nll], ignore_index=True)},
        k, os.path.join(a.out_dir, f"study_only_statistic_vs_gain_k{k}_K20_by_benchmark{sfx}"), summary)

    curves: list = []
    curve_main(main_df, k, os.path.join(a.out_dir, f"study_only_statistic_vs_gain_k{k}_K20_curve{sfx}"), curves)
    curve_by_benchmark(
        {"neg_mse": sim_reg,
         "accuracy": pd.concat([sim_acc, pcam_acc], ignore_index=True),
         "neg_nll": pd.concat([sim_nll, pcam_nll], ignore_index=True)},
        k, os.path.join(a.out_dir, f"study_only_statistic_vs_gain_k{k}_K20_by_benchmark_curve{sfx}"), curves)
    pd.concat(curves, ignore_index=True).to_csv(
        os.path.join(a.out_dir, f"study_only_statistic_k{k}_K20_curves{sfx}.csv"), index=False)
    summary.append("Binned curves: quantile bins of x_k per benchmark (10 regression / 8 classification / 4 PCam), "
                   "mean G20 per bin; band = " + ("empirical 2.5-97.5 percentiles of the configurations in the bin"
                                                  if BAND == "empirical" else
                                                  "95% bootstrap CI of the bin mean over its configurations") + ".")

    sim_only = pd.concat([sim_reg, sim_acc], ignore_index=True)
    st1 = tertile_bars(sim_only, k, os.path.join(a.out_dir, f"study_only_statistic_tertiles_k{k}_K20"),
                       summary, "simulated only")
    st2 = tertile_bars(main_df, k, os.path.join(a.out_dir, f"study_only_statistic_tertiles_k{k}_K20_with_pcam"),
                       summary, "simulated + PCam")
    st1.assign(pool="simulated").pipe(lambda s: pd.concat([s, st2.assign(pool="simulated+pcam")])) \
        .to_csv(os.path.join(a.out_dir, f"study_only_statistic_tertiles_k{k}_K20.csv"), index=False)

    with open(os.path.join(a.out_dir, f"study_only_statistic_k{k}_K20_summary.txt"), "w") as fh:
        fh.write("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
