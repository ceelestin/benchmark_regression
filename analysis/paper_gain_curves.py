#!/usr/bin/env python3
"""Paper figures: test sample gain G_K against the number of splits K on the simulated linear process, on the
mean squared error (redraws of the former R^2 figures aggregated_test_gain_k_*_sim_linear_*_r2_*.pdf).

Estimator (paper, benchmark-aware plug-in): G_K = Var_s(delta_1) / Var_s(mean_{k<=K} delta_k), delta_k = fold
MSE minus the benchmark MSE of the same predictor, variances across seeds; the single-holdout variance is that of
the first split of the same CV runs (so G_1 = 1). Bands: 95 % percentile intervals, 100 bootstrap resamples of seeds.

Usage (Jean-Zay): python analysis/paper_gain_curves.py --files A.parquet [B.parquet ...] --n-tr 1000 --kmax 20 \
    [--overlay-fixed] --out Images/aggregated_test_gain_k_1000_sim_linear_mse_K20.pdf
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

COLORS = {"Ridge": "#d62728", "MLP": "#2ca02c", "ExtraTrees": "#1f77b4", "GradientBoosting": "#ff7f0e",
          "MLP_GridSearch": "#8c564b", "MLP_RandomSearch": "#e377c2", "ExtraTrees_GridSearch": "#1f3fb4",
          "ExtraTrees_RandomSearch": "#ff9f3f", "Random features": "#9467bd"}
DISPLAY = {"KernelApprox": "Random features"}
COLS = ["solver_name", "p_dataset_seed", "idx_rep", "p_obj_train_size", "p_obj_fixed_split",
        "objective_neg_mse_test", "objective_neg_mse_bench", "objective_score_test", "objective_score_bench"]
METRIC = {"mse": ("objective_neg_mse_test", "objective_neg_mse_bench"), "r2": ("objective_score_test", "objective_score_bench")}


def short(name):
    s = name.split("[")[0].replace("_noHPO", "")
    return DISPLAY.get(s, s)


def load(files, n_tr, metric="mse"):
    d = pd.concat([pq.read_table(f, columns=[c for c in COLS if c in pq.read_schema(f).names]).to_pandas() for f in files], ignore_index=True)
    d = d[d.p_obj_train_size == n_tr]
    q, b = METRIC[metric]
    d["delta"] = d[q].astype(float) - d[b].astype(float)
    d["algo"] = d.solver_name.map(short)
    d["fixed"] = d.p_obj_fixed_split.astype(str) == "True"
    return d


def curves(g, kmax, rng, n_boot=100):
    E = g.pivot_table(index="p_dataset_seed", columns="idx_rep", values="delta").dropna(axis=0).values[:, :kmax]
    cm = np.cumsum(E, axis=1) / np.arange(1, E.shape[1] + 1)
    gain = np.var(E[:, 0], ddof=1) / np.var(cm, axis=0, ddof=1)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(E), len(E))
        bs.append(np.var(E[i, 0], ddof=1) / np.var(cm[i], axis=0, ddof=1))
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return np.arange(1, E.shape[1] + 1), gain, lo, hi, len(E)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--n-tr", type=int, required=True)
    ap.add_argument("--kmax", type=int, default=20)
    ap.add_argument("--overlay-fixed", action="store_true")
    ap.add_argument("--algos", nargs="*", default=None)
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="drop solvers whose full name contains any of these strings, e.g. 'hidden_layer_sizes=[32]'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default="mse", choices=list(METRIC))
    a = ap.parse_args()
    d = load(a.files, a.n_tr, a.metric)
    if a.algos:
        d = d[d.algo.isin(a.algos)]
    for pat in a.exclude:
        d = d[~d.solver_name.str.contains(pat, regex=False)]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(3.0, 2.9) if a.kmax <= 20 else (3.6, 2.9))
    groups = [(False, "-")] + ([(True, ":")] if a.overlay_fixed else [])
    finals = {}
    for fixed, ls in groups:
        sub = d[d.fixed == fixed] if a.overlay_fixed else d[~d.fixed]
        for algo, g in sub.groupby("algo"):
            k, gain, lo, hi, n = curves(g, a.kmax, rng)
            c = COLORS.get(algo, "black")
            ax.plot(k, gain, color=c, lw=2 if not fixed else 1.8, ls=ls)
            ax.fill_between(k, lo, hi, color=c, alpha=0.2, lw=0)
            if not fixed:
                finals[algo] = (gain[-1], n)
            print(f"{'fixed' if fixed else 'resampled'} {algo}: seeds {n}, G_{k[-1]} = {gain[-1]:.2f} [{lo[-1]:.2f}, {hi[-1]:.2f}]")
    # direct labels at the right end, above their curve, pushed apart to a minimum vertical gap
    ys = sorted(finals.items(), key=lambda kv: kv[1][0])
    lo_y, hi_y = ax.get_ylim()
    gap = 0.065 * (hi_y - lo_y)
    placed = []
    for algo, (y, _) in ys:
        y_lab = y + 0.015 * (hi_y - lo_y)
        if placed and y_lab < placed[-1] + gap:
            y_lab = placed[-1] + gap
        placed.append(y_lab)
        ax.annotate(algo, (a.kmax, y_lab), xytext=(-3, 0), textcoords="offset points", ha="right", va="bottom",
                    fontsize=7.5, fontweight="bold", color=COLORS.get(algo, "black"))
    if a.overlay_fixed:
        ax.plot([], [], color="0.3", ls="-", lw=2, label="Resampled split")
        ax.plot([], [], color="0.3", ls=":", lw=1.8, label="Fixed split")
        ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0.0, 0.9))
    ax.text(0.03, 0.97, f"$y = X\\beta$, $n_{{\\mathrm{{tr}}}} = {a.n_tr:,}$ (" + a.metric.upper().replace("R2", "$R^2$") + ")".replace(",", "{,}"), transform=ax.transAxes,
            ha="left", va="top", fontsize=7.5, fontweight="bold")
    ax.set_xlabel("Number of Splits (K)", fontsize=9)
    ax.set_ylabel(r"Test Sample Gain ($G^{\mathrm{test}}_K$)", fontsize=9)
    ticks = [1, 5, 10, 15, 20] if a.kmax <= 20 else [1, 50, 100, 150, 200]
    ax.set_xticks(ticks); ax.set_xlim(0, a.kmax + 0.5 if a.kmax <= 20 else a.kmax + 5)
    ax.grid(color="0.85", lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.3)
    fig.savefig(a.out); fig.savefig(a.out.replace(".pdf", ".png"), dpi=200)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
