#!/usr/bin/env python3
"""Draft 2D projection: sample gain G_20 vs study-only redundancy (analysis/out_omega/*.csv).

Rows: regression (gain on MSE, redundancy on squared error) and classification (gain on
accuracy, redundancy on 0-1 error). Columns: the paper's omega-hat at k = 3 (raw), and
t * rho_e at k = 3 (t = 0.2, rho_e = the loss-correlation factor of omega-hat) with the
reference curve K / (1 + (K - 1) x). Point = configuration mean over single runs; horizontal
whisker = 10-90% of single runs; vertical whisker = 5-95% bootstrap interval of G_20.
Color = train size (ordinal blue ramp, validated with the dataviz skill's validator).
Usage: python analysis/omega_gain_plot.py [--out analysis/out_omega/omega_vs_gain_K20.png]
"""
import argparse
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

T, K = 0.2, 20
BINS = [(0, 100, "50-100"), (101, 750, "300-750"), (751, 3000, "1000-3000"), (3001, 10**9, "5000-10000")]
COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#6b6b68", "#e6e5e1", "#fcfcfb"


def size_bin(n):
    for i, (lo, hi, _) in enumerate(BINS):
        if lo <= n <= hi:
            return i
    return len(BINS) - 1


def panel(ax, d, x, xlo, xhi, xlabel, curve):
    for i, (_, _, lab) in enumerate(BINS):
        g = d[d.bin == i]
        if g.empty:
            continue
        ax.errorbar(g[x], g.G_paper, xerr=[g[x] - g[xlo], g[xhi] - g[x]],
                    yerr=[g.G_paper - g.G_paper_lo, g.G_paper_hi - g.G_paper],
                    fmt="none", ecolor=COLORS[i], alpha=0.25, lw=0.8, zorder=1)
        ax.scatter(g[x], g.G_paper, s=18, color=COLORS[i], edgecolor=SURFACE, linewidth=0.6,
                   label=f"train size {lab}", zorder=2)
    if curve:
        xs = np.linspace(max(d[x].min(), -0.05), d[x].max() * 1.05, 300)
        xs = xs[1 / K + (1 - 1 / K) * xs > 0]
        ax.plot(xs, 1 / (1 / K + (1 - 1 / K) * xs), color=INK, lw=1.5, ls="--",
                label="K / (1 + (K-1) x)", zorder=3)
    ok = d[[x, "G_paper"]].dropna()
    r = spearmanr(ok[x], ok.G_paper).statistic
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel("sample gain $G^{test}_{20}$", color=INK)
    ax.text(0.98, 0.96, f"Spearman {r:+.2f}\n{len(ok)} configurations", transform=ax.transAxes,
            ha="right", va="top", color=MUTED, fontsize=8)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=3, help="number of splits the redundancy is read after")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    k = a.k
    out = a.out or f"analysis/out_omega/omega_vs_gain_K20_k{k}.png"
    d = pd.concat([pd.read_csv(f) for f in glob.glob("analysis/out_omega/omega_*.csv")], ignore_index=True)
    d["bin"] = d.p_obj_train_size.astype(int).map(size_bin)
    for q in ("mean", "q10", "q90"):
        d[f"x_{q}"] = T * d[f"rho_e_{q}_{k}" if q != "mean" else f"rho_e_mean_{k}"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), facecolor=SURFACE)
    rows = (("neg_mse", "Regression: gain on MSE, redundancy on squared error"),
            ("accuracy", "Classification: gain on accuracy, redundancy on 0-1 error"))
    for r, (gl, title) in enumerate(rows):
        x = d[d.gain_loss == gl]
        panel(axes[r, 0], x, f"omega_mean_{k}", f"omega_q10_{k}", f"omega_q90_{k}",
              f"paper redundancy score $\\hat\\omega^{{study}}_{k}$ (raw)", False)
        axes[r, 0].set_xscale("symlog", linthresh=0.1)
        panel(axes[r, 1], x, "x_mean", "x_q10", "x_q90",
              f"$t\\,\\hat\\rho_{{e,{k}}}$  (t = 0.2; loss-correlation factor of $\\hat\\omega$)", True)
        axes[r, 0].set_title(title, loc="left", color=INK, fontsize=10)
        for ax in axes[r]:
            ax.set_facecolor(SURFACE)
    axes[0, 1].legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left")
    fig.suptitle(f"Sample gain at K = 20 vs study-only redundancy at k = {k} (simulated data; point = "
                 "configuration mean, whiskers = single-run 10-90% / gain 90% CI)", color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
