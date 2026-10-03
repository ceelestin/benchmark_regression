#!/usr/bin/env python3
"""Paper figure: configuration-level gain G_20 (MSE) against the redundancy omega_k = t * rho_e,k on the
prospective holdout (job 101294, analysis/out_omega/omega_holdout1_neg_mse.csv), with the reference curve
K / (1 + (K-1) omega). Point = mean over the 50 single runs of a configuration; horizontal whisker = 10-90 %
of single runs; vertical whisker = 90 % bootstrap interval of the gain. Colour = training size.
Usage: python analysis/paper_redundancy_config_figure.py [--k 3] [--out ...pdf]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

K, T = 20, 0.2
SIZES, COLORS = (75, 750, 5000), ("#86b6ef", "#2a78d6", "#104281")   # ordinal blue, validated
INK, MUTED, GRID = "#0b0b0b", "#6b6b68", "#e6e5e1"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=3); ap.add_argument("--out", default=None)
    a = ap.parse_args(); k = a.k
    out = a.out or f"analysis/out_redundancy_triage_holdout1/paper/redundancy_vs_gain_config_k{k}.pdf"
    d = pd.read_csv("analysis/out_omega/omega_holdout1_neg_mse.csv")
    x, lo, hi = T * d[f"rho_e_mean_{k}"], T * d[f"rho_e_q10_{k}"], T * d[f"rho_e_q90_{k}"]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for n, c in zip(SIZES, COLORS):
        m = d.p_obj_train_size.astype(int) == n
        ax.errorbar(x[m], d.G_paper[m], xerr=[(x - lo)[m], (hi - x)[m]],
                    yerr=[(d.G_paper - d.G_paper_lo)[m], (d.G_paper_hi - d.G_paper)[m]],
                    fmt="none", ecolor=c, alpha=0.3, lw=0.7, zorder=1)
        ax.scatter(x[m], d.G_paper[m], s=14, color=c, edgecolor="white", linewidth=0.5, zorder=2,
                   label=f"$n_{{\\mathrm{{tr}}}}={n:,}$".replace(",", "{,}"))
    xs = np.linspace(-0.01, 0.205, 200)
    ax.plot(xs, 1 / (1 / K + (1 - 1 / K) * xs), color=INK, lw=1.3, ls="--", zorder=3,
            label=r"$K/(1+(K-1)\,\omega)$")
    r = spearmanr(x, d.G_paper).statistic
    ax.text(0.97, 0.96, f"Spearman {r:+.2f}\n{len(d)} configurations", transform=ax.transAxes, ha="right",
            va="top", fontsize=7, color=MUTED)
    ax.set_yscale("log")
    ax.set_xlabel(f"Redundancy $\\widehat{{\\omega}}^{{\\mathrm{{study}}}}_{{{k}}}$ (mean over runs)", color=INK, fontsize=9)
    ax.set_ylabel(r"$G^{\mathrm{test}}_{20}$ (MSE)", color=INK, fontsize=9)
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.legend(frameon=False, fontsize=7, loc="lower left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    print("wrote", out, "| Spearman", round(float(r), 3))


if __name__ == "__main__":
    main()
