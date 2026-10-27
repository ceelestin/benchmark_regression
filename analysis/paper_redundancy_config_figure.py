#!/usr/bin/env python3
"""Paper figure (Appendix D.2): configuration-level gain G_20 (MSE) against the redundancy omega_k = t * rho_e,k on
the pre-registered holdout (job 101294, analysis/out_omega/omega_holdout1_neg_mse.csv), with the reference curve
K / (1 + (K-1) omega). Point = mean over the 50 single runs of a configuration, colour = training size; black
markers = median and interquartile range of the configurations in each redundancy bin, placed at the bin's median redundancy; the dotted lines mark the
asymmetric reading of Table D.1: above omega_k = 0.10 no configuration reaches G_20 >= 8.
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
BINS = [-0.05, 0.05, 0.10, 0.15, 0.21]
W_CUT, G_CUT = 0.10, 8.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=3); ap.add_argument("--out", default=None)
    a = ap.parse_args(); k = a.k
    out = a.out or f"analysis/out_redundancy_triage_holdout1/paper/redundancy_vs_gain_config_k{k}.pdf"
    d = pd.read_csv("analysis/out_omega/omega_holdout1_neg_mse.csv")
    x, g = T * d[f"rho_e_mean_{k}"], d.G_paper
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.axhline(G_CUT, color=MUTED, lw=0.8, ls=":", zorder=1)
    ax.axvline(W_CUT, color=MUTED, lw=0.8, ls=":", zorder=1)
    for n, c in zip(SIZES, COLORS):
        m = d.p_obj_train_size.astype(int) == n
        ax.scatter(x[m], g[m], s=13, color=c, edgecolor="white", linewidth=0.4, zorder=2, alpha=0.9,
                   label=f"$n_{{\\mathrm{{tr}}}}={n:,}$".replace(",", "{,}"))
    b = pd.cut(x, BINS)
    st = g.groupby(b, observed=True).quantile([0.25, 0.5, 0.75]).unstack()
    mid = x.groupby(b, observed=True).median().loc[st.index].values   # median redundancy of each bin
    ax.errorbar(mid, st[0.5], yerr=[st[0.5] - st[0.25], st[0.75] - st[0.5]], fmt="s", color=INK, ms=4.5,
                lw=1.4, capsize=2.5, zorder=4, label="bin median (IQR)")
    xs = np.linspace(-0.01, 0.205, 200)
    ax.plot(xs, 1 / (1 / K + (1 - 1 / K) * xs), color=INK, lw=1.1, ls="--", zorder=3,
            label=r"$K/(1+(K-1)\,\omega)$")
    hi, lo = x > W_CUT, x <= W_CUT
    ax.text(0.98, 0.97, f"$\\widehat\\omega_{k}>{W_CUT:.2f}$: {(g[hi] >= G_CUT).sum()} of {hi.sum()} reach $G_{{20}}\\geq{G_CUT:.0f}$\n"
            f"$\\widehat\\omega_{k}\\leq{W_CUT:.2f}$: {(g[lo] >= G_CUT).sum()} of {lo.sum()}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(2, 40)
    ax.set_yticks([2, 4, 8, 16, 32]); ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(f"Redundancy $\\widehat{{\\omega}}^{{\\mathrm{{study}}}}_{{{k}}}$ (mean over runs)", color=INK, fontsize=9)
    ax.set_ylabel(r"$G^{\mathrm{test}}_{20}$ (MSE)", color=INK, fontsize=9)
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right", bbox_to_anchor=(1.0, 0.86), labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    r = spearmanr(x, g).statistic
    print("wrote", out, "| Spearman", round(float(r), 3), "| bins", st.round(2).to_dict())


if __name__ == "__main__":
    main()
