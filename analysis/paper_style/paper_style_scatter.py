#!/usr/bin/env python3
"""Figures 7-8 of FIGURE_STYLE_SPEC.md (2026-10-28): the two redundancy scatters in the paper's scatter style.

Values are unchanged from benchmark_regression analysis/paper_redundancy_main_figure.py (663 simulated
configurations + 36 PCam) and analysis/paper_redundancy_config_figure.py (whiskered version, 243 configurations
of the pre-registered holdout); only the formatting follows Section 4 of the spec.
Run in $WORK/paper_style_figs with `module load pytorch-gpu/py3/2.8.0`.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
from scipy.stats import spearmanr

import plot as P

K, T = 20, 0.2
REG = os.path.expandvars("$WORK/benchmark_regression/analysis")
PCAM = os.path.expandvars("$WORK/benchmark_pcam/analysis")
BLUES = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
BINS = [(0, 100, "50-100"), (101, 750, "300-750"), (751, 3000, "1,000-3,000"), (3001, 10**9, "5,000-10,000")]
PCAM_COLOR = "#E5AE38"
HALO = [pe.withStroke(linewidth=3.0, foreground="white")]
PT = dict(s=28, alpha=0.82, edgecolors="white", linewidths=0.35)
LEG = dict(frameon=False, fontsize=8, handletextpad=0.4, labelspacing=0.3, borderaxespad=0.3)


def bin_color(n):
    for i, (lo, hi, _) in enumerate(BINS):
        if lo <= n <= hi:
            return BLUES[i]
    raise ValueError(n)


def axes_common(ax, xlabel, gmin, gmax):
    xs = np.linspace(-0.012, 0.205, 200)
    ax.plot(xs, 1 / (1 / K + (1 - 1 / K) * xs), color="#262626", lw=1.1, dashes=(3.7, 1.6), zorder=1,
            label=r"$K/(1+(K-1)\,\omega)$")
    ax.set_yscale("log")
    ticks = [t for t in (2, 5, 10, 20, 50) if gmin <= t <= gmax]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([0.00, 0.05, 0.10, 0.15, 0.20])
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(r"Test Sample Gain ($G^{\mathrm{test}}_{20}$)", fontsize=12)
    ax.grid(True, which="major", linestyle=":", linewidth=0.7, color="#b0b0b0")
    ax.set_axisbelow(True)
    sns.despine(ax=ax)


def corr_text(ax, lines):
    ax.text(0.03, 0.04, "\n".join(lines), transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
            color="black", path_effects=HALO)


def fig7(out):
    d = pd.concat([pd.read_csv(f"{REG}/out_omega/omega_{s}_neg_mse.csv") for s in ("derivation", "holdout1", "holdout2", "holdout3")],
                  ignore_index=True)
    w, g, n = T * d["rho_e_mean_3"], d["G_paper"], d["p_obj_train_size"].astype(int)
    noisy = d.solver_name.str.contains("pred_noise|coef_noise")
    red = pd.read_csv(f"{PCAM}/out_round3_redundancy/redundancy_vs_gain.csv")
    gains = pd.concat([pd.read_csv(f"{PCAM}/out_round3/candidates_{m}.csv").assign(loss=m) for m in ("accuracy", "nll", "brier")])
    pc = red.merge(gains[["loss", "model", "train_size", "G_paper_20"]], left_on=["loss", "model", "size"],
                   right_on=["loss", "model", "train_size"], validate="1:1")
    assert len(d) == 663 and len(pc) == 36
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    cols = n.map(bin_color)
    for m, mk in ((~noisy, "o"), (noisy, "^")):
        ax.scatter(w[m], g[m], c=cols[m], marker=mk, zorder=2, **PT)
    ax.scatter(pc["omega3"], pc["G_paper_20"], c=PCAM_COLOR, marker="s", zorder=3, **PT)
    gall = pd.concat([g, pc["G_paper_20"]])
    axes_common(ax, r"Redundancy $\widehat{\omega}^{\mathrm{study}}_3$ after 3 splits", gall.min(), gall.max())
    h1 = [Line2D([], [], marker="o", ls="", markerfacecolor=c, markeredgecolor="white", markersize=5.3) for c in BLUES]
    l1 = ax.legend(h1, [f"$n_{{\\mathrm{{tr}}}}$ {b[2]}" for b in BINS], loc="upper right", **LEG)
    ax.add_artist(l1)
    # CE 2026-10-29: no second legend; markers and curve are described in the caption of Figure 6.
    rs, rp = spearmanr(w, g).statistic, spearmanr(pc["omega3"], pc["G_paper_20"]).statistic
    corr_text(ax, [f"simulated: rank corr. = {rs:.2f}, $n$ = {len(d)}", f"PCam: rank corr. = {rp:.2f}, $n$ = {len(pc)}"])
    fig.savefig(out, format="pdf", bbox_inches="tight")
    print("fig7", out, round(rs, 3), round(rp, 3))
    return fig


def fig8(out):
    d = pd.read_csv(f"{REG}/out_omega/omega_holdout1_neg_mse.csv")
    k = 3
    x, lo, hi = T * d[f"rho_e_mean_{k}"], T * d[f"rho_e_q10_{k}"], T * d[f"rho_e_q90_{k}"]
    g, n = d["G_paper"], d["p_obj_train_size"].astype(int)
    assert len(d) == 243
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    for size in sorted(n.unique()):
        m = n == size
        c = bin_color(size)
        ax.errorbar(x[m], g[m], xerr=[(x - lo)[m], (hi - x)[m]], yerr=[(g - d.G_paper_lo)[m], (d.G_paper_hi - g)[m]],
                    fmt="none", ecolor=c, alpha=0.3, elinewidth=0.6, capsize=0, zorder=1.5)
        ax.scatter(x[m], g[m], c=c, marker="o", zorder=2, label=f"$n_{{\\mathrm{{tr}}}}={size:,}$".replace(",", "{,}"), **PT)
    axes_common(ax, r"Mean redundancy $\widehat{\omega}^{\mathrm{study}}_3$",
                min(g.min(), d.G_paper_lo.min()), max(g.max(), d.G_paper_hi.max()))
    hs, ls = ax.get_legend_handles_labels()
    ax.legend(hs, ls, loc="upper right", **LEG)
    r = spearmanr(x, g).statistic
    corr_text(ax, [f"simulated: rank corr. = {r:.2f}, $n$ = {len(d)}"])
    fig.savefig(out, format="pdf", bbox_inches="tight")
    print("fig8", out, round(r, 3))
    return fig


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", required=True); a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    P.set_paper_style()
    fig7(f"{a.out_dir}/redundancy_vs_gain_all_k3.pdf")
    fig8(f"{a.out_dir}/redundancy_vs_gain_config_k3.pdf")


if __name__ == "__main__":
    main()
