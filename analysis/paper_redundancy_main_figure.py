#!/usr/bin/env python3
"""Main-text figure: configuration-level sample gain G_20 (MSE) against the redundancy omega_k = t * rho_e,k,
all simulated regression designs used for the redundancy (noise-injection grid, job 49236; pre-registered
holdout, job 101294; unstable-learner designs, jobs 121758 and 147428): 663 configurations, each the mean over
its single runs (30 or 50 seeds), with the reference curve K / (1 + (K-1) omega).
Usage: python analysis/paper_redundancy_main_figure.py [--k 3]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

K, T = 20, 0.2
SETS = ("derivation", "holdout1", "holdout2", "holdout3")
BINS = [(0, 100, "50-100"), (101, 750, "300-750"), (751, 3000, "1,000-3,000"), (3001, 10**9, "5,000-10,000")]
COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]          # ordinal blue, validated (dataviz skill)
INK, MUTED, GRID = "#0b0b0b", "#6b6b68", "#e6e5e1"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=3); a = ap.parse_args(); k = a.k
    d = pd.concat([pd.read_csv(f"analysis/out_omega/omega_{s}_neg_mse.csv") for s in SETS], ignore_index=True)
    d["w"] = T * d[f"rho_e_mean_{k}"]
    d["noisy"] = d.solver_name.str.contains("pred_noise|coef_noise")
    n = d.p_obj_train_size.astype(int)
    fig, ax = plt.subplots(figsize=(3.7, 2.7))
    for i, (lo, hi, lab) in enumerate(BINS):
        for noisy, mk in ((False, "o"), (True, "^")):
            m = n.between(lo, hi) & (d.noisy == noisy)
            ax.scatter(d.w[m], d.G_paper[m], s=11 if not noisy else 13, marker=mk, color=COLORS[i],
                       edgecolor="white", linewidth=0.35, alpha=0.9, zorder=2)
    xs = np.linspace(-0.012, 0.205, 200)
    ax.plot(xs, 1 / (1 / K + (1 - 1 / K) * xs), color=INK, lw=1.3, ls="--", zorder=3)
    # legends: training size (colour) and learner type (shape)
    hs = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=4.5) for c in COLORS]
    l1 = ax.legend(hs, [f"$n_{{\\mathrm{{tr}}}}$ {b[2]}" for b in BINS], frameon=False, fontsize=6.3, loc="upper right",
                   handletextpad=0.2, borderaxespad=0.2, labelspacing=0.25)
    ax.add_artist(l1)
    hs2 = [plt.Line2D([], [], marker="o", ls="", color=MUTED, markersize=4.5),
           plt.Line2D([], [], marker="^", ls="", color=MUTED, markersize=4.5),
           plt.Line2D([], [], ls="--", color=INK, lw=1.2)]
    ax.legend(hs2, ["learning algorithms", "Ridge + injected noise", r"$K/(1+(K-1)\,\omega)$"], frameon=False,
              fontsize=6.3, loc="lower left", handletextpad=0.3, borderaxespad=0.2, labelspacing=0.25)
    r = spearmanr(d.w, d.G_paper).statistic
    ax.text(0.97, 0.66, f"{len(d)} configurations\nrank corr. {r:+.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.3, color=MUTED)
    ax.set_yscale("log")
    ax.set_yticks([2, 5, 10, 20, 50]); ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(f"Redundancy $\\widehat{{\\omega}}^{{\\mathrm{{study}}}}_{{{k}}}$ after {k} splits", color=INK, fontsize=8)
    ax.set_ylabel(r"Sample gain $G^{\mathrm{test}}_{20}$", color=INK, fontsize=8)
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=6.5)
    fig.tight_layout(pad=0.3)
    out = f"analysis/out_redundancy_triage_holdout1/paper/redundancy_vs_gain_all_k{k}"
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=220)
    print("wrote", out, "| n", len(d), "| Spearman", round(float(r), 3), "| sizes", sorted(n.unique()))


if __name__ == "__main__":
    main()
