#!/usr/bin/env python3
"""Classification: G_20 vs t * rho_e after k splits -- earlier simulated reference (grey), the
classification sweep at PCam-like sizes (circles, blue by train size) and PCam (diamonds).
Point = configuration mean; horizontal whisker = 10-90 % of single runs; vertical = gain 90 % CI.
Usage: python analysis/sweep_pcam_plot.py [--k 5]
"""
import argparse
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

T, K = 0.2, 20
SIZES, COLORS = (100, 300, 800), ("#86b6ef", "#2a78d6", "#104281")
INK, MUTED, GRID, SURFACE, REF = "#0b0b0b", "#6b6b68", "#e6e5e1", "#fcfcfb", "#b9b8b4"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=5); a = ap.parse_args(); k = a.k
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), facecolor=SURFACE)
    for ax, (gl, pc, title) in zip(axes, (("accuracy", "accuracy", "gain on accuracy, redundancy on 0-1 error"),
                                          ("neg_nll", "nll", "gain on NLL, redundancy on NLL"))):
        ref = pd.concat([pd.read_csv(f) for f in glob.glob("analysis/out_omega/omega_*.csv")])
        ref = ref[(ref.gain_loss == gl) & ref.p_obj_train_size.astype(int).between(100, 1000)]
        ax.scatter(T * ref[f"rho_e_mean_{k}"], ref.G_paper, s=10, color=REF, marker="s", label="earlier simulations, train 100-1000", zorder=1)
        sw = pd.read_csv(f"analysis/out_omega_sweep/omega_sweep_clf_{gl}.csv")
        for n, c in zip(SIZES, COLORS):
            g = sw[sw.p_obj_train_size.astype(int) == n]
            x, lo, hi = T * g[f"rho_e_mean_{k}"], T * g[f"rho_e_q10_{k}"], T * g[f"rho_e_q90_{k}"]
            ax.errorbar(x, g.G_paper, xerr=[x - lo, hi - x], yerr=[g.G_paper - g.G_paper_lo, g.G_paper_hi - g.G_paper],
                        fmt="none", ecolor=c, alpha=0.3, lw=0.8, zorder=2)
            ax.scatter(x, g.G_paper, s=22, color=c, edgecolor=SURFACE, linewidth=0.6, label=f"sweep, train size {n}", zorder=3)
        pcam = pd.read_csv(f"analysis/out_pcam_consistency/pcam_{pc}_k{k}.csv")
        ax.errorbar(pcam.x, pcam.G_paper, xerr=[pcam.x - pcam.x_q10, pcam.x_q90 - pcam.x],
                    yerr=[pcam.G_paper - pcam.G_paper_lo, pcam.G_paper_hi - pcam.G_paper], fmt="none", ecolor=INK, alpha=0.25, lw=0.8, zorder=4)
        ax.scatter(pcam.x, pcam.G_paper, s=40, marker="D", facecolor="none", edgecolor=INK, linewidth=1.2, label="PCam (4 networks, train 100-800)", zorder=5)
        xs = np.linspace(-0.02, 0.21, 300)
        ax.plot(xs, 1 / (1 / K + (1 - 1 / K) * xs), color=INK, lw=1.5, ls="--", label="K / (1 + (K-1) x)", zorder=6)
        ax.set_yscale("log"); ax.set_xlim(-0.03, 0.22)
        ax.set_xlabel(f"$t\\,\\hat\\rho_{{e,{k}}}$  (t = 0.2, loss correlation after {k} splits)", color=INK)
        ax.set_ylabel("sample gain $G^{test}_{20}$", color=INK)
        ax.set_title(f"Classification: {title}", loc="left", color=INK, fontsize=10)
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_facecolor(SURFACE)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    fig.suptitle("Classification sweep (seeds 6000-6049, job 231717) and PCam against the simulated relation", color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = f"analysis/out_omega_sweep/sweep_pcam_vs_gain_K20_k{k}.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE); print("wrote", out)


if __name__ == "__main__":
    main()
