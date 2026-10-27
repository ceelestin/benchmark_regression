#!/usr/bin/env python3
"""Main-text figure: configuration-level sample gain G_20 (MSE) against the redundancy omega_k = t * rho_e,k,
all simulated regression designs used for the redundancy (noise-injection grid, job 49236; pre-registered
holdout, job 101294; unstable-learner designs, jobs 121758 and 147428): 663 configurations, each the mean over
its single runs (30 or 50 seeds), with the reference curve K / (1 + (K-1) omega).
Optionally overlays the PCam round-3 configurations (4 networks x 3 sizes x accuracy / NLL / Brier, 50 seeds):
redundancy = t * seed-mean rho after k splits on the metric's per-sample loss (benchmark_pcam
analysis/out_round3_redundancy/redundancy_vs_gain.csv) and the same variance-equivalent gain G_paper_20 as the
simulated points (benchmark_pcam analysis/out_round3/candidates_<metric>.csv).
Usage: python analysis/paper_redundancy_main_figure.py [--k 3] [--pcam-dir $WORK/benchmark_pcam/analysis]
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
PCAM = "#d9480f"                                                 # distinct hue for the real-data points
INK, MUTED, GRID = "#0b0b0b", "#6b6b68", "#e6e5e1"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--pcam-dir", default=None); a = ap.parse_args(); k = a.k
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
    pc = None
    if a.pcam_dir:
        red = pd.read_csv(f"{a.pcam_dir}/out_round3_redundancy/redundancy_vs_gain.csv")
        gains = pd.concat([pd.read_csv(f"{a.pcam_dir}/out_round3/candidates_{m}.csv").assign(loss=m)
                           for m in ("accuracy", "nll", "brier")], ignore_index=True)
        pc = red.merge(gains[["loss", "model", "train_size", "G_paper_20"]],
                       left_on=["loss", "model", "size"], right_on=["loss", "model", "train_size"], validate="1:1")
        assert len(pc) == len(red) == 36, len(pc)
        ax.scatter(pc[f"omega{k}"], pc.G_paper_20, s=16, marker="D", color=PCAM, edgecolor="white", linewidth=0.4,
                   zorder=4)
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
    labs2 = ["learning algorithms", "Ridge + injected noise", r"$K/(1+(K-1)\,\omega)$"]
    if pc is not None:
        hs2.insert(2, plt.Line2D([], [], marker="D", ls="", color=PCAM, markersize=4))
        labs2.insert(2, "PCam (acc., NLL, Brier)")
    ax.legend(hs2, labs2, frameon=False,
              fontsize=6.3, loc="lower left", handletextpad=0.3, borderaxespad=0.2, labelspacing=0.25)
    r = spearmanr(d.w, d.G_paper).statistic
    lab = f"{len(d)} simulated configurations\nrank corr. {r:+.2f}"
    if pc is not None:
        rp = spearmanr(pc[f"omega{k}"], pc.G_paper_20).statistic
        lab += f"\n{len(pc)} PCam configurations\nrank corr. {rp:+.2f}"
        print("PCam Spearman", round(float(rp), 3), "| per metric", {m: round(float(spearmanr(g[f"omega{k}"], g.G_paper_20).statistic), 2) for m, g in pc.groupby("loss")})
    ax.text(0.97, 0.66, lab, transform=ax.transAxes,
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
