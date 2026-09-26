#!/usr/bin/env python3
"""PCam consistency check, as pre-registered in analysis/PREREGISTRATION_pcam_consistency.md (c0bda0f).

x = 0.2 * mean over seeds of the study-only loss correlation after k splits; gain G_20 (order-averaged
paper gain). Reference band from the simulated sim_classification configurations with train size
100-1000 (analysis/out_omega/). Checks A1 coverage >= 70 %, A2 |median log offset| <= 0.2,
A3 Spearman <= -0.5.

Usage (Jean-Zay): python analysis/pcam_consistency_check.py [--k 5] [--gain accuracy|nll]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

T = 0.2
PCAM = os.path.expandvars("$WORK/benchmark_pcam")
SETS = {  # name -> (parquet glob, gains dir)
    "pcam_100seeds": (os.path.expandvars("$SCRATCH/pcam_calib100_links/*.parquet"),
                      f"{PCAM}/analysis/out_calibration_100seeds"),
    "pcam_round3": (f"{PCAM}/results_output_pcam/pcam_results_CV_shuffle_split_unstrat_*_nsplits_20_seeds_"
                    "1[0-4][0-9]-1[0-4][0-9]_sizes_[138]00_2026092[4-9]_*.parquet", f"{PCAM}/analysis/out_round3"),
}
RHO = {"accuracy": "study_error01_rho_cum_oof_intersection", "nll": "study_nll_rho_cum_oof_intersection"}
SIM_GAIN = {"accuracy": "accuracy", "nll": "neg_nll"}


def pcam_table(gain, k):
    out = []
    for name, (pattern, gdir) in SETS.items():
        cols = ["model", "study_size", "seed", "fold", RHO[gain]]
        df = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(pattern))], ignore_index=True)
        row = df[df["fold"] == k].groupby(["model", "study_size"])[RHO[gain]]
        st = row.agg(["mean", "count"]).reset_index().rename(columns={"study_size": "train_size"})
        st["q10"] = row.quantile(0.1).values; st["q90"] = row.quantile(0.9).values
        g = pd.read_csv(os.path.join(gdir, f"gains_{gain}.csv"))
        g = g[g.K == 20][["model", "train_size", "n_seeds", "G_paper", "G_paper_lo", "G_paper_hi", "G_paper_capped"]]
        st["train_size"] = st["train_size"].astype(int)
        t = st.merge(g, on=["model", "train_size"], how="inner")
        t.insert(0, "set", name)
        out.append(t)
    t = pd.concat(out, ignore_index=True)
    t["x"], t["x_q10"], t["x_q90"] = T * t["mean"], T * t["q10"], T * t["q90"]
    return t


def sim_reference(gain, k):
    d = pd.concat([pd.read_csv(f) for f in glob.glob("analysis/out_omega/omega_*.csv")], ignore_index=True)
    d = d[(d.gain_loss == SIM_GAIN[gain]) & d.p_obj_train_size.astype(int).between(100, 1000)]
    d = d.assign(x=T * d[f"rho_e_mean_{k}"])[["set", "p_obj_train_size", "solver_name", "x", "G_paper"]].dropna()
    return d


def band(sim, x):
    w = 0.02
    while True:
        nb = sim[(sim.x - x).abs() <= w]
        if len(nb) >= 10 or w > 0.2:
            return (np.percentile(nb.G_paper, 10), np.median(nb.G_paper), np.percentile(nb.G_paper, 90), len(nb), w)
        w += 0.01


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--gain", default="accuracy", choices=list(RHO))
    a = ap.parse_args()
    p, sim = pcam_table(a.gain, a.k), sim_reference(a.gain, a.k)
    b = np.array([band(sim, x) for x in p.x])
    p["band_lo"], p["centre"], p["band_hi"], p["n_neighbours"], p["half_width"] = b.T
    p["inside"] = (p.G_paper >= p.band_lo) & (p.G_paper <= p.band_hi)
    p["log_offset"] = np.log(p.G_paper / p.centre)
    p["ideal_curve"] = 1 / (1 / 20 + (1 - 1 / 20) * p.x)
    pd.set_option("display.width", 250)
    print(f"===== gain {a.gain}, statistic 0.2 * rho_e({RHO[a.gain].split('_')[1]}) after k = {a.k} splits; "
          f"simulated reference: {len(sim)} configurations (train size 100-1000) =====")
    print(p[["set", "model", "train_size", "count", "x", "x_q10", "x_q90", "G_paper", "band_lo", "centre", "band_hi",
             "n_neighbours", "inside", "ideal_curve"]].round(3).to_string(index=False))
    for name, g in [("ALL", p)] + list(p.groupby("set")):
        cov, off = g.inside.mean(), g.log_offset.median()
        r = spearmanr(g.x, g.G_paper).statistic
        tag = "" if name != "ALL" else "  <- pre-registered"
        print(f"{name:14s} n={len(g):2d} | A1 coverage {cov:.2f} {'PASS' if cov >= 0.7 else 'FAIL'} | "
              f"A2 median log offset {off:+.2f} {'PASS' if abs(off) <= 0.2 else 'FAIL'} | "
              f"A3 Spearman {r:+.2f} {'PASS' if r <= -0.5 else 'FAIL'}{tag}")
    os.makedirs("analysis/out_pcam_consistency", exist_ok=True)
    p.to_csv(f"analysis/out_pcam_consistency/pcam_{a.gain}_k{a.k}.csv", index=False)


if __name__ == "__main__":
    main()
