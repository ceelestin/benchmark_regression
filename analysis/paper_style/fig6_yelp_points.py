#!/usr/bin/env python3
"""Figure 6 (p. 9), item 33: Yelp points computed EXACTLY as the PCam points of the figure.

PCam points of Figure 6 (paper_style_figs/paper_style_scatter.py, fig7):
  gain       = G_paper_20 of benchmark_pcam analysis/pcam_gain_analysis.py (analyse): variance-equivalent gain on the
               benchmark-aware deltas q_k - b_k, single-split curve V1(alpha) from the first split's test fold and its
               outer chunks (hidden_test_<metric>s), averaged exactly over the order of the chunks, crossing with
               log-interpolation; written to out_round3/candidates_<metric>.csv
  redundancy = omega3 of benchmark_pcam analysis/pcam_redundancy_vs_gain.py: t * (mean over seeds of the cumulative
               OOF-intersection per-sample loss correlation after 3 splits, study_<loss>_rho_cum_oof_intersection
               at fold == 3), t = 0.2, on the gain's per-sample loss (0-1 error for accuracy)
This script imports the two PCam scripts (pcam_gain_analysis.py, pcam_redundancy_vs_gain.py,
benchmark_pcam fdb6873), first re-derives the 36 PCam points from the round-3 parquets and checks them against the
CSVs the figure reads, then applies the same functions to the Yelp re-run (BERT, XLM-RoBERTa; n_tr 1,000, 3,000;
50 seeds; K = 20). Yelp gains are on accuracy (as in the paper), so the redundancy is on the 0-1 error; NLL and Brier
are written too, for reference only.
Run in analysis/paper_style with `module load pytorch-gpu/py3/2.8.0`.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# the two PCam scripts of benchmark_pcam/analysis (fdb6873), unmodified; $BENCHMARK_PCAM_ANALYSIS overrides the path
sys.path.insert(0, os.environ.get("BENCHMARK_PCAM_ANALYSIS") or os.path.expandvars("$WORK/benchmark_pcam/analysis"))
import pcam_gain_analysis as GA          # noqa: E402
import pcam_redundancy_vs_gain as RV     # noqa: E402

W = os.environ["WORK"]
PCAM_GLOB = f"{W}/benchmark_pcam/results_output_pcam/pcam_results_CV_shuffle_split_unstrat_*_nsplits_20_seeds_1[0-4][0-9]-*_sizes_*.parquet"
YELP_GLOB = f"{W}/paper_style_figs/real_data/data/nlp/yelp_aligned/yelp_results_CV_shuffle_split_unstrat_*_nsplits_20_*.parquet"
PCAM_OUT = f"{W}/benchmark_pcam/analysis"
METRICS = ("accuracy", "nll", "brier")


def points(pattern):
    """(gain, redundancy) per (metric, model, n_tr) with the two PCam scripts' own functions."""
    d_red, n_files, n_kept = RV.load(pattern)                  # latest file per (model, seed, size)
    assert n_files == n_kept, (n_files, n_kept)                  # no duplicated runs
    df = GA.load(pattern)
    cand = pd.concat([GA.analyse(df, m)[1].assign(loss=m) for m in METRICS], ignore_index=True)
    rows = []
    for loss, (q, b, rc) in RV.LOSS.items():
        for (m, n), g in d_red.groupby(["model", "study_size"]):
            r = dict(loss=loss, model=m, size=int(n))
            for k in (3, 5):
                rho = g.loc[g.fold == k, rc].astype(float)
                r[f"omega{k}"], r[f"cov{k}"] = RV.T * rho.mean(), int(rho.notna().sum())
            rows.append(r)
    red = pd.DataFrame(rows)
    return red.merge(cand[["loss", "model", "train_size", "n_seeds", "G_paper_20"]], left_on=["loss", "model", "size"],
                     right_on=["loss", "model", "train_size"], validate="1:1")


def main():
    out = os.path.dirname(os.path.abspath(__file__))
    # 1. PCam: re-derive and compare with what Figure 6 reads
    pc = points(PCAM_GLOB)
    ref_red = pd.read_csv(f"{PCAM_OUT}/out_round3_redundancy/redundancy_vs_gain.csv")
    ref_g = pd.concat([pd.read_csv(f"{PCAM_OUT}/out_round3/candidates_{m}.csv").assign(loss=m) for m in METRICS])
    chk = pc.merge(ref_red[["loss", "model", "size", "omega3"]], on=["loss", "model", "size"], suffixes=("", "_ref")) \
            .merge(ref_g[["loss", "model", "train_size", "G_paper_20"]], on=["loss", "model", "train_size"], suffixes=("", "_ref"))
    assert len(chk) == 36
    print("PCam check: max |omega3 - ref| = %.2e, max |G20 - ref| = %.2e"
          % ((chk.omega3 - chk.omega3_ref).abs().max(), (chk.G_paper_20 - chk.G_paper_20_ref).abs().max()))
    # 2. Yelp: same functions
    y = points(YELP_GLOB)
    y.to_csv(f"{out}/fig6_yelp_points.csv", index=False)
    pd.set_option("display.width", 200)
    print(y.round(4).to_string(index=False))
    acc = y[y.loss == "accuracy"]
    print("Yelp accuracy: omega3 %.3f-%.3f, G20 %.2f-%.2f, Spearman %.2f (n=%d)"
          % (acc.omega3.min(), acc.omega3.max(), acc.G_paper_20.min(), acc.G_paper_20.max(),
             spearmanr(acc.omega3, acc.G_paper_20).statistic, len(acc)))
    acc = acc.assign(pred3=20 / (1 + 19 * acc.omega3))
    print(acc[["model", "size", "omega3", "G_paper_20", "pred3"]].assign(ratio=acc.pred3 / acc.G_paper_20).round(3).to_string(index=False))
    allp = pd.concat([pc[["omega3", "G_paper_20"]], y.loc[y.loss == "accuracy", ["omega3", "G_paper_20"]]])
    print("PCam (36) + Yelp accuracy (4): Spearman %.3f (n=%d); PCam alone %.3f"
          % (spearmanr(allp.omega3, allp.G_paper_20).statistic, len(allp), spearmanr(pc.omega3, pc.G_paper_20).statistic))
    for m in METRICS:
        g = y[y.loss == m]
        print(f"  Yelp {m}: Spearman {spearmanr(g.omega3, g.G_paper_20).statistic:+.2f}")


if __name__ == "__main__":
    main()
