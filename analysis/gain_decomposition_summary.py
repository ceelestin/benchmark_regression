#!/usr/bin/env python3
"""Pool the analysis/out_decomp_*/decomposition_*.csv tables (analysis/gain_decomposition.py) and
summarise, at one K, how each mechanism moves the observed gain away from the study-only closed
form, by factor (set, gain loss, statistic loss, train size, learner, noise, family, lever).

Usage: python analysis/gain_decomposition_summary.py [--K 20] [--out analysis/out_decomp_summary]
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

TERMS = ["A0_chunk_order", "A_oracle_floor", "B_ceiling", "C_model", "D_loss_mismatch",
         "E_study_vs_pop", "F_early_read", "H_seed_aggregation", "total"]


def load_all():
    frames = []
    for f in sorted(glob.glob("analysis/out_decomp_*/decomposition_*.csv")):
        m = re.match(r".*out_decomp_(\w+?)/decomposition_(\w+?)_stat_(\w+)\.csv", f)
        if not m:
            continue
        d = pd.read_csv(f)
        d["set"], d["gain_loss"], d["stat_loss"] = m.group(1), m.group(2), m.group(3)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["lever"] = d["solver_name"].str.extract(r"(pred_noise=[0-9.]+|coef_noise=[0-9.]+|logit_noise=[0-9.]+)")[0].fillna("none")
    d["learner"] = d["solver"] + np.where(d["solver_name"].str.contains(r"n_estimators=2\b"), "(2)",
                                   np.where(d["solver_name"].str.contains(r"n_estimators=5\b"), "(5)", ""))
    return d


def shares(g):
    v = g["total"].var()
    return {t: float(np.cov(g[t], g["total"])[0, 1] / v) if v > 0 else np.nan for t in TERMS[:-1]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--out", default="analysis/out_decomp_summary")
    a = ap.parse_args()
    d = load_all()
    d = d[(d.K == a.K) & (d.p_obj_train_source.fillna("study") == "study")]
    os.makedirs(a.out, exist_ok=True)
    d.to_csv(os.path.join(a.out, f"pooled_K{a.K}.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    key = ["set", "gain_loss", "stat_loss"]
    print(f"===== K = {a.K}: medians of the log terms (positive = observed gain above the closed form) =====")
    print(d.groupby(key)[TERMS].median().round(3).assign(n=d.groupby(key).size()).to_string())
    print("\n===== share of the across-configuration variance of 'total' carried by each term =====")
    print(pd.DataFrame({k: shares(g) for k, g in d.groupby(key)}).T.round(2).to_string())
    for by in ("p_obj_train_size", "learner", "p_dataset_noise", "family", "lever"):
        print(f"\n===== model term C (coupling) and total, median by {by} and set =====")
        t = d.pivot_table(index=by, columns=["set", "stat_loss"], values="C_model", aggfunc="median")
        print(t.round(2).to_string())
    print("\n===== C_obs (observed between-fold covariance ratio) vs C_cf = t * rho_b, median by size and learner (regression sets) =====")
    r = d[d.gain_loss == "neg_mse"]
    print(r.pivot_table(index="learner", columns="p_obj_train_size", values=["C_obs", "C_cf"], aggfunc="median").round(3).to_string())


if __name__ == "__main__":
    main()
