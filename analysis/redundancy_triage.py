#!/usr/bin/env python3
"""Single-run triage of the redundancy on the prospective holdout (k200_threshold_validation_holdout_50_*):
the paper's table "Single-run early-stopping validation" and figure "Study-only redundancy as a triage
signal", for the former redundancy score (C_g * rho_e * m_bar) and for the redundancy as now defined,
omega_k = t * rho_e,k (t = test fraction, rho_e,k = loss correlation after the first k splits, loss of the
gain's metric: squared error).

Per single run (one seed of one configuration): redundancy after k splits. Per configuration: the
order-averaged paper gains G_20, G_200 (gains.csv of derivation_levers_analysis.py, --gains-dir).
Tertiles are taken over all valid runs pooled.

Usage: python analysis/redundancy_triage.py --gains-dir analysis/out_holdout_mse [--solvers-exclude RandomForest]
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from derivation_levers_analysis import CONFIG_KEYS  # noqa: E402

GLOB = os.path.expandvars("$SCRATCH/ranking_outputs/k200_threshold_validation_holdout_50_*__c*.parquet")
CG = "objective_study_prediction_cov_all_oof_intersection"
RHO = "objective_study_squared_error_rho_all_oof_intersection"
MBAR = "objective_study_oof_intersection_mean_size_all"
KS = (2, 3, 5)


def load():
    import pyarrow.parquet as pq
    want = {CG, RHO, MBAR, "dataset_name", "p_dataset_seed", "idx_rep", "solver_name", *CONFIG_KEYS}
    fr = []
    for f in sorted(glob.glob(GLOB)):
        names = pq.read_schema(f).names
        d = pd.read_parquet(f, columns=[c for c in names if c in want])
        fr.append(d[d.idx_rep.isin([k - 1 for k in KS])])
    df = pd.concat(fr, ignore_index=True)
    df["family"] = df["dataset_name"].str.extract(r"^(\w+)\[")[0]
    if "p_obj_train_source" not in df:
        df["p_obj_train_source"] = "study"
    df["p_obj_train_source"] = df["p_obj_train_source"].fillna("study")
    df["k"] = df.idx_rep + 1
    df["former"] = df[CG].astype(float) * df[RHO].astype(float) * df[MBAR].astype(float)
    df["redundancy"] = df["p_obj_test_size"].astype(float) * df[RHO].astype(float)
    df["solver"] = df.solver_name.str.replace(r"\[.*$", "", regex=True)
    return df


def summarise(r, stat, k):
    x = r[(r.k == k) & r[stat].notna() & np.isfinite(r[stat])]
    if stat == "former":
        x = x[x[stat] > 0]                       # the former score is used on a log scale
        lc = pearsonr(np.log(x[stat]), np.log(x.G_paper_20)).statistic
        lc200 = pearsonr(np.log(x[stat]), np.log(x.G_paper_200)).statistic
    else:
        lc = pearsonr(x[stat], np.log(x.G_paper_20)).statistic
        lc200 = pearsonr(x[stat], np.log(x.G_paper_200)).statistic
    q1, q2 = x[stat].quantile([1 / 3, 2 / 3])
    x = x.assign(tertile=np.where(x[stat] <= q1, "low", np.where(x[stat] <= q2, "mid", "high")))
    rows = []
    for tgt, thr, lcv in (("G_paper_20", 10, lc), ("G_paper_200", 20, lc200)):
        rec = {"statistic": stat, "k": k, "target": tgt, "valid_runs": len(x), "log_corr": lcv,
               "rank_corr": spearmanr(x[stat], x[tgt]).statistic, "tertile_cut_1": q1, "tertile_cut_2": q2}
        for t in ("low", "mid", "high"):
            g = x[x.tertile == t]
            rec[f"median_gain_{t}"] = float(g[tgt].median())
            rec[f"P_large_{t}"] = float((g[tgt] >= thr).mean())
        rows.append(rec)
    probs = []
    for t in ("low", "mid", "high"):
        g = x[x.tertile == t]
        for thr in (7, 8, 10):
            p = float((g.G_paper_20 >= thr).mean())
            probs.append({"statistic": stat, "k": k, "tertile": t, "threshold": thr, "p": p,
                          "se": float(np.sqrt(p * (1 - p) / len(g))), "n": len(g)})
    return rows, probs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gains-dir", required=True)
    ap.add_argument("--solvers-exclude", nargs="*", default=[])
    ap.add_argument("--out-dir", default="analysis/out_redundancy_triage")
    a = ap.parse_args()
    r = load()
    r = r[~r.solver.isin(a.solvers_exclude)]
    g = pd.read_csv(os.path.join(a.gains_dir, "gains.csv"))
    keys = [c for c in CONFIG_KEYS if c in g.columns]
    wide = g[g.K.isin([20, 200])].pivot_table(index=keys, columns="K", values="G_paper").reset_index()
    wide.columns = [c if not isinstance(c, (int, np.integer)) else f"G_paper_{c}" for c in wide.columns]
    for c in keys:
        r[c], wide[c] = r[c].astype(str), wide[c].astype(str)
    r = r.merge(wide, on=keys, how="inner")
    print(f"runs x k: {len(r)} | configurations: {r.groupby(keys).ngroups} | solvers: {sorted(r.solver.unique())}")
    rows, probs = [], []
    for stat in ("former", "redundancy"):
        for k in KS:
            a_, b_ = summarise(r, stat, k); rows += a_; probs += b_
    t, p = pd.DataFrame(rows), pd.DataFrame(probs)
    os.makedirs(a.out_dir, exist_ok=True)
    tag = os.path.basename(a.gains_dir.rstrip("/"))
    t.to_csv(os.path.join(a.out_dir, f"table_{tag}.csv"), index=False)
    p.to_csv(os.path.join(a.out_dir, f"probabilities_{tag}.csv"), index=False)
    pd.set_option("display.width", 250)
    cols = ["statistic", "k", "target", "valid_runs", "log_corr", "rank_corr", "median_gain_low", "median_gain_mid",
            "median_gain_high", "P_large_low", "P_large_mid", "P_large_high"]
    print(t[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
