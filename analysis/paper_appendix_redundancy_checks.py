#!/usr/bin/env python3
"""Re-check two appendix claims of the paper with the redundancy omega_k = t * rho_e,k (squared error):
(1) HPO appendix: ordering of the redundancy for ExtraTrees with fixed parameters, grid search and random
    search (simulated linear, n_tr = 1000, K = 20, 1000 seeds; July 2026 runs).
(2) Small-budget appendix: at each training budget (5-100), how often the redundancy of a single run is
    defined after k splits, and how often a single run orders the two algorithms whose mean redundancies
    are furthest apart in the same way as their means (same seed = same study set).
Usage (Jean-Zay): python analysis/paper_appendix_redundancy_checks.py --dir $SCRATCH/paper_runs
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RHO = "objective_study_squared_error_rho_all_oof_intersection"
COLS = ["solver_name", "p_dataset_seed", "p_obj_train_size", "p_obj_test_size", "idx_rep", RHO,
        "objective_study_prediction_cov_all_oof_intersection", "objective_study_oof_intersection_mean_size_all"]


def load(files, ks=(3, 5)):
    fr = []
    for f in files:
        d = pq.read_table(f, columns=COLS).to_pandas()
        fr.append(d[d.idx_rep.isin([k - 1 for k in ks])].assign(file=os.path.basename(f)[13:30]))
    d = pd.concat(fr, ignore_index=True)
    d["k"] = d.idx_rep + 1
    d["redundancy"] = d.p_obj_test_size.astype(float) * d[RHO].astype(float)
    d["former"] = d[RHO] * d.objective_study_prediction_cov_all_oof_intersection * d.objective_study_oof_intersection_mean_size_all
    return d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); a = ap.parse_args()
    pd.set_option("display.width", 250)
    # (1) HPO
    fs = [f for f in glob.glob(os.path.join(a.dir, "benchopt_run_2026-07-*.parquet"))
          if any(s in str(pq.read_table(f, columns=["solver_name"]).column(0)[0]) for s in ("ExtraTrees",))]
    d = load(fs)
    d = d[(d.p_obj_train_size == 1000) & d.solver_name.str.startswith("ExtraTrees")]
    print("===== (1) HPO: ExtraTrees at n_tr = 1000, mean over seeds =====")
    print(d.groupby(["solver_name", "k"]).agg(n_seeds=("p_dataset_seed", "nunique"), redundancy=("redundancy", "mean"),
                                             former=("former", "mean")).round(4).to_string())
    # (2) small budgets
    fs = [f for f in glob.glob(os.path.join(a.dir, "benchopt_run_2026-07-31*.parquet"))]
    d = load(fs)
    print("\n===== (2) small budgets: share of runs with a defined redundancy, mean redundancy =====")
    s = d.groupby(["p_obj_train_size", "k", "solver_name"]).agg(defined=("redundancy", lambda x: x.notna().mean()),
                                                                mean=("redundancy", "mean")).reset_index()
    print(s.pivot_table(index=["p_obj_train_size", "k"], columns="solver_name", values="defined").round(2).to_string())
    print(s.pivot_table(index=["p_obj_train_size", "k"], columns="solver_name", values="mean").round(3).to_string())
    print("\nsingle-run agreement with the order of the two most separated algorithms (same seed):")
    rows = []
    for (n, k), g in d.groupby(["p_obj_train_size", "k"]):
        m = g.groupby("solver_name").redundancy.mean().dropna()
        if len(m) < 2:
            continue
        hi, lo = m.idxmax(), m.idxmin()
        w = g.pivot_table(index="p_dataset_seed", columns="solver_name", values="redundancy")[[hi, lo]].dropna()
        rows.append({"n_tr": n, "k": k, "higher": hi.split("[")[0], "lower": lo.split("[")[0], "gap_of_means": m[hi] - m[lo],
                     "seeds_both_defined": len(w), "agreement": float((w[hi] > w[lo]).mean()) if len(w) else np.nan})
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
