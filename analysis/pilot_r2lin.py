"""Pilot: does the R^2-linearized redundancy t * rho_r2lin predict the R^2 sample gains the way
t * rho_mse predicts the MSE gains? (configs/pilot_r2lin_k20.yml)

For each solver: G_K on R^2 and on MSE (benchmark-aware deltas, V1 = mean over splits of the across-seed
variance of a single split, VK = across-seed variance of the K-split mean), and omega_k = t * mean_seed rho
at k = 3 and 5 for both per-sample losses, with the curve prediction K / (1 + (K - 1) omega).

    python analysis/pilot_r2lin.py --files outputs/pilot_r2lin_k20__c0*.parquet
"""
import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RHO = {"mse": "objective_study_squared_error_rho_all_oof_intersection",
       "r2lin": "objective_study_r2lin_rho_all_oof_intersection"}
SCORE = {"mse": ("objective_neg_mse_test", "objective_neg_mse_bench"),
         "r2": ("objective_score_test", "objective_score_bench")}


def gain(g, col, K):
    q, b = SCORE[col]
    E = (g[q].astype(float) - g[b].astype(float))
    E = g.assign(d=E).pivot_table(index="p_dataset_seed", columns="idx_rep", values="d").dropna().values[:, :K]
    v1 = np.var(E, axis=0, ddof=1).mean()
    return v1 / np.var(E.mean(axis=1), ddof=1), len(E)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--K", type=int, default=20)
    a = ap.parse_args()
    cols = ["solver_name", "p_dataset_seed", "idx_rep", "p_obj_test_size"] + list(RHO.values()) + \
        [c for v in SCORE.values() for c in v]
    d = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in a.files], ignore_index=True)
    rows = []
    for s, g in d.groupby("solver_name"):
        t = float(g.p_obj_test_size.iloc[0])
        r = {"solver": s.split("[")[0].replace("_noHPO", "")}
        for m in SCORE:
            r[f"G{a.K}_{m}"], r["n_seeds"] = gain(g, m, a.K)
        for name, c in RHO.items():
            for k in (3, 5):
                w = t * g.loc[g.idx_rep == k - 1, c].astype(float).mean()
                r[f"omega{k}_{name}"] = w
                r[f"pred{k}_{name}"] = a.K / (1 + (a.K - 1) * w)
        rows.append(r)
    out = pd.DataFrame(rows).set_index("solver")
    pd.set_option("display.width", 200)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
