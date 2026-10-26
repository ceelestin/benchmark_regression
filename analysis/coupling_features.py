"""Screen study-only statistics as proxies of the train/test coupling term kappa (MSE).

For every configuration (solver, n_tr, noise) of the given runs with resampled splits: the oracle
kappa = rho_delta - omega_5, where rho_delta = (K/G_K - 1)/(K - 1) from the benchmark-aware MSE gain G_K
(split-averaged single-split variance) and omega_5 = t * mean_seed(rho_e) after five splits (OOF-intersection
squared-error correlation); and candidate study-only features after five splits, averaged over seeds:
  optimism   1 - mean MSE_train / mean MSE_test over the first five splits
  rho_full   squared-error correlation across splits over all study samples
  rho_rtm    same, after removing the train / test membership means of each split
  anova_fold, anova_icc  OOF-ANOVA fold-variance share and ICC (newer runs only)
Prints the Spearman correlation of kappa (and of the gain shortfall) with each feature and writes a CSV.

    python analysis/coupling_features.py --files ... --out analysis/out_coupling/coupling_features.csv
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

K = 20
COLS = ["solver_name", "p_dataset_seed", "idx_rep", "p_obj_train_size", "p_obj_test_size", "p_obj_fixed_split",
        "p_dataset_noise", "p_dataset_name", "objective_neg_mse_test", "objective_neg_mse_bench",
        "objective_neg_mse_train", "objective_study_squared_error_rho_all_oof_intersection",
        "objective_study_squared_error_rho_all_full", "objective_study_squared_error_rho_all_resid_train_membership",
        "objective_study_oof_anova_squared_error_fold_var", "objective_study_oof_anova_squared_error_sample_var",
        "objective_study_oof_anova_squared_error_icc"]


def load(f):
    names = pq.read_schema(f).names
    d = pq.read_table(f, columns=[c for c in COLS if c in names]).to_pandas()
    for c in COLS:
        if c not in d:
            d[c] = np.nan
    d["source"] = re.sub(r"__c\d+\.parquet$", "", os.path.basename(f))  # one run family across its chunks
    return d


def config_row(g):
    g = g[g.idx_rep < K]
    E = g.assign(x=g.objective_neg_mse_test.astype(float) - g.objective_neg_mse_bench.astype(float)) \
         .pivot_table(index="p_dataset_seed", columns="idx_rep", values="x").dropna()
    if E.shape[1] < K or len(E) < 100:
        return None
    E = E.values
    G = np.var(E, axis=0, ddof=1).mean() / np.var(E.mean(axis=1), ddof=1)
    t = float(g.p_obj_test_size.iloc[0])
    at = lambda c, k: g.loc[g.idx_rep == k - 1, c].astype(float).mean()
    first5 = g[g.idx_rep < 5]
    tr = -first5.groupby("p_dataset_seed").objective_neg_mse_train.mean().astype(float)
    te = -first5.groupby("p_dataset_seed").objective_neg_mse_test.mean().astype(float)
    r = dict(n_seeds=len(E), G20=G, rho_delta=(K / G - 1) / (K - 1),
             omega3=t * at("objective_study_squared_error_rho_all_oof_intersection", 3),
             omega5=t * at("objective_study_squared_error_rho_all_oof_intersection", 5),
             optimism=float((1 - tr / te).mean()),
             rho_full=at("objective_study_squared_error_rho_all_full", 5),
             rho_rtm=at("objective_study_squared_error_rho_all_resid_train_membership", 5),
             anova_icc=at("objective_study_oof_anova_squared_error_icc", 5))
    fv, sv = at("objective_study_oof_anova_squared_error_fold_var", 5), at("objective_study_oof_anova_squared_error_sample_var", 5)
    r["anova_fold"] = fv / (fv + sv) if fv == fv and sv == sv else np.nan
    r["kappa"] = r["rho_delta"] - r["omega5"]
    r["shortfall"] = np.log((K / (1 + (K - 1) * r["omega5"])) / G)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = pd.concat([load(f) for f in a.files], ignore_index=True)
    d = d[d.p_obj_fixed_split.astype(str) != "True"]
    rows = []
    for key, g in d.groupby(["source", "solver_name", "p_obj_train_size", "p_dataset_noise"], dropna=False):
        r = config_row(g)
        if r is not None:
            rows.append(dict(zip(["source", "solver", "n_tr", "noise"], key), **r))
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.to_csv(a.out, index=False)
    feats = ["optimism", "rho_full", "rho_rtm", "anova_fold", "anova_icc", "omega5", "n_tr"]
    print(f"{len(out)} configurations; kappa range {out.kappa.min():.3f} to {out.kappa.max():.3f}")
    for tgt in ("kappa", "shortfall"):
        print(tgt, "Spearman:", ", ".join(f"{f} {out[[tgt, f]].dropna().corr(method='spearman').iloc[0, 1]:+.2f} (n={out[[tgt, f]].dropna().shape[0]})" for f in feats))
    pd.set_option("display.width", 220)
    print(out.sort_values("kappa")[["solver", "n_tr", "G20", "omega5", "kappa", "optimism", "rho_full", "rho_rtm", "anova_fold"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
