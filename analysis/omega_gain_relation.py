#!/usr/bin/env python3
"""Paper redundancy score omega-hat vs the paper sample gain, per configuration.

omega-hat (PAPER.tex, study-only redundancy for early-stopping CV), for ONE CV run after k splits:
    omega_k = C_g * rho_e * m_bar
C_g = pair-averaged covariance of the two split predictors on I_ab (samples held out by both),
rho_e = pair-averaged loss covariance / pair-averaged loss variance on I_ab, m_bar = mean |I_ab|.
The objective records the three factors cumulatively (row idx_rep = k-1 holds the value over folds
1..k): objective_study_prediction_cov_all_oof_intersection, objective_study_<loss>_rho_all_oof_intersection,
objective_study_oof_intersection_mean_size_all. The loss of rho_e is the gain's metric
(neg_mse and R^2 score -> squared error; accuracy -> 0-1 error; neg_nll -> NLL).

Per configuration (seeds = single runs): mean of omega_k over runs, 10/50/90% quantiles across
runs, number of valid runs, joined with the order-averaged paper gain G_paper at K (gains.csv from
derivation_levers_analysis.py run with the same loss). ShuffleSplit, non-fixed splits, study-trained.

Usage: python analysis/omega_gain_relation.py --set holdout3 --gain-loss neg_mse \
          --glob "$SCRATCH/ranking_outputs/holdout3_largegain_sim_[lin]*__c*.parquet" \
          --gains analysis/out_holdout3_mse/gains.csv --out-dir analysis/out_omega
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from derivation_levers_analysis import CONFIG_KEYS  # noqa: E402

LOSS_OF_GAIN = {"neg_mse": "squared_error", "score": "squared_error", "accuracy": "error01", "neg_nll": "nll"}
CG = "objective_study_prediction_cov_all_oof_intersection"
MBAR = "objective_study_oof_intersection_mean_size_all"
KS = (2, 3, 5)


def load(pattern, loss):
    import pyarrow.parquet as pq
    rho = f"objective_study_{loss}_rho_all_oof_intersection"
    want = {CG, MBAR, rho, "dataset_name", "p_dataset_seed", "idx_rep", "solver_name",
            "p_dataset_logit_scale", *CONFIG_KEYS}
    frames = []
    for f in sorted(glob.glob(pattern)):
        names = pq.read_schema(f).names
        d = pd.read_parquet(f, columns=[c for c in names if c in want])
        frames.append(d[d["idx_rep"].isin([k - 1 for k in KS])])
    if not frames:
        raise SystemExit(f"no parquet matches {pattern}")
    df = pd.concat(frames, ignore_index=True)
    if "p_dataset_noise" not in df and "p_dataset_logit_scale" in df:
        df["p_dataset_noise"] = df["p_dataset_logit_scale"]
    if "p_obj_train_source" not in df:
        df["p_obj_train_source"] = "study"
    df["p_obj_train_source"] = df["p_obj_train_source"].fillna("study")
    df["family"] = df["dataset_name"].str.extract(r"^(\w+)\[")[0]
    df = df[(df.p_obj_procedure == "ShuffleSplit") & (df.p_obj_fixed_split.astype(str) != "True")
            & (df.p_obj_train_source == "study")]
    df["omega"] = df[CG].astype(float) * df[rho].astype(float) * df[MBAR].astype(float)
    df["C_g"], df["rho_e"], df["m_bar"] = df[CG].astype(float), df[rho].astype(float), df[MBAR].astype(float)
    return df


def summarise(df):
    rows = []
    for key, g in df.groupby(CONFIG_KEYS, dropna=False):
        rec = dict(zip(CONFIG_KEYS, key))
        for k in KS:
            x = g[g.idx_rep == k - 1]
            w = x["omega"].dropna()
            rec[f"n_runs_{k}"] = int(len(w))
            rec[f"omega_mean_{k}"] = float(w.mean()) if len(w) else np.nan
            for q in (10, 50, 90):
                rec[f"omega_q{q}_{k}"] = float(np.percentile(w, q)) if len(w) else np.nan
            for f in ("C_g", "rho_e", "m_bar"):
                rec[f"{f}_mean_{k}"] = float(x[f].mean())
            r = x["rho_e"].dropna()
            for q in (10, 50, 90):
                rec[f"rho_e_q{q}_{k}"] = float(np.percentile(r, q)) if len(r) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True)
    ap.add_argument("--gain-loss", required=True, choices=list(LOSS_OF_GAIN))
    ap.add_argument("--glob", required=True)
    ap.add_argument("--gains", required=True, help="gains.csv of derivation_levers_analysis.py, same loss")
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--out-dir", default="analysis/out_omega")
    a = ap.parse_args()
    loss = LOSS_OF_GAIN[a.gain_loss]
    om = summarise(load(a.glob, loss))
    g = pd.read_csv(a.gains)
    g = g[g.K == a.K][CONFIG_KEYS + ["n_seeds", "G_paper", "G_paper_capped", "G_paper_lo", "G_paper_hi", "G_paper_fixed", "G_err"]]
    for c in CONFIG_KEYS:                                  # align dtypes for the join
        om[c], g[c] = om[c].astype(str), g[c].astype(str)
    out = om.merge(g, on=CONFIG_KEYS, how="left")
    out.insert(0, "set", a.set); out.insert(1, "gain_loss", a.gain_loss); out.insert(2, "omega_loss", loss)
    out["solver"] = out["solver_name"].str.replace(r"\[.*$", "", regex=True)
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"omega_{a.set}_{a.gain_loss}.csv")
    out.to_csv(path, index=False)
    m = out["G_paper"].notna()
    print(f"{a.set} {a.gain_loss}: {len(out)} configurations, {int(m.sum())} with G_{a.K}; runs per config "
          f"{int(out.n_runs_3.min())}-{int(out.n_runs_3.max())}; wrote {path}")


if __name__ == "__main__":
    main()
