#!/usr/bin/env python3
"""Fold-common split of the fold-driven variance (derivation experiment, job 49236).
Usage: python analysis/fold_common_split.py   (reads $SCRATCH/ranking_merged/...shufflesplit.parquet,
writes analysis/out_derivation_mse/foldcommon_split.csv)

Result 2026-09-22: the within-study fold-common variance is <= 4% of the per-sample fold variance for
every lever (folds share 80% of their training data, so a fold's mean quality is nearly constant
within a study); f_err == f_anova, and the closed form predicts G_err with Spearman 0.81 (K=20) /
0.93 (K=200). G_raw carries an extra across-study floor (the models' true performance varies with
the training data) that no number of splits removes and that the formula does not include.

Method. Per (configuration, seed), with q_k = fold-k test MSE / Var(y_study):
Var_k(q_k) = sv (1-t)/m + fv/m + sig_c^2 (1 - 1/m)   with fv = sig_eps^2 + sig_c^2 (per-sample ANOVA
fold variance includes the fold-common shift). Solve for sig_c^2, then
  f_err   = (fv - sig_c^2) / (sv + fv - sig_c^2)                    (interaction only; target of G_err)
  f_total = ((fv - sig_c^2)/m + sig_c^2) / (sv/m + (fv - sig_c^2)/m + sig_c^2)   (procedure estimand)
"""
import os, numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.stats import spearmanr
P = "objective_"
cols = ["dataset_name", "p_dataset_noise", "p_obj_train_size", "p_dataset_seed", "solver_name", "idx_rep",
        P+"neg_mse_test", P+"neg_mse_bench", P+"split_test_size", P+"study_target_var",
        P+"study_oof_anova_squared_error_sample_var", P+"study_oof_anova_squared_error_fold_var"]
df = pq.read_table(os.path.expandvars("$SCRATCH/ranking_merged/derivation_redundancy_levers_shufflesplit.parquet"), columns=cols).to_pandas()
df["q"] = -df[P+"neg_mse_test"] / df[P+"study_target_var"]
df["b"] = -df[P+"neg_mse_bench"] / df[P+"study_target_var"]
key = ["p_dataset_noise", "p_obj_train_size", "solver_name"]
t = 0.2
rows = []
for k4, g in df.groupby(key + ["p_dataset_seed"]):
    cfg, seed = k4[:3], k4[3]
    g = g.sort_values("idx_rep")
    last = g.iloc[-1]
    sv = last[P+"study_oof_anova_squared_error_sample_var"]; fv = last[P+"study_oof_anova_squared_error_fold_var"]
    m = float(last[P+"split_test_size"]); varq = g["q"].var(ddof=1)
    sig_c2 = (varq - (sv*(1-t) + fv)/m) / (1 - 1/m)
    sig_c2 = min(max(sig_c2, 0.0), fv)
    eps2 = fv - sig_c2
    f_err = eps2/(sv + eps2) if sv + eps2 > 0 else np.nan
    f_tot = (eps2/m + sig_c2)/(sv/m + eps2/m + sig_c2) if sv + eps2 + sig_c2 > 0 else np.nan
    f_anova = fv/(sv+fv) if sv + fv > 0 else np.nan
    rows.append(dict(zip(key, cfg), seed=seed, f_err=f_err, f_tot=f_tot, f_anova=f_anova, sig_c2=sig_c2, fv=fv, sv=sv, m=m,
                     q1=g["q"].iloc[0], e1=g["q"].iloc[0]-g["b"].iloc[0],
                     q20=g["q"].iloc[:20].mean(), e20=(g["q"]-g["b"]).iloc[:20].mean(),
                     q200=g["q"].mean(), e200=(g["q"]-g["b"]).mean()))
r = pd.DataFrame(rows)
def G(f, K): return 1.0/((1-f)*(t+(1-t)/K) + f/K)
out = []
for cfg, g in r.groupby(key):
    rec = dict(zip(key, cfg)); rec["n_seeds"] = len(g)
    for f in ("f_err", "f_tot", "f_anova"): rec[f] = g[f].median()
    rec["sig_c2_share"] = (g["sig_c2"]/(g["fv"])).median()
    for K, qc, ec in ((20, "q20", "e20"), (200, "q200", "e200")):
        rec[f"G_err_{K}"] = g["e1"].var(ddof=1)/g[ec].var(ddof=1)
        rec[f"G_raw_{K}"] = g["q1"].var(ddof=1)/g[qc].var(ddof=1)
        rec[f"Gp_err_{K}"] = G(rec["f_err"], K); rec[f"Gp_tot_{K}"] = G(rec["f_tot"], K); rec[f"Gp_anova_{K}"] = G(rec["f_anova"], K)
    out.append(rec)
o = pd.DataFrame(out)
o["lev"] = o["solver_name"].str.replace(r"\[alpha=1.0(,)?", "[", regex=True).str.replace(r"\[\]", "", regex=True).str.replace(r"\[.*max_depth.*\]|\[n_estimators=200\]", "", regex=True)
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 200)
print(f"configurations: {len(o)}")
for K in (20, 200):
    for src, tgt in (("anova", "err"), ("err", "err"), ("tot", "raw"), ("anova", "raw")):
        pred, obs = o[f"Gp_{src}_{K}"], o[f"G_{tgt}_{K}"]; lr = np.log(pred/obs)
        print(f"K={K:3d}  f_{src:5s} -> G_{tgt:3s}: Spearman {spearmanr(pred, obs).statistic:+.2f} | median log(pred/obs) {lr.median():+.2f} | IQR {lr.quantile(.25):+.2f}..{lr.quantile(.75):+.2f}")
print("\nmedians by lever: fold-common share of fv, f_anova, f_err, G_err_200 obs / pred(anova) / pred(err), G_raw_200 obs / pred(tot)")
print(o.groupby("lev")[["sig_c2_share", "f_anova", "f_err", "G_err_200", "Gp_anova_200", "Gp_err_200", "G_raw_200", "Gp_tot_200"]].median().round(2).to_string())
o.to_csv(os.path.expandvars("$WORK/benchmark_regression/analysis/out_derivation_mse/foldcommon_split.csv"), index=False)
