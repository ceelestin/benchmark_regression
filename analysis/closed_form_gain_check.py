#!/usr/bin/env python3
"""Closed-form gain model checked against the derivation experiment (job 49236).

Model (ShuffleSplit with test fraction t, K splits, study of N samples, test folds
of m = tN samples). Write the per-sample fold-k loss as e_{k,i} = a_i + eps_{k,i}:
a_i is SAMPLE-driven (noise + shared bias at x_i, identical in every fold),
eps_{k,i} is FOLD-driven (training-set variability of the learner). The K-split
mean reuses the same N study samples, so the sample-driven variance can only be
averaged from sigma_a^2/m down to sigma_a^2/N (finite population: gain <= 1/t),
while the fold-driven variance averages as 1/K:

    Var(qbar_K)  = sigma_a^2/N * (1 + (1-t)/(K t)) + sigma_f^2 / K
    G_K = Var(q_1)/Var(qbar_K) = 1 / [ (1-f) (t + (1-t)/K) + f/K ]        (ShuffleSplit)
    G_K = 1 / [ (1-f) t + f/K ]                    (RepeatedKFold, complete repeats)

with f = fold-driven share of the single-split variance. f is estimated STUDY-ONLY
from the per-sample OOF ANOVA of the squared error emitted by objective.py
(f_hat = 1 - study_oof_anova_squared_error_icc). Usage:

    python analysis/closed_form_gain_check.py analysis/out_derivation_mse
"""
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
import sys
out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.expandvars("$WORK/benchmark_regression/analysis/out_derivation_mse")
c = pd.read_csv(os.path.join(out_dir, "candidates.csv"))
c["scheme"] = np.where(c["p_obj_fixed_split"].astype(str) == "True", "fixed", c["p_obj_procedure"])
c["lev"] = c["solver"] + c["lever"].fillna("").map(lambda s: " " + s if s else "")
t = 0.2
def G_ss(f, K): return 1.0 / ((1 - f) * (t + (1 - t) / K) + f / K)      # ShuffleSplit
def G_rkf(f, K): return 1.0 / ((1 - f) * t + f / K)                      # RepeatedKFold, complete repeats
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
for sch, G in (("ShuffleSplit", G_ss), ("RepeatedKFold", G_rkf)):
    d = c[c.scheme == sch].copy()
    print(f"\n################ {sch}: n={len(d)} ################")
    for k in (3, 20):
        d[f"fsq_{k}"] = 1 - d[f"icc_sq_{k}"]        # fold-driven share of the SQUARED error (per-sample ANOVA)
        d[f"fe_{k}"] = 1 - d[f"icc_an_{k}"]         # ... of the signed error
    for src in ("fsq_3", "fsq_20", "fe_20"):
        if d[src].isna().all(): print(f"{src}: not defined for this scheme at that k"); continue
        for K in (20, 200):
            pred = G(d[src], K); obs = d[f"G_err_{K}"]; ok = pred.notna() & obs.notna()
            lr = np.log(pred[ok] / obs[ok])
            print(f"f from {src:7s} -> G_{K:3d}: Spearman {spearmanr(pred[ok], obs[ok]).statistic:+.2f} | median log(pred/obs) {lr.median():+.2f} | IQR {lr.quantile(.25):+.2f}..{lr.quantile(.75):+.2f} | n={ok.sum()}")
    src = "fsq_20" if not d["fsq_20"].isna().all() else "fe_20"
    d["G20p"] = G(d[src], 20); d["G200p"] = G(d[src], 200)
    print(f"\nmedians by lever ({src}):")
    print(d.groupby("lev")[[src, "G_err_20", "G20p", "G_err_200", "G200p", "remaining_err_20_200"]].median().round(2).to_string())
    if sch == "ShuffleSplit":
        print("\n=== absolute rule on f read at k=3 (squared-error ANOVA): predict 'large further gain' ===")
        for target, thr_name in (("G_err_20", 10), ("G_err_200", 20)):
            large = d[target] >= thr_name
            for fthr in (0.5, 0.6, 0.7, 0.74, 0.8, 0.9):
                flag = d["fsq_3"] >= fthr           # flag = "keep splitting"
                tp = (flag & large).sum(); fn = (~flag & large).sum(); fp = (flag & ~large).sum(); tn = (~flag & ~large).sum()
                print(f"  {target}>={thr_name}: rule fsq_3>={fthr:.2f} -> miss rate P(large & not flagged)/P(large) = {fn/max(large.sum(),1):.2f}, false-alarm P(flagged & not large)/P(not large) = {fp/max((~large).sum(),1):.2f}  [large={large.sum()}, flagged={flag.sum()}]")
        print("\n=== the same with rho_e_3 (pre-registered ingredient) as 'redundant' indicator: stop if rho_e_3 > x ===")
        large = d["G_err_20"] >= 10
        for x in (0.05, 0.1, 0.2, 0.3, 0.5):
            stop = d["rho_e_3"] > x
            print(f"  stop if rho_e_3>{x}: P(large | stop) = {large[stop].mean():.2f} (n_stop={stop.sum()}), P(large | continue) = {large[~stop].mean():.2f}")
        print("\n=== Sdl_3 (pre-registered dimensionless S): stop if Sdl_3 > x ===")
        for x in (0.01, 0.03, 0.05, 0.1, 0.2):
            stop = d["Sdl_3"] > x
            print(f"  stop if Sdl_3>{x}: P(large | stop) = {large[stop].mean():.2f} (n_stop={stop.sum()}), P(large | continue) = {large[~stop].mean():.2f}")
