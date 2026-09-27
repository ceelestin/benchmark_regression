#!/usr/bin/env python3
"""Test-size check of the overlap factor (configs/testsize_check_*.yml, job 240569).

Paper: G_K ~ K / (1 + (K-1) rho_delta), rho_delta = tau_te / sigma2_te the fold-error correlation.
Hypothesis: rho_delta ~ f * rho_e with f = t, the mean overlap fraction of two random test folds.
From analysis/out_decomp_testsize (gain_decomposition.py), per configuration at K = 20:
C_obs (measured fold-error correlation), rho_b (loss correlation of the fold models on the benchmark
set, population), rho_sK (study-only, all K folds). Per test fraction t: slope f of C_obs on rho
through the origin (sum C rho / sum rho^2) and with an intercept (the intercept absorbs coupling).
Also the factor implied by the gain itself: f_G = (1/G_20 - 1/K) / ((1 - 1/K) rho) per configuration.
"""
import glob

import numpy as np
import pandas as pd

K = 20
pd.set_option("display.width", 250)
d = pd.concat([pd.read_csv(f).assign(gain_file=f.split("/")[-1]) for f in glob.glob("analysis/out_decomp_testsize/decomposition_*.csv")])
d = d[d.K == K].copy()
d["task"] = np.where(d.gain_file.str.contains("accuracy"), "classification (accuracy)", "regression (MSE)")
rows = []
for (task, t), g in d.groupby(["task", "p_obj_test_size"]):
    rec = {"task": task, "t": t, "n_configs": len(g)}
    for name, rho in (("pop", g.rho_b_gain), ("study", g.rho_sK)):
        c = g.C_obs
        rec[f"f_origin_{name}"] = float((c * rho).sum() / (rho ** 2).sum())
        slope, icpt = np.polyfit(rho, c, 1)
        rec[f"f_slope_{name}"], rec[f"intercept_{name}"] = float(slope), float(icpt)
    fg = (1 / g.G_paper - 1 / K) / ((1 - 1 / K) * g.rho_sK)
    rec["f_from_gain_median"] = float(fg[g.rho_sK > 0.2].median())
    rows.append(rec)
r = pd.DataFrame(rows)
print(r.round(3).to_string(index=False))
print("\nexpected under the overlap argument: f = t (0.1, 0.2, 0.3, 0.5)")
print("\nper configuration (C_obs vs t * rho_b):")
print(d.assign(pred=d.p_obj_test_size * d.rho_b_gain)[["task", "p_obj_train_size", "p_obj_test_size", "solver_name", "rho_b_gain", "rho_sK", "C_obs", "pred", "G_paper"]]
      .sort_values(["task", "p_obj_train_size", "solver_name", "p_obj_test_size"]).round(3).to_string(index=False))
r.to_csv("analysis/out_decomp_testsize/factor_by_test_size.csv", index=False)
