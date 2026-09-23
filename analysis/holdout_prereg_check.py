#!/usr/bin/env python3
"""Pre-registered checks of configs/k200_threshold_validation_holdout_50_*.yml (job 101294).

Rule (header of the YAMLs): S_3 = study_prediction_cov_all_oof_intersection x
study_squared_error_rho_all_oof_intersection at k = 3 (idx_rep = 2); if S_3 > x = 0.03 then
"no large further gain expected". Checks: P(G20 >= 10 | S_3 > 0.03) <= 0.05 and
P(G200 >= 20 | S_3 > 0.03) <= 0.05, G_K the exact configuration-level gains (bootstrap over the
50 seeds). Reported: G20/G5 saturation ratio. Secondary: S_3 / study_target_var, threshold
x_dimless derived on sim_linear only (tolerance 5%: smallest x with P(G20>=10 | S>x) <= 0.05),
validated on sim_interactions and sim_nonlinear.

Usage: python analysis/holdout_prereg_check.py analysis/out_holdout_<loss>   (candidates.csv from
derivation_levers_analysis.py; G_paper = the paper's variance-equivalent gain)."""
import os, sys
import numpy as np, pandas as pd

out = sys.argv[1]
c = pd.read_csv(os.path.join(out, "candidates.csv"))
c["family"] = c["family"].astype(str)
pd.set_option("display.width", 250)
print(f"configurations: {len(c)} | families: {sorted(c.family.unique())} | seeds per config: from gains.csv")
g = pd.read_csv(os.path.join(out, "gains.csv"))
print("n_seeds:", sorted(g.n_seeds.unique()), "| capped G_paper_200:", int(g[g.K == 200].G_paper_capped.sum()), "of", int((g.K == 200).sum()))
for gain in ("G_paper", "G_err"):
    print(f"\n===== gain = {gain} =====")
    stop = c["S_3"] > 0.03
    for K, thr in ((20, 10), (200, 20)):
        large = c[f"{gain}_{K}"] >= thr
        print(f"  PRIMARY  S_3 > 0.03: stop rate {stop.mean():.2f} | P(G{K}>={thr} | stop) = {large[stop].mean():.3f} ({int((stop & large).sum())}/{int(stop.sum())}) -> {'PASS' if large[stop].mean() <= 0.05 else 'FAIL'} | P(large | continue) = {large[~stop].mean():.2f}")
    print("  by family, P(G20>=10 | stop):", {f: round(float((c[(c.family == f) & stop][f'{gain}_20'] >= 10).mean()), 3) for f in sorted(c.family.unique())})
    # saturation ratio, reported not claimed
    print(f"  reported: median G20/G5 = {(c[f'{gain}_20'] / c[f'{gain}_5']).median():.2f}; share of configs gaining >= 25% from K=5 to 20: {((c[f'{gain}_20'] / c[f'{gain}_5']) >= 1.25).mean():.2f}")
    # secondary: dimensionless threshold derived on sim_linear only
    lin = c[c.family == "sim_linear"]
    xs = np.sort(lin["Sdl_3"].dropna().unique())
    x_dl = None
    for x in xs:
        st = lin["Sdl_3"] > x
        if st.sum() >= 5 and (lin.loc[st, f"{gain}_20"] >= 10).mean() <= 0.05:
            x_dl = float(x); break
    print(f"  SECONDARY x_dimless derived on sim_linear: {x_dl}")
    if x_dl is not None:
        for f in ("sim_interactions", "sim_nonlinear"):
            d = c[c.family == f]; st = d["Sdl_3"] > x_dl
            p20 = (d.loc[st, f"{gain}_20"] >= 10).mean() if st.any() else np.nan
            p200 = (d.loc[st, f"{gain}_200"] >= 20).mean() if st.any() else np.nan
            print(f"    validate on {f}: stop rate {st.mean():.2f} | P(G20>=10 | stop) = {p20:.3f} | P(G200>=20 | stop) = {p200:.3f}")
print("\nmedians by family x size:")
print(c.pivot_table(index="family", columns="p_obj_train_size", values=["S_3", "Sdl_3", "G_paper_20", "G_paper_200"], aggfunc="median").round(3).to_string())
