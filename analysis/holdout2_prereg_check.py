#!/usr/bin/env python3
"""Pre-registered checks of HOLDOUT 2 (analysis/PREREGISTRATION_f_rule_holdout2.md, job 121758).

Usage:
    python analysis/holdout2_prereg_check.py analysis/out_holdout2_mse analysis/out_holdout2_acc
(candidates.csv / gains.csv produced by derivation_levers_analysis.py with --loss neg_mse on the
three regression families and --loss accuracy on sim_classification).

Rule: STOP at k=3 iff rho_3 > 0.2631 (rho = OOF-intersection ICC of the matched loss;
<=> closed-form G_20(1 - rho_3) < 10, t = 0.2). Checks C1 P(G_20>=10 | stop) <= 0.05,
C1' P(G_200>=20 | stop) <= 0.05, C2 P(G_20(f_hat_3) >= G_20^paper) >= 0.75, power statement
base rate P(G_20>=10) >= 0.25. Comparisons: ANOVA-ICC form, S_3 > 0.03 rule, readings at k=5/20."""
import os
import sys

import numpy as np
import pandas as pd

T = 0.2
RHO_STAR = 0.2631


def G(f, K, t=T):
    return 1.0 / ((1 - f) * (t + (1 - t) / K) + f / K)


def check(c, rho, icc, label):
    print(f"\n########## {label}: {len(c)} configurations, families {sorted(c.family.unique())} ##########")
    large20, large200 = c["G_paper_20"] >= 10, c["G_paper_200"] >= 20
    print(f"power: base rate P(G_20>=10) = {large20.mean():.2f} ({int(large20.sum())}/{len(c)}), P(G_200>=20) = {large200.mean():.2f}"
          f" -> {'informative' if large20.mean() >= 0.25 else 'UNDER-POWERED (< 0.25)'}")
    for k in (3, 5, 20):
        r = c[f"{rho}_{k}"]
        stop = r > RHO_STAR
        pred = G(1 - c[f"{rho}_3"], 20)
        c1 = large20[stop].mean() if stop.any() else np.nan
        c1p = large200[stop].mean() if stop.any() else np.nan
        line = (f"  k={k:2d} rho-rule: stop rate {stop.mean():.2f} | C1 P(G20>=10|stop) = {c1:.3f} ({int((stop & large20).sum())}/{int(stop.sum())}) {'PASS' if c1 <= 0.05 else 'FAIL'}"
                f" | C1' P(G200>=20|stop) = {c1p:.3f} {'PASS' if c1p <= 0.05 else 'FAIL'} | P(large|continue) = {large20[~stop].mean() if (~stop).any() else np.nan:.2f}")
        if k == 3:
            c2 = (pred >= c["G_paper_20"]).mean()
            line += f" | C2 P(pred>=obs) = {c2:.2f} {'PASS' if c2 >= 0.75 else 'FAIL'} | median log(pred/obs) = {np.log(pred / c['G_paper_20']).median():+.2f}"
        print(line)
    from scipy.stats import spearmanr
    for name in (f"{rho}_3", f"{icc}_3", "S_3", "Sdl_3"):
        if name in c and c[name].notna().any():
            ok = c[[name, "G_paper_20"]].dropna()
            print(f"  Spearman({name}, G_20) = {spearmanr(ok.iloc[:, 0], ok.iloc[:, 1]).statistic:+.2f}", end="")
    print()
    if f"{icc}_3" in c and c[f"{icc}_3"].notna().any():
        stop = (1 - c[f"{icc}_3"]) < 1 - RHO_STAR
        print(f"  comparison ANOVA-ICC form at k=3: stop rate {stop.mean():.2f} | P(G20>=10|stop) = {large20[stop].mean() if stop.any() else np.nan:.3f}")
    stop = c["S_3"] > 0.03
    print(f"  comparison S_3 > 0.03: stop rate {stop.mean():.2f} | P(G20>=10|stop) = {large20[stop].mean() if stop.any() else np.nan:.3f} | P(large|continue) = {large20[~stop].mean() if (~stop).any() else np.nan:.2f}")
    c["lev"] = c["solver"] + c["lever"].fillna("").map(lambda s: " " + s if s else "")
    pd.set_option("display.width", 250)
    print(c.pivot_table(index="lev", columns="p_obj_train_size", values=["G_paper_20", f"{rho}_3"], aggfunc="median").round(2).to_string())
    print("  reported: median G20/G5 =", round(float((c["G_paper_20"] / c["G_paper_5"]).median()), 2))


def main():
    for out, rho, icc, label in ((sys.argv[1], "rho_e", "icc_sq", "REGRESSION families (MSE)"),
                                 (sys.argv[2], "rho_err01", "icc_err01", "sim_classification (0-1)")):
        c = pd.read_csv(os.path.join(out, "candidates.csv"))
        c = c[(c.p_obj_procedure == "ShuffleSplit") & (c.p_obj_fixed_split.astype(str) != "True")]
        check(c, rho, icc, label)
        if label.startswith("REGRESSION"):
            for fam, g in c.groupby("family"):
                stop = g["rho_e_3"] > RHO_STAR
                print(f"    {fam}: base rate {(g.G_paper_20 >= 10).mean():.2f} | stop rate {stop.mean():.2f} | C1 = {(g.G_paper_20[stop] >= 10).mean() if stop.any() else np.nan:.3f}")


if __name__ == "__main__":
    main()
