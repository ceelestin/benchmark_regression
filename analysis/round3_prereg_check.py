#!/usr/bin/env python3
"""Round-3 pre-registered checks (analysis/PREREGISTRATION_round3.md).

Usage:
    python analysis/round3_prereg_check.py <candidates.csv> <rho column prefix> [label] [--k200]
e.g.  python analysis/round3_prereg_check.py analysis/out_holdout3_mse/candidates.csv rho_e "holdout3 regression" --k200
      python analysis/round3_prereg_check.py ../benchmark_pcam/analysis/out_round3/candidates_accuracy.csv rho_nll "PCam round3, gain on accuracy, statistic NLL"

The rho column at fold k is "<prefix>_<k>" (benchmark_regression candidates: rho_e_3, rho_nll_3,
rho_err01_3; PCam candidates from pcam_gain_analysis.py --rho cum: rho_nll_3, rho_sq_3, rho_err01_3).
Rule: STOP at k=3 iff rho_3 > 0.3488 (<=> closed-form G_20 < 8.6, t = 0.2).
Checks: C1 P(G_20>=10 | stop) <= 0.05; C1' P(G_200>=20 | stop) <= 0.05 (--k200);
C2' |median log(pred/obs)| <= 0.15; C3 P(G_20>=10 | continue) >= 0.50; power: base rate >= 0.25."""
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

T, RHO_STAR, TARGET, STOP_LEVEL = 0.2, 0.3488, 10.0, 8.6


def G(f, K, t=T):
    return 1.0 / ((1 - f) * (t + (1 - t) / K) + f / K)


def main():
    path, rho = sys.argv[1], sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else path
    k200 = "--k200" in sys.argv
    c = pd.read_csv(path)
    if "p_obj_procedure" in c:
        c = c[(c.p_obj_procedure == "ShuffleSplit") & (c.p_obj_fixed_split.astype(str) != "True")]
    c = c[c[f"{rho}_3"].notna()]
    large = c["G_paper_20"] >= TARGET
    print(f"===== {label}: {len(c)} configurations, rule rho_3 > {RHO_STAR} (pred G_20 < {STOP_LEVEL}) =====")
    base = large.mean()
    print(f"power: base rate P(G_20>={TARGET:g}) = {base:.2f} ({int(large.sum())}/{len(c)}) -> {'informative' if base >= 0.25 else 'UNDER-POWERED'}")
    for k in (3, 5, 20):
        col = f"{rho}_{k}"
        if col not in c or c[col].isna().all():
            continue
        stop = c[col] > RHO_STAR
        c1 = large[stop].mean() if stop.any() else np.nan
        c3 = large[~stop].mean() if (~stop).any() else np.nan
        line = f"  k={k:2d}: stop rate {stop.mean():.2f} | C1 P(large|stop) = {c1:.3f} ({int((stop & large).sum())}/{int(stop.sum())}) {'PASS' if c1 <= 0.05 else 'FAIL'}"
        if k == 3:
            pred = G(1 - c[col], 20)
            mlr = np.log(pred / c["G_paper_20"]).median()
            line += f" | C2' median log(pred/obs) = {mlr:+.3f} {'PASS' if abs(mlr) <= 0.15 else 'FAIL'} | C3 P(large|continue) = {c3:.2f} {'PASS' if c3 >= 0.5 else 'FAIL'}"
            if k200 and "G_paper_200" in c:
                l200 = c["G_paper_200"] >= 20
                c1p = l200[stop].mean() if stop.any() else np.nan
                line += f" | C1' P(G_200>=20|stop) = {c1p:.3f} {'PASS' if c1p <= 0.05 else 'FAIL'}"
        else:
            line += f" | P(large|continue) = {c3:.2f}"
        print(line)
    ok = c[[f"{rho}_3", "G_paper_20"]].dropna()
    print(f"  Spearman({rho}_3, G_20) = {spearmanr(ok.iloc[:, 0], ok.iloc[:, 1]).statistic:+.2f} (n={len(ok)})")
    if "G_paper_5" in c:
        print(f"  reported: median G_20/G_5 = {(c['G_paper_20'] / c['G_paper_5']).median():.2f}")
    stop = c[f"{rho}_3"] > RHO_STAR
    if (stop & large).any():
        keep = [k for k in ("model", "solver", "lever", "train_size", "p_obj_train_size", "p_dataset_noise", "G_paper_20", "G_paper_200", f"{rho}_3") if k in c]
        print("  stopped-but-large configurations:")
        print(c.loc[stop & large, keep].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
