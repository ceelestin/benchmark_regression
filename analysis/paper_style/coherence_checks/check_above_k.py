"""Problem 2: verify the gains above K in Figure 6 and the noise-injection grid's implied fold-error correlation."""
import glob, os
import numpy as np, pandas as pd
A = os.path.expandvars("$WORK/benchmark_regression/analysis")
d = pd.concat([pd.read_csv(f"{A}/out_omega/omega_{s}_neg_mse.csv").assign(design=s) for s in ("derivation", "holdout1", "holdout2", "holdout3")], ignore_index=True)
print("Figure 6 simulated configs:", len(d), "| G_paper(K=20) > 20:", int((d.G_paper > 20).sum()), "| max:", round(d.G_paper.max(), 1))
print((d[d.G_paper > 20].groupby(["design", "solver_name"]).size()).to_string())
f = sorted(glob.glob(f"{A}/out_decomp_*/decomposition_neg_mse_stat_squared_error.csv"))
print("decomposition files:", [os.path.relpath(x, A) for x in f])
x = pd.concat([pd.read_csv(p).assign(src=os.path.relpath(p, A).split("/")[0]) for p in f], ignore_index=True)
print("columns:", [c for c in x.columns if "G" in c or "rho" in c or c in ("K", "solver_name")][:20])
k20 = x[x.K == 20] if "K" in x.columns else x
gt = [c for c in k20.columns if c.lower().startswith("g_true")][0]
hi = k20[k20[gt] > 20]
print(f"{gt} > 20 at K=20:", len(hi), "| by source:", hi.groupby("src").size().to_dict())
print("by solver:", hi.groupby(hi.solver_name.str.split("[").str[0]).size().to_dict())
rd = (20 / hi[gt] - 1) / 19
print("implied rho_delta: min %.3f median %.3f max %.3f" % (rd.min(), rd.median(), rd.max()))
