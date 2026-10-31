"""Text numbers from the all-seeds estimate (CE, 2026-10-31)."""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style_figs as F
V = os.path.expandvars("$WORK/paper_style_figs/v3")
est = lambda r: r[r.bootstrap_iter < 0]
# (a) App M
m = est(pd.read_parquet(f"{V}/final_sim_v5/tables/figM1.parquet"))
t = m[m.K.isin([5, 20])].pivot_table(index=["N", "solver_disp"], columns="K", values="sample_gain")
t["ratio"] = t[20] / t[5]; print(t.round(2).unstack("solver_disp").to_string())
print("ratio range", round(t.ratio.min(), 2), round(t.ratio.max(), 2), "| G20 range", round(t[20].min(), 2), round(t[20].max(), 2))
sm = t.reset_index(); print("G20 N<=10:", round(sm[sm.N <= 10][20].min(), 2), round(sm[sm.N <= 10][20].max(), 2))
print(sm[sm.N >= 20].groupby("solver_disp")[20].agg(["min", "max"]).round(2).to_string())
w = pd.read_csv(f"{V}/results_v4d/small_redundancy_k5.csv", index_col=0)
w = w[[c for c in w.columns if "64, 64" not in c]].rename(columns=lambda c: F.T.display_name(c)); w.index = w.index.astype(int)
kap = {}
for (n, l), r in t.iterrows():
    if l in w.columns: kap[(n, l)] = (20 / r[20] - 1) / 19 - w.loc[n, l]
k = pd.Series(kap).unstack(); print(k.round(3).to_string()); print("mean kappa by N:", k.mean(axis=1).round(3).to_dict())
# (b) App I: random-search MLP without its worst-fitting seeds
fr = F.read(sorted(glob.glob(f"{F.OUT}/hpo_mse_MLP_randomsearch__c*.parquet")))
d = pd.concat(fr); mse = -d.objective_neg_mse_test; bad = set(d[mse > 10 * mse.median()].p_dataset_seed.astype(int))
for lab, frames in (("all seeds", fr), (f"without {len(bad)} seeds", [x[~x.p_dataset_seed.astype(int).isin(bad)] for x in fr])):
    r = F.gains(F.prepare(frames), 0, 0.20, plugin=True); print("App I random-search MLP", lab, "G20 =", round(float(r[r.K == 20].sample_gain.iloc[0]), 2))
# (c) App L: MAE
for lab, files, solver in (("MLP(32)", sorted(glob.glob(f"{F.OUT}/paper_main_gain_mse_k200__c*.parquet")), F.MLP32),
                           ("RandomFeatures", sorted(glob.glob(f"{F.OUT}/paper_main_gain_mse_k200_rff50__c*.parquet")), None)):
    import pyarrow.parquet as pq
    names = pq.read_schema(files[0]).names
    for met in ("neg_mse", "neg_mae"):
        F.T.TEST, F.T.BENCH, F.T.OUTER = (f"objective_{met}_test", f"objective_{met}_bench", f"objective_outer_{met}")
        keep = [x for x in F.T.COLS if x in names] + [x for x in (F.T.TEST, F.T.BENCH, F.T.OUTER) if x not in F.T.COLS]
        fr2 = [pd.read_parquet(f, columns=keep) for f in files]
        fr2 = [x[(x.idx_rep < 20) & ((x.solver_name == solver) if solver else True)] for x in fr2]
        r = F.gains(F.prepare(fr2), 0, 0.20, plugin=True)
        print(f"App L {lab} {met}: G20 = {float(r[r.K == 20].sample_gain.iloc[0]):.2f}", flush=True)
