"""Coverage of the 95% percentile bootstrap intervals of G_K (CE, 2026-10-30).

Figure 4 data (n_tr = 1,000, 1,000 seeds; Ridge, ExtraTrees, MLP (32), RandomFeatures; K <= 20). Reference:
the gain computed on all 1,000 seeds (no resampling). Seeds are split into disjoint groups of m (50 = real-data
size, 200); in each group, the full-group estimate and its 95% percentile interval (B = 1,000, as in the figures)
are computed, and we count how often the interval contains the reference.
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style_figs as F
import compute

class Identity:                                   # "resampling" that returns the sample itself -> plug-in estimate
    def choice(self, a, size=None, replace=True):
        a = np.arange(a) if np.isscalar(a) else np.asarray(a)
        return a if size is None else a[:size]
_real_rng = np.random.default_rng
def plugin(data):
    compute.np.random.default_rng = lambda *_a, **_k: Identity()
    try:
        return F.gains(data, 1, 0.20)
    finally:
        compute.np.random.default_rng = _real_rng

KS = [5, 10, 20]
M6464 = "MLP[hidden_layer_sizes=[64, 64]]"
fr = F.read(sorted(glob.glob(f"{F.OUT}/paper_main_gain_mse_k200__c*.parquet")), drop={"GradientBoosting[n_estimators=200]"}) \
    + F.read(sorted(glob.glob(f"{F.OUT}/paper_main_gain_mse_k200_rff50__c*.parquet")))
fr = [f[f.idx_rep < 20] for f in fr]
data = F.prepare(fr)
lab = lambda r: r.assign(learner=r.solver_name.map(F.T.display_name))
ref = lab(plugin(data)); ref = ref[ref.K.isin(KS)].set_index(["learner", "K"]).sample_gain
print("reference (all seeds):\n", ref.round(2).to_string(), flush=True)
seeds = np.sort(data.seed.unique()); np.random.default_rng(0).shuffle(seeds)
rows = []
for m in (50, 200):
    for g, grp in enumerate(np.array_split(seeds, len(seeds) // m)):
        sub = data[data.seed.isin(grp)]
        est = lab(plugin(sub)); est = est[est.K.isin(KS)].set_index(["learner", "K"]).sample_gain
        bt = lab(F.gains(sub, 1000, 0.20)); bt = bt[bt.K.isin(KS)]
        q = bt.groupby(["learner", "K"]).sample_gain.agg(lo=lambda x: x.quantile(.025), hi=lambda x: x.quantile(.975),
                                                        mean="mean", at1=lambda x: (x <= 1.0001).mean())
        for (l, K), r in q.iterrows():
            rows.append({"m": m, "group": g, "learner": l, "K": K, "ref": ref[(l, K)], "est": est[(l, K)],
                         "lo": r.lo, "hi": r.hi, "boot_mean": r["mean"], "share_at_1": r.at1})
        print(f"m={m} group {g} done", flush=True)
d = pd.DataFrame(rows)
d["covered"] = (d.lo <= d.ref) & (d.ref <= d.hi)
d["miss_low"] = d.hi < d.ref; d["miss_high"] = d.lo > d.ref
out = os.path.expandvars("$WORK/paper_style_figs/v3/ci_coverage.csv"); d.to_csv(out, index=False)
print(d.groupby(["m", "learner", "K"])[["covered", "miss_low", "miss_high"]].mean().round(2).to_string())
print(d.groupby("m")[["covered", "miss_low", "miss_high"]].agg(["mean", "sum", "count"]).to_string())
d["rel_err_est"] = d.est / d.ref - 1; d["rel_err_bmean"] = d.boot_mean / d.ref - 1
print(d.groupby(["m", "learner"])[["rel_err_est", "rel_err_bmean"]].median().round(3).to_string())
