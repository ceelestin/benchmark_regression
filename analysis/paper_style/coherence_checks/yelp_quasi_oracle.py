"""Yelp aligned runs: BERT vs XLM-RoBERTa quasi-oracle and paired tests, two definitions.
(i) first split of each seed (paper Section 5: benchmark score averaged over single-split seeds);
(ii) all 20 splits of each seed (every split's model is benchmark-scored in these runs): per-seed mean over the splits.
Also the collapse counts (benchmark accuracy < 0.25) and the retrieval shares (single split, K = 5, 10, 20)."""
import glob, os
import numpy as np, pandas as pd
from scipy import stats
D = os.path.expandvars("$WORK/paper_style_figs/real_data/data/nlp/yelp_aligned")
d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{D}/*.parquet"))], ignore_index=True)
d = d.sort_values(["model", "study_size", "seed", "fold"])
for n, g in d.groupby("study_size"):
    print(f"\n=== n_tr = {n} ===")
    col = g[g.benchmark_accuracy < 0.25].groupby("model").apply(lambda x: f"{len(x)} splits in {x.seed.nunique()} seeds; first split: {int((x.fold == 1).sum())}")
    print("collapsed (bench acc < 0.25):", col.to_dict())
    first = g[g.fold == 1].pivot(index="seed", columns="model", values="benchmark_accuracy").dropna()
    allm = g.groupby(["seed", "model"]).benchmark_accuracy.mean().unstack().dropna()
    for tag, p in (("(i) first split", first), ("(ii) mean of 20 splits", allm)):
        diff = p["bert"] - p["roberta"]
        t, pv = stats.ttest_rel(p["bert"], p["roberta"])
        w = stats.wilcoxon(p["bert"], p["roberta"]).pvalue
        print(f"{tag:24s} n={len(p)} BERT {p['bert'].mean():.4f} XLM-R {p['roberta'].mean():.4f} D(BERT-XLMR)={diff.mean():+.5f} "
              f"sD={diff.std(ddof=1):.5f} t={t:+.3f} p={pv:.4f} (Wilcoxon p={w:.4f}); BERT higher in {100 * (diff > 0).mean():.0f}% of seeds")
    allmodels = g.groupby("model").benchmark_accuracy.agg(["mean", "median", "std"])
    print("all models (seeds x splits):", allmodels.round(4).to_dict("index"))
    # retrieval: share of seeds where the K-split CV mean test accuracy ranks BERT first
    g = g.assign(cum=g.groupby(["model", "seed"]).test_accuracy.cumsum() / g.fold)
    for K in (1, 5, 10, 20):
        p = g[g.fold == K].pivot(index="seed", columns="model", values="cum").dropna()
        print(f"  K={K:2d}: CV ranks BERT above XLM-R in {100 * (p['bert'] > p['roberta']).mean():.0f}% of {len(p)} seeds "
              f"(ties {100 * (p['bert'] == p['roberta']).mean():.0f}%)")
