"""Which data reproduce Table G.1's Yelp paired t-test columns (D-bar, s_D, t, p)?"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))  # code/datapaths.py
import datapaths
import numpy as np, pandas as pd
from scipy import stats
D = datapaths.data_path("nlp")
import nlp
d = nlp.load_nlp_data([os.path.join(D, x) for x in ("yelp_runs_a", "yelp_runs_b")])
d = d[d.model.isin(["bert", "roberta"])].copy()
d["cv"] = nlp._infer_cross_validation(d)
for col in ("benchmark_accuracy", "study_test_eval_accuracy"):
    d[col] = d[col].apply(nlp._extract_scalar)
print(d.groupby(["cv", "model", "study_size"]).agg(n=("seed", "nunique"), nb=("benchmark_accuracy", lambda x: x.notna().sum())).to_string())
for cv, sub in d.groupby("cv"):
    first = sub.sort_values("fold").groupby(["model", "study_size", "seed"]).head(1) if cv else sub
    for col in ("benchmark_accuracy", "study_test_eval_accuracy"):
        p = first.pivot_table(index=["study_size", "seed"], columns="model", values=col).dropna()
        for n, q in p.groupby(level=0):
            diff = q["bert"] - q["roberta"]
            t, pv = stats.ttest_rel(q["bert"], q["roberta"])
            print(f"cv={cv} {col:26s} N={n:6d} n={len(q):3d} Dbar={diff.mean():.5f} sD={diff.std(ddof=1):.5f} t={t:.3f} p={pv:.4f} "
                  f"bert>roberta {100 * (diff > 0).mean():.0f}%")
