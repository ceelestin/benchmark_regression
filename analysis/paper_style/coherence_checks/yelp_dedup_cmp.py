"""Yelp gains before and after the loader kept one run per seed (deterministic deduplication)."""
import os, sys
import pandas as pd
# usage: python yelp_dedup_cmp.py OUT_BEFORE OUT_AFTER (the --out-dir of two fig5_yelp runs of
# regen_real_data_gain_figures.py, before and after the loader kept one run per seed)
out = []
for tag, d in (("current", sys.argv[1]), ("dedup", sys.argv[2])):
    r = pd.read_parquet(f"{d}/tables/nlp_delta.parquet")
    est = r[r.bootstrap_iter < 0]; dr = r[r.bootstrap_iter >= 0]
    for (m, n), g in est.groupby(["model", "N"]):
        e = g.set_index("K").sample_gain
        q = dr[(dr.model == m) & (dr.N == n) & (dr.K == 20)].sample_gain.quantile([0.025, 0.975])
        out.append({"run": tag, "model": m, "N": n, "G5": round(e.get(5), 1), "G10": round(e.get(10), 1), "G20": round(e.get(20), 1),
                    "G20 95%": f"[{q.iloc[0]:.1f}, {q.iloc[1]:.1f}]"})
t = pd.DataFrame(out).sort_values(["model", "N", "run"])
print(t.to_string(index=False))
