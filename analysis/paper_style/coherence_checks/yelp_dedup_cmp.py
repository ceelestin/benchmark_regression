"""Item 13 side check: Yelp gains, current pipeline (v5c) vs deterministic deduplication."""
import os
import pandas as pd
V = os.path.expandvars("$WORK/paper_style_figs/v3")
out = []
for tag, d in (("current", "final_real_v5c"), ("dedup", "final_real_dedup")):
    r = pd.read_parquet(f"{V}/{d}/tables/nlp_delta.parquet")
    est = r[r.bootstrap_iter < 0]; dr = r[r.bootstrap_iter >= 0]
    for (m, n), g in est.groupby(["model", "N"]):
        e = g.set_index("K").sample_gain
        q = dr[(dr.model == m) & (dr.N == n) & (dr.K == 20)].sample_gain.quantile([0.025, 0.975])
        out.append({"run": tag, "model": m, "N": n, "G5": round(e.get(5), 1), "G10": round(e.get(10), 1), "G20": round(e.get(20), 1),
                    "G20 95%": f"[{q.iloc[0]:.1f}, {q.iloc[1]:.1f}]"})
t = pd.DataFrame(out).sort_values(["model", "N", "run"])
print(t.to_string(index=False))
