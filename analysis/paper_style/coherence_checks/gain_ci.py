"""Estimate (all seeds) and 95% percentile bootstrap interval of the sample gain at K = 2, 5, 10, 20, per model and
size, from a gain table (bootstrap_iter = -1: estimate; >= 0: draws), as in ci_appendix.py."""
import sys
import numpy as np, pandas as pd
t = pd.read_parquet(sys.argv[1])
rows = []
for (m, n), g in t.groupby(["model", "train_size"]):
    r = {"model": m, "n_tr": n}
    for K in (2, 5, 10, 20):
        d = g[g.K == K]
        e, b = d[d.bootstrap_iter < 0].sample_gain, d[d.bootstrap_iter >= 0].sample_gain
        if e.empty or b.empty:
            continue
        lo, hi = np.quantile(b, [.025, .975])
        r[f"K={K}"] = f"{float(e.iloc[0]):.1f} [{lo:.1f}, {hi:.1f}]" + (" floor" if (b <= 1.0001).mean() >= .025 else "")
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
