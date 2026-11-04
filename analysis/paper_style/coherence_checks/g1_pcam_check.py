"""Table M.1 PCam rows: is D-bar the mean of the same per-seed differences whose standard deviation is s_D?"""
import glob, itertools, os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))  # code/datapaths.py
import datapaths
import numpy as np, pandas as pd
from scipy import stats
D = datapaths.data_path("pcam", "nocv_jan")
fs = sorted(glob.glob(f"{D}/*.parquet"))
d0 = pd.read_parquet(fs[0]); print(d0.columns.tolist()); print(d0.iloc[0].to_dict())
d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
size = "study_size" if "study_size" in d.columns else [c for c in d.columns if "size" in c][0]
bench = [c for c in d.columns if c.startswith("benchmark") and "acc" in c][0]
test = [c for c in d.columns if "test" in c and "acc" in c and "hidden" not in c][0]
print("columns used:", size, bench, test, "| rows", len(d), "| per model/size", d.groupby(["model", size]).seed.nunique().unique())
for n, sub in d.groupby(size):
    b = sub.pivot_table(index="seed", columns="model", values=bench)
    t_ = sub.pivot_table(index="seed", columns="model", values=test)
    for m1, m2 in itertools.combinations(sorted(b.columns), 2):
        hi, lo = (m1, m2) if b[m1].mean() >= b[m2].mean() else (m2, m1)
        q = b[[hi, lo]].dropna(); diff = q[hi] - q[lo]
        tt, p = stats.ttest_rel(q[hi], q[lo])
        r = t_[[hi, lo]].dropna()
        print(f"n={n:4d} {hi:18s} > {lo:18s} n_seeds={len(q):3d} Dbar={diff.mean():.5f} sD={diff.std(ddof=1):.5f} t={tt:.3f} p={p:.4f} "
              f"single-split retrieval {100 * (r[hi] > r[lo]).mean():.0f}%")
