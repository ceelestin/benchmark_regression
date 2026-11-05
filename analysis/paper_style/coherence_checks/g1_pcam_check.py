"""Table M.1 PCam rows, every column: D-bar, s_D, t, p, single-split retrieval and K = 20 CV retrieval.

D-bar and s_D are the mean and standard deviation of the same per-seed differences. The paired t-test on the
benchmarking accuracy (D-bar, s_D, t, p) and the single-split retrieval come from the January 2026 single-split runs
(100 seeds; a sixth of each study set for testing, a sixth for validation), which also give the direction of each pair
(higher mean benchmarking accuracy first). The K = 20 CV retrieval comes from the March 2026 20-split MCCV
(shuffle_split) runs (50 seeds; test fold a fifth): the test accuracy averaged over the first 20 splits. A retrieval
share counts the seeds in which the displayed 'highest mean' network has the strictly higher test accuracy, so ties
count as failures. The study sets n = 30, 100, 300, 500, 800 printed here are the table's n_tr = 25, 83, 250, 416, 666.
"""
import argparse, glob, itertools, os
import numpy as np, pandas as pd
from scipy import stats

K = 20
ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--nocv-dir", default=os.path.expandvars("$WORK/paper_style_figs/v3/item4_data/pcam_nocv/results_output_pcam"),
                help="January single-split runs (default: the permanent copy; $SCRATCH/pcam_nocv_jan/results_output_pcam "
                     "holds the same files)")
ap.add_argument("--cv-dir", default=os.path.expandvars("$WORK/paper_style_figs/real_data/data/pcam/results_output_pcam"),
                help="March 20-split runs; only their MCCV (shuffle_split) files are read")
a = ap.parse_args()

D = a.nocv_dir
fs = sorted(glob.glob(f"{D}/*.parquet"))
d0 = pd.read_parquet(fs[0]); print(d0.columns.tolist()); print(d0.iloc[0].to_dict())
d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
size = "study_size" if "study_size" in d.columns else [c for c in d.columns if "size" in c][0]
bench = [c for c in d.columns if c.startswith("benchmark") and "acc" in c][0]
test = [c for c in d.columns if "test" in c and "acc" in c and "hidden" not in c][0]
print("columns used:", size, bench, test, "| rows", len(d), "| per model/size", d.groupby(["model", size]).seed.nunique().unique())

cfs = sorted(glob.glob(f"{a.cv_dir}/*_CV_shuffle_split_*.parquet"))
cv = pd.concat([pd.read_parquet(f, columns=["model", "study_size", "seed", "fold", "cv_method", "test_accuracy"]) for f in cfs],
               ignore_index=True)
cv = cv[cv.model.isin(d.model.unique())]
assert (cv.cv_method == "shuffle_split").all() and not cv.duplicated(["model", "study_size", "seed", "fold"]).any()
run = ["model", "study_size", "seed"]
assert cv.groupby(run).size().min() >= K, f"a run has fewer than {K} splits"
cvk = cv.sort_values("fold").groupby(run).head(K).groupby(["study_size", "seed", "model"]).test_accuracy.mean().unstack("model")
print(f"K={K} MCCV runs: {cv.groupby(run).ngroups} runs of {sorted(cv.model.unique())} in {a.cv_dir} "
      f"| splits per run {cv.groupby(run).size().unique()} "
      f"| seeds per model/size {cv.groupby(['model', 'study_size']).seed.nunique().unique()}")

for n, sub in d.groupby(size):
    b = sub.pivot_table(index="seed", columns="model", values=bench)
    t_ = sub.pivot_table(index="seed", columns="model", values=test)
    c_ = cvk.xs(n, level="study_size")
    for m1, m2 in itertools.combinations(sorted(b.columns), 2):
        hi, lo = (m1, m2) if b[m1].mean() >= b[m2].mean() else (m2, m1)
        q = b[[hi, lo]].dropna(); diff = q[hi] - q[lo]
        tt, p = stats.ttest_rel(q[hi], q[lo])
        r = t_[[hi, lo]].dropna()
        c = c_[[hi, lo]].dropna()
        print(f"n={n:4d} {hi:18s} > {lo:18s} n_seeds={len(q):3d} Dbar={diff.mean():.5f} sD={diff.std(ddof=1):.5f} t={tt:.3f} p={p:.4f} "
              f"single-split retrieval {100 * (r[hi] > r[lo]).mean():.0f}% "
              f"K={K} MCCV retrieval {100 * (c[hi] > c[lo]).mean():.0f}% ({len(c)} seeds)")
