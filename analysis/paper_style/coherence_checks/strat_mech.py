"""Item 6: is ViT-B/16 near a constant predictor at n_tr = 400? Benchmark (oracle) scores per network, and the spread
of fold-1 test accuracy across seeds (a constant predictor's test accuracy is the class share of the fold)."""
import glob, os
import numpy as np, pandas as pd
D = os.path.expandvars("$WORK/paper_style_figs/real_data/data/pcam/stratandnewmetrics")
cols = ["model", "study_size", "seed", "fold", "stratified", "test_accuracy", "benchmark_accuracy", "benchmark_nll", "benchmark_brier", "test_nll", "val_accuracy"]
d = pd.concat([pd.read_parquet(f, columns=cols) for f in glob.glob(f"{D}/*.parquet")], ignore_index=True)
d = d[d.study_size == 500]
g = d.groupby(["model", "stratified"])
out = pd.DataFrame({
    "bench acc mean": g.benchmark_accuracy.mean(), "bench acc q10": g.benchmark_accuracy.quantile(0.1), "bench acc q90": g.benchmark_accuracy.quantile(0.9),
    "share of fits with bench acc < 0.6": g.benchmark_accuracy.apply(lambda x: (x < 0.6).mean()),
    "bench NLL median": g.benchmark_nll.median(), "bench NLL q90": g.benchmark_nll.quantile(0.9), "bench Brier median": g.benchmark_brier.median(),
    "sd test acc (fold 1)": d[d.fold == 1].groupby(["model", "stratified"]).test_accuracy.std(),
}).round(3)
print(out.to_string())
# binomial sd of the class share of a 100-image fold, for reference
print("binomial sd of the class share in a 100-image test fold:", round(np.sqrt(0.25 / 100), 3))
