"""Sanity checks of the runs of nlpfinetuning_aligned.py: completeness, scores, collapsed splits, outer chunks,
redundancy columns, recorded runtimes (Table F.2)."""
import glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))  # code/datapaths.py
import datapaths
import numpy as np, pandas as pd
R = sys.argv[1] if len(sys.argv) > 1 else datapaths.nlp_repo_path("results_aligned")
fs = sorted(glob.glob(f"{R}/yelp_results_CV_*.parquet"))
d = pd.concat([pd.read_parquet(f).assign(_f=os.path.basename(f)) for f in fs], ignore_index=True)
print(f"{len(fs)} files, {len(d)} rows")
g = d.groupby(["model", "study_size"])
print(g.agg(seeds=("seed", "nunique"), rows=("seed", "size"), folds_max=("fold", "max"), bench_size=("benchmark_size", "max"),
            pool=("study_pool_size", "max")).to_string())
dup = d.groupby(["model", "study_size", "seed"])["_f"].nunique()
print("seeds with more than one file:", int((dup > 1).sum()))
bad = d.groupby(["model", "study_size", "seed"]).fold.agg(lambda x: sorted(x) == list(range(1, 21)))
print("seeds with folds != 1..20:", int((~bad).sum()))
# scores
print("\nbenchmark accuracy by (model, n_tr): mean, sd over all models; share of collapsed splits (bench acc < 0.25)")
print(g.benchmark_accuracy.agg(["mean", "std", "min", "max", lambda x: (x < 0.25).mean()]).round(4).to_string())
f1 = d[d.fold == 1]
print("\nfirst split: mean bench acc (quasi-oracle), mean test acc, outer chunks")
print(f1.groupby(["model", "study_size"]).agg(bench=("benchmark_accuracy", "mean"), test=("test_accuracy", "mean"),
      chunks=("hidden_test_accuracies", lambda x: x.apply(len).unique().tolist())).round(4).to_string())
print("\nfold > 1 have no outer chunks:", bool(d[d.fold > 1].hidden_test_accuracies.isna().all()))
cols = [c for c in d.columns if "rho" in c or "anova" in c]
print(f"\n{len(cols)} redundancy columns; NaN share at fold 3 of study_error01_rho_cum_oof_intersection:",
      round(float(d[d.fold == 3]["study_error01_rho_cum_oof_intersection"].isna().mean()), 3))
print(d[d.fold == 3].groupby(["model", "study_size"])["study_error01_rho_cum_oof_intersection"].mean().round(4).to_string())
# runtimes (hours)
rt = d.assign(eval=d[["val_runtime", "test_runtime", "benchmark_runtime"]].sum(axis=1) + d["hidden_runtime"].fillna(0))
print("\nrecorded runtimes, hours (train, eval, total) and per seed:")
t = rt.groupby("model").agg(train=("train_runtime", "sum"), eval=("eval", "sum"), seeds=("seed", "size"))
t[["train", "eval"]] /= 3600; t["total"] = t.train + t["eval"]; t["per_seed_h"] = t.total / (t.seeds / 20)
print(t.round(2).to_string())
