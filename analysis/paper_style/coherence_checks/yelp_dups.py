"""Item 13 side check: duplicated Yelp K = 20 runs (same model, size, seed in two files)."""
import glob, os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))  # code/datapaths.py
import datapaths
import pandas as pd
D = datapaths.data_path("nlp")
rows = []
for f in glob.glob(f"{D}/**/*.parquet", recursive=True):
    d = pd.read_parquet(f, columns=["model_arg_choice", "study_set_size", "n_splits_cv", "seed", "fold_index", "benchmark_eval_accuracy", "study_test_eval_accuracy"])
    d = d[d.n_splits_cv == 20]
    if len(d): d["file"] = os.path.relpath(f, D); rows.append(d)
d = pd.concat(rows, ignore_index=True)
g = d.groupby(["model_arg_choice", "study_set_size", "seed"]).file.nunique()
dup = g[g > 1].reset_index()
print("duplicated (model, size, seed):", len(dup), "| seeds range:", dup.seed.min(), "-", dup.seed.max(), "| sizes:", sorted(dup.study_set_size.unique()))
print("of which seeds < 20 (benchmark-scored):", int((dup.seed < 20).sum()))
ex = dup.iloc[0]
e = d[(d.model_arg_choice == ex.model_arg_choice) & (d.study_set_size == ex.study_set_size) & (d.seed == ex.seed)]
print("example:", ex.model_arg_choice, ex.study_set_size, ex.seed)
print(e.groupby("file").agg(folds=("fold_index", "nunique"), bench=("benchmark_eval_accuracy", lambda x: int(x.notna().sum())), test_mean=("study_test_eval_accuracy", "mean")).to_string())
same = []
for _, r in dup.iterrows():
    e = d[(d.model_arg_choice == r.model_arg_choice) & (d.study_set_size == r.study_set_size) & (d.seed == r.seed)]
    v = e.groupby("file").study_test_eval_accuracy.apply(lambda x: tuple(x.round(6)))
    same.append(len(set(v)) == 1)
print("duplicates with identical fold scores:", sum(same), "of", len(same))
