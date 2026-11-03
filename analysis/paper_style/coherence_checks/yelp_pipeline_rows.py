"""Item 13 side check: which duplicated Yelp run feeds the K-split side in nlp.py (Jean-Zay file order)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expandvars("$WORK/paper_style_figs/v3/code"))
import nlp
D = os.path.expandvars("$WORK/paper_style_figs/real_data/data/nlp")
dirs = [os.path.join(D, x) for x in sorted(os.listdir(D))]
files = nlp._find_parquet_files(dirs)
fb = [f for f in files if "cv_models_bert_nsplits_20_seeds_0-0_sizes_1250_" in f]
print("file order for BERT seed 0 size 1250 (K=20):", [os.path.basename(f)[-24:] for f in fb])
d = nlp.load_nlp_data(dirs)
d = d[(d.model == "bert") & (d.study_size == 1250) & (d.n_splits_cv == 20)].copy()
g = d.groupby(["seed", "model", "study_size", "cross_validation"], sort=False)
d["n_splits"] = g.cumcount() + 1
for s in (0, 5, 25):
    e = d[d.seed == s]
    first20 = e[e.n_splits <= 20]
    print(f"seed {s}: rows {len(e)}, first 20 rows: benchmark non-NaN {int(first20.benchmark_accuracy.notna().sum())}/20")
ok = d[d.n_splits <= 20].groupby("seed").benchmark_accuracy.apply(lambda x: x.notna().all())
print("seeds whose first 20 rows all have a benchmark score:", int(ok.sum()), "->", sorted(ok[ok].index.tolist())[:25])
