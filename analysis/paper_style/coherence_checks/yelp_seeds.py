"""Item 8: seeds per Yelp run set (CE copy real_data/data/nlp)."""
import glob, os
import pandas as pd, pyarrow.parquet as pq
D = os.path.expandvars("$WORK/paper_style_figs/real_data/data/nlp")
rows = []
for f in glob.glob(f"{D}/**/*.parquet", recursive=True):
    names = pq.read_schema(f).names
    cols = [c for c in ("model_arg_choice", "study_set_size", "n_splits_cv", "seed", "cross_validation", "fold_index") if c in names]
    d = pd.read_parquet(f, columns=cols)
    d["dir"] = os.path.relpath(f, D).split(os.sep)[0]
    d["cv_in_name"] = "NoCV" not in os.path.basename(f)
    rows.append(d)
d = pd.concat(rows, ignore_index=True)
m = "model_arg_choice"
s = "study_set_size"
g = d.groupby(["dir", m, s, "n_splits_cv", "cv_in_name"]).agg(seeds=("seed", "nunique"), smin=("seed", "min"), smax=("seed", "max"), rows=("seed", "size")).reset_index()
print(g.to_string(index=False))
