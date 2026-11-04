"""Structure of the Qwen3-0.6B Yelp runs: CV and single-split rows, seeds, benchmark scores, outer chunks."""
import os, sys
import numpy as np, pandas as pd
V = os.path.expandvars("$WORK/paper_style_figs/v3")
D = os.path.expandvars("$WORK/paper_style_figs/real_data/data/nlp/yelp_qwen3_0.6b")
sys.path.insert(0, f"{V}/code")
import nlp
print(sorted(os.listdir(D))[:5], len(os.listdir(D)))
d = nlp.load_nlp_data([D])
d["cv"] = nlp._infer_cross_validation(d)
d["n_hidden"] = d["hidden_test_accuracies"].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)
d["bench"] = d["benchmark_accuracy"].apply(nlp._extract_scalar)
print(d.groupby(["cv", "study_size"]).agg(rows=("seed", "size"), seeds=("seed", "nunique"), seed_min=("seed", "min"),
      seed_max=("seed", "max"), n_bench=("bench", lambda x: x.notna().sum()), max_hidden=("n_hidden", "max")).to_string())
print(d.groupby("cv")["_source_file"].apply(lambda x: sorted(set(x))[:2]).to_string())
