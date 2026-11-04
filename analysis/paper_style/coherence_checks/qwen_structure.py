"""Structure of the Qwen3-0.6B Yelp runs: CV and single-split rows, seeds, benchmark scores, outer chunks."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))  # code/datapaths.py
import datapaths
import numpy as np, pandas as pd
D = datapaths.data_path("nlp", "yelp_qwen3_0.6b")
import nlp
print(sorted(os.listdir(D))[:5], len(os.listdir(D)))
d = nlp.load_nlp_data([D])
d["cv"] = nlp._infer_cross_validation(d)
d["n_hidden"] = d["hidden_test_accuracies"].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)
d["bench"] = d["benchmark_accuracy"].apply(nlp._extract_scalar)
print(d.groupby(["cv", "study_size"]).agg(rows=("seed", "size"), seeds=("seed", "nunique"), seed_min=("seed", "min"),
      seed_max=("seed", "max"), n_bench=("bench", lambda x: x.notna().sum()), max_hidden=("n_hidden", "max")).to_string())
print(d.groupby("cv")["_source_file"].apply(lambda x: sorted(set(x))[:2]).to_string())
