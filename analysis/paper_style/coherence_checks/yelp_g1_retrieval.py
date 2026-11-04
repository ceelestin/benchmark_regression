"""Problem 1: Table G.1 Yelp K = 20 CV retrieval (share of seeds where 20-split CV ranks BERT above XLM-RoBERTa), deduplicated vs first-listed."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))  # code/datapaths.py
import datapaths
import numpy as np, pandas as pd
# usage: python yelp_g1_retrieval.py CODE_BEFORE CODE_AFTER (two copies of analysis/paper_style/code, before and
# after the Yelp loader kept one run per seed)
D = datapaths.data_path("nlp")
res = {}
for tag, code in (("first-listed (current)", sys.argv[1]), ("deduplicated", sys.argv[2])):
    sys.path.insert(0, code)
    for m in [k for k in list(sys.modules) if k in ("nlp", "compute", "pipeline_common", "plot")]:
        del sys.modules[m]
    import nlp
    d = nlp.load_nlp_data([os.path.join(D, x) for x in sorted(os.listdir(D))])
    sys.path.pop(0)
    d = d[(d.n_splits_cv == 20) & d.model.isin(["bert", "roberta"])].copy()
    d["k"] = d.groupby(["seed", "model", "study_size"], sort=False).cumcount() + 1
    d = d[d.k <= 20]
    cv = d.groupby(["study_size", "model", "seed"]).study_test_eval_accuracy.mean().unstack("model").dropna()
    res[tag] = (cv["bert"] > cv["roberta"]).groupby("study_size").agg(["mean", "size"])
for tag, r in res.items():
    print(tag); print((r.assign(share=lambda x: (100 * x["mean"]).round(0))[["share", "size"]]).to_string())
