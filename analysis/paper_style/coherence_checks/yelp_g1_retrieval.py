"""Problem 1: Table G.1 Yelp K = 20 CV retrieval (share of seeds where 20-split CV ranks BERT above XLM-RoBERTa), deduplicated vs first-listed."""
import os, sys
import numpy as np, pandas as pd
V = os.path.expandvars("$WORK/paper_style_figs/v3")
D = os.path.expandvars("$WORK/paper_style_figs/real_data/data/nlp")
res = {}
for tag, code in (("first-listed (current)", f"{V}/code"), ("deduplicated", f"{V}/code_dedup")):
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
