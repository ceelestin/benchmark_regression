"""Share of seeds in which a single split, with its test fold extended by m - 1 outer chunks, ranks BERT above
XLM-RoBERTa (n_tr = 1,000 and 3,000), for m = 1, 2, 5, 10, 20, 61."""
import glob, os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))  # code/datapaths.py
import datapaths
import numpy as np, pandas as pd
D = datapaths.data_path("nlp", "yelp_aligned")
d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{D}/*.parquet"))], ignore_index=True)
f1 = d[d.fold == 1]
for n, g in f1.groupby("study_size"):
    out = []
    for m in (1, 2, 5, 10, 20, 61):
        s = g.apply(lambda r: (r.test_accuracy + np.sum(r.hidden_test_accuracies[: m - 1])) / m, axis=1)
        p = g.assign(s=s).pivot(index="seed", columns="model", values="s").dropna()
        out.append(f"m={m}: BERT above {100 * (p.bert > p.roberta).mean():.0f}%, XLM-R above {100 * (p.roberta > p.bert).mean():.0f}%")
    print(f"n_tr={n}: " + "; ".join(out))
