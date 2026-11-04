"""Item 9: PCam n_tr = 24 (August batch): redundancy omega = t * mean_seed rho (OOF intersection, all 20 splits) per network x loss."""
import glob, os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))  # code/datapaths.py
import datapaths
import numpy as np, pandas as pd
D = datapaths.data_path("pcam", "ntr24")
RC = {"accuracy": "study_error01_rho_all_oof_intersection", "nll": "study_nll_rho_all_oof_intersection", "brier": "study_squared_error_rho_all_oof_intersection"}
d = pd.concat([pd.read_parquet(f, columns=["model", "seed", "fold", "benchmark_accuracy"] + list(RC.values())) for f in glob.glob(f"{D}/*.parquet")], ignore_index=True)
d = d[d.fold == 1]
rows = []
for m, g in d.groupby("model"):
    r = {"model": m, "seeds": g.seed.nunique(), "bench acc": round(g.benchmark_accuracy.mean(), 3)}
    for loss, c in RC.items():
        x = g[c].astype(float)
        r[f"omega {loss}"] = round(0.2 * x.mean(), 3); r[f"defined {loss}"] = int(x.notna().sum())
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
