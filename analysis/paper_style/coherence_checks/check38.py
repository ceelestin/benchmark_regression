"""Figure 6's configurations with G_true > 20 at K = 20 (MSE): which design, which are in the noise-injection grid."""
import os, pandas as pd
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))  # code/datapaths.py
import datapaths
A = datapaths.repo_path("analysis")
x = pd.concat([pd.read_csv(f"{A}/out_decomp_{s}/decomposition_neg_mse_stat_squared_error.csv").assign(design=s)
               for s in ("derivation", "holdout1", "holdout2", "holdout3")])
print(x.columns.tolist()[:40])
x = x[x.K == 20]
print("configs at K=20 per design:", x.design.value_counts().to_dict(), "total", len(x))
h = x[x.G_true > 20].copy()
h["rho_implied"] = (20 / h.G_true - 1) / 19
cols = [c for c in ("design", "solver", "dataset", "dataset_type", "noise", "train_size", "G_paper", "G_true", "rho_implied") if c in h.columns]
print(len(h), h.design.value_counts().to_dict()); print(h[cols].sort_values("rho_implied").to_string())
print("median implied rho, all:", round(h.rho_implied.median(), 4))
