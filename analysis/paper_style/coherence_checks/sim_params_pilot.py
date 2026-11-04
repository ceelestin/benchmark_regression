import glob, os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))  # code/datapaths.py
import datapaths
import pandas as pd, pyarrow.parquet as pq
OUT = datapaths.data_path("outputs")
for pat in ("pilot_persample_k20", "pilot_r2lin_k20"):
    fs = sorted(glob.glob(f"{OUT}/{pat}*.parquet"))
    if not fs: print(pat, "no files"); continue
    d = pd.concat([pd.read_parquet(f, columns=[c for c in ("dataset_name", "p_dataset_n_features", "p_dataset_noise", "p_dataset_seed", "p_obj_train_size", "solver_name") if c in pq.read_schema(f).names]) for f in fs])
    print(pat, len(fs), "files | d", sorted(d.p_dataset_n_features.unique().tolist()), "| noise", sorted(d.p_dataset_noise.astype(float).unique().tolist()),
          "| seeds", d.p_dataset_seed.nunique(), "| n_tr", sorted(d.p_obj_train_size.astype(int).unique().tolist()), "|", sorted({s.split("[")[0] for s in d.solver_name.unique()}))
