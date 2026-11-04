"""Item 7: dataset parameters, seeds and solvers of every run file in <data root>/paper_runs."""
import glob, os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))  # code/datapaths.py
import datapaths
import pandas as pd, pyarrow.parquet as pq
for f in sorted(glob.glob(datapaths.data_path("paper_runs", "*.parquet"))):
    names = pq.read_schema(f).names
    cols = [c for c in ("dataset_name", "p_dataset_n_features", "p_dataset_noise", "p_dataset_seed", "p_obj_train_size", "p_obj_n_splits", "solver_name") if c in names]
    d = pd.read_parquet(f, columns=cols)
    fam = sorted(d.dataset_name.str.extract(r"^([^\[]+)")[0].unique())
    solv = sorted({s.split("[")[0] for s in d.solver_name.unique()})
    print(os.path.basename(f).replace("benchopt_run_", ""), "| fam", ",".join(f.replace("sim_", "") for f in fam),
          "| d", sorted(d.p_dataset_n_features.unique().tolist()), "| noise", sorted(d.p_dataset_noise.astype(float).unique().tolist()),
          "| seeds", d.p_dataset_seed.nunique(), "| n_tr", sorted(d.p_obj_train_size.astype(int).unique().tolist()),
          "| K", sorted(d.p_obj_n_splits.astype(int).unique().tolist()), "|", ",".join(solv))
