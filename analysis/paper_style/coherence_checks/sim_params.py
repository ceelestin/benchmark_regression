"""Item 7 (CE, 2026-11-02): dataset parameters actually used by the simulated runs behind the paper's figures."""
import glob, os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))  # code/datapaths.py
import datapaths
import pandas as pd, pyarrow.parquet as pq
OUT = datapaths.data_path("outputs")
JULY = datapaths.data_path("paper_runs", "benchopt_run_2026-{}.parquet")
src = {
    "Fig 4/5 main k200 (n1000)": glob.glob(f"{OUT}/paper_main_gain_mse_k200__c*.parquet") + glob.glob(f"{OUT}/paper_main_gain_mse_k200_mlp64x64__c*.parquet") + glob.glob(f"{OUT}/paper_main_gain_mse_k200_rff50__c*.parquet"),
    "HPO (App I)": glob.glob(f"{OUT}/hpo_mse_*__c*.parquet"),
    "July n1000 12h40/12h47": [JULY.format("07-27_12h40m37"), JULY.format("07-27_12h47m49")],
    "n100 (App J) July 20h48/21h01 + rff/mlp64": [JULY.format("07-27_20h48m32"), JULY.format("07-27_21h01m05")] + glob.glob(f"{OUT}/paper_ntr100_k20_*__c*.parquet"),
    "frozen redrawn July 21h00": [JULY.format("07-27_21h00m32")],
    "small budget (App M) July 31": [JULY.format("07-31_19h07m06"), JULY.format("07-31_21h07m56"), JULY.format("07-31_19h02m17"), JULY.format("07-31_21h05m17")] + glob.glob(f"{OUT}/small_budget_mlp64_rff50__c*.parquet"),
}
for name, files in src.items():
    files = [f for f in files if os.path.exists(f)]
    rows = []
    for f in files:
        names = pq.read_schema(f).names
        cols = [c for c in names if c.startswith("p_dataset_") and c != "p_dataset_seed"] + [c for c in ("dataset_name", "p_obj_train_size", "p_obj_test_size") if c in names]
        d = pd.read_parquet(f, columns=cols).drop_duplicates()
        rows.append(d)
    d = pd.concat(rows, ignore_index=True).drop_duplicates()
    d["family"] = d["dataset_name"].str.extract(r"^([^\[]+)")[0]
    keep = [c for c in ("family", "p_dataset_n_features", "p_dataset_noise", "p_dataset_n_samples", "p_dataset_fixed", "p_obj_test_size") if c in d.columns]
    g = d.groupby(keep, dropna=False)["p_obj_train_size"].apply(lambda s: sorted(set(int(x) for x in s))).reset_index()
    print(f"== {name}: {len(files)} files"); print(g.to_string(index=False))
