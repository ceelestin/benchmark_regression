"""Redundancy (t * rho on the squared error, OOF intersection) for the App I / App M text pass (2026-10-30)."""
import glob, os
import numpy as np, pandas as pd, pyarrow.parquet as pq
O = os.path.expandvars("$WORK/benchmark_regression/outputs"); J = os.path.expandvars("$SCRATCH/paper_runs/benchopt_run_2026-{}.parquet")
RHO = "objective_study_squared_error_rho_all_oof_intersection"   # as analysis/paper_appendix_redundancy_checks.py
C = ["solver_name", "p_dataset_seed", "idx_rep", "p_obj_train_size", "p_obj_test_size", RHO]
def load(files, solver=None):
    d = pd.concat(pd.read_parquet(f, columns=[c for c in C if c in pq.read_schema(f).names]) for f in files)
    return d[d.solver_name.str.startswith(solver)] if solver else d
def red(d, k):
    x = d[d.idx_rep == k - 1]
    return (x.p_obj_test_size.astype(float) * x[RHO]).groupby([x.solver_name, x.p_obj_train_size]).mean()
print("== App I (n_tr = 1,000), after 3 / 5 splits")
sets = {"MLP(32) fixed": load(sorted(glob.glob(f"{O}/paper_main_gain_mse_k200__c0[0-3].parquet")), "MLP[hidden_layer_sizes=[32]"),
        "MLP(64,64) fixed": load(sorted(glob.glob(f"{O}/paper_main_gain_mse_k200_mlp64x64__c0[0-3].parquet"))),
        "ExtraTrees fixed": load(sorted(glob.glob(f"{O}/paper_main_gain_mse_k200__c0[0-3].parquet")), "ExtraTrees"),
        "MLP grid": load(sorted(glob.glob(f"{O}/hpo_mse_MLP_gridsearch__c*.parquet"))),
        "MLP random": load(sorted(glob.glob(f"{O}/hpo_mse_MLP_randomsearch__c*.parquet"))),
        "ExtraTrees grid/random": load([J.format("07-27_12h40m37"), J.format("07-27_12h47m49")])}
for lab, d in sets.items():
    print(f"  {lab:22s} k=3: {red(d, 3).round(3).to_dict()} | k=5: {red(d, 5).round(3).to_dict()}", flush=True)
print("== App M (small budgets), after 5 splits")
sm = pd.concat([load([J.format(t) for t in ("07-31_19h02m17", "07-31_21h05m17", "07-31_19h07m06", "07-31_21h07m56")]),
                load(sorted(glob.glob(f"{O}/small_budget_mlp64_rff50__c*.parquet")))])
w = red(sm, 5).unstack(0).round(3); print(w.to_string())
w.to_csv(os.path.expandvars("$WORK/paper_style_figs/v3/results_v4d/small_redundancy_k5.csv"))
