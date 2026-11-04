"""Appendix H: SD ratio frozen / reshuffled of the 20-fit benchmarking score on the MSE, n_tr = 100."""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))
import datapaths
J = datapaths.data_path("paper_runs", "benchopt_run_2026-{}.parquet"); O = datapaths.data_path("outputs")
C = ["solver_name", "p_dataset_seed", "idx_rep", "objective_neg_mse_test", "objective_neg_mse_bench"]
def per_seed(files):
    d = pd.concat(pd.read_parquet(f, columns=C) for f in files); d = d[d.idx_rep < 20]
    d["delta"] = d.objective_neg_mse_bench - d.objective_neg_mse_test          # test MSE - bench MSE
    d["score"] = -d.objective_neg_mse_test
    d["learner"] = d.solver_name.str.split("[").str[0].replace({"KernelApprox": "RandomFeatures", "ExtraTrees_noHPO": "ExtraTrees"})
    return d.groupby(["learner", "p_dataset_seed"])[["delta", "score"]].mean()
res = per_seed([J.format("07-27_21h01m05")] + sorted(glob.glob(f"{O}/paper_ntr100_k20_resampled_rff50__c*.parquet")))
arms = {"frozen, same dataset": per_seed([J.format("07-27_20h48m32")] + sorted(glob.glob(f"{O}/paper_ntr100_k20_fixed_rff50__c*.parquet"))),
        "frozen, data redrawn (Fig. H.1)": per_seed([J.format("07-27_21h00m32")] + sorted(glob.glob(f"{O}/paper_ntr100_k20_fixed_redrawn_rff50__c*.parquet")))}
rows = []
for arm, fz in arms.items():
    for l in sorted(set(fz.index.get_level_values(0))):
        for q in ("delta", "score"):
            rows.append({"frozen arm": arm, "learner": l, "quantity": q, "n_seeds": len(fz.loc[l]),
                         "sd_frozen": fz.loc[l][q].std(), "sd_reshuffled": res.loc[l][q].std(),
                         "ratio": fz.loc[l][q].std() / res.loc[l][q].std()})
print(pd.DataFrame(rows).round(4).to_string(index=False))
