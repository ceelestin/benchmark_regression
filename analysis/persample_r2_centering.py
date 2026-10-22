"""Why is the R^2 gain of a heavily shrunk Ridge so much larger than its MSE gain? Recompute the fold R^2
from the per-sample pilot (configs/pilot_persample_k20.yml) with its denominator centred on the fold's own
mean (sklearn's r2_score, as stored) or on the study mean, and compare G_K (benchmark-aware deltas).

    python analysis/persample_r2_centering.py --files outputs/pilot_persample_k20__c0*.parquet
"""
import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

COLS = ["solver_name", "p_dataset_seed", "idx_rep", "objective_score_test", "objective_score_bench",
        "objective_neg_mse_test", "objective_neg_mse_bench", "objective_persample_error",
        "objective_persample_test_mask", "objective_persample_y_study"]


def gain(D):
    return np.var(D, axis=0, ddof=1).mean() / np.var(D.mean(axis=1), ddof=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    a = ap.parse_args()
    d = pd.concat([pq.read_table(f, columns=COLS).to_pandas() for f in a.files], ignore_index=True)
    rows = []
    for s, gs in d.groupby("solver_name"):
        fold_c, study_c, stored, mse = [], [], [], []
        for _, g in gs.groupby("p_dataset_seed"):
            g = g.sort_values("idx_rep")
            y = np.asarray(g.objective_persample_y_study.iloc[0], dtype=float)
            rb = g.objective_score_bench.values.astype(float)
            fc, sc = [], []
            for e, te in zip(g.objective_persample_error, g.objective_persample_test_mask):
                te = te.astype(bool)
                sse = (np.asarray(e)[te] ** 2).sum()
                fc.append(1 - sse / ((y[te] - y[te].mean()) ** 2).sum())
                sc.append(1 - sse / ((y[te] - y.mean()) ** 2).sum())
            fold_c.append(np.array(fc) - rb)
            study_c.append(np.array(sc) - rb)
            stored.append(g.objective_score_test.values.astype(float) - rb)
            mse.append(g.objective_neg_mse_test.values.astype(float) - g.objective_neg_mse_bench.values.astype(float))
        rows.append({"solver": s.split("[")[0].replace("_noHPO", ""),
                     "G20_r2_stored": gain(np.array(stored)),
                     "G20_r2_fold_mean": gain(np.array(fold_c)),
                     "G20_r2_study_mean": gain(np.array(study_c)),
                     "G20_mse": gain(np.array(mse))})
    print(pd.DataFrame(rows).set_index("solver").round(2).to_string())


if __name__ == "__main__":
    main()
