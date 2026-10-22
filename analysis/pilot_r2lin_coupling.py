"""Offline check of a study-only coupling-inclusive estimate of rho_delta on the r2lin pilot.

Oracle (benchmark deltas delta_k = score_k - bench_k, variances across seeds):
    sigma2 = mean_k Var_seed(delta_k), tau = sigma2 - E_seed[Var_k(delta_k)], rho_delta = tau / sigma2,
    kappa = rho_delta - omega, omega = t * rho_e (OOF-intersection, k = 5).
Study-only (one study, first k splits, no benchmark):
    s2_w = Var_k(score_k) (spread of the fold scores), sigma2_hat = v_loss / n_te (per-sample loss variance),
    rho_hat = 1 - s2_w / sigma2_hat, G_hat = K / (1 + (K - 1) rho_hat).
The fold-score spread contains, on top of Var_k(delta_k), the spread of the benchmark scores across the
fitted models (model-to-model variation), which the benchmark deltas remove.

    python analysis/pilot_r2lin_coupling.py --files outputs/pilot_r2lin_k20__c0*.parquet
"""
import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

M = {  # metric: (study score, benchmark score, per-sample var column, per-sample rho column)
    "r2": ("objective_score_test", "objective_score_bench",
           "objective_study_r2lin_var_all_oof_intersection", "objective_study_r2lin_rho_all_oof_intersection"),
    "mse": ("objective_neg_mse_test", "objective_neg_mse_bench",
            "objective_study_squared_error_var_all_oof_intersection",
            "objective_study_squared_error_rho_all_oof_intersection"),
}


def wide(g, c):
    return g.pivot_table(index="p_dataset_seed", columns="idx_rep", values=c, dropna=False).reindex(columns=range(g.idx_rep.max() + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--K", type=int, default=20)
    a = ap.parse_args()
    cols = ["solver_name", "p_dataset_seed", "idx_rep", "p_obj_test_size", "objective_split_test_size", "objective_study_target_var"] + \
        sorted({c for v in M.values() for c in v})
    d = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in a.files], ignore_index=True)
    for c in cols[6:]:
        d[c] = d[c].astype(float)
    K = a.K
    rows = []
    for (s, m), g in [((s, m), g) for s, g in d.groupby("solver_name") for m in M]:
        q, b, vc, rc = M[m]
        t, n_te = float(g.p_obj_test_size.iloc[0]), float(g.objective_split_test_size.iloc[0])
        S, B = wide(g, q).values[:, :K], wide(g, b).values[:, :K]
        D = S - B
        sig2 = np.var(D, axis=0, ddof=1).mean()
        rho_d = 1 - np.var(D, axis=1, ddof=1).mean() / sig2
        G = sig2 / np.var(D.mean(axis=1), ddof=1)
        r = {"solver": s.split("[")[0].replace("_noHPO", ""), "metric": m, "G20": G,
             "rho_delta": rho_d, "G_from_rho": K / (1 + (K - 1) * rho_d)}
        w5 = t * g.loc[g.idx_rep == 4, rc].mean()
        r["omega5"], r["kappa"] = w5, rho_d - w5
        V = wide(g, vc).values
        if m == "mse":  # stored squared errors are divided by the study Var(y)
            V = V * (g.groupby("p_dataset_seed").objective_study_target_var.first().astype(float).values[:, None] ** 2)
        r["sig2_hat/sig2"] = (V[:, K - 1] / n_te).mean() / sig2
        r["modelvar_share"] = 1 - np.var(D, axis=1, ddof=1).mean() / np.var(S, axis=1, ddof=1).mean()
        # best case for any correction of the model-to-model term: use the benchmark deltas' spread
        r["rho_if_modelvar_known"] = 1 - np.var(D, axis=1, ddof=1).mean() / (V[:, K - 1] / n_te).mean()
        r["G_if_modelvar_known"] = K / (1 + (K - 1) * r["rho_if_modelvar_known"])
        for k in (5, 20):
            s2w = np.var(S[:, :k], axis=1, ddof=1)
            sh = V[:, k - 1] / n_te
            rh = 1 - s2w.mean() / sh.mean()
            r[f"rho_hat{k}"], r[f"G_hat{k}"] = rh, K / (1 + (K - 1) * rh)
            r[f"rho_hat{k}_q10_q90"] = "%.2f..%.2f" % tuple(np.percentile(1 - s2w / sh, [10, 90]))
        rows.append(r)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).set_index(["metric", "solver"]).sort_index().round(3).to_string())


if __name__ == "__main__":
    main()
