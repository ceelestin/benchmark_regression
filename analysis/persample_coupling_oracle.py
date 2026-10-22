"""Oracle decomposition of the split-pair covariance Cov(delta_a, delta_b) by sample class, on the
per-sample pilot (configs/pilot_persample_k20.yml, store_per_sample=True).

Per run (solver, seed) and split a, the per-sample deviation d_a(i) is such that the benchmark-aware
fold delta is delta_a = -mean_{i in T_a} d_a(i):
  MSE: d_a(i) = e_a(i)^2 - MSE_bench_a                                  (exact)
  R^2: d_a(i) = (e_a(i)^2 - M_a) / V - M_a ((y_i - ybar)^2 - V) / V^2  (first order in the fold MSE and
       fold variance around M_a = MSE_bench_a and V = Var_bench(y) = M_a / (1 - R2_bench_a); ybar = study
       mean); checked against the exact R^2 delta.
For a pair (a, b) the product delta_a delta_b splits over (i in T_a, j in T_b) into
  same      : i = j in T_a & T_b                       (overlap: the redundancy omega)
  sh_cross  : i != j, both in T_a & T_b
  sh_excl   : one in T_a & T_b, the other in the exclusive part of the other test set (trained on by
              the first split)
  excl_excl : i in T_a \\ T_b (in S_b), j in T_b \\ T_a (in S_a)
With fixed predictors every i != j term has mean 0, so the last three are the train/test coupling.
Each class is averaged over pairs and seeds and divided by sigma2 = mean_a Var_seed(delta_a); their sum
(minus the product of the mean deltas) is rho_delta.

    python analysis/persample_coupling_oracle.py --files outputs/pilot_persample_k20__c0*.parquet
"""
import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

COLS = ["solver_name", "p_dataset_seed", "idx_rep", "p_obj_test_size",
        "objective_score_test", "objective_score_bench", "objective_neg_mse_test", "objective_neg_mse_bench",
        "objective_study_r2lin_rho_all_oof_intersection", "objective_study_squared_error_rho_all_oof_intersection",
        "objective_persample_error", "objective_persample_train_mask", "objective_persample_test_mask",
        "objective_persample_y_study"]
CLASSES = ["same", "sh_cross", "sh_excl", "excl_excl"]


def run_arrays(g):
    g = g.sort_values("idx_rep")
    E = np.vstack(g.objective_persample_error.values)
    Te = np.vstack(g.objective_persample_test_mask.values).astype(bool)
    y = np.asarray(g.objective_persample_y_study.iloc[0], dtype=float)
    return E, Te, y, g


def deviations(E, y, g, metric):
    mse_b = -g.objective_neg_mse_bench.values.astype(float)
    if metric == "mse":
        return E ** 2 - mse_b[:, None]
    r2_b = g.objective_score_bench.values.astype(float)
    V = mse_b / (1 - r2_b)
    c = (y - y.mean()) ** 2
    return (E ** 2 - mse_b[:, None]) / V[:, None] - (mse_b / V ** 2)[:, None] * (c[None, :] - V[:, None])


def pair_classes(d, Te):
    K, n = Te.shape[0], Te[0].sum()
    out = np.zeros((K, K, len(CLASSES)))
    for a in range(K):
        for b in range(a + 1, K):
            sh = Te[a] & Te[b]
            xa, xb = Te[a] & ~Te[b], Te[b] & ~Te[a]
            sa, sb = d[a, sh].sum(), d[b, sh].sum()
            same = (d[a, sh] * d[b, sh]).sum()
            ea, eb = d[a, xa].sum(), d[b, xb].sum()
            out[a, b] = [same, sa * sb - same, sa * eb + ea * sb, ea * eb]
    return out / n ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    a = ap.parse_args()
    d = pd.concat([pq.read_table(f, columns=COLS).to_pandas() for f in a.files], ignore_index=True)
    rows = []
    for s, gs in d.groupby("solver_name"):
        t = float(gs.p_obj_test_size.iloc[0])
        for metric in ("r2", "mse"):
            deltas, deltas_exact, prods = [], [], []
            for _, g in gs.groupby("p_dataset_seed"):
                E, Te, y, g = run_arrays(g)
                dv = deviations(E, y, g, metric)
                deltas.append(np.array([-dv[k, Te[k]].mean() for k in range(len(Te))]))
                q, b = (("objective_score_test", "objective_score_bench") if metric == "r2"
                        else ("objective_neg_mse_test", "objective_neg_mse_bench"))
                deltas_exact.append(g[q].values.astype(float) - g[b].values.astype(float))
                prods.append(pair_classes(dv, Te))
            D, DX, P = np.array(deltas), np.array(deltas_exact), np.array(prods)
            K = D.shape[1]
            iu = np.triu_indices(K, 1)
            sig2 = np.var(D, axis=0, ddof=1).mean()
            m = D.mean(axis=0)
            cls = P[:, iu[0], iu[1], :].mean(axis=(0, 1)) / sig2
            meancorr = np.mean(m[iu[0]] * m[iu[1]]) / sig2
            rc = ("objective_study_r2lin_rho_all_oof_intersection" if metric == "r2"
                  else "objective_study_squared_error_rho_all_oof_intersection")
            r = {"metric": metric, "solver": s.replace("_noHPO", "").replace("hidden_layer_sizes=", "").replace("n_estimators=", "n=").replace(",min_samples_leaf=1", "").replace(",weights=uniform", ""),
                 "lin_vs_exact_corr": np.corrcoef(D.ravel(), DX.ravel())[0, 1],
                 "lin_vs_exact_var": D.var() / DX.var(),
                 "omega5": t * gs.loc[gs.idx_rep == 4, rc].astype(float).mean()}
            r.update({c: v for c, v in zip(CLASSES, cls)})
            r["sum-meancorr"] = cls.sum() - meancorr
            C = np.cov(D, rowvar=False, ddof=1)
            r["rho_delta"] = C[iu].mean() / sig2
            r["kappa"] = r["rho_delta"] - r["omega5"]
            r["G20_lin"] = sig2 / np.var(D.mean(axis=1), ddof=1)
            r["G20_exact"] = np.var(DX, axis=0, ddof=1).mean() / np.var(DX.mean(axis=1), ddof=1)
            rows.append(r)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).set_index(["metric", "solver"]).sort_index().round(3).to_string())


if __name__ == "__main__":
    main()
