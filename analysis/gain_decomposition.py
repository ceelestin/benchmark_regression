#!/usr/bin/env python3
"""Decompose the gap between the observed sample gain and the study-only closed form into
additive log terms, one per mechanism, using the oracle (benchmark-set) quantities.

Usage (Jean-Zay login node, benchcv venv):
    python analysis/gain_decomposition.py --loss neg_mse --glob "<parquets>" --out-dir <dir>
    python analysis/gain_decomposition.py --loss accuracy --stat-loss nll --glob ... --out-dir ...

The objective stores the per-sample squared error divided by Var(y); its variance is rescaled
by Var(y)^2 here. 0-1 error and NLL are stored raw.

Notation, per configuration (seeds = replicates), fold k: e_k = test-fold score error of model
m_k (q_k minus its true performance), beta_k = bench-score error (b_k minus true performance).
Paper gain uses E_k = q_k - b_k = e_k - beta_k, the two parts are independent (disjoint samples).
With s2 = per-sample loss variance on the bench set (mean over fold models) and rho_b = mean
pairwise correlation of per-sample losses between fold models on the bench set (population):
    Var(mean_k beta_k) = s2 / n_b * (1/K + (1 - 1/K) rho_b)          (bench-proxy part)
    T_K = Var_s(mean_k E_k) - Var(mean_k beta_k)                      (test-sampling part)
    G_true_K = V_chunk / T_K      (gain with a perfect oracle, no chunk ceiling; V_chunk = mean over the
                                   fresh outer chunks of Var_s(chunk score error) of the first-split model,
                                   exchangeable with the test fold T_1 but far less noisy)
Closed form: G_cf(rho) = 1 / (1/K + (1 - 1/K) t rho)  [= 1/((1-f)(t+(1-t)/K)+f/K), f = 1-rho].

G_paper is the paper gain with V1(alpha) averaged exactly over the order of the outer chunks
(derivation_levers_analysis.v1_alpha_curve); G_paper_fixed pools them in one fixed order (the analyses
before 2026-09-26), fixed_order_noise = log(G_paper_fixed / G_paper) is reported beside the sum.
log(G_paper / G_cf(rho_study at k=3, median over seeds)) = sum of
  A oracle_floor : log(G_paper / G_paper*)          finite bench set (V1(alpha) floor s2/n_b)
  B ceiling      : log(G_paper* / G_true)            chunk ceiling, alpha interpolation, departure from 1/alpha
  C model        : log(G_true / G_cf(rho_b))         closed form vs truth with the POPULATION rho
                                                     (train/test coupling across folds, misspecification)
  D loss_mismatch: log(G_cf(rho_b gain loss) / G_cf(rho_b stat loss))   0 when matched
  E study_vs_pop : log(G_cf(rho_b stat) / G_cf(rho_study stat at K))    OOF-intersection vs population
  F early_read   : log(G_cf(rho_study at K) / G_cf(rho_study at 3))     reading at k=3 instead of K
  H seed_aggr.   : log(G_cf(weighted rho_study at 3) / G_cf(median rho_study at 3))
Seed aggregation: a variance over seeds averages per-seed variances, so s2 is a MEAN over seeds and
every rho entering the closed form is weighted by the per-seed loss variance s2_s; the stop rules
used the plain median over seeds (term H).
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from derivation_levers_analysis import CONFIG_KEYS, _paper_gain, v1_alpha_curve  # noqa: E402

N_BENCH = 100_000
# loss -> (test col, bench col, outer list col, per-sample loss name in bench/study stats)
GAIN_LOSS = {
    "neg_mse": ("objective_neg_mse_test", "objective_neg_mse_bench", "objective_outer_neg_mse", "squared_error"),
    "accuracy": ("objective_accuracy_test", "objective_accuracy_bench", "objective_outer_accuracy", "error01"),
    "neg_nll": ("objective_neg_nll_test", "objective_neg_nll_bench", "objective_outer_neg_nll", "nll"),
}
STAT_LOSS = {"squared_error": "squared_error", "brier": "squared_error", "error01": "error01", "nll": "nll"}
K_LIST = (5, 20, 200)


def cols_for(name):
    return (f"objective_bench_{name}_var_all", f"objective_bench_{name}_rho_all",
            f"objective_study_{name}_rho_all_oof_intersection")


def load(pattern, loss, stat):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no parquet matches {pattern}")
    q_col, b_col, o_col, gname = GAIN_LOSS[loss]
    want = {q_col, b_col, o_col, "dataset_name", "p_dataset_seed", "idx_rep", "solver_name",
            "p_dataset_logit_scale", "objective_study_target_var", *CONFIG_KEYS, *cols_for(gname), *cols_for(stat)}
    frames = []
    for f in files:
        names = pq.read_schema(f).names
        frames.append(pd.read_parquet(f, columns=[c for c in names if c in want]))
    df = pd.concat(frames, ignore_index=True)
    if "p_dataset_noise" not in df and "p_dataset_logit_scale" in df:
        df["p_dataset_noise"] = df["p_dataset_logit_scale"]
    if "p_obj_train_source" not in df:
        df["p_obj_train_source"] = "study"
    df["p_obj_train_source"] = df["p_obj_train_source"].fillna("study")
    df["family"] = df["dataset_name"].str.extract(r"^(\w+)\[")[0]
    df["q"], df["b"] = df[q_col].astype(float), df[b_col].astype(float)
    df["outer"] = df[o_col]
    # the objective stores the per-sample squared error divided by Var(y) (study or bench set):
    # rescale its variance to the units of the score (Var(y) of bench and study sets agree)
    v = df[cols_for(gname)[0]].astype(float)
    df["s2_gain"] = v * df["objective_study_target_var"].astype(float) ** 2 if gname == "squared_error" else v
    return df


def G_cf(rho, K, t):
    return 1.0 / (1.0 / K + (1.0 - 1.0 / K) * t * rho)


def decompose(df, loss, stat):
    gname = GAIN_LOSS[loss][3]
    _, gr, _ = cols_for(gname)
    _, sr, ss = cols_for(stat)
    rows = []
    for key, g in df.groupby(CONFIG_KEYS, dropna=False):
        cfg = dict(zip(CONFIG_KEYS, key))
        if cfg["p_obj_procedure"] != "ShuffleSplit" or str(cfg["p_obj_fixed_split"]) == "True":
            continue
        piv = {c: g.pivot_table(index="p_dataset_seed", columns="idx_rep", values=c)
               for c in ("q", "b", "s2_gain", gr, sr, ss) if c in g}
        piv_q = piv["q"].dropna(axis=0)
        if len(piv_q) < 5:
            continue
        seeds = piv_q.index
        piv = {c: v.loc[seeds] for c, v in piv.items()}
        Q, Bs = piv["q"].values, piv["b"].values
        E = Q - Bs
        n_seeds, K_avail = E.shape
        t = 0.2                                              # all ShuffleSplit grids use t = 0.2
        n_tr = int(cfg["p_obj_train_size"])
        n_te = round(n_tr * t / (1 - t))
        first = g[g["idx_rep"] == 0].set_index("p_dataset_seed").loc[seeds]
        outer = np.vstack([np.asarray(o, dtype=float) for o in first["outer"].values])
        n_chunks = outer.shape[1]
        n_alpha = n_chunks + 1

        def at(c, k):                                        # per-seed column at fold k (1-based)
            m = piv[c]
            return m[k - 1].astype(float).values if (k - 1) in m else np.full(n_seeds, np.nan)

        s2_seed = at("s2_gain", 2)                           # var_all needs 2 folds
        s2_1 = float(np.nanmean(s2_seed))                    # a variance over seeds averages per-seed variances
        B1 = s2_1 / N_BENCH
        ref = Q[:, :1]
        # paper V1(alpha): test fold + first alpha-1 outer chunks (fixed order), as in the gain analysis
        pooled = np.concatenate([ref, (ref + np.cumsum(outer, axis=1)) / (np.arange(1, n_alpha)[None, :] + 1)], axis=1)
        v1_fixed = np.var(pooled - Bs[:, :1], axis=0, ddof=1)
        # same estimand averaged exactly over the order of the outer chunks (the test fold stays first)
        v1_avg = v1_alpha_curve(E[:, 0], outer - Bs[:, :1])
        chunk_var = np.var(outer - Bs[:, :1], axis=0, ddof=1) - B1
        V_chunk = float(np.mean(chunk_var))
        V1 = np.var(E[:, 0], ddof=1)
        T1 = V1 - B1
        alphas = np.arange(1, n_alpha + 1)
        okc = v1_avg - B1 > 0
        slope = float(np.polyfit(np.log(alphas[okc]), np.log((v1_avg - B1)[okc]), 1)[0]) if okc.sum() > 2 else np.nan

        def wmean(x, w):
            m = np.isfinite(x) & np.isfinite(w)
            return float(np.sum(x[m] * w[m]) / np.sum(w[m])) if m.any() else np.nan

        rho_s3_med = float(np.nanmedian(at(ss, 3)))          # what the stop rules used
        rho_s3_w = wmean(at(ss, 3), s2_seed)
        for K in [k for k in K_LIST if k <= K_avail]:
            s2K = at("s2_gain", K)
            rbg, rbs, rsK = at(gr, K), at(sr, K), at(ss, K)
            rho_b_gain, rho_b_stat, rho_sK = wmean(rbg, s2K), wmean(rbs, s2K), wmean(rsK, s2K)
            VK = np.var(E[:, :K].mean(axis=1), ddof=1)
            BK = float(np.nanmean(s2K * (1 / K + (1 - 1 / K) * rbg))) / N_BENCH
            TK = VK - BK
            G_fixed, capped = _paper_gain(VK, v1_fixed)
            G_avg, capped_avg = _paper_gain(VK, v1_avg)            # the paper gain (order-averaged)
            G_avg_star, capped_star = (_paper_gain(TK, np.clip(v1_avg - B1, 1e-300, None))
                                       if TK > 0 else (np.nan, False))
            G_true = V_chunk / TK if TK > 0 else np.nan
            cf_bg, cf_bs = G_cf(rho_b_gain, K, t), G_cf(rho_b_stat, K, t)
            cf_sK, cf_s3w, cf_s3 = G_cf(rho_sK, K, t), G_cf(rho_s3_w, K, t), G_cf(rho_s3_med, K, t)
            rec = dict(cfg, K=K, n_seeds=n_seeds, n_te=n_te, n_alpha=n_alpha,
                       G_paper=G_avg, G_paper_capped=capped_avg, G_paper_fixed=G_fixed, G_paper_fixed_capped=capped,
                       G_paper_star=G_avg_star, G_paper_star_capped=capped_star,
                       G_err=V1 / VK, G_true=G_true, G_true_T1=T1 / TK if TK > 0 else np.nan,
                       G_cf_rho_b=cf_bg, G_cf_rho_s3=cf_s3,
                       rho_b_gain=rho_b_gain, rho_b_stat=rho_b_stat, rho_sK=rho_sK,
                       rho_s3_w=rho_s3_w, rho_s3=rho_s3_med,
                       bench_share_V1=B1 / V1, bench_share_VK=BK / VK,
                       T1_over_theory=T1 / (s2_1 / n_te), chunk_over_theory=V_chunk / (s2_1 / n_te),
                       test_over_chunk=T1 / V_chunk, v1_alpha_slope=slope,
                       C_obs=(TK / V_chunk - 1 / K) / (1 - 1 / K), C_cf=t * rho_b_gain)
            rec["fixed_order_noise"] = np.log(G_fixed / G_avg)   # reported, not part of the sum
            rec["A_oracle_floor"] = np.log(G_avg / G_avg_star)
            rec["B_ceiling"] = np.log(G_avg_star / G_true)
            rec["C_model"] = np.log(G_true / cf_bg)
            rec["D_loss_mismatch"] = np.log(cf_bg / cf_bs)
            rec["E_study_vs_pop"] = np.log(cf_bs / cf_sK)
            rec["F_early_read"] = np.log(cf_sK / cf_s3w)
            rec["H_seed_aggregation"] = np.log(cf_s3w / cf_s3)
            rec["total"] = np.log(G_avg / cf_s3)
            rows.append(rec)
    out = pd.DataFrame(rows)
    out["solver"] = out["solver_name"].str.replace(r"\[.*$", "", regex=True)
    return out


TERMS = ["A_oracle_floor", "B_ceiling", "C_model", "D_loss_mismatch", "E_study_vs_pop",
         "F_early_read", "H_seed_aggregation", "total"]


def report(d, K):
    d = d[d.K == K]
    pd.set_option("display.width", 250)
    print(f"\n===== K = {K}: {len(d)} configurations; log terms (positive = observed gain above the closed form) =====")
    print("median over configurations:\n" + d[TERMS].median().round(3).to_string())
    chk = (d[TERMS[:-1]].sum(axis=1) - d["total"]).abs().max()
    print(f"(sum of terms - total: max abs {chk:.2e})")
    for by in ("p_obj_train_size", "solver", "p_dataset_noise", "family"):
        if by in d and d[by].nunique() > 1:
            print(f"\nby {by} (median):")
            print(d.groupby(by)[TERMS + ["bench_share_VK", "C_obs", "C_cf", "test_over_chunk", "v1_alpha_slope"]].median().round(3).to_string())
    print("\nsanity: T1 / (s2/n_te) median", round(float(d["T1_over_theory"].median()), 3),
          "| test/chunk", round(float(d["test_over_chunk"].median()), 3),
          "| chunk / (s2/n_te) median", round(float(d["chunk_over_theory"].median()), 3),
          "| capped: G_paper", int(d["G_paper_capped"].sum()), "G_paper_fixed", int(d["G_paper_fixed_capped"].sum()),
          "G_paper_star", int(d["G_paper_star_capped"].sum()))
    print("fixed-order noise log(G_paper_fixed / G_paper): median", round(float(d["fixed_order_noise"].median()), 3),
          "sd", round(float(d["fixed_order_noise"].std()), 3))
    print("medians of gains:", d[["G_paper", "G_paper_fixed", "G_paper_star", "G_true", "G_err", "G_cf_rho_b", "G_cf_rho_s3"]].median().round(2).to_dict())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--loss", default="neg_mse", choices=list(GAIN_LOSS))
    ap.add_argument("--stat-loss", default=None, choices=list(STAT_LOSS),
                    help="per-sample loss of the study statistic (default: the gain's loss)")
    args = ap.parse_args()
    stat = STAT_LOSS[args.stat_loss] if args.stat_loss else GAIN_LOSS[args.loss][3]
    df = load(args.glob, args.loss, stat)
    d = decompose(df, args.loss, stat)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"decomposition_{args.loss}_stat_{stat}.csv")
    d.to_csv(path, index=False)
    print(f"gain loss {args.loss}, statistic loss {stat}; wrote {path}")
    for K in K_LIST:
        if (d.K == K).any():
            report(d, K)


if __name__ == "__main__":
    main()
