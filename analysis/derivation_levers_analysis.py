#!/usr/bin/env python3
"""Analysis of the redundancy-score DERIVATION experiment
(configs/derivation_redundancy_levers_*.yml, job 49236).

Usage (login node, benchcv venv):
    python analysis/derivation_levers_analysis.py \
        --glob "$SCRATCH/ranking_outputs/derivation_redundancy_levers_*__c*.parquet" \
        --out-dir "$WORK/benchmark_regression/analysis/out_derivation"

Definitions (all per CONFIGURATION = dataset family x noise x train_size x
procedure x fixed_split x solver setting; seeds are the Monte-Carlo replicates):

* q_{s,k}  : CV score of fold k in seed s (``objective_score_test``, R^2 on the
             fold's test part; ``objective_neg_mse_test`` also available).
* b_{s,k}  : bench score of the SAME fitted model (``objective_score_bench``,
             R^2 on the 100k bench set) = that model's true performance.
* qbar_K   : mean of the first K fold scores (the K-split CV estimate).
* Two exact across-seed gains, both "variance-equivalent test sample gains":
    G_K^raw = Var_s(q_{s,1}) / Var_s(qbar_{s,K})
        variance of the CV estimate itself; includes the study-sampling floor
        (the target moves with the study), so it saturates.
    G_K^err = Var_s(q_{s,1} - b_{s,1}) / Var_s(qbar_{s,K} - bbar_{s,K})
        variance of the estimation ERROR of the CV estimate w.r.t. the true
        performance of the fitted models; removes the floor.
  Bootstrap over seeds gives percentile CIs.
* Study-only candidates, read at fold k (row idx_rep = k-1) and summarised by
  the median over seeds:
    rho_e  = study_squared_error_rho_all_oof_intersection
    covp   = study_prediction_cov_all_oof_intersection
    S      = covp * rho_e            (pre-registered S_3 at k=3)
    S_dl   = S / study_target_var    (dimensionless variant)
    icc_an = study_oof_anova_error_icc  (sample-driven / total OOF error var)
    icc_sq = study_oof_anova_squared_error_icc
    fv, sv = study_oof_anova_error_fold_var / _sample_var (in y^2 units)
    rho_full, rho_resid = *_full / *_resid_train_membership squared-error ICCs
    G_proxy = study_gain_proxy_squared_error_all_oof_intersection

Outputs: gains.csv (one row per configuration and K), candidates.csv (one row
per configuration with candidates at k=3/20/200 and the gains), rules.csv
(threshold-rule diagnostics), and PNG scatter plots.
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

K_GRID = [2, 3, 5, 10, 20, 50, 100, 200]
CAND_COLS = {
    "rho_e": "objective_study_squared_error_rho_all_oof_intersection",
    "covp": "objective_study_prediction_cov_all_oof_intersection",
    "icc_an": "objective_study_oof_anova_error_icc",
    "icc_sq": "objective_study_oof_anova_squared_error_icc",
    "fv": "objective_study_oof_anova_error_fold_var",
    "sv": "objective_study_oof_anova_error_sample_var",
    "rho_full": "objective_study_squared_error_rho_all_full",
    "rho_resid": (
        "objective_study_squared_error_rho_all_resid_train_membership"
    ),
    "G_proxy": (
        "objective_study_gain_proxy_squared_error_all_oof_intersection"
    ),
    "tvar": "objective_study_target_var",
}
# NB: dataset_name embeds the seed, so the configuration key uses the family
# (parsed from dataset_name) and the dataset parameters instead.
CONFIG_KEYS = ["family", "p_dataset_noise", "p_obj_train_size",
               "p_obj_procedure", "p_obj_fixed_split", "solver_name"]


def load(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no parquet matches {pattern}")
    cols = None
    frames = []
    for f in files:
        d = pd.read_parquet(f)
        if cols is None:
            cols = [c for c in d.columns if c in set(CAND_COLS.values())
                    or c in CONFIG_KEYS + ["dataset_name", "p_dataset_seed", "idx_rep",
                                            "objective_score_test",
                                            "objective_score_bench",
                                            "objective_neg_mse_test",
                                            "objective_neg_mse_bench",
                                            "objective_split_index", "time"]]
        frames.append(d[cols])
    df = pd.concat(frames, ignore_index=True)
    df["family"] = df["dataset_name"].str.extract(r"^(\w+)\[")[0]
    df["solver"] = df["solver_name"].str.replace(r"\[.*$", "", regex=True)
    df["lever"] = df["solver_name"].str.extract(r"(pred_noise=[0-9.]+|coef_noise=[0-9.]+)")[0].fillna("")
    return df


def gains(df, score="score_test", bench="score_bench"):
    """Across-seed gain curves per configuration."""
    q = f"objective_{score}"
    b = f"objective_{bench}"
    rows = []
    rng = np.random.default_rng(0)
    for key, g in df.groupby(CONFIG_KEYS, dropna=False):
        piv_q = g.pivot_table(index="p_dataset_seed", columns="idx_rep", values=q)
        piv_b = g.pivot_table(index="p_dataset_seed", columns="idx_rep", values=b)
        piv_q = piv_q.dropna(axis=0)
        piv_b = piv_b.loc[piv_q.index]
        n_seeds, K_avail = piv_q.shape
        if n_seeds < 5:
            continue
        Q = piv_q.values
        E = Q - piv_b.values
        v1_raw = np.var(Q[:, 0], ddof=1)
        v1_err = np.var(E[:, 0], ddof=1)
        for K in [k for k in K_GRID if k <= K_avail]:
            mq = Q[:, :K].mean(axis=1)
            me = E[:, :K].mean(axis=1)
            g_raw = v1_raw / np.var(mq, ddof=1)
            g_err = v1_err / np.var(me, ddof=1)
            # bootstrap over seeds
            bs_raw, bs_err = [], []
            for _ in range(200):
                idx = rng.integers(0, n_seeds, n_seeds)
                bs_raw.append(np.var(Q[idx, 0], ddof=1) / np.var(mq[idx], ddof=1))
                bs_err.append(np.var(E[idx, 0], ddof=1) / np.var(me[idx], ddof=1))
            rows.append(dict(zip(CONFIG_KEYS, key), K=K, n_seeds=n_seeds,
                             G_raw=g_raw, G_raw_lo=np.percentile(bs_raw, 5),
                             G_raw_hi=np.percentile(bs_raw, 95),
                             G_err=g_err, G_err_lo=np.percentile(bs_err, 5),
                             G_err_hi=np.percentile(bs_err, 95)))
    return pd.DataFrame(rows)


def candidates(df, gains_df):
    rows = []
    for key, g in df.groupby(CONFIG_KEYS, dropna=False):
        rec = dict(zip(CONFIG_KEYS, key))
        rec["solver"] = g["solver"].iloc[0]
        rec["lever"] = g["lever"].iloc[0]
        for k in (3, 20, 200):
            sub = g[g["idx_rep"] == k - 1]
            if sub.empty:
                continue
            med = {name: sub[col].astype(float).median()
                   for name, col in CAND_COLS.items() if col in sub}
            S = med["covp"] * med["rho_e"]
            rec[f"S_{k}"] = S
            rec[f"Sdl_{k}"] = S / med["tvar"] if med.get("tvar") else np.nan
            for name in ("rho_e", "icc_an", "icc_sq", "fv", "sv", "rho_full",
                         "rho_resid", "G_proxy"):
                rec[f"{name}_{k}"] = med.get(name, np.nan)
            rec[f"fv_dl_{k}"] = med["fv"] / med["tvar"] if med.get("tvar") else np.nan
        gk = (gains_df[(gains_df[CONFIG_KEYS] == pd.Series(rec)[CONFIG_KEYS]).all(axis=1)]
              if len(gains_df) else gains_df)
        for K in (5, 20, 200):
            r = gk[gk["K"] == K] if len(gk) else gk
            if len(r):
                rec[f"G_raw_{K}"] = float(r["G_raw"].iloc[0])
                rec[f"G_err_{K}"] = float(r["G_err"].iloc[0])
        if "G_err_20" in rec and "G_err_200" in rec:
            rec["remaining_err_20_200"] = rec["G_err_200"] / rec["G_err_20"]
            rec["remaining_raw_20_200"] = rec["G_raw_200"] / rec["G_raw_20"]
        rows.append(rec)
    return pd.DataFrame(rows)


def rules(cand):
    """Threshold-rule diagnostics: for each candidate read at k=3 and a grid of
    thresholds x, P(large further gain | candidate > x) and the coverage."""
    out = []
    needed = ("G_err_20", "G_err_200", "remaining_err_20_200")
    if any(c not in cand for c in needed):
        return pd.DataFrame(out)
    targets = {
        "G_err_20>=10": cand["G_err_20"] >= 10,
        "G_err_200>=20": cand["G_err_200"] >= 20,
        "remaining_err_20_200>=1.25": cand["remaining_err_20_200"] >= 1.25,
    }
    for name in ("S_3", "Sdl_3", "rho_e_3", "icc_an_3", "icc_sq_3", "fv_dl_3"):
        if name not in cand:
            continue
        x = cand[name].astype(float)
        for q in np.linspace(0.05, 0.95, 19):
            thr = x.quantile(q)
            above = x > thr
            rec = {"candidate": name, "threshold": thr, "quantile": q,
                   "n_above": int(above.sum())}
            for tname, t in targets.items():
                rec[f"P({tname}|above)"] = float(t[above].mean()) if above.any() else np.nan
                rec[f"P({tname}|below)"] = float(t[~above].mean()) if (~above).any() else np.nan
            out.append(rec)
    return pd.DataFrame(out)


def plots(cand, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    levers = cand["solver"] + cand["lever"].map(lambda s: f" {s}" if s else "")
    colors = {lv: plt.cm.tab20(i % 20) for i, lv in enumerate(sorted(levers.unique()))}
    for xname in ("S_3", "Sdl_3", "rho_e_3", "icc_an_3", "fv_dl_3", "rho_e_20", "icc_an_20"):
        if xname not in cand:
            continue
        for yname in ("G_err_20", "G_err_200", "remaining_err_20_200", "G_raw_200"):
            if yname not in cand:
                continue
            fig, ax = plt.subplots(figsize=(7, 5))
            for lv in sorted(levers.unique()):
                m = levers == lv
                ax.scatter(cand.loc[m, xname], cand.loc[m, yname], s=14, alpha=.75,
                           color=colors[lv], label=lv)
            ax.set_xlabel(xname); ax.set_ylabel(yname)
            if xname.startswith(("S_", "Sdl_", "fv")):
                ax.set_xscale("symlog", linthresh=1e-3)
            ax.set_yscale("log")
            ax.grid(alpha=.3); ax.legend(fontsize=6, ncol=2)
            ax.set_title(f"{yname} vs {xname} (median over seeds; one point per configuration)")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{yname}__vs__{xname}.png"), dpi=130)
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--score", default="score_test", choices=["score_test", "neg_mse_test"])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    df = load(args.glob)
    print(f"rows={len(df):,} configurations={df.groupby(CONFIG_KEYS).ngroups} "
          f"seeds={df['p_dataset_seed'].nunique()} K_max={df['idx_rep'].max() + 1}")
    bench = "score_bench" if args.score == "score_test" else "neg_mse_bench"
    g = gains(df, score=args.score, bench=bench)
    g.to_csv(os.path.join(args.out_dir, "gains.csv"), index=False)
    c = candidates(df, g)
    c.to_csv(os.path.join(args.out_dir, "candidates.csv"), index=False)
    r = rules(c)
    r.to_csv(os.path.join(args.out_dir, "rules.csv"), index=False)
    plots(c, args.out_dir)
    # quick console summary: rank correlations of each k=3 candidate with the gains
    from scipy.stats import spearmanr
    print("\nSpearman rank correlation across configurations (median-over-seeds candidate at k=3 vs gain):")
    for xname in ("S_3", "Sdl_3", "rho_e_3", "icc_an_3", "icc_sq_3", "fv_dl_3", "G_proxy_3"):
        if xname not in c:
            continue
        line = f"  {xname:10s}"
        for yname in ("G_err_20", "G_err_200", "remaining_err_20_200", "G_raw_200"):
            if yname not in c:
                continue
            ok = c[[xname, yname]].dropna()
            rho = spearmanr(ok[xname], ok[yname]).statistic if len(ok) > 3 else np.nan
            line += f"  {yname}: {rho:+.2f}"
        print(line)
    print(f"\nwrote {args.out_dir}/gains.csv, candidates.csv, rules.csv and plots")


if __name__ == "__main__":
    main()
