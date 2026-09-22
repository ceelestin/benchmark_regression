#!/usr/bin/env python3
"""Analysis of the redundancy-score DERIVATION experiments
(configs/derivation_redundancy_levers_*.yml: regression job 49236, classification analog).

Usage (login node, benchcv venv):
    python analysis/derivation_levers_analysis.py --loss neg_mse \
        --glob "$SCRATCH/ranking_outputs/derivation_redundancy_levers_[sfr]*__c*.parquet" \
        --out-dir analysis/out_derivation_mse
    python analysis/derivation_levers_analysis.py --loss neg_nll \
        --glob "$SCRATCH/ranking_outputs/derivation_redundancy_levers_classification_*__c*.parquet" \
        --out-dir analysis/out_classification_nll

Per CONFIGURATION (family x dataset noise x train_size x procedure x fixed_split x solver
setting), seeds being the Monte-Carlo replicates, with q_{s,k} the fold-k score on the
test fold (``objective_<loss>_test``), b_{s,k} the bench score of the same fitted model
(``objective_<loss>_bench``) and, on the first split only, the per-chunk scores of that
model on 200 outer chunks of the test-fold size (``objective_outer_<loss>``):

* G_K^paper (Definitions 3.1-3.2 of the paper). delta_HO(alpha) = pooled score of the
  first-split model on alpha x n_te fresh samples (test fold + alpha-1 outer chunks,
  per-sample means combine as (base + k mean_k)/(k+1)) minus its bench score;
  Delta_K = mean_k q_k - mean_k b_k. With V^delta_1(alpha) = Var_seeds(delta_HO(alpha))
  and V^delta_K = Var_seeds(Delta_K), N_equiv/n_te = min alpha with V^delta_1(alpha) <=
  V^delta_K (log-log interpolated between integer alphas; capped at the number of chunks).
* G_K^err = Var_s(q_1 - b_1) / Var_s(qbar_K - bbar_K): the same estimand under exact 1/N
  scaling of the single-split variance (used for the closed-form check).
* G_K^raw = Var_s(q_1) / Var_s(qbar_K): the CV estimate itself, including the across-study
  floor (the fitted models' true performance varies with the training data).
Bootstrap over seeds gives percentile CIs for G^err and G^raw.

Study-only candidates read at fold k (row idx_rep = k-1), median over seeds:
  rho_e (squared-error OOF-intersection ICC), covp, S = covp*rho_e, Sdl = S/target_var,
  icc_an / icc_sq (OOF-ANOVA ICC of the signed / squared error), fv, sv (its components),
  rho_full, rho_resid, G_proxy; for classification also rho_err01, icc_err01, rho_nll,
  icc_nll (0-1 error and NLL analogs). ``f_hat = 1 - icc_<loss>`` feeds the closed form
  (analysis/closed_form_gain_check.py).
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

K_GRID = [2, 3, 5, 10, 20, 50, 100, 200]
CAND_COLS = {
    "rho_e": "objective_study_squared_error_rho_all_oof_intersection",
    "covp": "objective_study_prediction_cov_all_oof_intersection",
    "icc_an": "objective_study_oof_anova_error_icc",
    "icc_sq": "objective_study_oof_anova_squared_error_icc",
    "fv": "objective_study_oof_anova_squared_error_fold_var",
    "sv": "objective_study_oof_anova_squared_error_sample_var",
    "rho_full": "objective_study_squared_error_rho_all_full",
    "rho_resid": "objective_study_squared_error_rho_all_resid_train_membership",
    "G_proxy": "objective_study_gain_proxy_squared_error_all_oof_intersection",
    "tvar": "objective_study_target_var",
    # classification only
    "rho_err01": "objective_study_error01_rho_all_oof_intersection",
    "icc_err01": "objective_study_oof_anova_error01_icc",
    "fv_err01": "objective_study_oof_anova_error01_fold_var",
    "sv_err01": "objective_study_oof_anova_error01_sample_var",
    "rho_nll": "objective_study_nll_rho_all_oof_intersection",
    "icc_nll": "objective_study_oof_anova_nll_icc",
    "fv_nll": "objective_study_oof_anova_nll_fold_var",
    "sv_nll": "objective_study_oof_anova_nll_sample_var",
}
# dataset_name embeds the seed: key on the family + dataset parameters instead.
CONFIG_KEYS = ["family", "p_dataset_noise", "p_obj_train_size",
               "p_obj_procedure", "p_obj_fixed_split", "solver_name"]
LOSS_COLS = {  # loss -> (test col, bench col, outer per-chunk list col)
    "neg_mse": ("objective_neg_mse_test", "objective_neg_mse_bench", "objective_outer_neg_mse"),
    "neg_nll": ("objective_neg_nll_test", "objective_neg_nll_bench", "objective_outer_neg_nll"),
    "accuracy": ("objective_accuracy_test", "objective_accuracy_bench", "objective_outer_accuracy"),
    "score": ("objective_score_test", "objective_score_bench", "objective_outer_scores"),
}


def load(pattern, loss):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no parquet matches {pattern}")
    q_col, b_col, o_col = LOSS_COLS[loss]
    base = set(CAND_COLS.values()) | {q_col, b_col, o_col, "dataset_name", "p_dataset_seed",
                                      "idx_rep", "objective_split_index", "objective_split_test_size",
                                      "time", "solver_name", "p_obj_train_size", "p_obj_procedure",
                                      "p_obj_fixed_split", "p_dataset_noise", "p_dataset_logit_scale"}
    frames = []
    for f in files:
        d = pd.read_parquet(f)
        cols = [c for c in d.columns if c in base]
        frames.append(d[cols])
    df = pd.concat(frames, ignore_index=True)
    if "p_dataset_noise" not in df and "p_dataset_logit_scale" in df:
        df["p_dataset_noise"] = df["p_dataset_logit_scale"]      # classification: noise lever
    df["family"] = df["dataset_name"].str.extract(r"^(\w+)\[")[0]
    df["solver"] = df["solver_name"].str.replace(r"\[.*$", "", regex=True)
    df["lever"] = df["solver_name"].str.extract(
        r"(pred_noise=[0-9.]+|coef_noise=[0-9.]+|logit_noise=[0-9.]+)")[0].fillna("")
    df["q"] = df[q_col].astype(float)
    df["b"] = df[b_col].astype(float)
    df["outer"] = df[o_col]
    return df


def _paper_gain(var_k, alpha_var):
    """Smallest alpha with V1(alpha) <= var_k, log-log interpolated; alpha_var[0] is alpha=1."""
    v = np.asarray(alpha_var, dtype=float)
    n = len(v)
    below = np.where(v <= var_k)[0]
    if len(below) == 0:
        return float(n), True                     # never reached: capped
    j = int(below[0])
    if j == 0:
        return 1.0, False
    # interpolate between alpha=j (v[j-1] > var_k) and alpha=j+1 (v[j] <= var_k)
    lv0, lv1 = np.log(v[j - 1]), np.log(v[j])
    la0, la1 = np.log(j), np.log(j + 1)
    frac = (lv0 - np.log(var_k)) / (lv0 - lv1) if lv0 > lv1 else 1.0
    return float(np.exp(la0 + frac * (la1 - la0))), False


def gains(df):
    rows = []
    rng = np.random.default_rng(0)
    for key, g in df.groupby(CONFIG_KEYS, dropna=False):
        piv_q = g.pivot_table(index="p_dataset_seed", columns="idx_rep", values="q").dropna(axis=0)
        piv_b = g.pivot_table(index="p_dataset_seed", columns="idx_rep", values="b").loc[piv_q.index]
        n_seeds, K_avail = piv_q.shape
        if n_seeds < 5:
            continue
        Q, B = piv_q.values, piv_b.values
        E = Q - B
        # pooled single-split estimation error on alpha x n_te samples (first split)
        first = g[g["idx_rep"] == 0].set_index("p_dataset_seed").loc[piv_q.index]
        outer = np.vstack([np.asarray(o, dtype=float) for o in first["outer"].values])   # seeds x chunks
        n_alpha = outer.shape[1] + 1
        cum = np.cumsum(outer, axis=1)
        pooled = np.concatenate([Q[:, :1], (Q[:, :1] + cum) / (np.arange(1, n_alpha)[None, :] + 1)], axis=1)
        delta_alpha_var = np.var(pooled - B[:, :1], axis=0, ddof=1)             # V^delta_1(alpha)
        v1_raw, v1_err = np.var(Q[:, 0], ddof=1), np.var(E[:, 0], ddof=1)
        for K in [k for k in K_GRID if k <= K_avail]:
            mq, me = Q[:, :K].mean(axis=1), E[:, :K].mean(axis=1)
            g_raw, g_err = v1_raw / np.var(mq, ddof=1), v1_err / np.var(me, ddof=1)
            g_paper, capped = _paper_gain(np.var(me, ddof=1), delta_alpha_var)
            bs_raw, bs_err, bs_paper = [], [], []
            for _ in range(200):
                idx = rng.integers(0, n_seeds, n_seeds)
                bs_raw.append(np.var(Q[idx, 0], ddof=1) / np.var(mq[idx], ddof=1))
                bs_err.append(np.var(E[idx, 0], ddof=1) / np.var(me[idx], ddof=1))
                bs_paper.append(_paper_gain(np.var(me[idx], ddof=1),
                                            np.var(pooled[idx] - B[idx, :1], axis=0, ddof=1))[0])
            rows.append(dict(zip(CONFIG_KEYS, key), K=K, n_seeds=n_seeds, n_alpha=n_alpha,
                             G_paper=g_paper, G_paper_capped=capped,
                             G_paper_lo=np.percentile(bs_paper, 5), G_paper_hi=np.percentile(bs_paper, 95),
                             G_raw=g_raw, G_raw_lo=np.percentile(bs_raw, 5), G_raw_hi=np.percentile(bs_raw, 95),
                             G_err=g_err, G_err_lo=np.percentile(bs_err, 5), G_err_hi=np.percentile(bs_err, 95)))
    return pd.DataFrame(rows)


def candidates(df, gains_df):
    rows = []
    for key, g in df.groupby(CONFIG_KEYS, dropna=False):
        rec = dict(zip(CONFIG_KEYS, key))
        rec["solver"], rec["lever"] = g["solver"].iloc[0], g["lever"].iloc[0]
        for k in (3, 20, 200):
            sub = g[g["idx_rep"] == k - 1]
            if sub.empty:
                continue
            med = {n: sub[c].astype(float).median() for n, c in CAND_COLS.items() if c in sub}
            S = med["covp"] * med["rho_e"]
            rec[f"S_{k}"], rec[f"Sdl_{k}"] = S, (S / med["tvar"] if med.get("tvar") else np.nan)
            for n in ("rho_e", "icc_an", "icc_sq", "fv", "sv", "rho_full", "rho_resid", "G_proxy",
                      "rho_err01", "icc_err01", "fv_err01", "sv_err01", "rho_nll", "icc_nll", "fv_nll", "sv_nll"):
                rec[f"{n}_{k}"] = med.get(n, np.nan)
            rec[f"fv_dl_{k}"] = med["fv"] / med["tvar"] if med.get("tvar") else np.nan
        gk = (gains_df[(gains_df[CONFIG_KEYS] == pd.Series(rec)[CONFIG_KEYS]).all(axis=1)]
              if len(gains_df) else gains_df)
        for K in (5, 20, 200):
            r = gk[gk["K"] == K] if len(gk) else gk
            if len(r):
                for name in ("G_paper", "G_raw", "G_err"):
                    rec[f"{name}_{K}"] = float(r[name].iloc[0])
        for name in ("G_paper", "G_err"):
            if f"{name}_20" in rec and f"{name}_200" in rec:
                rec[f"remaining_{name}_20_200"] = rec[f"{name}_200"] / rec[f"{name}_20"]
        rows.append(rec)
    return pd.DataFrame(rows)


def rules(cand, gain="G_paper"):
    out = []
    needed = (f"{gain}_20", f"{gain}_200", f"remaining_{gain}_20_200")
    if any(c not in cand for c in needed):
        return pd.DataFrame(out)
    targets = {f"{gain}_20>=10": cand[f"{gain}_20"] >= 10,
               f"{gain}_200>=20": cand[f"{gain}_200"] >= 20,
               f"remaining_{gain}_20_200>=1.25": cand[f"remaining_{gain}_20_200"] >= 1.25}
    for name in ("S_3", "Sdl_3", "rho_e_3", "icc_an_3", "icc_sq_3", "fv_dl_3", "rho_err01_3", "icc_err01_3", "rho_nll_3", "icc_nll_3"):
        if name not in cand or cand[name].isna().all():
            continue
        x = cand[name].astype(float)
        for qtl in np.linspace(0.05, 0.95, 19):
            thr = x.quantile(qtl)
            above = x > thr
            rec = {"candidate": name, "threshold": thr, "quantile": qtl, "n_above": int(above.sum())}
            for tname, t in targets.items():
                rec[f"P({tname}|above)"] = float(t[above].mean()) if above.any() else np.nan
                rec[f"P({tname}|below)"] = float(t[~above].mean()) if (~above).any() else np.nan
            out.append(rec)
    return pd.DataFrame(out)


def plots(cand, out_dir, gain="G_paper"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    levers = cand["solver"] + cand["lever"].map(lambda s: f" {s}" if s else "")
    colors = {lv: plt.cm.tab20(i % 20) for i, lv in enumerate(sorted(levers.unique()))}
    xs = ("S_3", "Sdl_3", "rho_e_3", "icc_an_3", "icc_sq_3", "rho_e_20", "icc_sq_20",
          "rho_err01_3", "icc_err01_3", "rho_nll_3", "icc_nll_3")
    ys = (f"{gain}_20", f"{gain}_200", f"remaining_{gain}_20_200", "G_err_200", "G_raw_200")
    for xname in xs:
        if xname not in cand or cand[xname].isna().all():
            continue
        for yname in ys:
            if yname not in cand:
                continue
            fig, ax = plt.subplots(figsize=(7, 5))
            for lv in sorted(levers.unique()):
                m = levers == lv
                ax.scatter(cand.loc[m, xname], cand.loc[m, yname], s=14, alpha=.75, color=colors[lv], label=lv)
            ax.set_xlabel(xname); ax.set_ylabel(yname); ax.set_yscale("log")
            if xname.startswith(("S_", "Sdl_")):
                ax.set_xscale("symlog", linthresh=1e-3)
            ax.grid(alpha=.3); ax.legend(fontsize=6, ncol=2)
            ax.set_title(f"{yname} vs {xname} (median over seeds; one point per configuration)")
            fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"{yname}__vs__{xname}.png"), dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--loss", default="neg_mse", choices=list(LOSS_COLS))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    df = load(args.glob, args.loss)
    print(f"loss={args.loss} rows={len(df):,} configurations={df.groupby(CONFIG_KEYS).ngroups} "
          f"seeds={df['p_dataset_seed'].nunique()} K_max={df['idx_rep'].max() + 1}")
    g = gains(df)
    g.to_csv(os.path.join(args.out_dir, "gains.csv"), index=False)
    c = candidates(df, g)
    c.to_csv(os.path.join(args.out_dir, "candidates.csv"), index=False)
    rules(c).to_csv(os.path.join(args.out_dir, "rules.csv"), index=False)
    plots(c, args.out_dir)
    from scipy.stats import spearmanr
    if "G_paper_200" in c:
        ok = c[["G_paper_200", "G_err_200", "G_raw_200"]].dropna()
        print(f"\nG_paper vs G_err at K=200: Spearman {spearmanr(ok.G_paper_200, ok.G_err_200).statistic:+.2f}, "
              f"median log(G_paper/G_err) {np.log(ok.G_paper_200 / ok.G_err_200).median():+.2f}; "
              f"capped configurations: {int(g[(g.K == 200)].G_paper_capped.sum())} of {int((g.K == 200).sum())}")
    print("\nSpearman rank correlation across configurations (median-over-seeds candidate at k=3 vs gain):")
    for xname in ("S_3", "Sdl_3", "rho_e_3", "icc_an_3", "icc_sq_3", "fv_dl_3", "rho_err01_3", "icc_err01_3", "rho_nll_3", "icc_nll_3"):
        if xname not in c or c[xname].isna().all():
            continue
        line = f"  {xname:11s}"
        for yname in ("G_paper_20", "G_paper_200", "remaining_G_paper_20_200", "G_err_200"):
            if yname not in c:
                continue
            ok = c[[xname, yname]].dropna()
            rho = spearmanr(ok[xname], ok[yname]).statistic if len(ok) > 3 else np.nan
            line += f"  {yname}: {rho:+.2f}"
        print(line)
    print(f"\nwrote {args.out_dir}/gains.csv, candidates.csv, rules.csv and plots")


if __name__ == "__main__":
    main()
