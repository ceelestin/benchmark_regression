"""Verify the D.1 claim 'the estimated rho_delta is positive in every configuration we examined (0.004 to 0.23)'.
ANOVA rho_delta = tau / sigma2 (tau = B - W/K, sigma2 = W + tau) on the first K = 20 splits, for:
  A. PCam round 3 (seeds 100-149), 4 networks x n_tr 100/300/800 x accuracy/NLL/Brier (Table D.2, Figure 6);
  B. the simulated designs of Figure 6 still on disk (holdout1, holdout3), mean squared error;
  C. the HPO runs of Appendix I (mean squared error);
  D. the implied rho (K/G_true - 1)/(K - 1) of the decomposition CSVs, by design (all four designs of Figure 6).
Read-only."""
import glob, os, sys
import numpy as np, pandas as pd
import pyarrow.parquet as pq

K = 20
W_ = os.path.expandvars("$WORK"); S_ = os.path.expandvars("$SCRATCH")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)


def anova(P):
    E = P.values
    W = E.var(axis=1, ddof=1).mean(); B = E.mean(axis=1).var(ddof=1); tau = B - W / K; s2 = W + tau
    return tau / s2, len(E)


part = sys.argv[1] if len(sys.argv) > 1 else "ABCD"

if "A" in part:
    D = f"{W_}/benchmark_pcam/results_output_pcam"
    LOSS = {"accuracy": ("test_accuracy", "benchmark_accuracy"), "nll": ("test_nll", "benchmark_nll"),
            "brier": ("test_brier", "benchmark_brier")}
    rows = []
    for model in ("densenet121", "mobilenet_v2", "wide_resnet101_2", "vit_b_16"):
        for size in (100, 300, 800):
            files = sorted(glob.glob(f"{D}/pcam_results_CV_shuffle_split_unstrat_{model}_nsplits_20_seeds_1[0-4][0-9]-*_sizes_{size}_*.parquet"))
            g = pd.concat([pd.read_parquet(f, columns=["seed", "fold", "n_splits", "study_size"] + [c for v in LOSS.values() for c in v])
                           for f in files], ignore_index=True)
            g = g[(g.n_splits == 20) & (g.study_size == size)].drop_duplicates(["seed", "fold"])
            for loss, (qc, bc) in LOSS.items():
                g["d"] = g[qc] - g[bc]
                P = g.pivot_table(index="seed", columns="fold", values="d").dropna()
                assert P.shape[1] == K, P.shape
                r, n = anova(P)
                rows.append({"model": model, "n_tr": size, "metric": loss, "seeds": n, "files": len(files), "rho_delta": r})
    a = pd.DataFrame(rows)
    print("== A. PCam round 3, ANOVA rho_delta at K = 20")
    print(a.pivot_table(index=["model", "n_tr"], columns="metric", values="rho_delta").round(4).to_string())
    print("seeds per config:", sorted(a.seeds.unique()), "| files:", sorted(a.files.unique()))
    for m, x in a.groupby("metric"):
        print(f"  {m}: min {x.rho_delta.min():.4f} ({x.loc[x.rho_delta.idxmin(), 'model']} {x.loc[x.rho_delta.idxmin(), 'n_tr']}), max {x.rho_delta.max():.4f}, negatives {int((x.rho_delta < 0).sum())}")
    print(f"  all 36: min {a.rho_delta.min():.4f}, max {a.rho_delta.max():.4f}, negatives {int((a.rho_delta < 0).sum())}")
    a.to_csv("rho_pcam_round3.csv", index=False)

KEYS = ["family", "p_dataset_noise", "p_obj_train_size", "p_obj_test_size", "p_obj_procedure", "p_obj_fixed_split",
        "p_obj_train_source", "solver_name"]


def sim_rho(pattern, label):
    files = sorted(glob.glob(pattern))
    want = {"dataset_name", "p_dataset_seed", "idx_rep", "objective_neg_mse_test", "objective_neg_mse_bench", *KEYS,
            "p_dataset_logit_scale"}
    fr = []
    for f in files:
        names = pq.read_schema(f).names
        x = pd.read_parquet(f, columns=[c for c in names if c in want])
        x = x[x.idx_rep < K]
        fr.append(x)
    df = pd.concat(fr, ignore_index=True)
    if "p_dataset_noise" not in df and "p_dataset_logit_scale" in df:
        df["p_dataset_noise"] = df["p_dataset_logit_scale"]
    for c in KEYS:
        if c not in df:
            df[c] = "na"
    df["p_obj_train_source"] = df["p_obj_train_source"].fillna("study")
    df["family"] = df["dataset_name"].str.extract(r"^(\w+)\[")[0]
    df = df[(df.p_obj_procedure == "ShuffleSplit") & (df.p_obj_fixed_split.astype(str) != "True")]
    df["d"] = df.objective_neg_mse_test.astype(float) - df.objective_neg_mse_bench.astype(float)
    rows = []
    for key, g in df.groupby(KEYS, dropna=False):
        P = g.pivot_table(index="p_dataset_seed", columns="idx_rep", values="d").dropna()
        if P.shape[1] < K or len(P) < 5:
            continue
        r, n = anova(P)
        rows.append(dict(zip(KEYS, key), seeds=n, rho_delta=r))
    out = pd.DataFrame(rows)
    out["solver"] = out.solver_name.str.replace(r"\[.*$", "", regex=True)
    print(f"== {label}: {len(files)} files, {len(out)} configurations; rho_delta min {out.rho_delta.min():.4f}, "
          f"median {out.rho_delta.median():.4f}, max {out.rho_delta.max():.4f}, negatives {int((out.rho_delta < 0).sum())}")
    print(out.groupby("solver").rho_delta.agg(["count", "min", "median", "max"]).round(4).to_string())
    if (out.rho_delta < 0).any():
        print(out[out.rho_delta < 0][["family", "p_dataset_noise", "p_obj_train_size", "p_obj_test_size", "solver", "seeds", "rho_delta"]].round(4).to_string(index=False))
    return out


if "B" in part:
    R = f"{S_}/ranking_outputs"
    b1 = sim_rho(f"{R}/k200_threshold_validation_holdout_50_*__c*.parquet", "B. holdout1 (k200_threshold_validation_holdout_50)")
    b1.assign(design="holdout1").to_csv("rho_holdout1.csv", index=False)
    b3 = sim_rho(f"{S_}/holdout3_in/holdout3_largegain_sim_[lin]*__c*.parquet", "B. holdout3 (regression)")
    b3.assign(design="holdout3").to_csv("rho_holdout3.csv", index=False)

if "C" in part:
    hp = sorted(glob.glob(f"{W_}/benchmark_regression/outputs/hpo_mse_*__c*.parquet"))
    print("HPO files in outputs:", len(hp), sorted({os.path.basename(f).split("__c")[0] for f in hp}))
    c = sim_rho(f"{W_}/benchmark_regression/outputs/hpo_mse_*__c*.parquet", "C. HPO (Appendix I)")
    c.to_csv("rho_hpo.csv", index=False)

if "D" in part:
    A = f"{W_}/benchmark_regression/analysis"
    print("== D. implied rho (K/G_true - 1)/(K - 1) from the decomposition CSVs, K = 20, mean squared error")
    for s in ("derivation", "holdout1", "holdout2", "holdout3"):
        f = f"{A}/out_decomp_{s}/decomposition_neg_mse_stat_squared_error.csv"
        if not os.path.exists(f):
            print("  missing", f); continue
        x = pd.read_csv(f); x = x[x.K == K]
        r = (K / x.G_true - 1) / (K - 1)
        print(f"  {s}: {len(x)} configs, implied rho min {r.min():.4f} median {r.median():.4f} max {r.max():.4f}, "
              f"negatives {int((r < 0).sum())}, G_true > 20: {int((x.G_true > 20).sum())}, G_paper > 20: {int((x.G_paper > 20).sum())}")
        if s != "derivation" and (r < 0).any():
            y = x[r < 0].assign(implied_rho=r[r < 0])
            print(y[["family", "p_dataset_noise", "p_obj_train_size", "solver", "n_seeds", "G_paper", "G_true", "implied_rho"]].round(3).to_string(index=False))
    # Figure 6 omega CSVs: number of simulated configurations per design (as in check_above_k.py)
    for s in ("derivation", "holdout1", "holdout2", "holdout3"):
        f = f"{A}/out_omega/omega_{s}_neg_mse.csv"
        x = pd.read_csv(f)
        print(f"  Figure 6 omega {s}: {len(x)} configs, G_paper > 20: {int((x.G_paper > 20).sum())}")
