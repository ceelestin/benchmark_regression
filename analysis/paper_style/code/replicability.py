"""
replicability.py — Significance-decision replicability curves (rebuttal experiment).

Answers the reviewer point that variance is ultimately used in significance tests: for
each headline pair (the comparisons in the paper's significance table) and study size,
we measure how often the K-repetition significance decision on the small study set
matches the oracle decision from the large benchmarking set, as a function of K — plus
the underlying Power(K) and false-positive rate FPR(K), for both the standard paired
t-test and the Nadeau-Bengio corrected resampled t-test (the Bouckaert & Frank family).

Scope (headline pairs = paper's Table `stat_significance`):
  * NLP (Yelp): BERT vs RoBERTa at N in {1000, 3000, 10000}.
  * PCam: DenseNet121/MobileNetV2, DenseNet121/WideResNet101_2, MobileNetV2/WideResNet101_2
    at N in {25, 83, 250, 416, 666}, using the ORIGINAL results_output_pcam data.
"""

# %%
import argparse
import os

import numpy as np
import pandas as pd

import compute
import plot as P
import nlp
import pcam

# %% ──────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

ALPHA = 0.05
K_GRID = list(range(2, 21))     # K=1 excluded: a paired across-fold test needs K>=2
SELF_NULL_KMAX = 10             # exact self-null needs 2K folds
DEFAULT_PLOTS_DIR = "plots/replicability"
RESULTS_DIR = "results"

# PCam — original benchmark data (study_size -> reported train size N)
PCAM_DATA_DIR = "/home/ceve/Documents/Run outputs/PCam/results_output_pcam"
# PCAM_DATA_DIR = "/home/ceve/Documents/Run outputs/PCam/stratandnewmetrics"
PCAM_PAIRS = [
    ("densenet121", "mobilenet_v2"),
    ("densenet121", "wide_resnet101_2"),
    ("mobilenet_v2", "wide_resnet101_2"),
]
PCAM_SIZE_TO_N = {30: 24, 100: 80, 300: 240, 500: 400, 800: 640}
PCAM_RHO = 0.25                 # test fold 1/5 of the study set -> n_test/n_train = 0.25

# NLP — Yelp
NLP_PAIRS = [("bert", "roberta")]
NLP_SIZE_TO_N = {1250: 1000, 3750: 3000, 12500: 10000}
NLP_RHO = 0.25                  # test_size = 0.2 -> n_test/n_train = 0.25

PCAM_DISPLAY = {
    "densenet121": "DenseNet121", "mobilenet_v2": "MobileNetV2",
    "wide_resnet101_2": "WideResNet101_2", "vit_b_16": "ViT-B/16",
}
NLP_DISPLAY = {"bert": "BERT", "roberta": "XLM-RoBERTa", "qwen3-0.6b": "Qwen3-0.6B"}


# %% ──────────────────────────────────────────────────────────────────────────
# Loaders — build a per-fold long table [seed, model, study_size(=N), fold,
# fold_score, bench_score]
# ─────────────────────────────────────────────────────────────────────────────

def _to_scalar_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(s.apply(pcam._extract_scalar), errors="coerce")


def _finalize(df: pd.DataFrame, size_to_n: dict[int, int]) -> pd.DataFrame:
    df = df.copy()
    df["N"] = df["study_size"].map(size_to_n)
    df = df.dropna(subset=["N", "fold_score", "bench_score"])
    df["N"] = df["N"].astype(int)
    out = df[["seed", "model", "N", "fold", "fold_score", "bench_score"]].copy()
    return out.rename(columns={"N": "study_size"})


def load_pcam_folds(data_dir: str) -> pd.DataFrame:
    d = pcam.load_cv_data(data_dir)
    if d.empty:
        return d
    if "cross_validation" in d.columns:
        d = d[d["cross_validation"] == True].copy()          # noqa: E712
    # MCCV (ShuffleSplit) only — exclude RepeatedKFold runs so the fold structure
    # (and thus the Nadeau-Bengio correction) is consistent.
    if "procedure" in d.columns:
        d = d[d["procedure"] == "MCCV"].copy()
    d["fold_score"] = _to_scalar_series(d["test_accuracy"])
    d["bench_score"] = _to_scalar_series(d["benchmark_accuracy"])
    return _finalize(d, PCAM_SIZE_TO_N)


def load_nlp_folds() -> pd.DataFrame:
    d = nlp.load_nlp_data(nlp.DEFAULT_DATA_DIRS)
    if d.empty:
        return d
    if "cross_validation" in d.columns:
        d = d[nlp._infer_cross_validation(d)].copy()
    d = d[d["model"].isin([m for pair in NLP_PAIRS for m in pair])].copy()
    d["fold_score"] = _to_scalar_series(d["study_test_eval_accuracy"])
    d["bench_score"] = _to_scalar_series(d["benchmark_accuracy"])
    return _finalize(d, NLP_SIZE_TO_N)


# %% ──────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def _run_dataset(name, df, pairs, rho, display, plots_dir):
    if df.empty:
        print(f"[{name}] no data — skipping.")
        return None, None
    sizes = sorted(df["study_size"].unique())
    print(f"[{name}] sizes(N)={sizes}  pairs={len(pairs)}  seeds≈{df['seed'].nunique()}")
    rep_df, pf_df = compute.significance_replicability_curves(
        df, pairs=pairs, sizes=sizes, k_grid=K_GRID,
        model_col="model", seed_col="seed", size_col="study_size", fold_col="fold",
        fold_score_col="fold_score", bench_score_col="bench_score",
        n_test_col=None, n_train_col=None, default_rho=rho,
        alpha=ALPHA, self_null_kmax=SELF_NULL_KMAX, dataset=name,
    )
    # Pretty pair labels for plots
    def _disp(p):
        a, b = p.split(" vs ")
        return f"{display.get(a, a)} vs {display.get(b, b)}"

    # Oracle summary (reproduces the significance table)
    orc = (rep_df.groupby(["pair", "N"])
           .agg(oracle_decision=("oracle_decision", "first"),
                oracle_significant=("oracle_significant", "first"),
                oracle_p=("oracle_p", "first")).reset_index())
    print(f"\n[{name}] Oracle decisions (benchmark paired t-test):")
    print(orc.to_string(index=False))

    P.set_paper_style()
    for (pair, N), _ in rep_df.groupby(["pair", "N"]):
        clean = pair.replace(" ", "_").replace("/", "_")
        P.plot_replicability_curves(
            rep_df, pf_df, pair=pair, N=N, alpha=ALPHA,
            dataset_label=name,
            output_path=f"{plots_dir}/replicability_{name}_{clean}_N{N}.pdf",
        )
    return rep_df, pf_df


def main(plots_dir=DEFAULT_PLOTS_DIR):
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    reps, pfs = [], []
    for name, loader, pairs, rho, display in (
        ("nlp", load_nlp_folds, NLP_PAIRS, NLP_RHO, NLP_DISPLAY),
        ("pcam", lambda: load_pcam_folds(PCAM_DATA_DIR), PCAM_PAIRS, PCAM_RHO, PCAM_DISPLAY),
    ):
        rep_df, pf_df = _run_dataset(name, loader(), pairs, rho, display, plots_dir)
        if rep_df is not None:
            rep_df.to_csv(f"{RESULTS_DIR}/replicability_{name}_rep.csv", index=False)
            pf_df.to_csv(f"{RESULTS_DIR}/replicability_{name}_power_fpr.csv", index=False)
            print(f"[{name}] wrote results/replicability_{name}_*.csv")
            reps.append(rep_df)
            pfs.append(pf_df)

    print("\nDone.")
    return (pd.concat(reps, ignore_index=True) if reps else pd.DataFrame(),
            pd.concat(pfs, ignore_index=True) if pfs else pd.DataFrame())


def _running_in_jupyter():
    try:
        get_ipython()  # noqa: F821
        return True
    except NameError:
        return False


if __name__ == "__main__":
    if _running_in_jupyter():
        main()
    else:
        ap = argparse.ArgumentParser(description=__doc__)
        ap.add_argument("--plots-dir", default=DEFAULT_PLOTS_DIR)
        args = ap.parse_args()
        main(plots_dir=args.plots_dir)

# %%
