#!/usr/bin/env python3
"""Paper-style MSE versions of the simulated gain-vs-K figures.

Redraws, on the mean squared error, the orchestrator's figures of 2026-10-22
(benchmark_regression analysis/paper_gain_curves.py, outputs in
$SCRATCH/paper_figs_mse on Jean-Zay) with the paper's own pipeline, so that they
match the existing Images/aggregated_test_gain_k_*_sim_linear_*_r2_*.pdf:

* estimator: compute.build_nocv_cache + compute.sample_gain_bootstrap, i.e. the
  variance-equivalent test size of the paper (first crossing of the single-split
  variance on test + outer chunks, delta scores, 100 bootstrap resamples of seeds,
  95% bands), exactly as simulated.py does for its 'neg_mse' metric;
* plotting: plot.plot_sample_gain_vs_splits with the arguments simulated.py uses
  (paper rcParams, figsize, in-axes formula title, direct labels, ticks, xlim);
* colours: tab10 by algorithm family as in the published figures (ExtraTrees blue,
  GradientBoosting orange, MLP green, Ridge red), random features purple. A fixed
  map is needed because simulated.py colours by sorted solver order, which would
  recolour MLP and Ridge once KernelApprox is added.

Data (same runs as the orchestrator):
  n_tr = 1000, K = 200, resampled: job 1426287 (Ridge, GB, ExtraTrees; its MLP[32] is
      dropped), MLP[64, 64] job 1446474, random features (KernelApprox, rff, 50) job 1453342
      -> Run outputs/Regression/data/mse_paper_figs/orchestrator_mse_runs_extract.parquet
  n_tr = 100, K = 20: Ridge / GB / ExtraTrees from the July runs (MLP[32] dropped),
      MLP[64, 64] and random features from the same extract.
      --fixed-source orchestrator (default): fixed arm = benchopt_run_2026-07-27_20h48m32,
          the run the orchestrator used, in which the DATASET is frozen too (p_dataset_fixed
          = True), like its new MLP / random-feature fixed runs.
      --fixed-source corrected: fixed arm = benchopt_run_2026-07-27_21h00m32 (dataset
          varies across seeds, the design the paper uses); MLP[64, 64] and random features
          then have no fixed arm, since no such run exists for them.
      Resampled arm: benchopt_run_2026-07-27_21h01m05 in both cases.

Usage: python mse_gain_figures.py [--fixed-source orchestrator|corrected] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import compute
import plot as P
import simulated as SIM

RUNS = os.path.expanduser("~/Documents/Run outputs/Regression/data")
EXTRACT = f"{RUNS}/mse_paper_figs/orchestrator_mse_runs_extract.parquet"
JULY_RESAMPLED = f"{RUNS}/fixed/benchopt_run_2026-07-27_21h01m05.parquet"
JULY_FIXED = {
    "orchestrator": f"{RUNS}/fixed/benchopt_run_2026-07-27_20h48m32.parquet",
    "corrected": f"{RUNS}/fixed/benchopt_run_2026-07-27_21h00m32.parquet",
}
DEFAULT_OUT = os.path.expanduser(
    "~/Documents/Papiers/Miens/Crossing the Validation Crisis/NeurIPS 2026 - Rebuttal/"
    "proposed_figures_mse_gain")

_TAB10 = plt.get_cmap("tab10").colors
FAMILY_COLOR = {"ExtraTrees": _TAB10[0], "GradientBoosting": _TAB10[1], "MLP": _TAB10[2],
                "Ridge": _TAB10[3], "RandomFeatures": _TAB10[4]}
DISPLAY = {"KernelApprox": "RandomFeatures"}
DROP_SOLVERS = {"MLP[hidden_layer_sizes=[32]]"}       # replaced by MLP[64, 64], as the orchestrator did
COLS = ["objective_name", "solver_name", "dataset_name", "idx_rep", "p_obj_procedure", "p_obj_n_splits",
        "p_obj_train_size", "p_obj_fixed_split", "p_dataset_seed", "p_dataset_noise", "p_dataset_fixed",
        "objective_neg_mse_test", "objective_neg_mse_bench", "objective_outer_neg_mse"]
TEST, BENCH, OUTER = "objective_neg_mse_test", "objective_neg_mse_bench", "objective_outer_neg_mse"


def display_name(solver: str) -> str:
    base = SIM.format_solver_label(solver)
    return DISPLAY.get(base, base)


def load(fixed_source: str) -> pd.DataFrame:
    frames = []
    ext = pd.read_parquet(EXTRACT)
    if fixed_source == "corrected":
        # The new MLP[64, 64] / random-feature fixed-split runs froze the dataset too: drop them.
        ext = ext[~ext["p_obj_fixed_split"].astype(str).str.lower().eq("true")]
    frames.append(ext[[c for c in COLS if c in ext.columns]])
    for path in (JULY_RESAMPLED, JULY_FIXED[fixed_source]):
        import pyarrow.parquet as pq
        names = pq.read_schema(path).names
        d = pd.read_parquet(path, columns=[c for c in COLS if c in names])
        frames.append(d[~d.solver_name.isin(DROP_SOLVERS)])
    data = pd.concat(frames, ignore_index=True)
    data = SIM.process_parquet(data)            # one row per (objective, solver, dataset, idx_rep)
    data["procedure"] = data["p_obj_procedure"]
    data["n_splits"] = data["p_obj_n_splits"].astype(int)
    data["train_size"] = data["p_obj_train_size"].astype(int)
    data["seed"] = data["p_dataset_seed"].astype(int)
    data["fixed_split"] = data["p_obj_fixed_split"].astype(str).str.lower().eq("true")
    family = data["dataset_name"].str.extract(r"^([^\[]+)")[0]
    noise = data["p_dataset_noise"].astype(float)
    data["dataset_type"] = np.where(noise > 0, family + "_noise" + noise.astype(str), family)
    return data


def gains(data: pd.DataFrame, n_bootstraps: int, test_size: float) -> pd.DataFrame:
    """simulated.py's per-metric block for 'neg_mse', unchanged."""
    data_cv = data[data["procedure"].isin(["ShuffleSplit", "RepeatedKFold"])].sort_values(
        ["objective_name", "solver_name", "dataset_name", "idx_rep"]).copy()
    nocv_base = data_cv[data_cv["idx_rep"] == 0].copy()
    data_cv["objective_score_test"] = data_cv[TEST]
    data_cv["objective_score_bench"] = data_cv[BENCH]
    grp = data_cv.groupby(["objective_name", "solver_name", "dataset_name"])
    data_cv["n_splits"] = grp.cumcount() + 1
    data_cv["objective_score_test"] = grp["objective_score_test"].cumsum() / data_cv["n_splits"]
    data_cv["objective_score_bench"] = grp["objective_score_bench"].cumsum() / data_cv["n_splits"]
    data_cv["delta_score_test"] = data_cv["objective_score_test"] - data_cv["objective_score_bench"]
    cache = compute.build_nocv_cache(
        nocv_base, model_col="solver_name", size_col="train_size", dataset_col="dataset_type",
        hidden_col=OUTER, test_col=TEST, bench_col=BENCH, use_delta=True,
        extra_key_cols=["fixed_split"], combination="average")
    res = compute.sample_gain_bootstrap(
        data_cv=data_cv, score_col_cv="delta_score_test", model_col="solver_name",
        n_bootstraps=n_bootstraps, method="absolute", test_size=test_size, nocv_cache=cache,
        models_to_analyze=None, study_sizes=sorted(data_cv["train_size"].unique()),
        size_col="train_size", dataset_col="dataset_type", extra_cv_cols=["fixed_split"],
        compute_train_size=False)
    res.attrs = {}
    return res


def figures(res: pd.DataFrame, out_dir: str, tag: str, n_splits_max: int = 200) -> list[str]:
    """The three plotting blocks of simulated.py (overlay, per-fixed-split full / K20 / K20_noGB_noRidge)."""
    P.set_paper_style()
    written = []
    step = max(1, int(round(n_splits_max / 5 / 10)) * 10) or max(1, n_splits_max // 5)
    ticks = list(range(0, n_splits_max + 1, step))
    ticks[0] = 1 if ticks[0] == 0 else ticks[0]
    if ticks[-1] != n_splits_max:
        ticks.append(n_splits_max)
    k20 = [1, 5, 10, 15, 20]
    res = res.copy()
    res["solver_disp"] = res["solver_name"].map(display_name)
    order = [f for f in FAMILY_COLOR if f in set(res["solver_disp"])]
    palette = {f: FAMILY_COLOR[f] for f in order}

    def draw(df, xtk, path, **extra):
        present = [f for f in order if f in set(df["solver_disp"])]
        fig = P.plot_sample_gain_vs_splits(
            df.rename(columns={"solver_name": "_solver_raw", "solver_disp": "solver_name"}),
            y_col="sample_gain", hue_col="solver_name", hue_order=present, palette=palette,
            xticks=xtk, xlim=(0, max(xtk)), direct_labels=True, direct_label_densify=10,
            figsize=(SIM.THREE_PANEL_SAMPLE_GAIN_FIGSIZE if max(xtk) == 20 else None),
            output_path=path, **extra)
        plt.close(fig)
        written.append(path)

    for ds in sorted(res["dataset_type"].unique()):
        rds = res[res["dataset_type"] == ds]
        formula = SIM.dataset_formula_label(ds)
        for n in sorted(rds["N"].unique()):
            sub = rds[(rds["N"] == n) & (rds["K"] <= 20)]
            if sub["fixed_split"].nunique() >= 2:             # fixed vs resampled overlay
                lab = {False: "Resampled split", True: "Fixed split"}
                sub = sub.assign(split_mode=sub["fixed_split"].map(lab))
                draw(sub, k20,
                     f"{out_dir}/aggregated_test_gain_k_{n}_{ds}_mse_fixedsplit_overlay_K20{tag}.pdf",
                     style_col="split_mode", style_order=["Resampled split", "Fixed split"],
                     dashes={"Resampled split": (), "Fixed split": (1, 1)}, style_legend=True,
                     style_legend_loc="upper left", style_legend_bbox=(0.02, 0.86),
                     title_text=f"{formula}, $n_{{\\mathrm{{tr}}}}={n}$", title_text_xy=(0.02, 0.97),
                     title_text_ha="left", title_text_va="top")
        for fs in sorted(rds["fixed_split"].unique()):
            rfs = rds[rds["fixed_split"] == fs]
            for n in sorted(rfs["N"].unique()):
                rn = rfs[rfs["N"] == n]
                k20_df = rn[rn["K"] <= 20]
                ranges = [("K20", k20, k20_df),
                          ("K20_noGB_noRidge", k20,
                           k20_df[~k20_df["solver_disp"].isin(["GradientBoosting", "Ridge"])])]
                if rn["K"].max() > 20:
                    ranges.insert(0, ("full", ticks, rn))
                for rtag, xtk, df in ranges:
                    if df.empty:
                        continue
                    draw(df, xtk,
                         f"{out_dir}/aggregated_test_gain_k_{n}_{ds}_fixed{fs}_mse_{rtag}{tag}.pdf",
                         title_text=f"{formula}, $n_{{\\mathrm{{tr}}}}={n}$", title_text_xy=(0.03, 0.97),
                         title_text_ha="left", title_text_va="top")
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed-source", choices=["orchestrator", "corrected"], default="orchestrator")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--n-bootstraps", type=int, default=100)
    ap.add_argument("--test-size", type=float, default=0.20)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    tag = "" if a.fixed_source == "orchestrator" else "_correctedfixed"
    data = load(a.fixed_source)
    print(data.groupby(["train_size", "fixed_split", "solver_name"]).seed.nunique().to_string())
    res = gains(data, a.n_bootstraps, a.test_size)
    res.to_parquet(f"{a.out_dir}/sample_gain_bootstrap_mse{tag}.parquet", index=False)
    summ = (res.groupby(["dataset_type", "N", "fixed_split", "solver_name", "K"])["sample_gain"]
              .agg(mean="mean", lo=lambda s: s.quantile(0.025), hi=lambda s: s.quantile(0.975)).reset_index())
    summ = summ[summ["K"].isin([1, 5, 10, 20, 50, 100, 200])]
    summ["solver_name"] = summ["solver_name"].map(display_name)
    summ.to_csv(f"{a.out_dir}/sample_gain_summary_mse{tag}.csv", index=False)
    for p in figures(res, a.out_dir, tag):
        print("Saved", p)


if __name__ == "__main__":
    main()
