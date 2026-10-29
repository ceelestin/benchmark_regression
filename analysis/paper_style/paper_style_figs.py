#!/usr/bin/env python3
"""Figures 1-6 of FIGURE_STYLE_SPEC.md (2026-10-28): paper-style MSE gain-vs-K figures, paper estimator.

Same pipeline as figure_style/code/mse_gain_figures.py (load -> gains() -> plot.plot_sample_gain_vs_splits),
with the Jean-Zay paths of the runs instead of CE's extract, the spec's fixed colours and names, figsize from
set_paper_style for every panel, and figures 3 (HPO) and 5-6 (small budgets) added.
Run in $WORK/paper_style_figs with `module load pytorch-gpu/py3/2.8.0` (seaborn); writes figures,
the bootstrap results and a K = 1 / 20 / 200 summary to --out-dir.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.ticker import MaxNLocator

import mse_gain_figures as T          # the spec's template: gains(), COLS, display_name()
import plot as P
import simulated as SIM

OUT = os.path.expandvars("$WORK/benchmark_regression/outputs")
JULY = os.path.expandvars("$SCRATCH/paper_runs/benchopt_run_2026-{}.parquet")
COLOR = {"ExtraTrees": "#1f77b4", "GradientBoosting": "#ff7f0e", "MLP": "#2ca02c", "Ridge": "#d62728",
         "RandomFeatures": "#9467bd", "MLP_GridSearch": "#8c564b", "MLP_RandomSearch": "#e377c2",
         "ExtraTrees_GridSearch": "#17becf", "ExtraTrees_RandomSearch": "#bcbd22", "MLP (32)": "#2ca02c"}
MLP32 = "MLP[hidden_layer_sizes=[32]]"
K20 = [1, 5, 10, 15, 20]
K200 = [1, 40, 80, 120, 160, 200]


def read(paths, drop=()):
    frames = []
    for f in paths:
        names = pq.read_schema(f).names
        d = pd.read_parquet(f, columns=[c for c in T.COLS if c in names])
        frames.append(d[~d.solver_name.isin(drop)])
    return frames


def prepare(frames):
    data = pd.concat(frames, ignore_index=True)
    data = SIM.process_parquet(data)
    data["procedure"] = data["p_obj_procedure"]
    data["n_splits"] = data["p_obj_n_splits"].astype(int)
    data["train_size"] = data["p_obj_train_size"].astype(int)
    data["seed"] = data["p_dataset_seed"].astype(int)
    data["fixed_split"] = data["p_obj_fixed_split"].astype(str).str.lower().eq("true")
    family = data["dataset_name"].str.extract(r"^([^\[]+)")[0]
    noise = data["p_dataset_noise"].astype(float)
    data["dataset_type"] = np.where(noise > 0, family + "_noise" + noise.astype(str), family)
    return data


def groups():
    g = {}
    g["n1000"] = read(sorted(glob.glob(f"{OUT}/paper_main_gain_mse_k200__c*.parquet")), drop={MLP32}) \
        + read(sorted(glob.glob(f"{OUT}/paper_main_gain_mse_k200_mlp64x64__c*.parquet"))) \
        + read(sorted(glob.glob(f"{OUT}/paper_main_gain_mse_k200_rff50__c*.parquet"))) \
        + read(sorted(glob.glob(f"{OUT}/hpo_mse_MLP_*__c*.parquet"))) \
        + read([JULY.format("07-27_12h40m37"), JULY.format("07-27_12h47m49")])
    g["n100"] = read([JULY.format("07-27_20h48m32"), JULY.format("07-27_21h01m05")], drop={MLP32}) \
        + read(sorted(glob.glob(f"{OUT}/paper_ntr100_k20_*_mlp64x64__c*.parquet"))) \
        + read(sorted(glob.glob(f"{OUT}/paper_ntr100_k20_*_rff50__c*.parquet")))
    small = read([JULY.format(t) for t in ("07-31_19h02m17", "07-31_19h07m06")])
    g["small"] = [f[f["p_obj_train_size"].astype(int).isin([30, 100])] for f in small]
    return g


def disp(res, mlp_label="MLP"):
    res = res.copy()
    res["solver_disp"] = res["solver_name"].map(T.display_name)
    if mlp_label != "MLP":
        res.loc[res["solver_disp"] == "MLP", "solver_disp"] = mlp_label
    return res


def draw(df, names, xtk, path, title, title_xy=(0.03, 0.97), save=True, **extra):
    present = [n for n in names if n in set(df["solver_disp"])]
    fig = P.plot_sample_gain_vs_splits(
        df.rename(columns={"solver_name": "_solver_raw", "solver_disp": "solver_name"}),
        y_col="sample_gain", hue_col="solver_name", hue_order=present, palette={n: COLOR[n] for n in present},
        xticks=xtk, xlim=(0, max(xtk)), direct_labels=True, direct_label_densify=10,
        title_text=title, title_text_xy=title_xy, title_text_ha="left", title_text_va="top",
        output_path=path if save else None, **extra)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-bootstraps", type=int, default=100)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    P.set_paper_style()
    res = {}
    for name, frames in groups().items():
        data = prepare(frames)
        print(name, data.groupby(["train_size", "fixed_split", "solver_name"]).seed.nunique().to_string(), flush=True)
        r = T.gains(data, a.n_bootstraps, 0.20)
        r.to_parquet(f"{a.out_dir}/sample_gain_bootstrap_{name}.parquet", index=False)
        res[name] = r
    f = lambda t: f"$y = X\\beta$, $n_{{\\mathrm{{tr}}}}={t}$"
    base = ["ExtraTrees", "GradientBoosting", "MLP", "Ridge", "RandomFeatures"]
    hpo = ["MLP", "MLP_GridSearch", "MLP_RandomSearch", "ExtraTrees", "ExtraTrees_GridSearch", "ExtraTrees_RandomSearch"]
    r1 = disp(res["n1000"]); r1 = r1[~r1["fixed_split"]]
    main_ = r1[r1["solver_disp"].isin(base)]
    figs = []
    fig = draw(main_, base, K200, f"{a.out_dir}/aggregated_test_gain_k_1000_sim_linear_fixedFalse_mse_full.pdf", f(1000)); figs.append(fig)
    fig = draw(main_[main_["K"] <= 20], base, K20, f"{a.out_dir}/aggregated_test_gain_k_1000_sim_linear_fixedFalse_mse_K20.pdf", f(1000)); figs.append(fig)
    h = r1[r1["solver_disp"].isin(hpo) & (r1["K"] <= 20)]
    fig = draw(h, hpo, K20, f"{a.out_dir}/aggregated_test_gain_k_1000_sim_linear_fixedFalse_mse_K20_hpo.pdf", f(1000)); figs.append(fig)
    r4 = disp(res["n100"]); r4 = r4[r4["K"] <= 20]
    r4 = r4.assign(split_mode=r4["fixed_split"].map({False: "Resampled split", True: "Fixed split"}))
    fig = draw(r4, base, K20, f"{a.out_dir}/aggregated_test_gain_k_100_sim_linear_mse_fixedsplit_overlay_K20.pdf", f(100),
               title_xy=(0.02, 0.97), style_col="split_mode", style_order=["Resampled split", "Fixed split"],
               dashes={"Resampled split": (), "Fixed split": (1, 1)}, style_legend=True,
               style_legend_loc="upper left", style_legend_bbox=(0.02, 0.86)); figs.append(fig)
    r5 = disp(res["small"], mlp_label="MLP (32)"); r5 = r5[~r5["fixed_split"] & (r5["K"] <= 20)]
    small_names = ["ExtraTrees", "GradientBoosting", "MLP (32)", "Ridge"]
    hi = r5.groupby(["N", "solver_disp", "K"])["sample_gain"].quantile(0.975)
    ymax = math.ceil(float(hi.max()))
    panels = []
    for n in (30, 100):
        fig = draw(r5[r5["N"] == n], small_names, K20, None, f(n), save=False)
        panels.append((n, fig))
    for n, fig in panels:
        ax = fig.axes[0]
        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        P.save_figure(fig, f"{a.out_dir}/small_label_gain_vs_K_N{n}_mse.pdf", format="pdf", bbox_inches="tight")
        print("panel", n, "ylim", ax.get_ylim(), "yticks", list(ax.get_yticks()))
    # K = 1 / 20 / 200 means per learner (checklist item 1)
    rows = []
    for name, r in (("n1000", r1), ("n100", r4), ("small", r5)):
        s = r.groupby(["N", "fixed_split", "solver_disp", "K"])["sample_gain"].agg(
            mean="mean", lo=lambda x: x.quantile(0.025), hi=lambda x: x.quantile(0.975)).reset_index()
        rows.append(s[s["K"].isin([1, 20, 200])].assign(group=name))
    summ = pd.concat(rows)
    summ.to_csv(f"{a.out_dir}/summary_K1_K20_K200.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
