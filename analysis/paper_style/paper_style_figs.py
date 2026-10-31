#!/usr/bin/env python3
"""[v4, 2026-10-30 (CE): GradientBoosting dropped; Figure J.1 as one panel per learner (2 x 2); Figure M.1 with the
main-text learners (Ridge and ExtraTrees from the July sweeps, MLP (64, 64) and RandomFeatures from
small_budget_mlp64_rff50), gains for all eight budgets; --groups / --from-results to recompute only some groups.]
[v3, 2026-10-29: 1,000 bootstrap draws with random_state 0; bold mean + 20 hairlines per curve
(code/plot.py GAIN_DISPLAY); Figure M.1 panels share the y-axis top (larger 99th percentile of the two).]

Figures 1-6 of FIGURE_STYLE_SPEC.md (2026-10-28): paper-style MSE gain-vs-K figures, paper estimator.

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

import compute
import mse_gain_figures as T          # the spec's template: COLS, display_name(); gains() copied below (v3)
import plot as P
import simulated as SIM

N_BOOT, SEED, HAIR = 1000, 0, 20      # v3


def gains(data, n_bootstraps, test_size, random_state=SEED, plugin=False):
    """mse_gain_figures.gains, unchanged except for random_state (v3)."""
    data_cv = data[data["procedure"].isin(["ShuffleSplit", "RepeatedKFold"])].sort_values(
        ["objective_name", "solver_name", "dataset_name", "idx_rep"]).copy()
    nocv_base = data_cv[data_cv["idx_rep"] == 0].copy()
    data_cv["objective_score_test"] = data_cv[T.TEST]
    data_cv["objective_score_bench"] = data_cv[T.BENCH]
    grp = data_cv.groupby(["objective_name", "solver_name", "dataset_name"])
    data_cv["n_splits"] = grp.cumcount() + 1
    data_cv["objective_score_test"] = grp["objective_score_test"].cumsum() / data_cv["n_splits"]
    data_cv["objective_score_bench"] = grp["objective_score_bench"].cumsum() / data_cv["n_splits"]
    data_cv["delta_score_test"] = data_cv["objective_score_test"] - data_cv["objective_score_bench"]
    cache = compute.build_nocv_cache(
        nocv_base, model_col="solver_name", size_col="train_size", dataset_col="dataset_type",
        hidden_col=T.OUTER, test_col=T.TEST, bench_col=T.BENCH, use_delta=True,
        extra_key_cols=["fixed_split"], combination="average")
    res = compute.sample_gain_bootstrap(
        data_cv=data_cv, score_col_cv="delta_score_test", model_col="solver_name",
        n_bootstraps=n_bootstraps, method="absolute", test_size=test_size, nocv_cache=cache,
        models_to_analyze=None, study_sizes=sorted(data_cv["train_size"].unique()),
        size_col="train_size", dataset_col="dataset_type", extra_cv_cols=["fixed_split"],
        compute_train_size=False, random_state=random_state, plugin_estimate=plugin)
    res.attrs = {}
    return res

OUT = os.path.expandvars("$WORK/benchmark_regression/outputs")
JULY = os.path.expandvars("$SCRATCH/paper_runs/benchopt_run_2026-{}.parquet")
# v4e (CE, 2026-10-30): the four simulated-data learners take matplotlib C0-C3 (ExtraTrees C0, RandomFeatures C1, MLP C2, Ridge C3).
COLOR = {"ExtraTrees": "#1f77b4", "GradientBoosting": "#ff7f0e", "MLP": "#2ca02c", "Ridge": "#d62728",
         "RandomFeatures": "#ff7f0e", "MLP_GridSearch": "#8c564b", "MLP_RandomSearch": "#e377c2",
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
    g["small"] = read([JULY.format("07-31_19h07m06"), JULY.format("07-31_21h07m56")]) \
        + read(sorted(glob.glob(f"{OUT}/small_budget_mlp64_rff50__c*.parquet")))
    # v4c (CE, 2026-10-30): Ridge grid/random search for Figure I.1, in its own group so that the bootstrap
    # draws (one RNG per group) of the n1000 curves already in the paper do not change.
    g["hpo_ridge"] = read(sorted(glob.glob(f"{OUT}/hpo_mse_Ridge_*__c*.parquet")))
    # v4e (CE, 2026-10-30): Figure M.1 back to the one-hidden-layer MLP (32 units) of the July small-budget runs,
    # in its own group so that the bootstrap draws of the other small-budget curves do not change.
    g["small_mlp32"] = read([JULY.format("07-31_19h02m17"), JULY.format("07-31_21h05m17")], drop={"GradientBoosting[n_estimators=200]"})
    # v4f (CE, 2026-10-30): one MLP throughout the paper, the one-hidden-layer MLP (32 units) of the redundancy
    # experiments; its runs at n_tr = 1,000 (job 1426287) and n_tr = 100 (July sweeps) get their own groups so that
    # the bootstrap draws of the other curves do not change. MLP [64, 64] rows are dropped from every figure.
    g["mlp32_n1000"] = [f[f.solver_name == MLP32] for f in read(sorted(glob.glob(f"{OUT}/paper_main_gain_mse_k200__c*.parquet")))]
    g["mlp32_n100"] = [f[f.solver_name == MLP32] for f in read([JULY.format("07-27_20h48m32"), JULY.format("07-27_21h01m05")])]
    # v4g (CE, 2026-10-30): Figure J.1's frozen arm from runs that redraw the data at every seed, as the reshuffled
    # arm does (July 21h00m32 for Ridge / ExtraTrees / MLP (32), job 1747862 for RandomFeatures), in its own group.
    g["frozen_redrawn"] = read([JULY.format("07-27_21h00m32")], drop={"GradientBoosting[n_estimators=200]"}) \
        + read(sorted(glob.glob(f"{OUT}/paper_ntr100_k20_fixed_redrawn_rff50__c*.parquet")))
    g["hpo_rff"] = read(sorted(glob.glob(f"{OUT}/hpo_mse_RandomFeatures_*__c*.parquet")))    # v4d, same reason
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
        xticks=xtk, xlim=(0, max(xtk)), direct_labels=True, direct_label_densify=10, hairlines=HAIR,
        title_text=title, title_text_xy=title_xy, title_text_ha="left", title_text_va="top",
        output_path=path if save else None, **extra)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-bootstraps", type=int, default=N_BOOT)
    ap.add_argument("--from-results", default=None, help="load sample_gain_bootstrap_<group>.parquet from this dir instead of computing")
    ap.add_argument("--plugin-groups", nargs="*", default=[],
                    help="v5: compute only the estimate on all seeds (bootstrap_iter = -1) for these groups")
    ap.add_argument("--hpo-learners", nargs="+", default=["MLP", "ExtraTrees"], help="learners of Figure I.1 (v4d)")
    ap.add_argument("--y-top", choices=["quantile", "mean"], default="mean")
    ap.add_argument("--groups", nargs="*", default=["n1000", "n100", "small"], help="groups to compute (others are loaded; none = redraw only)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    P.set_paper_style()
    P.GAIN_DISPLAY["y_top"] = a.y_top
    res = {}
    M6464 = "MLP[hidden_layer_sizes=[64, 64]]"
    swap = lambda r, m32: pd.concat([r[r["solver_name"] != M6464], m32], ignore_index=True)       # v4f
    for name in ("n1000", "n100", "small", "hpo_ridge", "hpo_rff", "small_mlp32", "mlp32_n1000", "mlp32_n100", "frozen_redrawn"):
        if name not in a.groups:
            res[name] = pd.read_parquet(f"{a.from_results}/sample_gain_bootstrap_{name}.parquet")
    # v5 (CE, 2026-10-31): the thick line is the estimate on all seeds. It is computed separately (no draws) and
    # stored as sample_gain_plugin_<group>.parquet; the stored draws are unchanged, since computing the estimate
    # does not use the random generator.
    if a.plugin_groups:
        for name, frames in groups().items():
            if name in a.plugin_groups:
                pe = gains(prepare(frames), 0, 0.20, plugin=True)
                pe.to_parquet(f"{a.out_dir}/sample_gain_plugin_{name}.parquet", index=False)
                print("plugin", name, len(pe), flush=True)
    for name in list(res):
        for d_ in (a.from_results, a.out_dir):
            f_ = f"{d_}/sample_gain_plugin_{name}.parquet" if d_ else None
            if f_ and os.path.exists(f_):
                res[name] = pd.concat([res[name], pd.read_parquet(f_)], ignore_index=True)
                break
    for name, frames in groups().items():
        if name not in a.groups:
            continue
        data = prepare(frames)
        print(name, data.groupby(["train_size", "fixed_split", "solver_name"]).seed.nunique().to_string(), flush=True)
        r = gains(data, a.n_bootstraps, 0.20)
        r.to_parquet(f"{a.out_dir}/sample_gain_bootstrap_{name}.parquet", index=False)
        res[name] = r
    f = lambda t: f"$y = X\\beta$, $n_{{\\mathrm{{tr}}}}={t}$"
    base = ["ExtraTrees", "MLP", "Ridge", "RandomFeatures"]                     # v4: no GradientBoosting
    hpo = [f"{n}{p}" for n in a.hpo_learners for p in ("", "_GridSearch", "_RandomSearch")]
    r1 = disp(pd.concat([swap(res["n1000"], res["mlp32_n1000"]), res["hpo_ridge"], res["hpo_rff"]], ignore_index=True)); r1 = r1[~r1["fixed_split"]]
    main_ = r1[r1["solver_disp"].isin(base)]
    figs = []
    tdir = f"{a.out_dir}/tables"; os.makedirs(tdir, exist_ok=True)                        # v5: tables for the CI appendix
    save = lambda df, tag: df.drop(columns=[c for c in df.columns if df[c].dtype == object and c not in
                                            ("solver_disp", "procedure_disp", "split_mode", "solver_name", "dataset_type")],
                                   errors="ignore").to_parquet(f"{tdir}/{tag}.parquet", index=False)
    save(main_, "fig4")
    fig = draw(main_, base, K200, f"{a.out_dir}/aggregated_test_gain_k_1000_sim_linear_fixedFalse_mse_full.pdf", f(1000)); figs.append(fig)
    fig = draw(main_[main_["K"] <= 20], base, K20, f"{a.out_dir}/aggregated_test_gain_k_1000_sim_linear_fixedFalse_mse_K20.pdf", f(1000)); figs.append(fig)
    # v4b (CE, 2026-10-30): colour = learner as in every other figure, line style = tuning procedure.
    proc = {f"{n}{p}": (n, lab) for n in a.hpo_learners
            for p, lab in (("", "Fixed"), ("_GridSearch", "Grid search"), ("_RandomSearch", "Random search"))}
    h = r1[r1["solver_disp"].isin(hpo) & (r1["K"] <= 20)].copy(); save(h, "figI1")
    h["procedure_disp"] = h["solver_disp"].map(lambda x: proc[x][1])
    h["solver_disp"] = h["solver_disp"].map(lambda x: proc[x][0])
    fig = draw(h, a.hpo_learners, K20, f"{a.out_dir}/aggregated_test_gain_k_1000_sim_linear_fixedFalse_mse_K20_hpo.pdf", f(1000),
               style_col="procedure_disp", style_order=["Fixed", "Grid search", "Random search"],
               dashes={"Fixed": (), "Grid search": (4, 2), "Random search": (1, 1)}, style_legend=True,
               style_legend_loc="upper left", style_legend_bbox=(0.02, 0.86)); figs.append(fig)
    r4 = swap(res["n100"], res["mlp32_n100"]); r4 = r4[~r4["fixed_split"].astype(bool)]           # v4g: reshuffled arm
    r4 = disp(pd.concat([r4, res["frozen_redrawn"]], ignore_index=True)); r4 = r4[(r4["K"] <= 20) & r4["solver_disp"].isin(base)]
    r4 = r4.assign(split_mode=r4["fixed_split"].map({False: "Resampled split", True: "Fixed split"})); save(r4, "figJ1")
    fig = P.plot_sample_gain_facets(                                                  # v4: one panel per learner
        r4, panel_col="solver_disp", panel_order=base, style_col="split_mode",
        style_order=["Resampled split", "Fixed split"], dashes={"Resampled split": (), "Fixed split": (1, 1)},
        palette={n: COLOR[n] for n in base}, xticks=K20, hairlines=30, ncols=2,
        axes_size=(2.502, 2.409), axis_label_fontsize=12,    # v4g: each panel the size of a single figure's axes
        output_path=f"{a.out_dir}/aggregated_test_gain_k_100_sim_linear_mse_fixedsplit_overlay_K20.pdf"); figs.append(fig)
    r5 = disp(swap(res["small"], res["small_mlp32"]))                                # v4f: MLP (32) labelled MLP
    r5 = r5[~r5["fixed_split"] & (r5["K"] <= 20)]
    small_names = base
    save(r5, "figM1")
    top = max(P._y_top(r5[r5["N"] == n], "K", "sample_gain", ["solver_disp"]) for n in (30, 100))
    panels = []
    for n in (30, 100):
        fig = draw(r5[r5["N"] == n], small_names, K20, None, f(n), save=False, ylim_top=top)
        panels.append((n, fig))
    for n, fig in panels:
        ax = fig.axes[0]
        ax.set_ylim(0, top)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        P.save_figure(fig, f"{a.out_dir}/small_label_gain_vs_K_N{n}_mse.pdf", format="pdf", bbox_inches="tight")
        print("panel", n, "ylim", ax.get_ylim(), "yticks", list(ax.get_yticks()))
    # K = 1 / 20 / 200 means per learner (checklist item 1)
    rows = []
    for name, r in (("n1000", r1), ("n100", r4), ("small", r5)):
        keys = ["N", "fixed_split", "solver_disp", "K"]                                        # v5
        est = r[r["bootstrap_iter"] < 0].groupby(keys)["sample_gain"].mean().rename("estimate")
        s = r[r["bootstrap_iter"] >= 0].groupby(keys)["sample_gain"].agg(
            mean="mean", lo=lambda x: x.quantile(0.025), hi=lambda x: x.quantile(0.975),
            at_floor=lambda x: (x <= 1.0001).mean()).join(est).reset_index()
        rows.append(s[s["K"].isin([1, 20, 200])].assign(group=name))
    summ = pd.concat(rows)
    summ.to_csv(f"{a.out_dir}/summary_K1_K20_K200.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
