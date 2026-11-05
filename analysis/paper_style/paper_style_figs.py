#!/usr/bin/env python3
"""Simulated sample-gain figures of the paper (mean squared error) and the gain tables behind their intervals.

Figures written to --out-dir:
  aggregated_test_gain_k_1000_sim_linear_fixedFalse_mse_full.pdf        Figure 4 (n_tr = 1,000, K up to 200)
  aggregated_test_gain_k_1000_sim_linear_fixedFalse_mse_K20.pdf         Figure 5 (left; the paper's panel is redrawn
                                                                        with the axis of the other two panels by
                                                                        fig5_common_axis.py)
  aggregated_test_gain_k_1000_sim_linear_fixedFalse_mse_K20_hpo.pdf     Figure G.1 (fixed parameters, grid and random search)
  aggregated_test_gain_k_100_sim_linear_mse_fixedsplit_overlay_K20.pdf  Figure H.1 (frozen and resampled splits)
  small_label_gain_vs_K_N30_mse.pdf, small_label_gain_vs_K_N100_mse.pdf Figure L.1 (small budgets)
It also writes the bootstrap results of each group of runs (sample_gain_bootstrap_<group>.parquet), the gain tables
read by ci_appendix.py (tables/fig4, figI1, figJ1, figM1.parquet, for Tables E.1, E.3, E.4 and E.8; the names follow
the LaTeX labels of those tables) and a K = 1 / 20 / 200 summary (summary_K1_K20_K200.csv).

Estimator: the paper's variance-equivalent test sample gain (code/compute.py) on the benchmark-adjusted MSE, with the
single-split curve averaged exactly over the order of the outer chunks (chunk_order 'average', the default;
--chunk-order stored gives the values of the stored chunk order). Thick line: the estimate on all seeds; thin lines:
20 of the 1,000 bootstrap resamples of the seeds (random_state 0). _group_loaders() lists the runs of each group
under the data root; configs/README.md gives the config of every run. The shared code is ./code next to this script,
or $PAPER_STYLE_CODE.

The figures are built in two passes, so that the stored bootstrap draws do not change when the estimate is added:
  python paper_style_figs.py --out-dir A --groups n1000 n100 small hpo_ridge hpo_rff small_mlp32 \
      mlp32_n1000 mlp32_n100 frozen_redrawn                                # bootstrap draws of every group
  python paper_style_figs.py --out-dir B --from-results A --groups \
      --plugin-groups n1000 n100 small hpo_ridge hpo_rff small_mlp32 mlp32_n1000 mlp32_n100 frozen_redrawn
The second pass computes the estimate on all seeds of every group and draws the figures in B. To stay under a
per-process CPU-time limit, the draws of a group can be computed in ranges, --bootstrap-range START STOP, each written
as sample_gain_bootstrap_<group>.part<START>-<STOP>.parquet; --from-results assembles the parts, and the parts
concatenated are identical to one run.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

sys.path.insert(0, os.environ.get("PAPER_STYLE_CODE") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.ticker import MaxNLocator

import compute
import datapaths
import mse_gain_figures as T          # COLS, TEST / BENCH / OUTER, display_name(); gains() adapted below
import plot as P
import simulated as SIM

N_BOOT, SEED, HAIR = 1000, 0, 20      # bootstrap draws, their random_state, thin lines per curve


def gains(data, n_bootstraps, test_size, random_state=SEED, plugin=False, chunk_order=None, bootstrap_range=None):
    """mse_gain_figures.gains with a random_state, the order of the outer chunks (chunk_order) and a range of draws
    (bootstrap_range); plugin=True adds the estimate on all seeds (bootstrap_iter = -1)."""
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
        compute_train_size=False, random_state=random_state, plugin_estimate=plugin,
        chunk_order=chunk_order, bootstrap_range=bootstrap_range)
    res.attrs = {}
    return res



def run_locations(root=None):
    """The benchopt outputs of the chunked runs, and the pattern of the July 2026 single-file runs, under the data root
    (--data-root, else $DATA_ROOT, else data/ at the repository root; see code/datapaths.py)."""
    return (datapaths.data_path("outputs", root=root),
            datapaths.data_path("paper_runs", "benchopt_run_2026-{}.parquet", root=root))


OUT, JULY = run_locations()
# The four simulated-data learners take matplotlib C0-C3 in every figure (ExtraTrees C0, RandomFeatures C1, MLP C2, Ridge C3).
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


def _group_loaders():
    """The groups of runs as loaders (nothing is read until called). Each group has its own random generator, so that
    adding a group does not change the bootstrap draws of the others."""
    g = {}
    # n_tr = 1,000 (Figures 4, 5 left and G.1): paper_main_gain_mse_k200{,_mlp64x64,_rff50}, hpo_mse_MLP_*, and the July
    # runs of config_hpo_test_{grid,random}search.yml (ExtraTrees grid and random search).
    g["n1000"] = lambda: read(sorted(glob.glob(f"{OUT}/paper_main_gain_mse_k200__c*.parquet")), drop={MLP32}) \
        + read(sorted(glob.glob(f"{OUT}/paper_main_gain_mse_k200_mlp64x64__c*.parquet"))) \
        + read(sorted(glob.glob(f"{OUT}/paper_main_gain_mse_k200_rff50__c*.parquet"))) \
        + read(sorted(glob.glob(f"{OUT}/hpo_mse_MLP_*__c*.parquet"))) \
        + read([JULY.format("07-27_12h40m37"), JULY.format("07-27_12h47m49")])
    # n_tr = 100 (Figure H.1): the July runs of linear_1000_K20_new_fixed_samedata.yml (frozen split, same data) and
    # linear_1000_K20_new_unfixed.yml (resampled), and paper_ntr100_k20_{fixed,resampled}_{mlp64x64,rff50}.
    g["n100"] = lambda: read([JULY.format("07-27_20h48m32"), JULY.format("07-27_21h01m05")], drop={MLP32}) \
        + read(sorted(glob.glob(f"{OUT}/paper_ntr100_k20_*_mlp64x64__c*.parquet"))) \
        + read(sorted(glob.glob(f"{OUT}/paper_ntr100_k20_*_rff50__c*.parquet")))
    # Small budgets (Figure L.1): the July runs of linear_1000_half1_K20_smalldata{,2}.yml (Ridge, ExtraTrees) and
    # small_budget_mlp64_rff50.
    g["small"] = lambda: read([JULY.format("07-31_19h07m06"), JULY.format("07-31_21h07m56")]) \
        + read(sorted(glob.glob(f"{OUT}/small_budget_mlp64_rff50__c*.parquet")))
    # Ridge grid / random search (Figure G.1), in its own group so that the draws of the n1000 curves do not change.
    g["hpo_ridge"] = lambda: read(sorted(glob.glob(f"{OUT}/hpo_mse_Ridge_*__c*.parquet")))
    # MLP (32) of the July small-budget runs (linear_1000_half2_K20_smalldata{,2}.yml) for Figure L.1, in its own group
    # so that the draws of the other small-budget curves do not change.
    g["small_mlp32"] = lambda: read([JULY.format("07-31_19h02m17"), JULY.format("07-31_21h05m17")], drop={"GradientBoosting[n_estimators=200]"})
    # One MLP throughout the paper, the one-hidden-layer MLP (32 units) of the redundancy experiments; its runs at
    # n_tr = 1,000 (paper_main_gain_mse_k200) and n_tr = 100 (July runs) get their own groups so that the draws of the
    # other curves do not change. MLP [64, 64] rows are dropped from every figure.
    g["mlp32_n1000"] = lambda: [f[f.solver_name == MLP32] for f in read(sorted(glob.glob(f"{OUT}/paper_main_gain_mse_k200__c*.parquet")))]
    g["mlp32_n100"] = lambda: [f[f.solver_name == MLP32] for f in read([JULY.format("07-27_20h48m32"), JULY.format("07-27_21h01m05")])]
    # Frozen arm of Figure H.1: runs that redraw the data at every seed, as the resampled arm does (the July run of
    # linear_1000_K20_new_fixed.yml for Ridge / ExtraTrees / MLP (32), paper_ntr100_k20_fixed_redrawn_rff50 for
    # RandomFeatures), in its own group.
    g["frozen_redrawn"] = lambda: read([JULY.format("07-27_21h00m32")], drop={"GradientBoosting[n_estimators=200]"}) \
        + read(sorted(glob.glob(f"{OUT}/paper_ntr100_k20_fixed_redrawn_rff50__c*.parquet")))
    g["hpo_rff"] = lambda: read(sorted(glob.glob(f"{OUT}/hpo_mse_RandomFeatures_*__c*.parquet")))    # Figure G.1, own group
    return g


def groups(only=None):
    """The groups of runs, read (only those named in *only*; None = all)."""
    return {k: f() for k, f in _group_loaders().items() if only is None or k in only}


def each_group(names):
    """(name, frames) for the groups in *names*, in the order of _group_loaders, one group in memory at a time."""
    for k, f in _group_loaders().items():
        if k in names:
            yield k, f()


def load_draws(d, name, n_bootstraps):
    """sample_gain_bootstrap_<name>.parquet, or its parts (--bootstrap-range runs) concatenated in order."""
    f = f"{d}/sample_gain_bootstrap_{name}.parquet"
    if os.path.exists(f):
        return pd.read_parquet(f)
    parts = sorted(glob.glob(f"{d}/sample_gain_bootstrap_{name}.part*-*.parquet"))
    if not parts:
        raise SystemExit(f"no draws for group {name} in {d}")
    rng_ = [tuple(int(v) for v in os.path.basename(p).split(".part")[1].split(".")[0].split("-")) for p in parts]
    order = sorted(range(len(parts)), key=lambda i: rng_[i])
    edges = [rng_[i] for i in order]
    if edges[0][0] != 0 or edges[-1][1] != n_bootstraps or any(a[1] != b[0] for a, b in zip(edges, edges[1:])):
        raise SystemExit(f"group {name}: parts {edges} do not cover 0..{n_bootstraps} contiguously")
    return pd.concat([pd.read_parquet(parts[i]) for i in order], ignore_index=True)


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
    global OUT, JULY
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-root", default=None,
                    help="run outputs: outputs/ and paper_runs/ under it (default: $DATA_ROOT, else data/ at the repository root)")
    ap.add_argument("--n-bootstraps", type=int, default=N_BOOT)
    ap.add_argument("--from-results", default=None, help="load sample_gain_bootstrap_<group>.parquet from this dir instead of computing")
    ap.add_argument("--plugin-groups", nargs="*", default=[],
                    help="compute the estimate on all seeds (bootstrap_iter = -1, the thick line) for these groups")
    ap.add_argument("--hpo-learners", nargs="+", default=["MLP", "ExtraTrees"], help="learners of Figure G.1")
    ap.add_argument("--y-top", choices=["quantile", "mean"], default="mean")
    ap.add_argument("--groups", nargs="*", default=["n1000", "n100", "small"], help="groups to compute (others are loaded; none = redraw only)")
    ap.add_argument("--chunk-order", choices=compute.CHUNK_ORDERS, default=None,
                    help="V1(alpha) averaged over the order of the chunks (default) or in their stored order; default $GAIN_CHUNK_ORDER, else average")
    ap.add_argument("--bootstrap-range", nargs=2, type=int, default=None, metavar=("START", "STOP"),
                    help="compute only these draws of --groups, write them as parts and stop (no figures)")
    a = ap.parse_args()
    OUT, JULY = run_locations(a.data_root)
    a.chunk_order = compute.resolve_chunk_order(a.chunk_order)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(f"{a.out_dir}/CHUNK_ORDER.txt", "a") as fh:                                           # provenance
        fh.write(f"chunk_order={a.chunk_order} n_bootstraps={a.n_bootstraps} groups={a.groups} "
                 f"plugin_groups={a.plugin_groups} bootstrap_range={a.bootstrap_range}\n")
    if a.bootstrap_range is not None:                                                               # one range of draws
        lo, hi = a.bootstrap_range
        for name, frames in each_group(a.groups):
            if name in a.groups:
                r = gains(prepare(frames), a.n_bootstraps, 0.20, chunk_order=a.chunk_order, bootstrap_range=(lo, hi))
                r.to_parquet(f"{a.out_dir}/sample_gain_bootstrap_{name}.part{lo:05d}-{hi:05d}.parquet", index=False)
                print("draws", name, lo, hi, len(r), flush=True)
        return
    P.set_paper_style()
    P.GAIN_DISPLAY["y_top"] = a.y_top
    res = {}
    M6464 = "MLP[hidden_layer_sizes=[64, 64]]"
    swap = lambda r, m32: pd.concat([r[r["solver_name"] != M6464], m32], ignore_index=True)       # MLP (32) rows replace MLP [64, 64]
    for name in ("n1000", "n100", "small", "hpo_ridge", "hpo_rff", "small_mlp32", "mlp32_n1000", "mlp32_n100", "frozen_redrawn"):
        if name not in a.groups:
            res[name] = load_draws(a.from_results, name, a.n_bootstraps)                            # or its parts
    # The thick line is the estimate on all seeds. It is computed separately (no draws) and stored as
    # sample_gain_plugin_<group>.parquet; the stored draws are unchanged, since computing the estimate does not use the
    # random generator. It is merged into the groups loaded with --from-results (see the module docstring).
    if a.plugin_groups:
        for name, frames in each_group(a.plugin_groups):                                        # one group in memory at a time
            if name in a.plugin_groups:
                pe = gains(prepare(frames), 0, 0.20, plugin=True, chunk_order=a.chunk_order)
                pe.to_parquet(f"{a.out_dir}/sample_gain_plugin_{name}.parquet", index=False)
                print("plugin", name, len(pe), flush=True)
                del frames
    for name in list(res):
        for d_ in (a.from_results, a.out_dir):
            f_ = f"{d_}/sample_gain_plugin_{name}.parquet" if d_ else None
            if f_ and os.path.exists(f_):
                res[name] = pd.concat([res[name], pd.read_parquet(f_)], ignore_index=True)
                break
    for name, frames in each_group(a.groups):                                                    # one group in memory at a time
        if name not in a.groups:
            continue
        data = prepare(frames)
        print(name, data.groupby(["train_size", "fixed_split", "solver_name"]).seed.nunique().to_string(), flush=True)
        r = gains(data, a.n_bootstraps, 0.20, chunk_order=a.chunk_order)
        r.to_parquet(f"{a.out_dir}/sample_gain_bootstrap_{name}.parquet", index=False)
        res[name] = r
    # Top-left text of a panel: the data process and n_tr on two lines, as on the real-data panels ("PCam" /
    # "n_tr = 400"). Figure G.1 keeps one line (f1): its style legend sits just below the text.
    f = lambda t: f"$y = X\\beta$\n$n_{{\\mathrm{{tr}}}}={t}$"
    f1 = lambda t: f"$y = X\\beta$, $n_{{\\mathrm{{tr}}}}={t}$"
    # Next to the two-line text, the automatic placement would move the RandomFeatures label of Figure 4 and of
    # Figure L.1 (n_tr = 100) onto the thin lines: it keeps the place it had next to the one-line text.
    keep_rf = {"fig4": {"RandomFeatures": (79.2064, 12.5388, "left", "top")},
               100: {"RandomFeatures": (12.1947, 8.7846, "right", "center")}}
    base = ["ExtraTrees", "MLP", "Ridge", "RandomFeatures"]                     # GradientBoosting is not shown
    hpo = [f"{n}{p}" for n in a.hpo_learners for p in ("", "_GridSearch", "_RandomSearch")]
    r1 = disp(pd.concat([swap(res["n1000"], res["mlp32_n1000"]), res["hpo_ridge"], res["hpo_rff"]], ignore_index=True)); r1 = r1[~r1["fixed_split"]]
    main_ = r1[r1["solver_disp"].isin(base)]
    figs = []
    tdir = f"{a.out_dir}/tables"; os.makedirs(tdir, exist_ok=True)  # tables for ci_appendix.py (Appendix E)
    save = lambda df, tag: df.drop(columns=[c for c in df.columns if df[c].dtype == object and c not in
                                            ("solver_disp", "procedure_disp", "split_mode", "solver_name", "dataset_type")],
                                   errors="ignore").to_parquet(f"{tdir}/{tag}.parquet", index=False)
    save(main_, "fig4")
    fig = draw(main_, base, K200, f"{a.out_dir}/aggregated_test_gain_k_1000_sim_linear_fixedFalse_mse_full.pdf", f(1000),
               direct_label_positions=keep_rf["fig4"]); figs.append(fig)
    fig = draw(main_[main_["K"] <= 20], base, K20, f"{a.out_dir}/aggregated_test_gain_k_1000_sim_linear_fixedFalse_mse_K20.pdf", f(1000)); figs.append(fig)
    # Figure G.1: colour = learner as in every other figure, line style = tuning procedure.
    proc = {f"{n}{p}": (n, lab) for n in a.hpo_learners
            for p, lab in (("", "Fixed"), ("_GridSearch", "Grid search"), ("_RandomSearch", "Random search"))}
    h = r1[r1["solver_disp"].isin(hpo) & (r1["K"] <= 20)].copy(); save(h, "figI1")
    h["procedure_disp"] = h["solver_disp"].map(lambda x: proc[x][1])
    h["solver_disp"] = h["solver_disp"].map(lambda x: proc[x][0])
    fig = draw(h, a.hpo_learners, K20, f"{a.out_dir}/aggregated_test_gain_k_1000_sim_linear_fixedFalse_mse_K20_hpo.pdf", f1(1000),
               style_col="procedure_disp", style_order=["Fixed", "Grid search", "Random search"],
               dashes={"Fixed": (), "Grid search": (4, 2), "Random search": (1, 1)}, style_legend=True,
               style_legend_loc="upper left", style_legend_bbox=(0.02, 0.86)); figs.append(fig)
    r4 = swap(res["n100"], res["mlp32_n100"]); r4 = r4[~r4["fixed_split"].astype(bool)]           # Figure H.1, resampled arm
    r4 = disp(pd.concat([r4, res["frozen_redrawn"]], ignore_index=True)); r4 = r4[(r4["K"] <= 20) & r4["solver_disp"].isin(base)]
    r4 = r4.assign(split_mode=r4["fixed_split"].map({False: "Resampled split", True: "Fixed split"})); save(r4, "figJ1")
    fig = P.plot_sample_gain_facets(                                                  # Figure H.1: one panel per learner
        r4, panel_col="solver_disp", panel_order=base, style_col="split_mode",
        style_order=["Resampled split", "Fixed split"], dashes={"Resampled split": (), "Fixed split": (1, 1)},
        palette={n: COLOR[n] for n in base}, xticks=K20, hairlines=30, ncols=2,
        axes_size=(2.502, 2.409), axis_label_fontsize=12,    # each panel the size of a single figure's axes
        output_path=f"{a.out_dir}/aggregated_test_gain_k_100_sim_linear_mse_fixedsplit_overlay_K20.pdf"); figs.append(fig)
    r5 = disp(swap(res["small"], res["small_mlp32"]))                                # Figure L.1; MLP (32) labelled MLP
    r5 = r5[~r5["fixed_split"] & (r5["K"] <= 20)]
    small_names = base
    save(r5, "figM1")
    top = max(P._y_top(r5[r5["N"] == n], "K", "sample_gain", ["solver_disp"]) for n in (30, 100))
    panels = []
    for n in (30, 100):
        fig = draw(r5[r5["N"] == n], small_names, K20, None, f(n), save=False, ylim_top=top,
                   direct_label_positions=keep_rf.get(n))
        panels.append((n, fig))
    for n, fig in panels:
        ax = fig.axes[0]
        ax.set_ylim(0, top)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        P.save_figure(fig, f"{a.out_dir}/small_label_gain_vs_K_N{n}_mse.pdf", format="pdf", bbox_inches="tight")
        print("panel", n, "ylim", ax.get_ylim(), "yticks", list(ax.get_yticks()))
    # K = 1 / 20 / 200 summary per learner
    rows = []
    for name, r in (("n1000", r1), ("n100", r4), ("small", r5)):
        keys = ["N", "fixed_split", "solver_disp", "K"]
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
