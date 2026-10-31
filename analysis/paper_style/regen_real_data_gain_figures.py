#!/usr/bin/env python3
"""[v3, 2026-10-29: 1,000 bootstrap draws with random_state 0; bold mean + hairlines (code/plot.py GAIN_DISPLAY);
Figures F.1 and K.1 as one panel per network (plot.plot_sample_gain_facets); Figure 5 PCam panel MCCV only
(in pcam.py). Changes from CE's driver are marked "v3".]

Rebuild the paper's five real-data sample-gain figures with the paper's current estimator
(paired + interpolated, Code/compute.py) and 95 % bootstrap percentile bands (Code/plot.py).

Each figure is produced exactly as the version now in the paper was produced (same data, same
selection, same plotting call); only the data and output locations are parameters.

    Figure 5, PCam panel  aggregated_test_gain_k_500_delta_linear_accuracy_stratFalse.pdf   pcam.py on data/pcam/results_output_pcam
    Figure 5, Yelp panel  aggregated_test_gain_k_1000_delta_linear.pdf                       nlp.py  on data/nlp/*
    Figure F.1            aggregated_test_gain_k_300_delta_linear_cv_procedures.pdf          data/pcam/results_output_pcam
    Figure K.1            aggregated_test_gain_k_500_delta_linear_nll_vs_strat_baseline.pdf  data/pcam/stratandnewmetrics
    Figure L.1            sample_gain_by_metric_wide_resnet101_2_N500_stratTrue.pdf          data/pcam/stratandnewmetrics

Usage:
    python regen_real_data_gain_figures.py --code-dir ../code --data-root ./data --out-dir OUT [--only fig5_pcam ...]
pcam.py and nlp.py are run as subprocesses with their working directory inside OUT/work/, so the
tables they write to ./results/ stay there.
"""
import argparse
import os
import shutil
import subprocess
import sys

FIGS = ["fig5_pcam", "fig5_yelp", "figF1", "figK1", "figL1"]
N_BOOT, SEED, HAIR, HAIR_FACET = 1000, 0, 20, 30                                   # v3
PIPE_ARGS = ["--n-bootstraps", str(N_BOOT), "--random-state", str(SEED), "--hairlines", str(HAIR),   # v3
             "--plugin-estimate"]                     # v5 (CE, 2026-10-31): thick line = estimate on all seeds
SINGLE_AXES = (2.502, 2.409)   # v5: inches, axes of a 3.2 x 3.0 in single-panel gain figure
PCAM_COLORS = {"DenseNet121": "#1f77b4", "MobileNetV2": "#ff7f0e", "WideResNet101_2": "#2ca02c", "ViT-B/16": "#d62728"}  # v3


def _save_table(res, out, tag):
    """v5: gain table (estimate rows bootstrap_iter = -1, draws >= 0) for the CI appendix."""
    os.makedirs(os.path.join(out, "tables"), exist_ok=True)
    res.to_parquet(os.path.join(out, "tables", f"{tag}.parquet"), index=False)


def run_pipeline(code_dir, script, args, work, plots):
    os.makedirs(work, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=code_dir, MPLBACKEND="Agg")
    cmd = [sys.executable, os.path.join(code_dir, script), "--plots-dir", plots] + args
    with open(os.path.join(work, "run.log"), "w") as log:
        rc = subprocess.call(cmd, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
    if rc != 0:
        raise SystemExit(f"{script} failed (exit {rc}), see {work}/run.log")


def fig5_pcam(code_dir, data, out):
    work = os.path.join(out, "work", "pcam"); plots = os.path.join(work, "plots")
    run_pipeline(code_dir, "pcam.py", ["--data-dir", os.path.join(data, "pcam", "results_output_pcam")] + PIPE_ARGS + ["--save-gains", os.path.join(out, "tables")], work, plots)
    shutil.copy(os.path.join(plots, "aggregated_test_gain_k_500_delta_linear_accuracy_stratFalse.pdf"), out)


def fig5_yelp(code_dir, data, out):
    work = os.path.join(out, "work", "nlp"); plots = os.path.join(work, "plots")
    dirs = [os.path.join(data, "nlp", d) for d in ("yelp_runs_a", "yelp_runs_b", "yelp_qwen3_0.6b")]
    run_pipeline(code_dir, "nlp.py", ["--data-dirs", *dirs] + PIPE_ARGS + ["--save-gains", os.path.join(out, "tables")], work, plots)
    shutil.copy(os.path.join(plots, "aggregated_test_gain_k_1000_delta_linear.pdf"), out)


def figF1(code_dir, data, out):
    """Figure F.1: PCam MCCV vs Repeated 5-Fold, n_tr = 300, accuracy (former cvproc.py)."""
    import pandas as pd, pcam, compute, pipeline_common as C, plot as P
    work = os.path.join(out, "work", "figF1"); os.makedirs(work, exist_ok=True)
    data_ = pcam.load_cv_data(os.path.join(data, "pcam", "results_output_pcam"))
    if "stratified" not in data_.columns:
        data_["stratified"] = False
    tc, bc, hc = "test_accuracy", "benchmark_accuracy", "hidden_test_accuracies"
    gc = ["seed", "model", "study_size", "cross_validation", "procedure", "stratified"]
    dp = data_[[c for c in gc + ["fold", tc, bc, hc] if c in data_.columns]].copy()
    dp = dp.drop_duplicates(subset=gc + ["fold"]).sort_values(gc + ["fold"])
    dp = C.coerce_scalar_columns(dp, columns=[tc, bc], scalar_extractor=pcam._extract_scalar)
    g = dp.groupby(gc, sort=False); dp["n_splits"] = g.cumcount() + 1
    dp["objective_score_test"] = g[tc].cumsum() / dp["n_splits"]
    dp["objective_score_bench"] = g[bc].cumsum() / dp["n_splits"]
    cv = dp[dp["cross_validation"] & (dp["study_size"] == 300)].copy()
    cvm_delta = C.build_delta_cv_table(cv, cumulative_test_col="objective_score_test",
                                       cumulative_bench_col="objective_score_bench", fold_test_col=tc, fold_bench_col=bc)
    nocv = dp[(dp["cross_validation"]) & (dp["study_size"] == 300) & (dp["n_splits"] == 1) & (dp["procedure"] == "MCCV")].copy()
    dfk = compute.expand_hidden_scores(nocv, scores_col=hc, id_cols=["seed", "model", "study_size"])
    _, comb = compute.build_nocv_combined_df(dfk, nocv, model_col="model", score_raw_col="objective_score_test",
                                             bench_col="objective_score_bench", include_k0=True)
    res = compute.sample_gain_bootstrap(data_cv=cvm_delta, score_col_cv="delta_score_test", model_col="model",
                                        n_bootstraps=N_BOOT, random_state=SEED, plugin_estimate=True,                              # v3
                                        method="absolute", test_size=0.2, df_nocv_combined=comb,
                                        score_col_nocv="delta_score_combined", models_to_analyze=pcam.DEFAULT_MODELS,
                                        study_sizes=[300], compute_train_size=False)
    _save_table(res, out, "figF1")                                                                        # v5
    res = res[res["K"] <= 20].copy()
    res["procedure_display"] = res["cv_procedure"].map({"MCCV": "MCCV", "RepeatedKFold": "Repeated 5-Fold"})
    cnn = ["densenet121", "mobilenet_v2", "wide_resnet101_2"]          # the figure excludes ViT
    res = res[res["model"].isin(cnn)].copy()
    P.set_paper_style()
    res["model_disp"] = res["model"].map(pcam.MODEL_DISPLAY_NAMES)                                      # v3: one panel per network
    order = [pcam.MODEL_DISPLAY_NAMES[m] for m in cnn]
    P.plot_sample_gain_facets(
        res, panel_col="model_disp", panel_order=order, style_col="procedure_display",
        style_order=["MCCV", "Repeated 5-Fold"], dashes={"MCCV": (), "Repeated 5-Fold": (4, 2)},
        palette={m: PCAM_COLORS[m] for m in order}, xticks=[1, 5, 10, 15, 20], hairlines=HAIR_FACET,
        axes_size=SINGLE_AXES, axis_label_fontsize=12,                                                  # v5: 1 x 3
        output_path=os.path.join(out, "aggregated_test_gain_k_300_delta_linear_cv_procedures.pdf"))


def figK1(code_dir, data, out):
    """Figure K.1: PCam NLL, n_tr = 500, unstratified and stratified CV against the stratified baseline."""
    import pandas as pd, pcam, compute, pipeline_common as C, plot as P
    work = os.path.join(out, "work", "figK1"); os.makedirs(work, exist_ok=True)
    d = pcam.load_cv_data(os.path.join(data, "pcam", "stratandnewmetrics")); d = d[d.study_size == 500].copy()
    tc, bc, hc = "test_nll", "benchmark_nll", "hidden_test_nlls"
    gc = ["seed", "model", "study_size", "cross_validation", "procedure", "stratified"]
    dp = d[[c for c in gc + ["fold", tc, bc, hc] if c in d.columns]].copy()
    dp = C.coerce_scalar_columns(dp, columns=[tc, bc], scalar_extractor=pcam._extract_scalar).sort_values(gc + ["fold"])
    g = dp.groupby(gc, sort=False); dp["n_splits"] = g.cumcount() + 1
    dp["objective_score_test"] = g[tc].cumsum() / dp["n_splits"]; dp["objective_score_bench"] = g[bc].cumsum() / dp["n_splits"]
    cvd = C.build_delta_cv_table(dp, cumulative_test_col="objective_score_test", cumulative_bench_col="objective_score_bench",
                                 fold_test_col=tc, fold_bench_col=bc)
    base = dp[(dp.n_splits == 1) & (dp.procedure == "MCCV") & (dp.stratified == True)].copy()
    dfk = compute.expand_hidden_scores(base, scores_col=hc, id_cols=["seed", "model", "study_size"])
    _, comb = compute.build_nocv_combined_df(dfk, base, model_col="model", score_raw_col="objective_score_test",
                                             bench_col="objective_score_bench", include_k0=True)
    df_cross = pd.concat([comb.assign(stratified=sv) for sv in [False, True]], ignore_index=True)
    res = compute.sample_gain_bootstrap(data_cv=cvd, score_col_cv="delta_score_test", model_col="model", n_bootstraps=N_BOOT,
                                        random_state=SEED, plugin_estimate=True,                                                   # v3
                                        method="absolute", test_size=1 / 5, df_nocv_combined=df_cross,
                                        score_col_nocv="delta_score_combined", study_sizes=[500],
                                        extra_cv_cols=["stratified"], compute_train_size=False)
    _save_table(res, out, "figK1")                                                                        # v5
    res = res[res.K <= 20].copy(); res["stratified_display"] = res["stratified"].map({False: "Unstratified", True: "Stratified"})
    P.set_paper_style()
    res["model_disp"] = res["model"].map(pcam.MODEL_DISPLAY_NAMES)                                      # v3: one panel per network
    order = [pcam.MODEL_DISPLAY_NAMES[m] for m in pcam.DEFAULT_MODELS if m in set(res["model"])]
    P.plot_sample_gain_facets(
        res, panel_col="model_disp", panel_order=order, style_col="stratified_display",
        style_order=["Unstratified", "Stratified"], dashes={"Unstratified": (), "Stratified": (4, 2)},
        palette={m: PCAM_COLORS[m] for m in order}, xticks=[1, 5, 10, 15, 20], hairlines=HAIR_FACET,
        ncols=2, axes_size=SINGLE_AXES, axis_label_fontsize=12,                                         # v5: 2 x 2
        output_path=os.path.join(out, "aggregated_test_gain_k_500_delta_linear_nll_vs_strat_baseline.pdf"))


def figL1(code_dir, data, out):
    """Figure L.1: PCam WideResNet101_2, n_tr = 500, stratified: gain by metric (accuracy, Brier, NLL)."""
    import pandas as pd, seaborn as sns, matplotlib.pyplot as plt, pcam, compute, pipeline_common as C, plot as P
    d = pcam.load_cv_data(os.path.join(data, "pcam", "stratandnewmetrics"))
    d = d[(d.model == "wide_resnet101_2") & (d.study_size == 500) & (d.stratified == True)].copy()
    gc = ["seed", "model", "study_size", "cross_validation", "procedure", "stratified"]
    metrics = [("Accuracy", "test_accuracy", "benchmark_accuracy", "hidden_test_accuracies"),
               ("Brier", "test_brier", "benchmark_brier", "hidden_test_briers"),
               ("NLL", "test_nll", "benchmark_nll", "hidden_test_nlls")]
    allres = []
    for disp, tc, bc, hc in metrics:
        dp = d[[c for c in gc + ["fold", tc, bc, hc] if c in d.columns]].copy()
        dp = C.coerce_scalar_columns(dp, columns=[tc, bc], scalar_extractor=pcam._extract_scalar).sort_values(gc + ["fold"])
        g = dp.groupby(gc, sort=False); dp["n_splits"] = g.cumcount() + 1
        dp["objective_score_test"] = g[tc].cumsum() / dp["n_splits"]; dp["objective_score_bench"] = g[bc].cumsum() / dp["n_splits"]
        cvd = C.build_delta_cv_table(dp, cumulative_test_col="objective_score_test", cumulative_bench_col="objective_score_bench",
                                     fold_test_col=tc, fold_bench_col=bc)
        nocv = dp[(dp.n_splits == 1) & (dp.procedure == "MCCV")].copy()
        dfk = compute.expand_hidden_scores(nocv, scores_col=hc, id_cols=["seed", "model", "study_size"])
        _, comb = compute.build_nocv_combined_df(dfk, nocv, model_col="model", score_raw_col="objective_score_test",
                                                 bench_col="objective_score_bench", include_k0=True)
        comb["stratified"] = True
        r = compute.sample_gain_bootstrap(data_cv=cvd, score_col_cv="delta_score_test", model_col="model", n_bootstraps=N_BOOT,
                                          random_state=SEED, plugin_estimate=True,                                                 # v3
                                          method="absolute", test_size=1 / 5, df_nocv_combined=comb,
                                          score_col_nocv="delta_score_combined", study_sizes=[500],
                                          extra_cv_cols=["stratified"], compute_train_size=False)
        r["metric"] = disp; allres.append(r)
    res = pd.concat(allres, ignore_index=True); _save_table(res, out, "figL1"); res = res[res.K <= 20]    # v5
    pal = dict(zip(["Accuracy", "Brier", "NLL"], sns.color_palette("colorblind", 3)))
    P.set_paper_style()
    fig = P.plot_sample_gain_vs_splits(
        res, y_col="sample_gain", hue_col="metric", hue_order=["Accuracy", "Brier", "NLL"], palette=pal,
        xticks=[1, 5, 10, 15, 20], title_text=r"PCam WideResNet101_2, $n_{\mathrm{tr}}=500$",
        title_text_xy=(0.5, 0.02), title_text_ha="center", title_text_va="bottom", xlim=(0, 20), direct_labels=True,
        hairlines=HAIR,                                                                                       # v3
        output_path=os.path.join(out, "sample_gain_by_metric_wide_resnet101_2_N500_stratTrue.pdf"))
    plt.close(fig)
    k20 = res[res.K == 20]
    est = k20[k20.bootstrap_iter < 0].groupby("metric").sample_gain.mean().round(2).to_dict()             # v5
    ci = k20[k20.bootstrap_iter >= 0].groupby("metric").sample_gain.quantile([.025, .975]).round(2).unstack().to_dict("index")
    print("Figure L.1 caption values, G_20 by metric (estimate on all seeds):", est, "95% percentile:", ci)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--code-dir", default=os.path.join(here, "..", "code"))
    ap.add_argument("--data-root", default=os.path.join(here, "data"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--only", nargs="+", choices=FIGS, default=FIGS)
    a = ap.parse_args()
    code_dir, data, out = (os.path.abspath(p) for p in (a.code_dir, a.data_root, a.out_dir))
    os.makedirs(out, exist_ok=True)
    sys.path.insert(0, code_dir)
    import matplotlib; matplotlib.use("Agg")
    import warnings; warnings.filterwarnings("ignore")
    for name in a.only:
        print(f"--- {name}", flush=True)
        globals()[name](code_dir, data, out)
        print(f"    done", flush=True)


if __name__ == "__main__":
    main()
