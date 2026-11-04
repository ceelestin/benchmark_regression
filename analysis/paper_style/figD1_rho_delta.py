#!/usr/bin/env python3
"""Figure D.1: fold-error correlation against the sample gain, from the current runs.

G_20 comes from the gain tables of the order-averaged estimator (--sim-gain-dir, --pcam-gain, --yelp-aligned-gain,
--qwen-gain); the y axis goes to 15 (ViT-B/16 at n_tr = 24 reaches 14.0). The code directory is ./code next to this
script (or $PAPER_STYLE_CODE); the simulated runs are read as in paper_style_figs.py.

Estimated fold-error correlation rho_delta = tau_te / sigma2_te|K against the observed sample gain at K = 20, with the
ICC curve K / (1 + (K - 1) rho_delta). tau_te and sigma2_te|K are the ANOVA estimates of compute.compute_tau
(tau = B - W / K, sigma2 = W + tau) on the per-split delta-scores (test-fold score minus the same predictor's
benchmarking-set score) of the first K = 20 splits of every seed. G_20 is the estimate on all seeds (bootstrap_iter
= -1) that the paper's figures and Appendix E report.

Configurations, each on the metric of its gain figure:
  simulated linear process, mean squared error: the four learners of Figure 4 (Ridge, ExtraTrees, MLP (32),
    RandomFeatures) at n_tr = 1,000 (1,000 seeds) and n_tr = 100 (resampled arm of Figure H.1, 1,000 seeds);
  PCam, accuracy: the four networks at n_tr = 24, 80, 240, 400, 640 (MCCV, unstratified, 50 seeds; Figure 5 middle,
    Figure I.1);
  Yelp, accuracy: BERT and XLM-RoBERTa at n_tr = 1,000 and 3,000 (re-run on the PCam protocol, 50 seeds), Qwen3-0.6B
    at n_tr = 1,000 (50 seeds).
Writes figD1_rho_delta.csv and rho_delta_vs_gain.pdf to --out-dir. Reads the run outputs under the data root and the
gain tables written by paper_style_figs.py (--sim-gain-dir) and regen_real_data_gain_figures.py (--real-dir).
"""
import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("PAPER_STYLE_CODE") or os.path.join(HERE, "code"))
sys.path.insert(0, HERE)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import compute
import datapaths
import mse_gain_figures as T
import plot as P
import paper_style_figs as H

K = 20
SIM_SOLVERS = {"Ridge[alpha=1613]": "Ridge", "ExtraTrees_noHPO[n_estimators=200]": "ExtraTrees",
               "MLP[hidden_layer_sizes=[32]]": "MLP", "KernelApprox[alpha=0.001,gamma=0.1,kind=rff,n_components=50]": "RandomFeatures"}
PCAM = {"densenet121": "DenseNet121", "mobilenet_v2": "MobileNetV2", "wide_resnet101_2": "WideResNet101_2", "vit_b_16": "ViT-B/16"}
NLP = {"bert": "BERT", "roberta": "XLM-RoBERTa", "qwen3-0.6b": "Qwen3-0.6B"}
# colours: simulated learners as everywhere in the paper (ExtraTrees C0, RandomFeatures C1, MLP C2, Ridge C3); PCam
# networks as in Figure 5 (middle); Yelp networks as in Figure 5 (right)
COLOR = {("simulated", "ExtraTrees"): "#1f77b4", ("simulated", "RandomFeatures"): "#ff7f0e",
         ("simulated", "MLP"): "#2ca02c", ("simulated", "Ridge"): "#d62728",
         ("PCam", "DenseNet121"): "#1f77b4", ("PCam", "MobileNetV2"): "#ff7f0e", ("PCam", "WideResNet101_2"): "#2ca02c",
         ("PCam", "ViT-B/16"): "#d62728", ("Yelp", "BERT"): "#1f77b4", ("Yelp", "XLM-RoBERTa"): "#ff7f0e",
         ("Yelp", "Qwen3-0.6B"): "#2ca02c"}
MARKER = {"simulated": "o", "PCam": "s", "Yelp": "D"}  # Figure 6 uses triangles for noisy Ridge
SIZE = {"simulated": 34, "PCam": 28, "Yelp": 22}


def simulated_rows(sim_gain_dir):
    out = H.OUT
    sets = {1000: sorted(glob.glob(f"{out}/paper_main_gain_mse_k200__c*.parquet"))
                  + sorted(glob.glob(f"{out}/paper_main_gain_mse_k200_rff50__c*.parquet")),
            100: [H.JULY.format("07-27_20h48m32"), H.JULY.format("07-27_21h01m05")]
                 + sorted(glob.glob(f"{out}/paper_ntr100_k20_resampled_rff50__c*.parquet"))}
    rows = []
    for n_tr, paths in sets.items():
        frames = [f[f.solver_name.isin(SIM_SOLVERS)] for f in H.read(paths)]
        raw = pd.concat(frames, ignore_index=True)
        # one row per (learner, seed, split) and arm: process_parquet would silently average a duplicated run
        dup = raw.duplicated(["solver_name", "p_dataset_seed", "idx_rep", "p_obj_fixed_split", "p_obj_train_size",
                              "p_obj_procedure", "p_dataset_noise"])
        assert not dup.any(), f"{int(dup.sum())} duplicated (seed, split) rows at n_tr = {n_tr}"
        d = H.prepare(frames)
        d = d[(~d.fixed_split) & d.procedure.eq("ShuffleSplit") & (d.train_size == n_tr) & (d.idx_rep < K)].copy()
        d["delta_fold"] = d[T.TEST].astype(float) - d[T.BENCH].astype(float)
        n_folds = d.groupby(["solver_name", "seed"]).idx_rep.nunique()
        assert (n_folds == K).all(), n_folds.value_counts()
        tau = compute.compute_tau(d, score_col="delta_fold", seed_col="seed", group_cols=["solver_name"], n_bootstraps=0)
        for r in tau.itertuples():
            rows.append({"pipeline": "simulated", "learner": SIM_SOLVERS[r.solver_name], "n_tr": n_tr, "metric": "MSE",
                         "tau_te": r.tau, "sigma2_te_K": r.sigma2, "n_seeds": int(d[d.solver_name == r.solver_name].seed.nunique())})
    sim = pd.DataFrame(rows)
    # G_20: estimate on all seeds from the gain tables of the simulated figures
    g = []
    for grp, n_tr in (("n1000", 1000), ("mlp32_n1000", 1000), ("n100", 100), ("mlp32_n100", 100)):
        t = pd.read_parquet(f"{sim_gain_dir}/sample_gain_plugin_{grp}.parquet")
        t = t[(t.K == K) & (~t.fixed_split.astype(bool)) & t.solver_name.isin(SIM_SOLVERS) & (t.bootstrap_iter < 0)]
        g += [{"learner": SIM_SOLVERS[s], "n_tr": n_tr, "G20": float(v)} for s, v in zip(t.solver_name, t.sample_gain)]
    return sim.merge(pd.DataFrame(g), on=["learner", "n_tr"], how="left", validate="one_to_one")


def pcam_rows(tau_table, gain_table):
    t = pd.read_parquet(tau_table)
    t = t[(t.procedure == "MCCV") & (~t.stratified.astype(bool))]
    g = pd.read_parquet(gain_table)
    g = g[(g.K == K) & (g.bootstrap_iter < 0) & (g.cv_procedure == "MCCV") & (~g.stratified.astype(bool))]
    g = g.rename(columns={"model": "algorithm", "N": "study_size", "sample_gain": "G20"})[["algorithm", "study_size", "G20"]]
    m = t.merge(g, on=["algorithm", "study_size"], how="left", validate="one_to_one")
    return pd.DataFrame({"pipeline": "PCam", "learner": m.algorithm.map(PCAM), "n_tr": (0.8 * m.study_size).round().astype(int),
                         "metric": "accuracy", "tau_te": m.tau_te, "sigma2_te_K": m.sigma2_te_K, "n_seeds": m.n_seeds,
                         "G20": m.G20})


def yelp_rows(sources):
    out = []
    for tau_table, gain_table, models in sources:
        t = pd.read_parquet(tau_table)
        t = t[t.algorithm.isin(models)]
        g = pd.read_parquet(gain_table)
        g = g[(g.K == K) & (g.bootstrap_iter < 0) & g.model.isin(models)]
        g = g.rename(columns={"model": "algorithm", "N": "study_size", "sample_gain": "G20"})[["algorithm", "study_size", "G20"]]
        m = t.merge(g, on=["algorithm", "study_size"], how="left", validate="one_to_one")
        out.append(pd.DataFrame({"pipeline": "Yelp", "learner": m.algorithm.map(NLP),
                                 "n_tr": (0.8 * m.study_size).round().astype(int), "metric": "accuracy",
                                 "tau_te": m.tau_te, "sigma2_te_K": m.sigma2_te_K, "n_seeds": m.n_seeds, "G20": m.G20}))
    return pd.concat(out, ignore_index=True)


def draw(df, path):
    """Drawn at its printed size: included at 0.6\\linewidth = 3.3 in (NeurIPS text width 5.5 in), no rescaling."""
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    from scipy.stats import spearmanr
    P.set_paper_style()
    plt.rcParams.update({"axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7})
    fig, ax = plt.subplots(figsize=(3.3, 3.5), layout="constrained")
    x = np.geomspace(0.03, 0.25, 200)
    icc, = ax.plot(x, K / (1 + (K - 1) * x), color="k", ls="--", lw=1.2, zorder=1,
                   label=r"$K/(1+(K-1)\,\widehat\rho_\delta)$, $K=20$")
    order = {"simulated": ["ExtraTrees", "RandomFeatures", "MLP", "Ridge"],
             "PCam": ["DenseNet121", "MobileNetV2", "WideResNet101_2", "ViT-B/16"],
             "Yelp": ["BERT", "XLM-RoBERTa", "Qwen3-0.6B"]}
    draw_order = {"simulated": ["Ridge", "ExtraTrees", "MLP", "RandomFeatures"]}   # ExtraTrees on top of Ridge
    for pipe in ("simulated", "PCam", "Yelp"):
        for learner in draw_order.get(pipe, order[pipe]):
            g = df[(df.pipeline == pipe) & (df.learner == learner)]
            c = COLOR[(pipe, learner)]
            ax.scatter(g.rho_delta, g.G20, marker=MARKER[pipe], s=SIZE[pipe], facecolor=to_rgba(c, 0.45), edgecolor=c,
                       linewidth=0.9, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(0.03, 0.25)
    ax.xaxis.set_major_locator(FixedLocator([0.03, 0.05, 0.1, 0.2]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylim(3, 15)
    ax.set_yticks([4, 6, 8, 10, 12, 14])
    ax.set_xlabel(r"Fold-error correlation $\widehat\rho_\delta$")
    ax.set_ylabel(r"Test sample gain $G^{\mathrm{test}}_{20}$")
    ax.grid(color="0.9", lw=0.6)
    ax.legend(handles=[icc], loc="upper right", frameon=False, handlelength=2.4, fontsize=7)
    ax.text(0.03, 0.03, f"Rank corr. = ${spearmanr(df.rho_delta, df.G20)[0]:.2f}$, $n = {len(df)}$", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7, color="0.3")
    # legend below the axes: one column per data set (bold header), rows sharing a colour across the columns
    title = {"simulated": "Simulated (MSE)", "PCam": "PCam (accuracy)", "Yelp": "Yelp (accuracy)"}
    handles = []
    for pipe in ("simulated", "PCam", "Yelp"):
        col = [Line2D([], [], ls="", marker="", label=title[pipe])]
        col += [Line2D([], [], ls="", marker=MARKER[pipe], markersize=5 if pipe != "Yelp" else 4.2,
                       markerfacecolor=to_rgba(COLOR[(pipe, l)], 0.45), markeredgecolor=COLOR[(pipe, l)],
                       markeredgewidth=0.9, label=l) for l in order[pipe]]
        col += [Line2D([], [], ls="", marker="", label=" ")] * (5 - len(col))
        handles += col
    leg = fig.legend(handles=handles, ncol=3, loc="outside lower center", frameon=False, handletextpad=0.3,
                     columnspacing=0.9, labelspacing=0.25, handlelength=1.0, fontsize=6.5)
    for t in leg.get_texts():
        if t.get_text() in title.values():
            t.set_fontweight("bold")
    fig.savefig(path, pad_inches=0.02)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-root", default=None,
                    help="run outputs: outputs/ and paper_runs/ under it (default: $DATA_ROOT, else data/ at the repository root)")
    ap.add_argument("--sim-gain-dir", default=datapaths.figs_path("sim"), help="--out-dir of paper_style_figs.py")
    ap.add_argument("--real-dir", default=datapaths.figs_path("real"),
                    help="--out-dir of regen_real_data_gain_figures.py; the defaults of the four options below are under it")
    ap.add_argument("--pcam-tau", default=None, help="default: <real-dir>/work/pcam/results/tau_gain_pcam.parquet")
    ap.add_argument("--pcam-gain", default=None, help="default: <real-dir>/tables/pcam_accuracy.parquet")
    ap.add_argument("--yelp-aligned-tau", default=None, help="default: <real-dir>/work/nlp_aligned/results/tau_gain_nlp.parquet")
    ap.add_argument("--yelp-aligned-gain", default=None, help="default: <real-dir>/tables_aligned/nlp_delta.parquet")
    ap.add_argument("--qwen-tau", default=None, help="default: <real-dir>/work/nlp/results/tau_gain_nlp.parquet")
    ap.add_argument("--qwen-gain", default=None, help="default: <real-dir>/tables/nlp_delta.parquet")
    a = ap.parse_args()
    H.OUT, H.JULY = H.run_locations(a.data_root)
    os.makedirs(a.out_dir, exist_ok=True)
    rd = a.real_dir
    a.pcam_tau = a.pcam_tau or f"{rd}/work/pcam/results/tau_gain_pcam.parquet"
    a.pcam_gain = a.pcam_gain or f"{rd}/tables/pcam_accuracy.parquet"
    a.yelp_aligned_tau = a.yelp_aligned_tau or f"{rd}/work/nlp_aligned/results/tau_gain_nlp.parquet"
    a.yelp_aligned_gain = a.yelp_aligned_gain or f"{rd}/tables_aligned/nlp_delta.parquet"
    a.qwen_tau = a.qwen_tau or f"{rd}/work/nlp/results/tau_gain_nlp.parquet"
    a.qwen_gain = a.qwen_gain or f"{rd}/tables/nlp_delta.parquet"
    yelp = yelp_rows([(a.yelp_aligned_tau, a.yelp_aligned_gain, ["bert", "roberta"]),
                      (a.qwen_tau, a.qwen_gain, ["qwen3-0.6b"])])
    df = pd.concat([simulated_rows(a.sim_gain_dir), pcam_rows(a.pcam_tau, a.pcam_gain), yelp], ignore_index=True)
    df["rho_delta"] = df.tau_te / df.sigma2_te_K
    df["icc_gain"] = K / (1 + (K - 1) * df.rho_delta)
    assert df.G20.notna().all() and len(df) == 33, (len(df), df[df.G20.isna()])
    df.to_csv(f"{a.out_dir}/figD1_rho_delta.csv", index=False)
    print(df.round(4).to_string(index=False))
    from scipy.stats import spearmanr
    print(f"\n{len(df)} configurations; rho_delta {df.rho_delta.min():.4f} to {df.rho_delta.max():.4f}; "
          f"Spearman(rho_delta, G20) = {spearmanr(df.rho_delta, df.G20)[0]:.2f}; "
          f"median |log(G20 / ICC)| = {np.median(np.abs(np.log(df.G20 / df.icc_gain))):.3f}")
    for p, g in df.groupby("pipeline"):
        print(f"  {p}: n={len(g)}, rho {g.rho_delta.min():.3f}-{g.rho_delta.max():.3f}, Spearman {spearmanr(g.rho_delta, g.G20)[0]:.2f}, "
              f"G20/ICC {np.min(g.G20 / g.icc_gain):.2f}-{np.max(g.G20 / g.icc_gain):.2f}")
    draw(df, f"{a.out_dir}/rho_delta_vs_gain.pdf")


if __name__ == "__main__":
    main()
