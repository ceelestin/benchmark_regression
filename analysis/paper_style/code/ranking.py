# %% ──────────────────────────────────────────────────────────────────────────
# ranking.py — Ranking-Equivalent Sample Gain (paper Appendix H)
#
# Computes the *ranking-equivalent sample gain* G^rank_{K,alpha} for the
# ExtraTrees family of solvers and reproduces the two panels of Figure H.1:
#
#   (a) the distribution of G^rank_{K,alpha=0.95} as a function of the number of
#       splits K, aggregated over the family's ordered model pairs;
#   (b) the empirical retrieval-probability curves pi_K(N) as a function of the
#       study size N, for several split counts K, with the alpha=0.95 threshold.
#
# The oracle ranking inside a family is fixed by hyperparameter monotonicity:
# for ExtraTrees, more estimators -> lower oracle risk -> better, so the oracle
# order is n_estimators 200 > 20 > 2. This pins the per-pair oracle direction
# without any benchmark-set estimate (cf. Appendix H).
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import compute
import plot as P

# %% ──────────────────────────────────────────────────────────────────────────
# Default configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DATA_PATHS = [
    '/home/ceve/Documents/Run outputs/Regression/data/ranking/benchopt_run_2026-05-31_05h37m39.parquet', # ranking_100seeds.yml
    '/home/ceve/Documents/Run outputs/Regression/data/ranking/benchopt_run_2026-06-01_17h39m30.parquet', # ranking_100seeds_2.yml
    '/home/ceve/Documents/Run outputs/Regression/data/ranking/benchopt_run_2026-06-02_01h44m44.parquet', # ranking_100seeds_3.yml
    '/home/ceve/Documents/Run outputs/Regression/data/ranking/benchopt_run_2026-06-02_01h59m30.parquet', # ranking_100seeds_4.yml

    '/home/ceve/Documents/Run outputs/Regression/data/ranking/benchopt_run_2026-06-03_03h41m16.parquet', # ranking_100-300seeds.yml
    '/home/ceve/Documents/Run outputs/Regression/data/ranking/benchopt_run_2026-06-03_03h19m37.parquet', # ranking_100-300seeds_2.yml
    '/home/ceve/Documents/Run outputs/Regression/data/ranking/benchopt_run_2026-06-03_14h09m10.parquet', # ranking_100-300seeds_3.yml
    '/home/ceve/Documents/Run outputs/Regression/data/ranking/benchopt_run_2026-06-03_14h29m46.parquet', # ranking_100-300seeds_4.yml

    '/home/ceve/Documents/Run outputs/Regression/data/ranking/benchopt_run_2026-06-03_21h11m35.parquet', # ranking_300-500seeds.yml
    '/home/ceve/Documents/Run outputs/Regression/data/ranking/benchopt_run_2026-06-03_20h49m15.parquet', # ranking_300-500seeds_2.yml
    '/home/ceve/Documents/Run outputs/Regression/data/ranking/benchopt_run_2026-06-04_01h55m39.parquet', # ranking_300-500seeds_3.yml
    '/home/ceve/Documents/Run outputs/Regression/data/ranking/benchopt_run_2026-06-04_02h56m57.parquet', # ranking_300-500seeds_4.yml
]

DEFAULT_PLOTS_DIR = "plots/ranking"
DEFAULT_RESULTS_DIR = "results"

# Number of CV splits K shown on the plots (x-axis of H.1a, curve set of H.1b).
SPLIT_COUNTS = [1, 2, 3, 5, 10, 20, 50, 100, 200]

# Ranking-retrieval confidence level alpha (the dashed threshold of H.1b).
ALPHA = 0.95

# Bootstrap resamples over seeds used to build the gain distribution (H.1a).
N_BOOTSTRAPS = 500

# Family specification: how to recognise a solver family and how its
# hyperparameter maps to oracle quality.
#   higher_is_better -> True  : larger hyperparameter value is the oracle-better
#                               model (e.g. ExtraTrees n_estimators).
#   higher_is_better -> False : smaller value is better (e.g. Ridge alpha).
FAMILIES = {
    "ExtraTrees": {"prefix": "ExtraTrees", "param": "n_estimators", "higher_is_better": True},
    "Ridge":      {"prefix": "Ridge",      "param": "alpha",        "higher_is_better": False},
}

DEFAULT_FAMILIES = ["ExtraTrees"]

# Ranking modes to run:
#   'pairwise' — paper Appendix H: each ordered pair scored independently.
#   'full'     — a retrieval counts only if the *whole* family ranking is
#                recovered (e.g. n_estimators 200 > 20 > 2).
DEFAULT_RANKING_MODES = ["pairwise", "full"]

# A green for the ExtraTrees family, echoing the paper figure palette.
FAMILY_COLORS = {"ExtraTrees": "#4E9D63", "Ridge": "#E08A3C"}

# Columns we actually need from the (large) parquet files.
_NEEDED_COLUMNS = [
    "solver_name", "p_obj_train_size", "p_dataset_seed",
    "idx_rep", "objective_score_test", "objective_score_bench",
]


# %% ──────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def _running_in_jupyter() -> bool:
    """True when invoked from an IPython kernel (Jupyter / VSCode cells)."""
    try:
        get_ipython()  # noqa: F821
        return True
    except NameError:
        return False


def _show_if_interactive() -> None:
    """``plt.show()`` only when an IPython kernel is driving us.

    A plain ``python ranking.py`` shell has no GUI event loop, so a blocking
    ``plt.show()`` would hang; gating it keeps interactive cells working while
    making headless runs portable.
    """
    if _running_in_jupyter():
        plt.show()


def _to_scalar_series(s: pd.Series) -> pd.Series:
    """Coerce a score column to scalars, skipping the apply when numeric."""
    if pd.api.types.is_numeric_dtype(s):
        return s
    return s.apply(compute.to_scalar)


def _log_thin_1_2_5(sizes):
    """Study sizes on a clean 1-2-5 log ladder (first significant digit in
    {1, 2, 5}), always keeping the smallest and largest available size.

    Declutters the retrieval plot: closely-spaced study sizes differ mostly by
    seed noise and create spurious local spikes in the pi_K(N) curves.
    """
    s = np.sort(np.unique(np.asarray(list(sizes), dtype=float)))
    if s.size == 0:
        return set()
    mant = s / 10.0 ** np.floor(np.log10(s))
    keep = np.any([np.isclose(mant, t) for t in (1.0, 2.0, 5.0)], axis=0)
    keep[0] = keep[-1] = True
    return set(s[keep].astype(int).tolist())


# %% ──────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_scores(data_paths, families) -> pd.DataFrame:
    """Load per-split scores for the requested solver families.

    Reads only the handful of columns this analysis needs, concatenates the
    parquets, parses the family + integer hyperparameter out of ``solver_name``,
    and keeps only solvers belonging to *families*.

    Returns a tidy long frame with columns
    ``['family', 'param', 'train_size', 'seed', 'idx_rep', 'score', 'bench']``.
    """
    specs = {f: FAMILIES[f] for f in families}
    frames = []
    for path in data_paths:
        df = pd.read_parquet(path, columns=_NEEDED_COLUMNS)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    # Resolve each solver to a (family, hyperparameter value) pair.
    data["family"] = pd.NA
    data["param"] = np.nan
    for fam, spec in specs.items():
        mask = data["solver_name"].str.startswith(spec["prefix"])
        if not mask.any():
            continue
        vals = data.loc[mask, "solver_name"].str.extract(
            rf"{spec['param']}=([0-9.]+)"
        )[0].astype(float)
        data.loc[mask, "family"] = fam
        data.loc[mask, "param"] = vals.values

    data = data[data["family"].notna()].copy()
    data["score"] = _to_scalar_series(data["objective_score_test"]).astype(float)
    data["bench"] = _to_scalar_series(data["objective_score_bench"]).astype(float)
    data = data.rename(columns={
        "p_obj_train_size": "train_size",
        "p_dataset_seed": "seed",
    })
    data["train_size"] = data["train_size"].astype(int)
    data["seed"] = data["seed"].astype(int)
    data["idx_rep"] = data["idx_rep"].astype(int)
    return data[["family", "param", "train_size", "seed", "idx_rep", "score", "bench"]]


def oracle_order(params, higher_is_better: bool):
    """Return family members ordered best -> worst by hyperparameter monotonicity."""
    return sorted(params, reverse=higher_is_better)


def ordered_pairs(params, higher_is_better: bool):
    """All ``(best, worse)`` ordered pairs within a family."""
    ranked = oracle_order(params, higher_is_better)
    return [(ranked[i], ranked[j])
            for i in range(len(ranked)) for j in range(i + 1, len(ranked))]


# %% ──────────────────────────────────────────────────────────────────────────
# Cross-validation estimator R_K (cumulative mean over the first K splits)
# ─────────────────────────────────────────────────────────────────────────────

def add_cv_means(df: pd.DataFrame, split_counts) -> pd.DataFrame:
    """Add the K-split MCCV estimator R_K for every K in *split_counts*.

    R_K is the running mean of the per-split test score over the first K splits
    (paper eq. 7). We compute the full cumulative mean once and keep the rows
    whose split count is requested.
    """
    df = df.sort_values(["family", "param", "train_size", "seed", "idx_rep"])
    grp = df.groupby(["family", "param", "train_size", "seed"], sort=False)
    df["K"] = grp.cumcount() + 1
    df["cv_mean"] = grp["score"].cumsum() / df["K"]
    return df[df["K"].isin(split_counts)].copy()


# %% ──────────────────────────────────────────────────────────────────────────
# Ranking retrieval (paper eq. H.1)
# ─────────────────────────────────────────────────────────────────────────────

def comparison_groups(params, higher_is_better, mode):
    """Comparison groups whose ranking we test, as ``(label, members)`` tuples.

    ``members`` is the oracle order best -> worst.

    - ``mode='pairwise'`` (paper Appendix H): one 2-member group per ordered
      pair, e.g. ``(200, 20)``; retrieval is correct when that pair is ordered
      right.
    - ``mode='full'``: a single group holding the whole family, e.g.
      ``(200, 20, 2)``; retrieval is correct only when the *entire* ranking is
      recovered (every consecutive inequality holds).
    """
    ranked = oracle_order(params, higher_is_better)
    if mode == "full":
        label = " > ".join(f"{m:g}" for m in ranked)
        return [(label, tuple(ranked))]
    return [(f"{ranked[i]:g} vs {ranked[j]:g}", (ranked[i], ranked[j]))
            for i in range(len(ranked)) for j in range(i + 1, len(ranked))]


def build_correct(df_fam: pd.DataFrame, groups, split_counts):
    """Per-(N, seed) ranking-retrieval indicators for every group and split count.

    For a group with members ordered best -> worst, ``correct`` is 1 when the
    K-split estimator R_K reproduces that order, i.e. every consecutive
    inequality ``R_K[m_i] > R_K[m_{i+1}]`` holds (R_K is an R² utility, so a
    larger value is "better"). A 2-member group is the pairwise test; a 3-member
    group is the full-ranking test. Members are paired by seed and study size.

    Returns
    -------
    correct_df : DataFrame
        ``['comparison', 'K', 'train_size', 'seed', 'correct']`` pooled over
        groups.
    matrices : dict
        ``(label, K) -> (N_grid, M)`` where ``M`` is a ``[n_seeds, n_N]``
        float matrix of retrieval indicators, used for the seed bootstrap.
    """
    correct_rows = []
    matrices = {}
    for K in split_counts:
        sub = df_fam[df_fam["K"] == K]
        wide = sub.pivot_table(
            index=["train_size", "seed"], columns="param", values="cv_mean"
        ).reset_index()
        for label, members in groups:
            ok = np.ones(len(wide), dtype=bool)
            for better, worse in zip(members[:-1], members[1:]):
                ok &= (wide[better] > wide[worse]).to_numpy()
            tmp = pd.DataFrame({
                "comparison": label,
                "K": K,
                "train_size": wide["train_size"].to_numpy(),
                "seed": wide["seed"].to_numpy(),
                "correct": ok.astype(float),
            })
            correct_rows.append(tmp)
            mat = tmp.pivot_table(index="seed", columns="train_size", values="correct")
            matrices[(label, K)] = (
                mat.columns.to_numpy(dtype=float),
                mat.to_numpy(dtype=float),
            )
    correct_df = pd.concat(correct_rows, ignore_index=True)
    return correct_df, matrices


def family_retrieval_curve(correct_df: pd.DataFrame, n_bootstraps: int = 0,
                           random_state: int = 0) -> pd.DataFrame:
    """Pool retrieval over all groups and seeds -> family curve pi_K(N) (H.1b).

    With ``n_bootstraps > 0``, also returns 95% ``ci_low``/``ci_high`` columns
    from a non-parametric bootstrap over seeds (the comparison groups are first
    averaged within each seed, then seeds are resampled with replacement).
    """
    curve = (
        correct_df.groupby(["K", "train_size"])["correct"].mean()
        .reset_index().rename(columns={"correct": "pi"})
    )
    if n_bootstraps and n_bootstraps > 0:
        per_seed = (
            correct_df.groupby(["K", "train_size", "seed"])["correct"].mean()
            .reset_index()
        )
        rng = np.random.default_rng(random_state)
        ci_rows = []
        for K, sub in per_seed.groupby("K"):
            mat = sub.pivot_table(index="seed", columns="train_size", values="correct")
            Ns = mat.columns.to_numpy(dtype=float)
            M = mat.to_numpy(dtype=float)  # [n_seeds, n_N]
            idx = rng.integers(0, M.shape[0], size=(n_bootstraps, M.shape[0]))
            boot = np.nanmean(M[idx], axis=1)  # [n_bootstraps, n_N]
            ci_rows.append(pd.DataFrame({
                "K": K, "train_size": Ns,
                "ci_low": np.nanpercentile(boot, 2.5, axis=0),
                "ci_high": np.nanpercentile(boot, 97.5, axis=0),
            }))
        curve = curve.merge(pd.concat(ci_rows, ignore_index=True),
                            on=["K", "train_size"], how="left")
    return curve.sort_values(["K", "train_size"]).reset_index(drop=True)


# %% ──────────────────────────────────────────────────────────────────────────
# Threshold-crossing study size N*_{K,alpha} (paper Def H.1)
# ─────────────────────────────────────────────────────────────────────────────

def threshold_crossing_N(N_grid, pi, alpha: float) -> float:
    """Smallest study size at which retrieval reaches *alpha* (paper eq. H.2).

    The retrieval probability is monotone non-decreasing in N in theory, so we
    take a running maximum over increasing N to absorb finite-sample noise, then
    read off the first crossing with linear interpolation in ``log10(N)``. If the
    threshold is never reached within the grid, the value is right-censored to
    the largest available study size.
    """
    N = np.asarray(N_grid, dtype=float)
    p = np.maximum.accumulate(np.asarray(pi, dtype=float))
    reached = p >= alpha
    if not reached.any():
        return float(N[-1])
    idx = int(np.argmax(reached))
    if idx == 0:
        return float(N[0])
    p0, p1 = p[idx - 1], p[idx]
    x0, x1 = np.log10(N[idx - 1]), np.log10(N[idx])
    if p1 <= p0:
        return float(N[idx])
    frac = (alpha - p0) / (p1 - p0)
    return float(10.0 ** (x0 + frac * (x1 - x0)))


def threshold_table(matrices, groups, split_counts, alpha) -> pd.DataFrame:
    """Population (full-sample) N*_{K,alpha} for every comparison group."""
    rows = []
    for label, _ in groups:
        for K in split_counts:
            N_grid, M = matrices[(label, K)]
            pi = np.nanmean(M, axis=0)
            rows.append({
                "comparison": label,
                "K": K,
                "N_star": threshold_crossing_N(N_grid, pi, alpha),
            })
    return pd.DataFrame(rows)


# %% ──────────────────────────────────────────────────────────────────────────
# Ranking-equivalent sample gain + seed bootstrap (paper Def H.2, Figure H.1a)
# ─────────────────────────────────────────────────────────────────────────────

def compute_gain_distribution(
    matrices, groups, split_counts, alpha, n_bootstraps, random_state=0
) -> pd.DataFrame:
    """Bootstrap distribution of G^rank_{K,alpha} = N*_{1} / N*_{K}.

    The numerator N*_{1} is fixed to its population value (the figure's
    "numerator fixed to population mean"); the denominator N*_{K} is recomputed
    on each seed bootstrap resample, so the spread reflects estimation noise on
    the K-split threshold-crossing size. Results are pooled over the family's
    comparison groups (3 ordered pairs in pairwise mode, a single full-ranking
    group in full mode).
    """
    rng = np.random.default_rng(random_state)
    rows = []
    for label, _ in groups:
        N_grid_1, M1 = matrices[(label, 1)]
        N_star_1 = threshold_crossing_N(N_grid_1, np.nanmean(M1, axis=0), alpha)
        for K in split_counts:
            N_grid, M = matrices[(label, K)]
            n_seeds = M.shape[0]
            idx = rng.integers(0, n_seeds, size=(n_bootstraps, n_seeds))
            boot_pi = np.nanmean(M[idx], axis=1)  # [n_bootstraps, n_N]
            for b in range(n_bootstraps):
                N_star_K = threshold_crossing_N(N_grid, boot_pi[b], alpha)
                rows.append({
                    "comparison": label,
                    "K": K,
                    "gain": N_star_1 / N_star_K,
                })
    return pd.DataFrame(rows)


# %% ──────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_retrieval_vs_studysize(
    curve_df, alpha, split_counts, family, mode="pairwise", thin=True,
    output_path=None
):
    """Figure H.1(b): family retrieval curves pi_K(N) vs study size N.

    With ``thin=True`` the curves are plotted only on a 1-2-5 log ladder of
    study sizes (see :func:`_log_thin_1_2_5`); the threshold-crossing N* and the
    gain are still computed elsewhere on the full study-size grid.
    """
    if thin:
        keep = _log_thin_1_2_5(curve_df["train_size"].unique())
        curve_df = curve_df[curve_df["train_size"].isin(keep)]
    ks = [k for k in split_counts if k in curve_df["K"].unique()]
    colors = plt.cm.viridis(np.linspace(0.0, 0.92, len(ks)))

    has_ci = "ci_low" in curve_df.columns
    # fig, ax = plt.subplots(figsize=(3.8, 3.0), layout="tight")
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    for color, K in zip(colors, ks):
        sub = curve_df[curve_df["K"] == K].sort_values("train_size")
        if has_ci and sub["ci_low"].notna().any():
            ax.fill_between(sub["train_size"], sub["ci_low"], sub["ci_high"],
                            color=color, alpha=0.15, linewidth=0)
        ax.plot(
            sub["train_size"], sub["pi"],
            marker="o", markersize=3, linewidth=1.4, color=color, label=str(K),
        )
    ax.axhline(alpha, linestyle=":", linewidth=1.2, color="0.35")
    ax.text(
        0.01, alpha + 0.01, f"$\\alpha={alpha:g}$",
        transform=ax.get_yaxis_transform(), va="bottom", ha="left",
        fontsize=8, color="0.35",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Study Size ($N$)")
    ax.set_ylabel("Oracle-ranking retrieval  $\\widehat{\\pi}_K(N)$")
    ax.set_ylim(0.4, 1.02)
    leg = ax.legend(title="$K$ (splits)", fontsize=7, title_fontsize=8,
                    ncol=2, loc="lower right", frameon=False)
    leg.get_title().set_fontsize(8)
    if output_path:
        P.save_figure(fig, output_path)
    _show_if_interactive()
    return fig


def plot_ranking_gain_violin(gain_df, split_counts, family, mode="pairwise",
                             output_path=None):
    """Figure H.1(a): distribution of G^rank_{K,alpha} vs number of splits K."""
    ks = [k for k in split_counts if k in gain_df["K"].unique()]
    color = FAMILY_COLORS.get(family, "#4E9D63")
    order = [str(k) for k in ks]
    gdf = gain_df.copy()
    gdf["K_str"] = gdf["K"].astype(int).astype(str)

    # fig, ax = plt.subplots(figsize=(3.8, 3.0), layout="tight")
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    sns.violinplot(
        data=gdf, x="K_str", y="gain", order=order,
        color=color, cut=0, inner=None, linewidth=0.6, ax=ax,
    )
    for art in ax.collections:
        art.set_alpha(0.75)
    # 95% interval (2.5–97.5 percentiles of the bootstrap gains) + median dot.
    q_lo = gdf.groupby("K")["gain"].quantile(0.025).reindex(ks)
    q_hi = gdf.groupby("K")["gain"].quantile(0.975).reindex(ks)
    ax.vlines(range(len(ks)), q_lo.to_numpy(), q_hi.to_numpy(),
              color="0.2", linewidth=1.0, zorder=4)
    medians = gdf.groupby("K")["gain"].median().reindex(ks)
    ax.scatter(range(len(ks)), medians.to_numpy(), color="white",
               edgecolor="black", s=16, zorder=5)

    ax.set_yscale("log")
    ax.set_xlabel("Number of Splits ($K$)")
    ax.set_ylabel("Rank.-Eq. Sample Gain ($G^{\\mathrm{rank}}_{K,0.95}$)")
    if output_path:
        P.save_figure(fig, output_path)
    _show_if_interactive()
    return fig


# %% ──────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def analyze_family(data, family, split_counts, alpha, n_bootstraps,
                   plots_dir, results_dir, ranking_mode="pairwise"):
    """Run the full Appendix-H analysis for one family and emit plots + CSVs."""
    spec = FAMILIES[family]
    df_fam = data[data["family"] == family].copy()
    params = sorted(df_fam["param"].unique())
    groups = comparison_groups(params, spec["higher_is_better"], ranking_mode)

    ranked = oracle_order(params, spec["higher_is_better"])
    print(f"\n=== {family} family ({ranking_mode} ranking) ===")
    print(f"  members (best -> worst): "
          + " > ".join(f"{spec['param']}={p:g}" for p in ranked))
    print(f"  comparison groups: " + ", ".join(label for label, _ in groups))

    # Sanity check the oracle direction against mean benchmark scores.
    bench_mean = df_fam.groupby("param")["bench"].mean()
    bench_rank = list(bench_mean.sort_values(ascending=False).index)
    if bench_rank == ranked:
        print("  oracle direction confirmed by mean benchmark scores.")
    else:
        print(f"  WARNING: benchmark order {bench_rank} != hyperparameter order {ranked}")

    # K=1 is always needed for the gain numerator even if not displayed.
    build_splits = sorted(set(split_counts) | {1})
    df_cv = add_cv_means(df_fam, build_splits)
    correct_df, matrices = build_correct(df_cv, groups, build_splits)
    curve_df = family_retrieval_curve(correct_df, n_bootstraps=n_bootstraps)
    thresh_df = threshold_table(matrices, groups, split_counts, alpha)
    gain_df = compute_gain_distribution(
        matrices, groups, split_counts, alpha, n_bootstraps
    )

    # ── Persist numbers ──────────────────────────────────────────────────────
    tag = f"{family.lower()}_{ranking_mode}"
    curve_df.assign(family=family).to_csv(
        os.path.join(results_dir, f"ranking_retrieval_curves_{tag}.csv"), index=False)
    thresh_df.assign(family=family).to_csv(
        os.path.join(results_dir, f"ranking_threshold_sizes_{tag}.csv"), index=False)
    gain_df.assign(family=family).to_csv(
        os.path.join(results_dir, f"ranking_gain_distribution_{tag}.csv"), index=False)

    # ── Console summary (paper sanity checks) ────────────────────────────────
    gain_med = gain_df.groupby("K")["gain"].median()
    print("  median ranking-equivalent gain by K:")
    print("   " + "  ".join(f"K={k}:{gain_med[k]:.1f}" for k in split_counts
                            if k in gain_med.index))

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_retrieval_vs_studysize(
        curve_df, alpha, split_counts, family, mode=ranking_mode,
        output_path=os.path.join(plots_dir, f"ranking_retrieval_curves_{tag}.pdf"))
    plot_ranking_gain_violin(
        gain_df, split_counts, family, mode=ranking_mode,
        output_path=os.path.join(plots_dir, f"ranking_equivalent_gain_violin_{tag}.pdf"))

    return curve_df, thresh_df, gain_df


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Ranking-equivalent sample gain (paper Appendix H).")
    parser.add_argument("--data-paths", nargs="+", default=DEFAULT_DATA_PATHS)
    parser.add_argument("--plots-dir", default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--n-bootstraps", type=int, default=N_BOOTSTRAPS)
    parser.add_argument("--split-counts", type=int, nargs="+", default=SPLIT_COUNTS)
    parser.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES,
                        choices=list(FAMILIES.keys()))
    parser.add_argument("--ranking-mode", nargs="+", default=DEFAULT_RANKING_MODES,
                        choices=["pairwise", "full"],
                        help="'pairwise' (paper Appendix H) and/or 'full' "
                             "(whole-family ranking must be exactly recovered).")
    return parser.parse_args(argv)


def main(
    data_paths=None,
    plots_dir=DEFAULT_PLOTS_DIR,
    results_dir=DEFAULT_RESULTS_DIR,
    alpha=ALPHA,
    n_bootstraps=N_BOOTSTRAPS,
    split_counts=None,
    families=None,
    ranking_modes=None,
):
    if data_paths is None:
        data_paths = DEFAULT_DATA_PATHS
    if split_counts is None:
        split_counts = list(SPLIT_COUNTS)
    if families is None:
        families = list(DEFAULT_FAMILIES)
    if ranking_modes is None:
        ranking_modes = list(DEFAULT_RANKING_MODES)

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    P.set_paper_style()

    data = load_scores(data_paths, families)
    n_seeds = data["seed"].nunique()
    n_sizes = data["train_size"].nunique()
    print(f"Loaded {len(data):,} rows | families={families} | "
          f"seeds={n_seeds} | study sizes={n_sizes} | "
          f"N range={data['train_size'].min()}..{data['train_size'].max()}")
    print(f"Split counts K={split_counts} | alpha={alpha} | "
          f"bootstraps={n_bootstraps} | ranking modes={ranking_modes}")

    for family in families:
        for ranking_mode in ranking_modes:
            analyze_family(data, family, split_counts, alpha, n_bootstraps,
                           plots_dir, results_dir, ranking_mode=ranking_mode)


# %% ──────────────────────────────────────────────────────────────────────────
# Run the pipeline.
# - VSCode cells / Jupyter: calls main() with defaults (argparse would otherwise
#   choke on the kernel's own argv).
# - CLI (python ranking.py --data-paths ...): parses arguments first.
# - import ranking: does nothing.

if __name__ == "__main__":
    if _running_in_jupyter():
        main()
    else:
        args = parse_args()
        main(
            data_paths=args.data_paths,
            plots_dir=args.plots_dir,
            results_dir=args.results_dir,
            alpha=args.alpha,
            n_bootstraps=args.n_bootstraps,
            split_counts=args.split_counts,
            families=args.families,
            ranking_modes=args.ranking_mode,
        )

# %%
