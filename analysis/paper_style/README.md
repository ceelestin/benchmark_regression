# Figure and table scripts of the paper

The scripts here draw the gain figures of the paper and compute the tables and numbers built on them, from the run
outputs. Shared code is in `code/`: the estimator (`compute.py`), the plotting style (`plot.py`), the PCam, Yelp and
simulated pipelines (`pcam.py`, `nlp.py`, `simulated.py`, `pipeline_common.py`), the ranking-equivalent gain
(`ranking.py`), the significance curves (`replicability.py`) and where the inputs are (`datapaths.py`). Every script
puts `code/` (next to it) first on `sys.path`; `$PAPER_STYLE_CODE` overrides it.

Environment: `pip install -r requirements-figures.txt` (the paper's figures were drawn with Python 3.12 and the versions
pinned there). Inputs: the run outputs under the data root (`--data-root`, else `$DATA_ROOT`, else `data/` at the
repository root; layout in the main README) and, for the scripts that read what other scripts wrote, the figure root
(`$FIGS_DIR`, default `figs/`).

**Estimator.** Every sample gain uses the single-split curve V1(alpha) averaged exactly over the choice of the
alpha - 1 outer chunks (`compute.v1_alpha_curve`, closed form `[C00 + 2m c0 + m d + m(m-1) o] / alpha^2`, m = alpha - 1,
from the covariance across seeds of the test-fold error and the chunk errors): `chunk_order='average'` is the default
of `compute.sample_gain_bootstrap`. `chunk_order='stored'` (or `$GAIN_CHUNK_ORDER=stored`, which the drivers pass to
the `pcam.py` / `nlp.py` subprocesses) uses the chunks in their stored order. Checks: the default reproduces the
order-averaged gains of Figure 6 and Table D.2 (benchmark_pcam `analysis/pcam_gain_analysis.py`, 60 cells x 3 losses,
relative difference <= 1.2e-14), and `'stored'` reproduces the tables of the stored order exactly (estimate and
draws).

Order (each step reads what the previous ones wrote):

```bash
cd analysis/paper_style
# simulated gain figures: draws of every group, then the estimate on all seeds and the figures
G="n1000 n100 small hpo_ridge hpo_rff small_mlp32 mlp32_n1000 mlp32_n100 frozen_redrawn"
python paper_style_figs.py --out-dir ../../figs/sim_draws --groups $G
python paper_style_figs.py --out-dir ../../figs/sim --from-results ../../figs/sim_draws --groups --plugin-groups $G
# real-data gain figures (PCam, Yelp)
python regen_real_data_gain_figures.py --out-dir ../../figs/real
# figures and tables built on the gain tables or on other analyses
python fig5_common_axis.py --sim-tables ../../figs/sim/tables --real-tables ../../figs/real/tables --out-dir ../../figs/fig5
python fig2_pedagogical.py --out-dir ../../figs/fig2
python fig6_yelp_points.py    # writes fig6_yelp_points.csv here; needs the PCam round-3 tables (see the table below)
python fig6_redundancy_vs_gain.py --yelp fig6_yelp_points.csv --out ../../figs/fig6/redundancy_vs_gain_all_k3.pdf
python figD1_rho_delta.py --out-dir ../../figs/figD1
python ci_appendix.py > ../../figs/ci_tables.tex
```

`paper_style_figs.py --bootstrap-range START STOP` computes a range of the 1,000 draws of a group, to stay under a
per-process CPU-time limit; `--from-results` assembles the parts.

| Script | Produces |
|---|---|
| `paper_style_figs.py` | Figures 4, 5 (left), G.1, H.1 and L.1; gain tables `tables/fig4`, `figI1`, `figJ1`, `figM1` (Tables E.1, E.3, E.4, E.8; the names follow the LaTeX labels of those tables) |
| `regen_real_data_gain_figures.py` | Figures 5 (middle, right), 7, I.1, J.1 and K.1; gain tables for Tables E.2, E.5, E.6, E.7; the Yelp rows of Table M.1 (`work/nlp_aligned/results/pairwise_ttest_nlp.csv`) |
| `fig5_common_axis.py` | Figure 5: its three panels redrawn from the tables of the two scripts above, with one y range and the same ticks |
| `fig2_pedagogical.py`, `fig2_layouts.py` | Figure 2 (PCam MobileNetV2, n_tr = 400, MCCV) with the order-averaged single-split curve |
| `fig6_redundancy_vs_gain.py`, `fig6_yelp_points.py` | Figure 6 (redundancy after 3 splits against G_20), from the simulated configuration tables (`analysis/out_omega/`), the PCam round-3 tables of benchmark_pcam (`analysis/out_round3/candidates_<loss>.csv` and `analysis/out_round3_redundancy/redundancy_vs_gain.csv`, written by its `analysis/pcam_gain_analysis.py` and `analysis/pcam_redundancy_vs_gain.py`) and the Yelp points (`fig6_yelp_points.py` computes them with the PCam functions of benchmark_pcam and writes `fig6_yelp_points.csv`) |
| `figD1_rho_delta.py` | Figure D.1 (fold-error correlation against G_20, 33 configurations) and its table |
| `ci_appendix.py` | Tables E.1 to E.8 (LaTeX, to stdout) |
| `ci_coverage.py` | coverage of the 95% percentile intervals (Appendix E; groups of 50 and 200 seeds) |
| `sd_ratio_mse.py` | Appendix H: standard-deviation ratios, frozen and reshuffled splits |
| `redundancy_numbers.py` | Appendices G and L: redundancy values (t times the squared-error OOF correlation); writes `small_redundancy_k5.csv` for `text_numbers_v5.py` |
| `text_numbers_v5.py` | numbers quoted in Appendices G, K and L (computed from the tables of the stored chunk order) |
| `appL_r2.py` | Appendix K: Ridge gains on R^2 and on the MSE from the same runs |
| `code/ranking.py` | Figure O.1 (ranking-equivalent sample gain), from the 12 ranking runs |
| `code/replicability.py` | Table N.1 (significance within one CV run) |
| `coherence_checks/` | the scripts behind other numbers of the text; see its README |

Figure 7 (retrieval of the quasi-oracle ranking on Yelp) is written by `nlp.py` through
`regen_real_data_gain_figures.py --only fig5_yelp_aligned`: 95% Wilson intervals over the 50 seeds
(`pipeline_common.run_retrieval_analysis(retrieval_ci_method="wilson")`), the published y range, and each
"CV, K=..." label above its line unless two lines are closer than one label height (`plot.plot_retrieval_curve`).

`paper_style_scatter.py` is the earlier version of the Figure 6 scatter (simulated and PCam points only), with the
whiskered scatter of the pre-registered holdout. `code/mse_gain_figures.py` is the template of `paper_style_figs.py`
(column names, display names, the gain call), and `code/simulated.py` the earlier simulated-data pipeline, whose
`process_parquet` it uses (importing `simulated.py` pulls in `code/gain_relationships.py`). `code/proxy_tau_te.py`,
`code/tau_vs_gain.py` and `code/study_only_statistic_figures.py` are exploratory analyses that no figure or table of
the paper uses.
