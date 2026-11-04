# Figure and table scripts of the paper

The scripts here draw the gain figures of the paper and compute the tables and numbers built on them, from the run
outputs. Shared code is in `code/`: the estimator (`compute.py`), the plotting style (`plot.py`), the PCam, Yelp and
simulated pipelines (`pcam.py`, `nlp.py`, `simulated.py`, `pipeline_common.py`), the ranking-equivalent gain
(`ranking.py`), the significance curves (`replicability.py`) and where the inputs are (`datapaths.py`).

Environment: `pip install -r requirements-figures.txt` (the paper's figures were drawn with Python 3.12 and the versions
pinned there). Inputs: the run outputs under the data root (`--data-root`, else `$DATA_ROOT`, else `data/` at the
repository root; layout in the main README) and, for the scripts that read what other scripts wrote, the figure root
(`$FIGS_DIR`, default `figs/`).

Order (each step reads what the previous ones wrote):

```bash
cd analysis/paper_style
# simulated gain figures: draws of every group, then the estimate on all seeds and the figures
G="n1000 n100 small hpo_ridge hpo_rff small_mlp32 mlp32_n1000 mlp32_n100 frozen_redrawn"
python paper_style_figs.py --out-dir ../../figs/sim_draws --groups $G
python paper_style_figs.py --out-dir ../../figs/sim --from-results ../../figs/sim_draws --groups --plugin-groups $G
# real-data gain figures (PCam, Yelp)
python regen_real_data_gain_figures.py --out-dir ../../figs/real
# tables and figures built on the gain tables
python ci_appendix.py > ../../figs/ci_tables.tex
python figD1_rho_delta.py --out-dir ../../figs/figD1
```

| Script | Produces |
|---|---|
| `paper_style_figs.py` | Figures 4, 5 (left), G.1, H.1 and L.1; gain tables `tables/fig4`, `figI1`, `figJ1`, `figM1` (Tables E.1, E.3, E.4, E.8; the names follow the LaTeX labels of those tables) |
| `regen_real_data_gain_figures.py` | Figures 5 (middle, right), 7, I.1, J.1 and K.1 (and, in the plots of `pcam.py`, Figure 2); gain tables for Tables E.2, E.5, E.6, E.7; the Yelp rows of Table M.1 (`work/nlp_aligned/results/pairwise_ttest_nlp.csv`) |
| `ci_appendix.py` | Tables E.1 to E.8 (LaTeX, to stdout) |
| `ci_coverage.py` | coverage of the 95% percentile intervals (Appendix E) |
| `figD1_rho_delta.py` | Figure D.1 and its table `figD1_rho_delta.csv` |
| `paper_style_scatter.py` | the redundancy scatter of Figure 6 from the simulated and PCam configuration tables (`analysis/out_omega*/`, benchmark_pcam `analysis/out_round3*/`) |
| `sd_ratio_mse.py` | Appendix H: standard-deviation ratios, frozen and reshuffled splits |
| `redundancy_numbers.py` | Appendices G and L: redundancy values (t times the squared-error OOF correlation); writes `small_redundancy_k5.csv` for `text_numbers_v5.py` |
| `text_numbers_v5.py` | numbers quoted in Appendices G, K and L |
| `appL_r2.py` | Appendix K: Ridge gains on R^2 and on the MSE from the same runs |
| `code/ranking.py` | Figure O.1 (ranking-equivalent sample gain), from the 12 ranking runs |
| `code/replicability.py` | Table N.1 (significance within one CV run) |
| `coherence_checks/` | the scripts behind other numbers of the text; see its README |

`code/mse_gain_figures.py` is the template of `paper_style_figs.py` (column names, display names, the gain call), and
`code/simulated.py` the earlier simulated-data pipeline, whose `process_parquet` it uses (importing `simulated.py`
pulls in `code/gain_relationships.py`). `code/proxy_tau_te.py`, `code/tau_vs_gain.py` and
`code/study_only_statistic_figures.py` are exploratory analyses that no figure or table of the paper uses.
