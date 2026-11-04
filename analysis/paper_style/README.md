# Paper-style gain figures and numbers

Run on Jean-Zay with `module load pytorch-gpu/py3/2.8.0`; every script puts `code/` (next to it) first on
`sys.path` (`$PAPER_STYLE_CODE` overrides it). `code/` is CE's `figure_style/code` with the changes in
`figure_style_code.diff` (display: thick line = estimate on all seeds via
`compute.sample_gain_bootstrap(..., plugin_estimate=True)`, thin lines = bootstrap draws, y-axis cut 8% above the
highest thick line; single-figure-sized facet panels; `--plugin-estimate`, `--save-gains`, `--random-state`,
`--hairlines` in `pcam.py` / `nlp.py`) and the later changes below.

**Estimator (2026-11-04, CE: "go for the order-averaged estimator everywhere").** Every sample gain uses the
single-split curve V1(alpha) averaged exactly over the choice of the alpha - 1 outer chunks
(`compute.v1_alpha_curve`, closed form `[C00 + 2m c0 + m d + m(m-1) o] / alpha^2`, m = alpha - 1, from the
covariance across seeds of the test-fold error and the chunk errors): `chunk_order='average'` is the default of
`compute.sample_gain_bootstrap`. `chunk_order='stored'` (or `$GAIN_CHUNK_ORDER=stored`, which the drivers pass to the
`pcam.py` / `nlp.py` subprocesses) reproduces the values before, with the chunks in their stored order. Checks run
before the switch: the default reproduces the order-averaged G_paper of Figure 6 / Table D.2
(benchmark_pcam `analysis/pcam_gain_analysis.py`, 60 cells x 3 losses, relative difference <= 1.2e-14) and
`'stored'` reproduces every previous table exactly (estimate and draws).

| Script | Produces (current paper numbering) |
|---|---|
| `paper_style_figs.py` | simulated gain figures 4, 5 (left), G.1, H.1, L.1 and their tables (`--bootstrap-range` splits the 1,000 draws into chunks under the login-node CPU limit; `--from-results` assembles them; `--plugin-groups` adds the estimates) |
| `regen_real_data_gain_figures.py` | real-data figures 5 (middle, right), I.1, J.1, K.1, 7 and their tables (data: CE's `figure_style/real_data/data` as `./data`) |
| `fig5_common_axis.py` | Figure 5: the three panels redrawn from the tables of the two drivers with one y range and the same ticks |
| `fig2_pedagogical.py`, `fig2_layouts.py` | Figure 2 (PCam MobileNetV2, n_tr = 400, MCCV), order-averaged single-split curve, previous design |
| `figD1_rho_delta.py` | Figure D.1 (fold-error correlation against G_20, 33 configurations) |
| `fig6_redundancy_vs_gain.py`, `fig6_yelp_points.py`, `fig6_yelp_points.csv` | Figure 6 (redundancy against G_20; boxed keys; Yelp points computed with the PCam functions of benchmark_pcam) |
| `ci_appendix.py` | the LaTeX tables E.1 to E.8 of Appendix E (estimate and 95% percentile interval per figure) |
| `ci_coverage.py` | coverage check of the intervals of Appendix E (groups of 50 and 200 seeds) |
| `sd_ratio_mse.py` | Appendix H standard-deviation ratios (frozen / reshuffled, MSE) |
| `redundancy_numbers.py` | Appendix G and L redundancy values (t * squared-error OOF correlation) |
| `text_numbers_v5.py` | Appendix G, K and L values quoted in the text (stored-order outputs of 2026-10-31) |
| `compare_v5.py` | before / after renders (thick line = mean of draws vs estimate on all seeds) |
| `pdfcheck.py` | page sizes and fonts of the figure PDFs |
| `appL_r2.py` | Appendix K: Ridge gains on R^2 and on the MSE from the same July runs |

Figure 7 (retrieval of the quasi-oracle ranking on Yelp) is written by `nlp.py` through
`regen_real_data_gain_figures.py --only fig5_yelp_aligned`: 95% Wilson intervals over the 50 seeds
(`pipeline_common.run_retrieval_analysis(retrieval_ci_method="wilson")`), the published y range, and each
"CV, K=..." label above its line unless two lines are closer than one label height (`plot.plot_retrieval_curve`).
