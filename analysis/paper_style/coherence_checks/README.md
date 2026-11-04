# Checks behind numbers quoted in the paper

Short scripts that compute or verify statements of the text. Each reads the run outputs under the data root
(`$DATA_ROOT`, see `../code/datapaths.py`), the outputs of this repository's analysis scripts (`analysis/out_*/`), the
outputs of the figure scripts (`$FIGS_DIR`) or a checkout of a companion repository (`$BENCHMARK_PCAM`,
`$BENCHMARK_NLP`); a few take a folder as argument. Run them from any directory with the environment of
`requirements-figures.txt`.

| Script | Supports |
|---|---|
| `sim_params.py`, `sim_params_all.py`, `sim_params_pilot.py` | Appendix F.1: d = 5 and sigma = 0 for the main simulated runs (and the 200-seed run of Appendix K); parameters, seeds and learners of every single-file run |
| `check_above_k.py`, `check38.py`, `rho_all.py` | Appendix D.1, estimates above K: the configurations of Figure 6 with G_20 > 20 and the fold-error correlation they imply; rho_delta positive over Figure D.1, Table D.2, Table G.1 and the pre-registered holdout |
| `strat_effect.py`, `strat_table.py` | Appendix J and Table J.1: unstratified / stratified variance ratios on PCam (paired 100 seeds, n_tr = 400) |
| `strat_mech.py`, `vit_march.py`, `vit_all.py` | Section 4 and Appendix J: ViT-B/16 close to a constant predictor (benchmark accuracy 0.51-0.59 in every run set, CNNs 0.60-0.85) |
| `fold_geometry.py` | Appendix L: share of runs in which the redundancy is defined after 3 and 5 splits |
| `pcam24_redundancy.py`, `pcam24_gains.py` | Appendix L, PCam at n_tr = 24 (gains from `code/pcam.py --data-dir <the n_tr = 24 runs> --n-bootstraps 200 --random-state 0 --plugin-estimate --save-gains $FIGS_DIR/pcam24/tables`) |
| `yelp_seeds.py`, `yelp_bench.py`, `yelp_times.py` | Appendix F.3 and Table F.2: seeds of the first Yelp runs, which of them are scored on the benchmarking set, recorded runtimes |
| `yelp_dups.py`, `yelp_pipeline_rows.py`, `yelp_dedup_cmp.py`, `yelp_g1_retrieval.py` | the first Yelp K = 20 runs of BERT and XLM-RoBERTa: seeds 0-19 were run twice (once without benchmarking scores); the loader of `code/nlp.py` keeps one run per seed. `yelp_dedup_cmp.py` and `yelp_g1_retrieval.py` compare two versions of the pipeline given as arguments |
| `yelp_aligned_check.py`, `yelp_aligned_times.py`, `yelp_quasi_oracle.py`, `yelp_retrieval_curve.py`, `yelp_paper_snippets.py`, `qwen_structure.py`, `gain_ci.py` | Yelp runs on the PCam protocol (benchmark_nlp `nlpfinetuning_aligned.py`): sanity checks, Table F.2 runtimes, the two quasi-oracle definitions, the single-split shares of Section 5, the LaTeX of the Yelp rows of Tables M.1, E.2 and F.2 (`yelp_paper_snippets.py <--out-dir of regen_real_data_gain_figures.py --only fig5_yelp_aligned>`), the structure of the Qwen3-0.6B runs, gain intervals (`gain_ci.py <gain table>`) |
| `g1_ttest_source.py`, `g1_pcam_check.py` | Table M.1: the Yelp paired t-test columns, and the PCam rows from the January single-split runs (`<data root>/pcam/nocv_jan`) |

Figure D.1 is drawn by `../figD1_rho_delta.py`.
