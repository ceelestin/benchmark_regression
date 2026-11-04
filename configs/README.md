# Run configurations

Every simulated run behind a figure, a table or a number of the paper has its benchopt configuration here. Figure,
table and appendix numbers are those of the paper (Appendix F.2 describes the redundancy designs, Table F.2 lists the
runs and their compute).

Two kinds of runs:

- **Chunked runs** (Jean-Zay CPU nodes, September-October 2026). Each configuration was split into seed chunks with
  `make_chunks.py`; the chunks and their manifest are in `chunks_<design>/` and the SLURM array that ran them is
  `submit_<design>.slurm` at the root (see [../docs/jean-zay.md](../docs/jean-zay.md)). Each chunk writes
  `outputs/<config>__cNN.parquet`.
- **Single runs** (one 128-core machine outside Jean-Zay, May-July 2026): `benchopt run . --config configs/<file>.yml`. benchopt names
  the output after the time the run was saved, `benchopt_run_<date>_<time>.parquet`; the figure scripts read these
  files under that name (in `<data root>/paper_runs/` or `<data root>/ranking/`, see the main README).

## Chunked runs

| Config(s) | Launcher, chunks | Used in the paper |
|---|---|---|
| `paper_main_gain_mse_k200.yml` | `submit_paper_main_mse.slurm`, `chunks_paper_main_mse/` | Figures 4 and 5 (left), D.1, G.1 (fixed parameters), Tables E.1, F.2; Appendix K (MAE) |
| `paper_main_gain_mse_k200_rff50.yml`, `paper_ntr100_k20_fixed_rff50.yml`, `paper_ntr100_k20_resampled_rff50.yml` | `submit_paper_rff50.slurm`, `chunks_paper_rff50/` | RandomFeatures in Figures 4, 5 (left), D.1 and H.1 (resampled arm); Appendix H; Table F.2 |
| `paper_ntr100_k20_fixed_redrawn_rff50.yml` | `submit_paper_rff50_redrawn.slurm`, `chunks_paper_rff50_redrawn/` | RandomFeatures, frozen arm of Figure H.1; Table F.2 |
| `hpo_mse_MLP_gridsearch.yml`, `hpo_mse_MLP_randomsearch.yml` | `submit_hpo_mse_mlp.slurm`, `chunks_hpo_mse_mlp/` | Figure G.1, Tables G.1, E.3, F.2 |
| `small_budget_mlp64_rff50.yml` | `submit_small_budget.slurm`, `chunks_small_budget/` | RandomFeatures in Figure L.1, Table E.8; Table F.2 |
| `timing_*.yml` | `submit_timing.slurm`, `chunks_timing/` | Table F.2: 5% of the seeds of the July runs re-run with the evaluation time recorded |
| `derivation_redundancy_levers_{shufflesplit,repeatedkfold,fixedsplit}.yml` | `submit_derivation.slurm`, `chunks_derivation/` | Noise-injection grid of Figure 6 (Appendix F.2); Table F.2 |
| `k200_threshold_validation_holdout_50_{linear,interactions,nonlinear}.yml` | `submit_holdout1.slurm`, `chunks_holdout1/` | Pre-registered holdout: Table D.1, Figure 6; Table F.2 (plan: `analysis/PREREGISTRATION_f_rule.md`) |
| `holdout2_largegain_*.yml` | `submit_holdout2.slurm`, `chunks_holdout2/` | First unstable-learner design of Figure 6 (regression families); Table F.2 (`analysis/PREREGISTRATION_f_rule_holdout2.md`) |
| `holdout3_largegain_*.yml` | `submit_holdout3.slurm`, `chunks_holdout3/` | Second unstable-learner design of Figure 6; Table F.2 (`analysis/PREREGISTRATION_round3.md`) |
| `paper_rerun_may_design_{linear,interactions,nonlinear}.yml` | `submit_paper_rerun.slurm`, `chunks_paper_rerun/` | Fresh-seed confirmation of Appendix D.2; Table F.2 (`analysis/PREREGISTRATION_paper_rerun.md`) |
| `testsize_check_{regression,classification}.yml` | `submit_testsize.slurm`, `chunks_testsize/` | Test-fraction check of Appendix D.2; Table F.2 |
| `pilot_persample_k20.yml` | `submit_pilot_persample.slurm`, `chunks_pilot_persample/` | Appendix K, the 200-seed run with per-sample errors (`analysis/persample_r2_centering.py`); Table F.2 |
| `paper_main_gain_mse_k200_mlp64x64.yml`, `paper_ntr100_k20_{fixed,resampled}_mlp64x64.yml` | `submit_paper_mlp64x64.slurm`, `chunks_paper_mlp64x64/` | Exploratory (Table F.2); read by `paper_style_figs.py`, then dropped from every figure |
| `paper_main_gain_mse_k20_mlp64.yml` | `submit_paper_mlp64.slurm`, `chunks_paper_mlp64/` | Exploratory (Table F.2) |
| `hpo_mse_Ridge_*.yml`, `hpo_mse_RandomFeatures_*.yml` | `submit_hpo_mse_ridge.slurm`, `submit_hpo_mse_rff.slurm` | Exploratory (Table F.2); computed by `paper_style_figs.py`, not drawn in Figure G.1 |
| `derivation_redundancy_levers_classification_*.yml` | `submit_derivation_classification.slurm`, `chunks_classification/` | Classification analog of the noise-injection grid (exploratory) |
| `pilot_r2lin_k20.yml`, `pilot_highgain_k20.yml`, `pilot_realistic_k20.yml`, `pilot_family2_k20.yml`, `pilot_rff_confirm_k20.yml` | `submit_pilot_*.slurm`, `chunks_pilot_*/` | Pilots: R^2-linearised redundancy, choice of the high-gain learners of the main runs (exploratory) |
| `coupling_test_{regression,classification}.yml` | `submit_coupling.slurm`, `chunks_coupling/` | Train/test coupling diagnostic (exploratory) |
| `sweep_clf_pcam_sizes.yml` | `submit_sweep_clf.slurm`, `chunks_sweep_clf/` | Classification sweep at PCam-like sizes (exploratory) |
| `preflight_{derivation,classification,coupling}.yml` | none | One-seed checks that run in minutes; useful to test an installation |

## Single runs

All on the linear process (d = 5, no noise, 150,000 samples), ShuffleSplit with test fraction 0.2, seeds 0-999 unless
stated. The headers of the files give the details.

| Config | Output file (`benchopt_run_...parquet`) | Used in the paper |
|---|---|---|
| `ranking_100seeds.yml`, `_2`, `_3`, `_4` | `2026-05-31_05h37m39`, `2026-06-01_17h39m30`, `2026-06-02_01h44m44`, `2026-06-02_01h59m30` | Figure O.1 (ExtraTrees and Ridge, seeds 0-99, K = 200; the four files cover the 25 training sizes) |
| `ranking_100-300seeds.yml`, `_2`, `_3`, `_4` | `2026-06-03_03h41m16`, `2026-06-03_03h19m37`, `2026-06-03_14h09m10`, `2026-06-03_14h29m46` | Figure O.1 (ExtraTrees, seeds 100-299) |
| `ranking_300-500seeds.yml`, `_2`, `_3`, `_4` | `2026-06-03_21h11m35`, `2026-06-03_20h49m15`, `2026-06-04_01h55m39`, `2026-06-04_02h56m57` | Figure O.1 (ExtraTrees, seeds 300-499) |
| `config_hpo_test_gridsearch.yml` | `2026-07-27_12h40m37` | ExtraTrees grid search: Figure G.1, Tables G.1, E.3, F.2 |
| `config_hpo_test_randomsearch.yml` | `2026-07-27_12h47m49` | ExtraTrees random search: Figure G.1, Tables G.1, E.3, F.2 |
| `linear_1000_K20_new_fixed.yml` | `2026-07-27_21h00m32` | n_tr = 100, frozen split, data redrawn: frozen arm of Figure H.1, Table E.4 |
| `linear_1000_K20_new_fixed_samedata.yml` (reconstructed) | `2026-07-27_20h48m32` | n_tr = 100, frozen split on the same data: read by `paper_style_figs.py`, `figD1_rho_delta.py` and `sd_ratio_mse.py`; Table F.2 |
| `linear_1000_K20_new_unfixed.yml` | `2026-07-27_21h01m05` | n_tr = 100, resampled: Figure H.1, Table E.4, Figure D.1; Table F.2 |
| `linear_1000_half1_K20_smalldata.yml`, `linear_1000_half1_K20_smalldata2.yml` | `2026-07-31_19h07m06`, `2026-07-31_21h07m56` | Ridge and ExtraTrees in Figure L.1, Table E.8; Table F.2 |
| `linear_1000_half2_K20_smalldata.yml`, `linear_1000_half2_K20_smalldata2.yml` | `2026-07-31_19h02m17`, `2026-07-31_21h05m17` | MLP (32) in Figure L.1, Table E.8; Table F.2 |
| `linear_1000_half1_K20_N1000.yml` (reconstructed) | `2026-07-24_10h46m24` | Appendix K: Ridge gain on R^2 and on the MSE (`analysis/paper_style/appL_r2.py`) |
| `k200_study_rule_prospective_holdout_50.yml` | `2026-05-04_12h45m52` | Earlier redundancy design (216 configurations, d = 20, seeds 1000-1049) that the fresh-seed confirmation repeats; Table F.2 |

The YAML of every single-run file is that of the file that produced the run, matched to the run's parquet on its
`p_*` parameter columns; only the header comments were added. Two files are reconstructions, because their original
was edited after the run: each differs from its sibling in one line, set to the value recorded in the parquet (dataset
`fixed: [True]`, `train_size: [1000]`). These runs used the objective of their time; the current `objective.py`
records more columns.

`legacy/` holds the early configurations of the benchmark, which no paper run uses.
