# Crossing the Validation Crisis: code of the simulated benchmark and of the paper's figures

This repository is the entry point to the code of *Crossing the Validation Crisis: Cross-Validation Reduces
Benchmarking Variance Surprisingly Well* (Célestin Eve, Gaël Varoquaux, Thomas Moreau, 2026; `<link to the paper>`).

| Repository | Contents |
|---|---|
| **benchmark_regression** (this one) | the [benchopt](https://benchopt.github.io/) benchmark of the simulated runs (`objective.py`, `datasets/`, `solvers/`, `benchmark_utils/`), the configuration of every simulated run (`configs/`), the SLURM launchers, and the scripts that draw every figure and table of the paper from the run outputs, PCam and Yelp included (`analysis/`) |
| [ceelestin/benchmark_pcam](https://github.com/ceelestin/benchmark_pcam) | the PatchCamelyon (PCam) training loop and its run configurations |
| [ceelestin/benchmark_nlp](https://github.com/ceelestin/benchmark_nlp) | the Yelp Review Full fine-tuning loops and their run configurations |

The benchmark estimates, for a learning algorithm, how much a K-split cross-validation estimate of its performance
gains over a single train/test split: the *variance-equivalent test sample gain*. Each run draws a study set from a
large simulated dataset, evaluates the algorithm with K splits of the study set, and scores every fitted model on a
held-out benchmarking set of 100,000 observations (the oracle) and on outer chunks of fresh data; the analysis scripts
compute the gains, their bootstrap intervals and the study-only redundancy statistics from these scores.

## Install

Python 3.11 (the runs used 3.11.5):

```bash
git clone https://github.com/ceelestin/benchmark_regression
cd benchmark_regression
pip install -r requirements.txt
```

`requirements.txt` pins **benchopt 1.9.0**. From benchopt 1.9.1 on, every cross-validation repetition gets its own copy
of the objective, so the statistics that the objective accumulates across the splits of a run (the `study_*` and
redundancy columns) come out empty. The other pins are the versions the paper's runs used. Install with pip rather
than `benchopt install`, which would also install the requirements of solvers that no run uses (`solvers/TabPFN.py`
pulls torch).

The figure and table scripts read the run outputs and do not need benchopt; they ran with Python 3.12 and the versions
of `requirements-figures.txt` (`pip install -r requirements-figures.txt`).

## Run a simulated configuration

```bash
# a one-seed check that runs in minutes
benchopt run . --config configs/preflight_derivation.yml --no-plot --no-cache --no-timeout -j 4
# one seed chunk (25 of the 1,000 seeds) of the main simulated experiment of Figures 4 and 5, on a 40-core node
benchopt run . --config configs/chunks_paper_main_mse/paper_main_gain_mse_k200__c00.yml \
    --no-plot --no-cache --no-timeout -j 40 --output paper_main_gain_mse_k200__c00
```

benchopt writes `outputs/<name>.parquet`: one row per (dataset, solver, objective parameters, split), with the
parameters in `p_dataset_*`, `p_solver_*` and `p_obj_*` columns and the scores and statistics in `objective_*`
columns. `python check_run_integrity.py outputs/<name>.parquet` checks the split bookkeeping of a result file.

## How the paper's runs were made

- **Chunked runs.** Most configurations were split into seed chunks (`make_chunks.py`); the chunks of every paper run
  and their manifests are tracked in `configs/chunks_<design>/`. Each chunk ran as one task of a SLURM job array
  (`submit_<design>.slurm`, written for the CPU nodes of the Jean-Zay cluster: set the account, the array range and
  the environment, and `mkdir -p logs` first), with `benchopt run ... -j 40 --output <config>__cNN`.
  `merge_chunks.py` merges the chunk outputs per config and `check_run_integrity.py` checks them. Step by step:
  [docs/jean-zay.md](docs/jean-zay.md).
- **Single runs.** The ranking runs of Figure O.1 and the July 2026 runs (hyper-parameter search, frozen vs resampled
  splits, small budgets) ran as one `benchopt run . --config configs/<file>.yml` each on a 128-core machine. benchopt
  named their outputs `benchopt_run_<date>_<time>.parquet`, and the figure scripts read them under these names.

[configs/README.md](configs/README.md) lists every configuration, the run or launcher it belongs to, and where the
paper uses it.

## Data layout

The figure and table scripts read the run outputs under one **data root**: the `--data-root` argument of the scripts
that take one, else the `DATA_ROOT` environment variable, else `data/` at the repository root
(`analysis/paper_style/code/datapaths.py`). The run outputs are not distributed with this repository; the runs above
produce them. Expected layout:

```
<data root>/
  outputs/        <config>__cNN.parquet: benchopt outputs of the chunked runs (the outputs/ folder of benchopt)
  merged/         <config>.parquet: merge_chunks.py --in-dir outputs --out-dir <data root>/merged
  paper_runs/     benchopt_run_2026-05-04_12h45m52.parquet and benchopt_run_2026-07-*.parquet: single runs of
                  configs/k200_study_rule_prospective_holdout_50.yml, configs/config_hpo_test_*.yml and
                  configs/linear_1000_*.yml (configs/README.md maps file names to configs)
  ranking/        benchopt_run_2026-05-31_05h37m39.parquet ... benchopt_run_2026-06-04_02h56m57.parquet: the 12 runs
                  of configs/ranking_*.yml
  pcam/
    results_output_pcam/  PCam K-split runs (benchmark_pcam: rebuttal_shufflesplit.txt, rebuttal_vit.txt,
                          rebuttal_repeatedkfold.txt)
    stratandnewmetrics/   PCam stratified and unstratified runs with Brier score and NLL
                          (benchmark_pcam: stratification*_newmetrics_full*.txt)
    nocv_jan/             PCam single-split runs (benchmark_pcam: configs_adapted_100seeds_5study.txt), Table M.1
    ntr24/                PCam runs at n_tr = 24 (benchmark_pcam: redundancy_smalldata.txt), Appendix L
    calib100/             PCam 100-seed calibration runs (analysis/pcam_consistency_check.py only)
  nlp/
    yelp_runs_a/, yelp_runs_b/  first BERT and XLM-RoBERTa runs (two output folders of benchmark_nlp)
    yelp_qwen3_0.6b/            Qwen3-0.6B runs (benchmark_nlp: qwen_0.6b_1000.txt, qwen_0.6b_1000_nocv.txt)
    yelp_aligned/               BERT and XLM-RoBERTa on the PCam protocol (benchmark_nlp: aligned_bert.txt,
                                aligned_roberta.txt)
  mse_paper_figs/, holdout3_in/   inputs of two side scripts (code/mse_gain_figures.py, coherence_checks/rho_all.py)
```

Each sub-folder can be a link to wherever the outputs are. Three more locations, also set by environment variables:
`FIGS_DIR` (default `figs/`) holds what the figure scripts write and the table scripts read back; `BENCHMARK_PCAM` and
`BENCHMARK_NLP` (default: `benchmark_pcam/` and `benchmark_nlp/` next to this repository) are checkouts of the
companion repositories, whose analysis outputs some scripts read. The outputs of this repository's own analysis
scripts go to `analysis/out_<name>/`.

## Figures and tables of the paper

Scripts are in `analysis/paper_style/` unless the path says otherwise; [analysis/paper_style/README.md](analysis/paper_style/README.md)
gives the order in which to run them. Configs are in `configs/` here, or in the companion repository named.

| Paper | Runs | Script |
|---|---|---|
| Figure 1 | OpenML dataset sizes | not in this repository |
| Table 1, Figure 3 | (literature table, diagram) | none |
| Figure 2 | PCam K-split runs (benchmark_pcam `rebuttal_shufflesplit.txt`) | `code/pcam.py`, run by `regen_real_data_gain_figures.py --only fig5_pcam` |
| Figure 4, Figure 5 (left), Table E.1 | `paper_main_gain_mse_k200.yml`, `paper_main_gain_mse_k200_rff50.yml` | `paper_style_figs.py`; intervals `ci_appendix.py` |
| Figure 5 (middle), Table E.2 | benchmark_pcam `rebuttal_shufflesplit.txt`, `rebuttal_vit.txt` | `regen_real_data_gain_figures.py --only fig5_pcam` |
| Figure 5 (right), Figure 7, Table E.2 | benchmark_nlp `aligned_bert.txt`, `aligned_roberta.txt`; Qwen3-0.6B: `qwen_0.6b_1000.txt`, `qwen_0.6b_1000_nocv.txt` | `regen_real_data_gain_figures.py --only fig5_yelp_aligned` |
| Figure 6 | noise-injection grid `derivation_redundancy_levers_*.yml`; pre-registered holdout `k200_threshold_validation_holdout_50_*.yml`; unstable-learner designs `holdout2_largegain_*.yml`, `holdout3_largegain_*.yml`; PCam: benchmark_pcam `redundancy_round3_50seeds_3sizes.txt`; Yelp: the aligned runs | `analysis/derivation_levers_analysis.py` then `analysis/omega_gain_relation.py` (tables `analysis/out_omega/`), then `paper_style_scatter.py`, which draws the simulated and PCam points (the Yelp points are not yet drawn by a script of this repository) |
| Figure D.1 | runs of Figure 4; `linear_1000_K20_new_unfixed.yml`, `paper_ntr100_k20_resampled_rff50.yml`; the PCam and Yelp runs of Figure 5 | `figD1_rho_delta.py` |
| Table D.1 | `k200_threshold_validation_holdout_50_*.yml` (plan: `analysis/PREREGISTRATION_f_rule.md`); fresh-seed confirmation `paper_rerun_may_design_*.yml` | gains: `analysis/derivation_levers_analysis.py --loss neg_mse --glob '<data root>/outputs/k200_threshold_validation_holdout_50_*__c*.parquet' --out-dir analysis/out_holdout_mse`; then `analysis/redundancy_triage.py --gains-dir analysis/out_holdout_mse --out-dir analysis/out_redundancy_triage_holdout1` and `analysis/redundancy_triage_paper_outputs.py --dir analysis/out_redundancy_triage_holdout1` |
| Table D.2 | benchmark_pcam `redundancy_round3_50seeds_3sizes.txt` | benchmark_pcam `analysis/pcam_gain_analysis.py`, `analysis/pcam_redundancy_vs_gain.py` |
| Tables E.1 to E.8 | the runs of Figures 4, 5, G.1, H.1, I.1, J.1, K.1 and L.1 | `ci_appendix.py` (coverage check: `ci_coverage.py`) |
| Table F.2 | the runs above; `timing_*.yml` re-measure 5% of the July runs | `submit_timing.slurm`; `coherence_checks/yelp_times.py`, `coherence_checks/yelp_aligned_times.py` |
| Figure G.1, Tables G.1 and E.3 | `hpo_mse_MLP_gridsearch.yml`, `hpo_mse_MLP_randomsearch.yml`, `config_hpo_test_gridsearch.yml`, `config_hpo_test_randomsearch.yml` (ExtraTrees); fixed parameters from `paper_main_gain_mse_k200.yml` | `paper_style_figs.py`; redundancy `redundancy_numbers.py`, `analysis/paper_appendix_redundancy_checks.py`; text `text_numbers_v5.py` |
| Figure H.1, Table E.4 | `linear_1000_K20_new_fixed.yml`, `linear_1000_K20_new_unfixed.yml` (Ridge, ExtraTrees, MLP); `paper_ntr100_k20_fixed_redrawn_rff50.yml`, `paper_ntr100_k20_resampled_rff50.yml` (RandomFeatures) | `paper_style_figs.py`; standard-deviation ratios `sd_ratio_mse.py` |
| Figure I.1, Table E.5 | benchmark_pcam `rebuttal_shufflesplit.txt`, `rebuttal_repeatedkfold.txt` | `regen_real_data_gain_figures.py --only figF1` |
| Table J.1, Figure J.1, Table E.6 | benchmark_pcam `stratification*_newmetrics_full*.txt` | `coherence_checks/strat_table.py` (Table J.1); `regen_real_data_gain_figures.py --only figK1` |
| Figure K.1, Table E.7 | the same PCam runs; for the text: `linear_1000_half1_K20_N1000.yml` (Ridge, R^2 and MSE), `paper_main_gain_mse_k200*.yml` (MAE), `pilot_persample_k20.yml` (R^2 recentred) | `regen_real_data_gain_figures.py --only figL1`; `appL_r2.py`, `text_numbers_v5.py`, `analysis/persample_r2_centering.py` |
| Figure L.1, Table E.8 | `linear_1000_half{1,2}_K20_smalldata{,2}.yml`, `small_budget_mlp64_rff50.yml`; PCam at n_tr = 24: benchmark_pcam `redundancy_smalldata.txt` | `paper_style_figs.py`; text `text_numbers_v5.py`, `redundancy_numbers.py`, `coherence_checks/fold_geometry.py`, `coherence_checks/pcam24_*.py` |
| Table M.1 | PCam: benchmark_pcam `configs_adapted_100seeds_5study.txt` (single split) and the K-split runs; Yelp: the aligned runs | Yelp rows: `code/nlp.py` via `regen_real_data_gain_figures.py --only fig5_yelp_aligned`, `coherence_checks/yelp_paper_snippets.py`; PCam rows: `coherence_checks/g1_pcam_check.py` |
| Table N.1 | benchmark_pcam `rebuttal_shufflesplit.txt` | `code/replicability.py` |
| Figure O.1 | `ranking_100seeds*.yml`, `ranking_100-300seeds*.yml`, `ranking_300-500seeds*.yml` | `code/ranking.py` |

Other numbers of the text come from `coherence_checks/` (see its README), `redundancy_numbers.py` and
`text_numbers_v5.py`. `analysis/PREREGISTRATION_*.md` are the analysis plans written before the corresponding runs
(their commit dates time-stamp them), and `analysis/ADDENDUM_order_averaged_gain.md` the post-hoc re-evaluation of their
checks.

## Repository layout

```
objective.py, datasets/, solvers/, benchmark_utils/   the benchopt benchmark
configs/                    run configurations, chunks of the chunked runs (chunks_<design>/), legacy/ (early configs)
submit_<design>.slurm       SLURM job arrays of the chunked runs (Jean-Zay)
make_chunks.py, merge_chunks.py, check_run_integrity.py
analysis/                   analyses of the simulated runs (redundancy, gain decomposition, pre-registered checks)
analysis/paper_style/       figure and table scripts of the paper; shared code in code/; checks in coherence_checks/
docs/jean-zay.md            running the chunked runs as SLURM arrays
tests/                      tests of the objective's incremental statistics (pytest)
learning_ranking.py         analysis of the early configs in configs/legacy/ (needs dask; not used by the paper)
```

## Citation

```bibtex
@article{eve2026crossing,
  title  = {Crossing the Validation Crisis: Cross-Validation Reduces Benchmarking Variance Surprisingly Well},
  author = {Eve, C{\'e}lestin and Varoquaux, Ga{\"e}l and Moreau, Thomas},
  year   = {2026},
  note   = {<venue or arXiv identifier>}
}
```

## License

BSD 3-Clause, see [LICENSE](LICENSE). The PCam data are released under CC0 and the Yelp Review Full data under the Yelp
Dataset Terms of Use (non-commercial use); see the companion repositories.
