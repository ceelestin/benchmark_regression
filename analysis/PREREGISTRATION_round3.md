# Pre-registration, round 3: study-only stopping rule with a probabilistic loss and a margin

Written 2026-09-24 by the orchestrating agent for CE, BEFORE launching any round-3 job and
before computing any gain on the round-3 seeds. Builds on analysis/PREREGISTRATION_f_rule.md
(rounds 1-2 on PCam) and analysis/PREREGISTRATION_f_rule_holdout2.md.

## What rounds 1-2 established (data: jobs 49236, 96950, 101451, 101294, 121758, 121977)

1. The study-only OOF-intersection ICC rho_k of a per-sample loss ranks the exact
   variance-equivalent test sample gain well when the loss is probabilistic (squared error,
   NLL, Brier): Spearman -0.75/-0.72 on holdout 2 (50 seeds), -0.95/-0.76 on the 100-seed
   PCam set with NLL/Brier, versus -0.50 with the thresholded 0-1 error. The 0-1 error hides
   fold-to-fold variability that the probabilistic losses see (ViT-B/16, size 800: rho_err01
   0.39 vs rho_nll 0.09, true G_20 = 11-18).
2. The closed form G_K(f) = 1/[(1-f)(t+(1-t)/K)+f/K] with f = 1 - rho predicts the LEVEL of
   the gain without bias (median log(pred/obs) within +-0.1 on every set) but it is NOT an
   upper bound (P(pred >= obs) = 0.46-0.62 everywhere). A stop placed exactly at the target
   therefore misses true gains just above it at roughly the noise rate.
3. Exploratory on holdout 2: stopping only when the predicted G_20 is below ~8.6 (rho_3 >
   0.35) gave P(G_20 >= 10 | stop) of 0.018 (regression) and 0.000 (classification).

## The rule, version 3 (computable from the study set alone)

ShuffleSplit with test fraction t (0.2 here), K splits. After k splits:

1. Loss for the redundancy statistic: a PROBABILISTIC per-sample loss -- NLL for classifiers
   (primary), Brier (= squared error on P(y=1)) secondary; squared error for regression --
   even when the metric of interest is the accuracy. rho_k = OOF-intersection ICC of that
   loss across the k folds (`study_<loss>_rho_cum_oof_intersection` in PCam,
   `study_<loss>_rho_all_oof_intersection` at fold k in benchmark_regression).
2. f_hat_k = 1 - rho_k; predicted gain G_K(f_hat_k) by the closed form above.
3. **Decision at k = 3: STOP iff G_20(f_hat_3) < 8.6, i.e. iff f_hat_3 < 0.6512, i.e. iff
   rho_3 > 0.3488** (t = 0.2). The 8.6 is a ~15% margin under the target gain of 10, sized on
   the 50-seed scatter of holdout 2. Otherwise CONTINUE.
   Secondary readings: k = 5 and k = 20; ANOVA-ICC form (f_hat = 1 - study_oof_anova_<loss>_icc).

## Pre-registered checks (gain = paper definition G_K^paper, bootstrap over seeds)

* C1 (safety, primary): P(G_20 >= 10 | STOP at k = 3) <= 0.05.
* C1' (K = 200 sets only): P(G_200 >= 20 | STOP at k = 3) <= 0.05.
* C2' (calibration, replaces the bound check of rounds 1-2): |median log(G_20(f_hat_3) /
  G_20^paper)| <= 0.15 over all configurations.
* C3 (usefulness): P(G_20 >= 10 | CONTINUE at k = 3) >= 0.50 -- the rule must let through
  mostly configurations that do gain.
* Power statement: base rate P(G_20 >= 10) >= 0.25 for the checks to count as informative.
* Reported, not claimed: stop rate, Spearman(rho_3, G_20), the same at k = 5 / 20, the
  ANOVA-ICC and 0-1 variants, G_20/G_5.

## Validation sets (all new seeds; nothing in them has been analysed)

A. PCam, round 3: DenseNet121, MobileNetV2, WideResNet101_2, ViT-B/16 x train sizes
   100 / 300 / 800 x seeds 100-149 (50 seeds; extendable to 100 if the checks are borderline),
   K = 20, t = 0.2, 30 epochs. 12 configurations. Loss for the statistic: NLL (primary), Brier
   (secondary); the gain is computed on accuracy (as in the paper) AND on NLL/Brier.
   Cost change vs. rounds 1-2, decided before launch: the oracle (benchmarking set) is a fixed
   stratified subset of 60,000 of the ~240k benchmarking images (`--bench-max-samples 60000`,
   random_state 43), which divides the ViT evaluation cost per fold by 4; the oracle's
   sampling error (Var <= 0.25/60000) stays two orders of magnitude below the fold-level
   variances involved. Config: configs/redundancy_round3_50seeds_3sizes.txt (600 lines),
   launcher launch_round3.sh.
B. Simulations, HOLDOUT 3 (large-gain design v2): sim_linear, sim_interactions, sim_nonlinear
   (MSE; statistic on squared error) and sim_classification (accuracy metric; statistic on
   NLL primary, Brier secondary). Train sizes 1000 / 3000 / 10000, where holdout 2 found the
   large gains; learners DecisionTree, ExtraTrees {2, 5} trees, RandomForest {2, 5} trees,
   Ridge control (regression); DecisionTreeClf, ExtraTreesClf {2, 5}, LogReg control
   (classification). Noise 0.3 / 1.0 (logit_scale 1.0 / 3.0). Seeds 5000-5049. K = 200,
   ShuffleSplit, t = 0.2. Configs: configs/holdout3_largegain_*.yml (6,600 runs).

## Analysis

benchmark_pcam analysis/pcam_gain_analysis.py --rho cum --loss {accuracy,nll,brier} and
benchmark_regression analysis/derivation_levers_analysis.py (--loss neg_mse / accuracy), then
analysis/round3_prereg_check.py on the candidates.csv files with rho* = 0.3488, target 10,
predicted-gain stop level 8.6.

---

## OUTCOME, validation set A (PCam round 3) -- added 2026-09-24 after the analysis; text above unchanged

Job 148283: 600/600 tasks, seeds 100-149, sizes 100/300/800, 4 models, K = 20, oracle subset
60,000. Analysis: benchmark_pcam analysis/pcam_gain_analysis.py --rho cum, outputs
analysis/out_round3/; checks: analysis/round3_prereg_check.py. 12 configurations, 50 seeds.

| gain metric / statistic | base rate | stop rate | C1 P(G20>=10 \| stop) | C2' median log(pred/obs) | C3 P(G20>=10 \| continue) | Spearman(rho_3, G_20) |
|---|---|---|---|---|---|---|
| accuracy / NLL (PRIMARY) | 0.33 | 0.50 | **0.167 (1/6) FAIL** | **+0.27 FAIL** | 0.50 PASS | -0.73 |
| accuracy / Brier | 0.33 | 0.67 | 0.125 (1/8) FAIL | +0.21 FAIL | 0.75 PASS | -0.71 |
| **NLL / NLL** | 0.25 | 0.50 | **0.000 (0/6) PASS** | **+0.12 PASS** | 0.50 PASS | -0.49 |
| accuracy / 0-1 (reference) | 0.33 | 0.75 | 0.111 (1/9) FAIL | +0.21 FAIL | 1.00 | -0.62 |

The single stopped-but-large cell under every accuracy row is DenseNet121 at size 800:
G_20(accuracy) = 10.7 with rho_3 of 0.46-0.51 on all three losses (predicted 6.4-7.0).

Reading. (1) When the statistic and the gain use the SAME probabilistic loss (NLL/NLL), all
three pre-registered checks pass on PCam: no unsafe stop, calibrated level (+0.12), and half
of the continued cells gain. (2) When the gain is measured on ACCURACY while the statistic uses
NLL or Brier, the closed form under-predicts the accuracy gain by ~25-30% (C2' fails) and one
cell just above the target is stopped: the mismatch between the loss of the statistic and the
metric of the gain is not free. (3) The 0-1 statistic is not a fix: it under-predicts as well
(+0.21) and, in round 2, hid ViT's fold-driven variability. (4) Ranking with the probabilistic
statistics is strong on 12 cells (-0.71 to -0.73 against the accuracy gain).

Verdict on set A: the round-3 rule is validated for a probabilistic reported metric (NLL)
and NOT validated for accuracy as the reported metric; the primary check fails by one cell
whose gain (10.7) sits inside the noise band of the target at 50 seeds. Set B (holdout 3,
job 147428) pending.

Erratum to the set-A reading, added 2026-09-25: C2' is median log(pred/obs), so the +0.27 (NLL)
and +0.21 (Brier) of the accuracy rows mean the closed form OVER-predicts the accuracy gain by
~23-31% in the median, not "under-predicts" as written in Reading (2). The stopped DenseNet121
size-800 cell is an individual under-prediction (predicted 6.4-7.0, observed 10.7). Numbers and
verdict unchanged.

---

## OUTCOME, validation set B (holdout 3) -- added 2026-09-25 after the analysis; text above unchanged

Job 147428: 40/40 tasks COMPLETED (7-10 min each, after ~36 h pending); check_run_integrity OK
on all 40 chunk parquets (6,600 runs, 0 contaminated); merged to one parquet per family.
Analyses analysis/derivation_levers_analysis.py --loss neg_mse (sim_linear, sim_interactions,
sim_nonlinear; analysis/out_holdout3_mse, 108 configurations) and --loss accuracy
(sim_classification; analysis/out_holdout3_acc, 24 configurations), 50 seeds each; checks by
analysis/round3_prereg_check.py --k200.

| group / statistic | base rate | stop rate | C1 P(G20>=10 \| stop) | C1' P(G200>=20 \| stop) | C2' median log(pred/obs) | C3 P(G20>=10 \| continue) | Spearman(rho_3, G_20) |
|---|---|---|---|---|---|---|---|
| regression, squared error (PRIMARY) | 0.28 | 0.51 | **0.091 (5/55) FAIL** | 0.000 PASS | +0.070 PASS | **0.47 FAIL** | -0.60 |
| classification, NLL (PRIMARY) | 0.58 | 0.25 | **0.000 (0/6) PASS** | 0.000 PASS | **+0.265 FAIL** | 0.78 PASS | -0.60 |
| classification, Brier (secondary) | 0.58 | 0.25 | 0.000 (0/6) PASS | 0.000 PASS | +0.111 PASS | 0.78 PASS | -0.46 |
| classification, 0-1 (reference) | 0.58 | 0.25 | 0.000 (0/6) | 0.000 | +0.019 | 0.78 | -0.65 |

Readings at k = 20 are identical to k = 3 up to one configuration. Reported: median G_20/G_5 =
1.88 (regression), 2.16 (classification).

The five stopped-but-large regression configurations are all 2- or 5-tree RandomForest /
ExtraTrees at train size 3000 or 10000: sim_linear RF 10000 noise 0.3 (G_20 12.6, rho_3 0.47),
sim_linear ET 3000 noise 1.0 (11.0, 0.46), sim_nonlinear RF 10000 noise 0.3 (16.0, 0.54),
sim_nonlinear ET 10000 noise 1.0 (16.0, 0.55), sim_nonlinear RF 10000 noise 1.0 (11.2, 0.57).
The two 16.0 values sit at the 16-chunk ceiling of G_paper_20, so their gain is at least 16.

Exploratory, NOT pre-registered:
* Calibration drifts with train size in regression. Median log(pred/obs) is +0.31 / +0.07 /
  -0.29 at sizes 1000 / 3000 / 10000, so the pooled C2' pass hides a trend. Median rho_3 of the
  small ensembles hardly moves with size (RF 0.42 / 0.39 / 0.39) while their G_20 grows
  (6.1 / 8.7 / 13.3). The unsafe stops are where the closed form under-predicts.
* By regression family, C1 = 0.000 (sim_interactions, 0/18), 0.111 (sim_linear, 2/18), 0.158
  (sim_nonlinear, 3/19).
* With the variance-ratio gain G_err_20 instead of G_paper_20, regression C1 = 0.036 (2/55); the
  stopped-but-large cells have G_err_20 of 5.8-12.9.
* In classification the NLL statistic over-predicts at every size (+0.30 / +0.18 / +0.29),
  strongest for ExtraTreesClf (+0.33). Over-prediction is the conservative direction for the
  stop decision; the Brier statistic is calibrated (+0.11).

Verdict on set B: the round-3 rule is NOT validated for regression: it fails C1 (0.091) and
narrowly C3 (0.47), with the failures concentrated on small tree ensembles at large train size.
For classification it passes the safety checks C1 and C1' and usefulness C3, and fails only the
calibration check C2' with the primary NLL statistic, in the conservative direction; with the
Brier statistic it passes all four checks.

Across both round-3 sets: no unsafe stop in classification when the statistic is probabilistic
(PCam with an NLL gain; holdout 3 with an accuracy gain); unsafe stops remain where the gain
grows with train size faster than the study-only statistic registers it (small ensembles in
regression here, DenseNet121 at size 800 on PCam with an accuracy gain).

---

Post-hoc note (2026-09-26): these checks were re-evaluated with the order-averaged paper gain, outside the pre-registration; see analysis/ADDENDUM_order_averaged_gain.md. The text above is unchanged.
