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
