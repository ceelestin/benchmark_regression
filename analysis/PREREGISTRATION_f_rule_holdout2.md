# Pre-registration: study-only f-rule on HOLDOUT 2 (large-gain design), simulations

Written 2026-09-23 by the orchestrating agent for CE, BEFORE launch of the holdout-2 arrays.
Companion to analysis/PREREGISTRATION_f_rule.md (rule and closed form unchanged; PCam
outcome there: C1/C2 failed on an under-powered set whose gains sat at the threshold).

## Why a second holdout

The pre-registered K=200 holdout (job 101294) had a 5.3% base rate of G_20 >= 10 and its
rule stopped 97% of the time: no discriminating power. Holdout 2 is designed so that large
gains occur for REAL learners (no synthetic noise levers): single trees, 2- and 5-tree
ExtraTrees / RandomForest, 1-NN and 5-NN, a small MLP, at train sizes 300 / 1000 / 3000 where
the derivation grids showed single trees at G_20 = 12-17; one stable control per family
(Ridge / LogReg). Families sim_linear, sim_interactions, sim_nonlinear (MSE loss) and
sim_classification (0-1 loss primary). NEW seeds 4000-4049 (50). ShuffleSplit, K = 200, t = 0.2.
Configs: configs/holdout2_largegain_*.yml (9,300 runs).

## Rule (identical to PREREGISTRATION_f_rule.md)

rho_k = OOF-intersection ICC of the matched per-sample loss after k folds; f_hat_k = 1 - rho_k;
G_K(f) = 1 / [(1 - f)(t + (1 - t)/K) + f/K]. Decision at k = 3: STOP iff G_20(f_hat_3) < 10,
i.e. iff rho_3 > 0.2631. Secondary reading at k = 5 and k = 20.

Also evaluated, for comparison only: the ANOVA-ICC form (f_hat = 1 - study_oof_anova_<loss>_icc)
and the S_3 > 0.03 rule of the first holdout.

## Pre-registered checks (per family and pooled; gain = paper definition G_K^paper, 50 seeds)

* C1 (safety): P(G_20 >= 10 | STOP at k = 3) <= 0.05, pooled over the 3 regression families
  (MSE) and separately on sim_classification (0-1).
* C1' : P(G_200 >= 20 | STOP at k = 3) <= 0.05, same groupings.
* C2 (bound): P(G_20(f_hat_3) >= G_20^paper) >= 0.75 pooled; the train/test coupling only lowers
  the true gain, so the closed form should be an upper bound (holds at 0.63-0.77 on the
  derivation grids at 30 seeds; 50 seeds here).
* Power statement: the base rate P(G_20 >= 10) over all configurations is reported; the checks
  are considered informative only if it is >= 0.25.
* Reported, not claimed: stop rate, P(large | CONTINUE), Spearman(rho_3, G_20), median
  log(pred/obs) by size and learner, G_20/G_5.

## Analysis

benchmark_regression/analysis/derivation_levers_analysis.py --loss neg_mse (regression
families) and --loss accuracy (sim_classification), then analysis/holdout2_prereg_check.py.
