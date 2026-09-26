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

---

## OUTCOME (added 2026-09-23 after the analysis; text above unchanged)

Job 121758: 40/40 tasks in 100 min, integrity OK on all parquets. Analyses
analysis/out_holdout2_mse (162 regression configurations) and analysis/out_holdout2_acc
(24 classification configurations), 50 seeds each; checks by analysis/holdout2_prereg_check.py.

| group | base rate G_20>=10 | stop rate (k=3) | C1 P(G20>=10 \| stop) | C1' P(G200>=20 \| stop) | C2 P(pred>=obs) | median log(pred/obs) | Spearman(rho_3, G_20) |
|---|---|---|---|---|---|---|---|
| regression, MSE | 0.18 (under-powered per pre-reg) | 0.78 | 0.056 (7/126) FAIL | 0.040 PASS | 0.50 FAIL | +0.00 | -0.75 |
| classification, 0-1 | 0.54 (informative) | 0.42 | 0.200 (2/10) FAIL | 0.200 FAIL | 0.46 FAIL | -0.02 | -0.72 |

By regression family: sim_linear C1 = 0.028 (base rate 0.24), sim_interactions 0.071 (0.22),
sim_nonlinear 0.062 (0.07). Reported: median G20/G5 = 1.79 (regression), 2.62 (classification).
Comparisons at k=3: ANOVA-ICC form C1 = 0.055 / 0.111; S_3 > 0.03 stops 100% of regression
configurations (C1 = 0.179) and 25% of classification ones (C1 = 0.000); Spearman(S_3, G_20)
= -0.32 / -0.43 versus -0.75 / -0.72 for rho_3.

Reading. (1) The design goal was only met for classification: single trees reached G_20 of
8.5/10.4/14.0 at 300/1000/3000 in regression (12-17 were expected from the derivation grids),
so the regression base rate stayed at 0.18. (2) The closed form is UNBIASED in level at 50
seeds (median log ratio 0.00 / -0.02) but it is NOT an upper bound: C2 sits at 0.50 / 0.46. The
train/test-coupling argument for conservativeness does not carry over to these learners; the
prediction scatters symmetrically around the truth. (3) Consequently a stop threshold placed
exactly at the target (predicted G_20 < 10) lets configurations whose true gain is just above
10 through at roughly the noise rate: the stopped-but-large cases are 2- and 5-tree
RandomForest / ExtraTrees with rho_3 in 0.26-0.51 and observed G_20 in 10.5-18.7. (4) Ranking
is confirmed on real learners at 50 seeds: rho_3 -0.75 / -0.72, three times S_3.

Exploratory, NOT pre-registered: with the same statistic, a stricter threshold rho_3 > 0.35
(predicted G_20 < 8.6) gives C1 = 0.018 (regression) and 0.000 (classification) at stop rates
0.68 and 0.25; rho_3 > 0.50 gives 0.011 / 0.000 at 0.57 / 0.25. A safety margin of ~15% on
the predicted gain is what the 50-seed scatter requires.

Verdict: the pre-registered rule fails C1 (marginally in regression, clearly in classification)
and fails C2 in both; the study-only statistic ranks gains well and predicts their level
without bias, but it is not conservative, and a validated rule needs an explicit margin.

---

Post-hoc note (2026-09-26): these checks were re-evaluated with the order-averaged paper gain, outside the pre-registration; see analysis/ADDENDUM_order_averaged_gain.md. The text above is unchanged.
