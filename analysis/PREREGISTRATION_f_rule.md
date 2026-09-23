# Pre-registration: study-only stopping rule for repeated CV splits, validated on PCam

Written 2026-09-23 by the orchestrating agent for CE, BEFORE any PCam gain was computed
(the PCam calibration parquets, job 9433 + 18278, were only inspected for column names).
Derivation data: benchmark_regression jobs 49236 (sim_linear, regression), 96950
(sim_classification) and 101451 (train/test coupling test), seeds 3000-3029.

## The rule (everything is computable from the study set alone)

Run K splits of ShuffleSplit with test fraction t (here t = 0.2). After k splits (k >= 2):

1. Redundancy: rho_k = OOF-intersection intra-class correlation of the per-sample loss
   across folds, on the samples that are out-of-fold for both folds of each pair, averaged
   over the k(k-1)/2 pairs. Matched loss: 0-1 error for accuracy, NLL for the log score,
   squared error for the Brier / MSE score. (PCam column: `study_<loss>_rho_cum_oof_intersection`;
   benchmark_regression column: `study_<loss>_rho_all_oof_intersection`.)
2. Fold-driven share: f_hat_k = 1 - rho_k.
3. Predicted variance-equivalent test sample gain of K splits (closed form; derivation
   analysis/closed_form_gain_check.py):

       G_K(f) = 1 / [ (1 - f) (t + (1 - t) / K) + f / K ]        (ShuffleSplit)
       G_K(f) = 1 / [ (1 - f) t + f / K ]                        (RepeatedKFold, complete repeats)

   The sample-driven part of the variance (1 - f) can only be averaged down to its
   full-study value (finite population: gain <= 1/t); the fold-driven part f averages as 1/K.
4. **Decision at k = 3 (primary):** STOP adding splits iff G_20(f_hat_3) < 10,
   i.e. iff f_hat_3 < 0.7369, i.e. iff **rho_3 > 0.2631** (t = 0.2). Otherwise CONTINUE.

Direction of the approximation. The train/test coupling across folds (a study sample is a
training point for some folds and a test point for others) adds a K-independent variance
floor that no within-sample study-only statistic can see (benchmark_regression job 101451:
the excess disappears when every fold trains on a pool disjoint from the study). It only
LOWERS the true gain, so G_K(f_hat) is an optimistic bound and a STOP is conservative:
it should never hide a true large gain. The cost is late stopping; its size is reported,
not claimed.

## Pre-registered checks on PCam (redundancy-calibration set: DenseNet121, MobileNetV2,
WideResNet101_2, ViT-B/16; train sizes 100/300/500/800; 25 seeds; K = 20; t = 0.2)

Exact gain: G_K^paper per (model, size) from the 25 seeds, Definitions 3.1-3.2 of the paper
(Delta_K = mean_k q_k - mean_k b_k with b_k the benchmark-set score of the same model;
V_1(alpha) from the hidden-test chunks of the first fold pooled with the test fold).
Primary loss: accuracy (0-1). Secondary: NLL, Brier.

* C1 (safety, primary): among (model, size) configurations STOPPED at k = 3,
  P(G_20^paper >= 10) <= 0.05.
* C2 (bound): P(G_20(f_hat_3) >= G_20^paper) >= 0.75 over all 16 configurations
  (the level observed on the simulated classification grids, 0.75-0.77).
* Reported, not claimed: the stop rate, the median log(G_20(f_hat_3) / G_20^paper)
  (late-stopping cost), the Spearman rank correlation between rho_3 and G_20^paper,
  and the same quantities at k = 5.

## Derivation numbers, for the record (analysis/out_*; 30 seeds, 150 ShuffleSplit
configurations per grid; gains carry ~0.4 log noise at 30 seeds)

| grid, loss            | stop rate (k=3) | P(G_20>=10 \| stop) | P(G_200>=20 \| stop) | P(bound holds) | median log(pred/obs) |
|-----------------------|-----------------|---------------------|----------------------|----------------|----------------------|
| sim_linear, MSE       | 0.35            | 0.113               | 0.132                | 0.63           | +0.18                |
| sim_classif., 0-1     | 0.66            | 0.051               | 0.010                | 0.76           | +0.35                |
| sim_classif., NLL     | 0.71            | 0.047               | 0.000                | 0.76           | +0.41                |
| sim_classif., Brier   | 0.72            | 0.046               | 0.000                | 0.75           | +0.34                |

Spearman(rho_3, G_20^paper): -0.89 (sim_linear), -0.90 / -0.88 / -0.90 (0-1 / NLL / Brier).
Fixed-split control (identical folds): G = 1.00 everywhere.

Excess variance floor observed/closed-form (train/test coupling), median: classification
1.7-2.1x at n = 50-100, 1.4 at 1000, 1.0 at 3000; regression 1.85 at n = 50, ~1.1 for
n >= 300. The floor is K-independent (corr 0.89 between K = 20 and 200 estimates) and
scales roughly as n^-0.3, not as p/n; no robust study-only correction was found, hence the
bound formulation above.

## Not part of the pre-registration

The S_3 = cov x rho rule of configs/k200_threshold_validation_holdout_50_*.yml (job 101294)
keeps its own pre-registration and is reported separately. On the derivation grids S
ranked the gains at Spearman -0.84 (regression) and -0.66 (classification), rho at -0.89.
