# Pre-registration: PCam consistency check of the simulated gain / redundancy relation

Written 2026-09-26 BEFORE computing the statistic below on PCam. Requested by CE. This is a
CONSISTENCY check, not a fresh validation: the PCam seeds used here (0-149) were already analysed
in rounds 1-3 with related statistics (median over seeds of the loss correlation at k = 3 and 20).
A genuine validation would need new PCam seeds (GPU, CE's decision).

## Quantities (paper definitions; redundancy on the gain's own metric)

* Gain: paper sample gain G_20 (variance-equivalent test size / n_te, on delta scores = fold score
  minus benchmark score of the same predictor), single-split curve averaged over the order of the
  hidden chunks (analysis/ADDENDUM_order_averaged_gain.md), K = 20.
* Statistic: x = t * rho_e with t = 0.2, rho_e = the loss-correlation factor of the paper's
  redundancy score omega-hat (pair-averaged covariance / variance of per-sample losses on the
  samples held out by both splits), read after the first k splits, averaged over seeds (mean).
  Primary: accuracy gain with the 0-1 error correlation, k = 5. Secondary: k = 3; NLL gain with the
  NLL correlation.

## Data

* PCam: 100-seed set (job 121977 + 138125; seeds 0-99; DenseNet121, MobileNetV2,
  WideResNet101_2, ViT-B/16 x train sizes 100 / 800; 8 configurations) and round-3 set (job 148283;
  seeds 100-149; train sizes 100 / 300 / 800; 12 configurations; oracle = 60,000-image stratified
  subset of the benchmarking set). 20 configurations. The statistic uses
  `study_error01_rho_cum_oof_intersection` (NLL: `study_nll_rho_cum_oof_intersection`) on the row
  of fold k. The 25-seed grid (job 9433) is excluded: it lacks the cumulative columns.
* Simulated reference: every simulated sim_classification configuration with train size between
  100 and 1000 available when the check is run (derivation grid, holdouts 2-3, study arm of the
  coupling test; analysis/out_omega/), same gain and statistic definitions. The planned
  classification sweep (seeds 6000-6049) will be a second, secondary reference when it exists.

## Reference band and checks

For each PCam configuration with statistic x: neighbours = simulated configurations with
|x_sim - x| <= 0.02, widened by 0.01 until there are at least 10; band = 10th-90th percentile of
their G_20; centre = median of their G_20.

* A1 coverage: at least 70 % of the PCam configurations (14 of 20) have G_20 inside their band
  (about 80 % expected if PCam follows the simulated relation).
* A2 offset: |median over PCam configurations of log(G_20 / centre)| <= 0.2.
* A3 ranking: Spearman(x, G_20) over the 20 PCam configurations <= -0.5.

Reported, not claimed: each check per PCam set, at k = 3, and for the NLL gain; the position of every
configuration against the band; the ideal curve K / (1 + (K - 1) x).

## Analysis

benchmark_regression analysis/pcam_consistency_check.py (to be written after this file is
committed): PCam per-seed statistics read from the parquets, gains from benchmark_pcam
analysis/out_calibration_100seeds (seeds 0-99 only) and analysis/out_round3 (gains_accuracy.csv,
gains_nll.csv), simulated reference from analysis/out_omega/.

---

## OUTCOME (added 2026-09-26 after running analysis/pcam_consistency_check.py; text above unchanged)

20 PCam configurations (8 from the 100-seed set, 12 from round 3); simulated reference: 132
sim_classification configurations with train size 100-1000. Outputs analysis/out_pcam_consistency/.

| reading | A1 coverage (>= 0.70) | A2 median log offset (\|.\| <= 0.2) | A3 Spearman (<= -0.5) |
|---|---|---|---|
| accuracy gain, k = 5 (PRIMARY) | 0.65 (13/20) FAIL | +0.30 FAIL | -0.80 PASS |
| accuracy gain, k = 3 | 0.45 FAIL | +0.26 FAIL | -0.88 PASS |
| NLL gain, k = 5 | 0.55 FAIL | +0.27 FAIL | -0.92 PASS |
| NLL gain, k = 3 | 0.55 FAIL | +0.29 FAIL | -0.94 PASS |

By PCam set (primary): 100-seed set coverage 0.50, offset +0.39; round 3 coverage 0.75, offset +0.21;
Spearman -0.86 / -0.92.

Verdict: the ranking carries over to PCam (A3 passes on every reading), but PCam gains sit about 30 %
above the simulated classification band at the same redundancy, so the simulated relation as
pre-registered is NOT consistent with PCam in level (A1, A2 fail).

Exploratory, NOT pre-registered: PCam follows the ideal curve K / (1 + (K - 1) x) closely (median
log(G / curve) -0.02 to +0.04; 80-85 % of configurations within +-25 % of it). The simulated
reference sits 0.30 below that curve, and the offset depends on the simulated set: derivation grid
-0.34 and coupling test -0.36 (dominated by LogReg with noise levers at small train sizes), holdout 3
-0.16, holdout 2 0.00 (natural unstable learners at 300-1000). The composition of the simulated
reference, not the redundancy statistic, drives the level mismatch.
