# Decomposing the redundancy / sample-gain relation (2026-09-26)

Exploratory, not pre-registered. Goal set by CE: explain the relation between the study-only
redundancy score and the variance-equivalent test sample gain, and the factors acting on it,
before any absolute threshold. Scripts: `analysis/gain_decomposition.py` (one data set) and
`analysis/gain_decomposition_summary.py` (pooled, by factor). Outputs: `analysis/out_decomp_*/`.
Data: every ShuffleSplit set run so far (derivation grids, holdouts 1-3, coupling test; regression
and sim_classification), 50 seeds (30 for derivation and coupling), K = 20 unless stated.

## The identity

With t the test fraction and rho a between-fold correlation of per-sample losses, the closed form is
G_cf(rho) = 1 / (1/K + (1 - 1/K) t rho)  [= 1/((1-f)(t+(1-t)/K)+f/K), f = 1 - rho].
Using the oracle quantities the objective already records (every fold model is also scored on the
100k benchmark set, and the first-split model on up to 200 fresh outer chunks), the gap between the
observed paper gain and the study-only prediction splits EXACTLY (sum checked to 1e-15) into:

| term | what it isolates |
|---|---|
| A0 chunk order | the paper gain pools the outer chunks in one fixed order; vs. V1(alpha) averaged over random orders |
| A oracle floor | the benchmark score's own sampling error, s2 / 100,000 |
| B ceiling | chunk ceiling (16 chunks at train size 10000), alpha interpolation |
| C model | oracle gain vs. closed form with the POPULATION correlation rho_b (benchmark set) |
| D loss mismatch | population rho of the gain's loss vs. of the statistic's loss (0 when matched) |
| E study vs population | study OOF-intersection rho at K vs. rho_b |
| F early reading | study rho at K vs. at k = 3 |
| H seed aggregation | variance-weighted vs. plain median over seeds (what the stop rules used) |

Two implementation details matter. The objective stores the per-sample squared error divided by
Var(y), so its variance is rescaled by Var(y)^2; and a variance over seeds averages per-seed
variances, so the loss variance is a mean over seeds and every rho is weighted by the per-seed loss
variance. After both, the single-split variance matches s2 / n_test (median ratio 1.01).

## Findings

1. **The study-only statistic is not the bottleneck.** E, F and H are within +-0.02 on every set:
   the OOF-intersection correlation read at k = 3 recovers the population between-fold correlation.
2. **The closed form's one structural gap is train/test coupling across folds (term C).** A sample
   tested in one fold is a training point of the other folds' models. This is intrinsic to
   cross-validation and part of the estimand we study, not a defect: the closed form is what is
   incomplete, since it only credits redundancy through test samples shared between folds. The
   coupling test (a diagnostic, not a target) confirms that C is this effect: with folds trained on fresh outer data the median of C is +0.01 (K = 20) / +0.10 (K = 200),
   against -0.35 / -0.47 with study-trained folds. C carries 44-64% of the spread of the gap across
   configurations when statistic and gain use the same loss.
3. **Coupling depends on train size, learner and task, not on the data family per se.** It usually
   adds between-fold covariance (gain below the closed form): -0.4 to -0.6 at train sizes 50-300 in
   classification and for single trees, fading towards 0 by 1000-3000 in regression. It can reverse:
   in sim_nonlinear at 10000 the oracle gain exceeds the closed form for every learner, Ridge
   included (bootstrap 90% CI of C excludes 0 for 8 of 12 configurations). Not explained yet.
   Noise-injection levers leave C near 0 in regression; in classification they weaken it (median -0.08
   to -0.38 by lever, against -0.41 without a lever).
4. **About a third of the spread is measurement noise in the paper gain as implemented (term A0).**
   The fixed chunk order makes V1(alpha) swing between 0.5 and 1.2 times its expectation at 50
   seeds, and the chunks are the same samples for every learner of a given size, so the error is
   shared across configurations and looks like a size effect (A0 median -0.19 at size 1000, -0.10 at
   3000 on holdout 3). Averaging V1(alpha) over chunk orders is the same estimand and removes it.
5. **The finite benchmark set and the chunk ceiling matter only at large train size** (A up to
   +0.08 and a hard cap at 16 at size 10000, where 37,500 outer samples make 15 chunks).
6. **Reporting accuracy while measuring redundancy on NLL or Brier costs a loss-mismatch term D**
   (27-35% of the spread on holdouts 2-3). With NLL its median is -0.08: 0-1 errors are more
   correlated across folds than NLL, so the NLL-based closed form over-predicts the accuracy gain.
   With Brier the median is +0.01 to +0.02, the bias is gone but the scatter remains. This is the likely
   mechanism of the PCam C2' over-prediction (+0.27), not testable there yet (see below).

The five unsafe stops of holdout 3 split in two: sim_linear RandomForest at 10000 and ExtraTrees at
3000 are measurement (oracle gain 7.3 / 6.7 vs. closed form 7.2 / 6.9; A0 + A + B = +0.55 / +0.49);
the three sim_nonlinear ensembles at 10000 are coupling reversal (C = +0.49 to +0.90).

## Open questions / next steps

* Extend the closed form with a coupling term, and find a study-only estimate of it: split the
  pairwise covariance of per-sample losses by membership (sample tested in fold k and trained on in
  fold l vs. tested in both). The objective already
  computes a `rho_resid_train_membership` statistic worth checking against C.
* Theory of C: a stability argument (influence of one training point on another fold's test loss)
  should give its 1/n scaling and learner dependence; the sign reversal needs its own explanation.
* Measurement: use the order-averaged gain (G_paper_avg) and a larger outer set at large train size.
* PCam: record the per-sample benchmark losses across folds (population rho_b) to decompose the real
  data the same way; A0 and A are computable from the existing parquets.
