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
observed paper gain and the study-only prediction splits EXACTLY (sum checked to 1e-15) into the
terms below. The paper gain here is the order-averaged one (see finding 4); its difference from the
former fixed-order value is reported beside the sum as `fixed_order_noise`.

| term | what it isolates |
|---|---|
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
   against -0.35 / -0.47 with study-trained folds. C carries 67-94% of the spread of the gap across
   configurations when statistic and gain use the same loss (44-64% before the chunk-order noise
   was removed from the gain).
3. **Coupling depends on train size, learner and task, not on the data family per se.** It usually
   adds between-fold covariance (gain below the closed form): -0.4 to -0.6 at train sizes 50-300 in
   classification and for single trees, fading towards 0 by 1000-3000 in regression. It can reverse:
   in sim_nonlinear at 10000 the oracle gain exceeds the closed form for every learner, Ridge
   included (bootstrap 90% CI of C excludes 0 for 8 of 12 configurations). Not explained yet.
   Noise-injection levers leave C near 0 in regression; in classification they weaken it (median -0.08
   to -0.38 by lever, against -0.41 without a lever).
4. **The paper gain as first implemented carried measurement noise, now removed.** It pooled the
   outer chunks in one fixed order, which made V1(alpha) swing between 0.5 and 1.2 times its
   expectation at 50 seeds; the chunks are the same samples for every learner of a given size, so
   the error was shared across configurations and looked like a size effect (log G_20 sd 0.19-0.20
   per configuration, median -0.19 / -0.09 / 0.00 at sizes 1000 / 3000 / 10000 on holdout 3). Since
   2026-09-26 every analysis averages V1(alpha) exactly over the order of the chunks (same estimand);
   the effect on the pre-registered checks is in analysis/ADDENDUM_order_averaged_gain.md.
5. **The finite benchmark set and the chunk ceiling matter only at large train size** (A up to
   +0.08 and a hard cap at 16 at size 10000, where 37,500 outer samples make 15 chunks).
6. **Reporting accuracy while measuring redundancy on NLL or Brier costs a loss-mismatch term D**
   (29-49% of the spread on holdouts 2-3). With NLL its median is -0.08: 0-1 errors are more
   correlated across folds than NLL, so the NLL-based closed form over-predicts the accuracy gain.
   With Brier the median is +0.01 to +0.02, the bias is gone but the scatter remains. This is the likely
   mechanism of the PCam C2' over-prediction (+0.20 with the order-averaged gain, +0.27 before), not
   testable there yet (see below).

With the order-averaged gain, holdout 3 regression has four unsafe stops: sim_interactions
ExtraTrees at 10000 (gain 10.6, oracle gain 7.9, closed form 7.0: benchmark floor and chunk ceiling)
and three sim_nonlinear ensembles at 10000 where coupling raises the gain above the closed form
(C = +0.49 to +0.90). The two sim_linear stops of the fixed-order analysis were chunk-order noise.

## Open questions / next steps

* Extend the closed form with a coupling term, and find a study-only estimate of it: split the
  pairwise covariance of per-sample losses by membership (sample tested in fold k and trained on in
  fold l vs. tested in both). The objective already
  computes a `rho_resid_train_membership` statistic worth checking against C.
* Theory of C: a stability argument (influence of one training point on another fold's test loss)
  should give its 1/n scaling and learner dependence; the sign reversal needs its own explanation.
* Measurement: done for the chunk order (order-averaged gain everywhere); a larger outer set and
  benchmark set would remove the floor and ceiling at large train size.
* PCam: record the per-sample benchmark losses across folds (population rho_b) to decompose the real
  data the same way; the benchmark-floor term is computable from the existing parquets.
