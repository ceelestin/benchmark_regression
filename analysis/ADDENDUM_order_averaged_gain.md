# Addendum (post hoc, 2026-09-26): pre-registered checks re-evaluated with the order-averaged gain

NOT pre-registered. Decided by CE on 2026-09-26 after the gain decomposition
(analysis/GAIN_DECOMPOSITION.md) showed that the paper gain as implemented pooled the outer
(hidden) chunks in ONE fixed order. At 50 seeds that single-split curve V1(alpha) swings between
0.5 and 1.2 times its expectation, and the chunks are the same samples for every learner of a given
size, so the error is shared across configurations: sd 0.19-0.20 of log G_20 per configuration on
holdout 3, median -0.19 / -0.09 / 0.00 at train sizes 1000 / 3000 / 10000.

Change. G_paper now uses V1(alpha) averaged over the order of the chunks: the exact expectation over
a uniformly random choice of the alpha - 1 chunks, from their covariance matrix across seeds
(`v1_alpha_curve` in derivation_levers_analysis.py and pcam_gain_analysis.py; checked against 20,000
random orders, max relative difference 0.5 %). Same estimand, same data, same crossing rule. The
fixed-order value is kept in every output as G_paper_fixed. Every check below was run twice with the
unchanged check scripts, on the new gains (NEW) and on a copy with G_paper <- G_paper_fixed (OLD);
OLD reproduces every published outcome exactly. The pre-registered texts and outcomes are unchanged.

| pre-registration, set, check | OLD (published) | NEW (order-averaged) |
|---|---|---|
| holdout 1 (S_3 > 0.03), C1 P(G20>=10 \| stop) | 0.055 FAIL | **0.047 PASS** |
| holdout 1, C1' P(G200>=20 \| stop) | 0.051 FAIL | **0.047 PASS** |
| f-rule, PCam 25 seeds (read at k=20), C1 accuracy / NLL / Brier | 0.385 / 0.40 / 0.27 FAIL | 0.308 / 0.100 / 0.273 FAIL |
| f-rule, PCam 100 seeds (k=3), C1 accuracy (primary) / NLL / Brier | 0.167 FAIL / 0.000 / 0.000 | 0.167 FAIL / 0.000 / 0.000 |
| f-rule, PCam 100 seeds, C2 P(pred >= obs) accuracy / NLL / Brier | 0.62 / 0.62 / 0.50 FAIL | 0.38 / **0.88 PASS** / 0.25 |
| holdout 2 regression, C1 / C1' / C2 | 0.056 FAIL / 0.040 / 0.50 FAIL | 0.071 FAIL / 0.016 / 0.52 FAIL |
| holdout 2 regression, base rate (power) | 0.18 under-powered | 0.23 under-powered |
| holdout 2 classification (0-1), C1 / C1' / C2 | 0.200 / 0.200 / 0.46 FAIL | 0.300 / 0.100 / 0.38 FAIL |
| round 3 set A PCam, accuracy/NLL (PRIMARY): C1 / C2' / C3 | 0.167 FAIL / +0.27 FAIL / 0.50 | **0.000 PASS** / +0.20 FAIL / 0.50 |
| round 3 set A, accuracy/Brier: C1 / C2' / C3 | 0.125 FAIL / +0.21 FAIL / 0.75 | **0.000 / +0.09 / 0.75, all PASS** |
| round 3 set A, NLL/NLL: C1 / C2' / C3 | 0.000 / +0.12 / 0.50, all PASS | 0.000 / -0.02 / 0.83, all PASS |
| round 3 set A, accuracy/0-1 (reference): C1 / C2' | 0.111 / +0.21 | 0.000 / +0.03 |
| round 3 set B regression: C1 / C1' / C2' / C3 | 0.091 FAIL / 0.000 / +0.07 / 0.47 FAIL | 0.073 FAIL / 0.000 / -0.02 / **0.60 PASS** |
| round 3 set B classification, NLL (PRIMARY): C1 / C1' / C2' / C3 | 0.000 / 0.000 / +0.27 FAIL / 0.78 | 0.000 / 0.000 / +0.35 FAIL / 0.56 |
| round 3 set B classification, Brier: C1 / C1' / C2' / C3 | 0.000 / 0.000 / +0.11 / 0.78, all PASS | 0.000 / 0.000 / **+0.20 FAIL** / 0.56 |

Reading. Removing the chunk-order noise moves several marginal verdicts, in both directions: it was
large enough to decide checks whose margins were a few configurations. Holdout 1 now passes its
safety checks; round 3 set A passes C1 on every loss pairing and all checks with the Brier statistic;
round 3 set B regression still fails C1 (4/55: sim_interactions ExtraTrees at 10000, gain 10.6 with an
oracle gain of 7.9, i.e. benchmark floor and chunk ceiling; and the three sim_nonlinear ensembles at
10000 where coupling raises the gain above the closed form); holdout 2 gets worse. The pattern that
survives is the one in GAIN_DECOMPOSITION.md: the closed form with a probabilistic-loss statistic
over-predicts the accuracy gain of classifiers (C2' +0.20 to +0.35) and the remaining unsafe stops
come from train/test coupling and from the oracle's limits at large train size.

Also fixed in passing: the 100-seed PCam re-analysis must be restricted to seeds 0-99, because its
original file pattern also matches the round-3 parquets written later (seeds 100-149).
