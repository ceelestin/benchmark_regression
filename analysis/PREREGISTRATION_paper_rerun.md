# Analysis plan for the fresh-seed rerun of the paper's prospective redundancy holdout (job 533660)

Written 2026-10-03 after the job finished and its integrity check passed (30/30 chunks), BEFORE any gain
or redundancy is computed on it. Purpose: confirm, on fresh seeds (8000-8049) never used before, the
single-run and configuration-level results that the paper reports for the redundancy
omega_k = t * rho_e,k on the pre-registered holdout of job 101294 (Appendix D.3 of the paper).

Design (configs/paper_rerun_may_design_*.yml): 3 families x noise {0.1, 0.7, 1.5} x n_tr {50, 300, 2000}
x 8 solvers, 20 features, K = 200, test fraction 0.2, 50 seeds: 216 configurations, 10,800 single runs.

Analysis, with the scripts used for the paper and nothing tuned on this run:
1. Gains: analysis/derivation_levers_analysis.py --loss neg_mse (order-averaged G_paper at K = 20, 200).
2. Single-run table (analysis/redundancy_triage.py): redundancy on the squared error after k = 3 and 5
   splits; correlation with log G_20 / log G_200, rank correlation, median gain and P(G_20 >= 10),
   P(G_200 >= 20) by tertile.
3. Configuration level (analysis/omega_gain_relation.py): rank correlation of the mean redundancy with
   G_20, median log(G_20 / (K / (1 + (K-1) omega))).

What would count as confirming the paper's claims (stated before looking):
- C1 the redundancy is informative: correlation with log G_200 after 3 splits <= -0.4;
- C2 high redundancy rules out large gains: P(G_20 >= 10 | highest tertile, k = 3) <= 0.02;
- C3 the reference curve holds in level: |median log(G_20 / curve)| <= 0.15 at configuration level.
All numbers will be reported whatever they are.

---

## OUTCOME (added 2026-10-03 after the analysis; text above unchanged)

216 configurations, 10,800 single runs (10,786 with a defined redundancy after 3 splits). Outputs:
analysis/out_paper_rerun_mse/, analysis/out_omega/omega_paper_rerun_neg_mse.csv, analysis/out_redundancy_triage_rerun/.

| check | result | verdict |
|---|---|---|
| C1 correlation of the redundancy (3 splits) with log G_200 | -0.62 (rank -0.42); -0.73 after 5 splits | PASS |
| C2 P(G_20 >= 10 \| highest tertile, 3 splits) | 0.001 (low / mid / high tertiles: 0.163 / 0.003 / 0.001) | PASS |
| C3 \|median log(G_20 / reference curve)\| at configuration level | 0.203 (3 splits), 0.181 (5 splits) | FAIL |

Also reported: P(G_200 >= 20) by tertile 0.083 / 0.000 / 0.000; median G_20 by tertile 5.6 / 3.8 / 3.8;
configuration-level rank correlation -0.42. The former three-factor score is uninformative on the same runs
(correlation with log G_200 +0.05 after 3 splits).

Reading. The redundancy confirms on fresh seeds as a ranking and early-stopping signal (C1, C2). The
reference curve K/(1+(K-1) omega) does not hold in level on this design: gains sit below it, by a median
log-ratio of -0.27 / -0.22 / -0.12 at n_tr = 50 / 300 / 2000, and least for Ridge (-0.05). The pre-registered
holdout of job 101294 shows the same size pattern (-0.23 / -0.17 / +0.05 at 75 / 750 / 5000). This is the
train/test coupling term kappa of the paper's derivation, which lowers the gain most at small training sizes.
The paper must not claim that configurations follow the curve in level without this qualification.
