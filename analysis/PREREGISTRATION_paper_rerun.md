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
