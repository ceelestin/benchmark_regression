# Prompt for the agent editing the paper (copy everything below this line)

---

You are editing the LaTeX source of the paper *Crossing the Validation Crisis* (main file `PAPER.tex`).
The author has changed the definition of the study-only **redundancy** and the metric used for the
simulated regression gains. Your job is to adapt the paper text, equations, figure inclusions,
tables and captions to these decisions. Do not change anything else (the sample-gain definitions,
the PCam and NLP results, the ranking analyses) unless an item below requires it. Line numbers
refer to the snapshot of 2026-09-26 and may have shifted; locate passages by their content.

## 1. Decisions taken by the author

1. **The redundancy is redefined.** From now on the word *redundancy* and the symbol
   $\widehat\omega^{\mathrm{study}}_k$ denote
   $$\widehat\omega^{\mathrm{study}}_k = t\,\widehat\rho^{\mathrm{study}}_{e,k},$$
   where $t = n_{\text{te}}/N$ is the test fraction and $\widehat\rho^{\mathrm{study}}_{e,k}$ is the
   pair-averaged loss correlation already defined in the paper (Appendix D.3,
   `eq:study-loss-correlation`: mean over split pairs of the covariance of the per-sample losses on
   the observations held out by both splits, divided by the mean of their variances), computed on the
   first $k$ splits of a single run. Keep the symbol and the label `eq:study-redundancy-score` for
   the new definition, so existing references keep working.
2. **The former definition is deleted.** The product
   $\widehat C^{\mathrm{study}}_{g,k,s}\,\widehat\rho^{\mathrm{study}}_{e,k,s}\,\bar m^{\mathrm{study}}_{k,s}$
   (prediction covariance times loss correlation times mean intersection size) must disappear from
   the paper, together with the definitions of $\widehat C^{\mathrm{study}}_{g,k,s}$ and
   $\bar m^{\mathrm{study}}_{k,s}$ and every sentence that interprets them (for example "the splits
   produce similar predictions *and* err on the same observations", "the mean size of $I_{ab}$",
   "how much direct overlap supports this comparison"). Do not keep it as an alternative.
3. **Same metric.** The per-sample loss of the redundancy is always the loss of the metric whose
   gain is targeted: squared error for an MSE gain, 0-1 error for an accuracy gain, negative
   log-likelihood for an NLL gain (Brier score for a Brier gain). State this explicitly where the
   redundancy is defined. AUC-ROC remains excluded (no per-sample loss), as the paper already says.
4. **Simulated regression gains switch from $R^2$ to MSE.** $R^2$ divides each fold's error by that
   fold's own target variance and has no per-sample loss, so it cannot be paired with a redundancy
   on the same metric. All simulated regression gains reported in the paper must be on the mean
   squared error (benchmark-adjusted delta scores, as before). See Section 5 below for what this
   implies.
5. **Readings after 3 and 5 splits.** Replace "two or three splits often suffice" by reporting the
   redundancy after $k = 3$ and $k = 5$ splits: configuration-level relations are the same, while the
   spread of a single run's reading roughly halves from 3 to 5 splits (cost: two more fits).

## 2. New mathematical content to integrate

A standalone derivation is in
`/data/parietal/store3/work/ceve/benchmark_regression/analysis/notes/gain_vs_loss_correlation.tex`
(compiled: `gain_vs_loss_correlation.pdf` next to it). Integrate its Sections 2-6 into Appendix D
(the natural place is a new subsection of "Estimating and anticipating sample gains", right after
"Benchmark-aware estimation of sample gain", or a rewrite of D.3), adapting notation where needed.
Its key results:

- Fold-error correlation: $\rho_\delta = \tau^{\mathrm{te}}/\sigma^{2,\mathrm{te}} = t\,\rho_e + \kappa$,
  where $t\,\rho_e$ comes from the observations that two splits both test (on average
  $t\,n_{\text{te}}$ of them) and $\kappa$ from the train/test coupling across splits (intrinsic to
  cross-validation, either sign).
- Hence $G^{\text{test}}_K \approx K/\bigl(1+(K-1)(\omega+\kappa)\bigr)$ with $\omega = t\,\rho_e$ the
  (population) redundancy; with $\kappa = 0$ this is the reference curve $K/(1+(K-1)\omega)$.
  Limits: $\omega \to 0$ gives $G \to K$; $\rho_e \to 1$ gives a ceiling $1/t = N/n_{\text{te}}$.
- The redundancy is dimensionless and lies in $[-t, t]$. The former score is approximately
  $\widehat C^{\mathrm{study}}_{g,k}\, n_{\text{te}}\,\widehat\omega^{\mathrm{study}}_k$: it multiplies the
  redundancy by the prediction covariance (squared units of the target, signal strength) and by
  the test-set size (grows with the study), which is why no absolute threshold can be placed on it.
  You may say this in one or two sentences when explaining the change of definition; do not
  re-introduce the former score as a definition.
- Assumptions to state: negligible benchmark noise and $1/m$ scaling (already in the paper); the
  study estimate has a bias of order $1/|I_{ab}|$; the overlap factor $t$ is specific to MCCV (in
  repeated $K$-fold, splits of the same repeat have disjoint test sets); the reference curve omits
  $\kappa$.

## 3. Edits by location

**Main text, Section 5.1, paragraph "Measuring how sample gains settle in" (about lines 592-614).**
- Rewrite the definition sentence (line ~596): the redundancy is $t$ times the correlation of the
  per-sample losses of two split predictors on the observations both held out, on the gain's
  metric. Remove the three-factor description and the sentence "A large product means that the
  splits produce similar predictions and err on the same observations...". Replace it by: a high
  redundancy means two splits err alike on shared observations, so additional splits mostly average
  redundant information; the reference curve $K/(1+(K-1)\omega)$ links it to the gain.
- Line ~598 ("low values indicate weaker coupling between splits"): the word *coupling* now names
  the train/test coupling $\kappa$; rephrase as "low values indicate that splits err more
  independently".
- Paragraph "Early stopping cross-validation" (line ~610): "two or three often suffice" becomes the
  3- and 5-split readings (decision 5).
- Line ~612: replace "already at $k=2$, the score correlates with $G^{\text{test}}_{200}$ at $-0.69$"
  by the new numbers (Section 4 below), and replace "216 simulated configurations" only if you also
  update the count (it stays 216 configurations, 10,800 single runs).
- Lines ~612-613 on the label budget ("reliable per-study signal above roughly 20 labels"): this was
  established with the former score. Keep the sentence but mark it with a `% TODO(author): re-verify
  with the new redundancy` comment; do not invent a new number.

**Figure `fig:study_redundancy_high_gain_probability` (lines ~601-608).** Replace the file
`Images/study_redundancy_single_seed_early_stop_K20.pdf` by the new figure (Section 4). Keep the
label. New caption: "Study-only redundancy as a triage signal: probability that the gain at 20
splits reaches 7, 8 or 10, by redundancy tertile after 3 splits (prospective simulated holdout,
MSE gains, single runs)." If space allows, show the 5-split version as a second panel or in the
appendix.

**Appendix D introduction (line ~1395).** "a study-only redundancy statistic" stays; no change
needed beyond consistency.

**Appendix D.3 "Study-only redundancy for early-stopping cross-validation" (lines ~1702-1830).**
- Paragraph "Single-run redundancy score": keep the definitions of $I_{ab}$, $e_{a,s,i}$,
  $\widehat C^{\mathrm{study}}_{e,k,s}$, $\widehat V^{\mathrm{study}}_{e,k,s}$ and
  $\widehat\rho^{\mathrm{study}}_{e,k,s}$ (`eq:study-loss-correlation`). Delete the paragraph
  defining $\widehat C^{\mathrm{study}}_{g,k,s}$ and $\bar m^{\mathrm{study}}_{k,s}$. The boxed
  equation `eq:study-redundancy-score` becomes
  $\widehat\omega^{\mathrm{study}}_{k,s}(F_\lambda,n_{\text{tr}}) = t\,\widehat\rho^{\mathrm{study}}_{e,k,s}$.
  Add the same-metric sentence (decision 3). Replace "which is the squared error in the synthetic
  regression experiments" consistently with MSE gains (it stays the squared error).
- Paragraph "Mathematical intuition" (lines ~1782-1793): rewrite from the derivation
  ($\rho_\delta = t\rho_e + \kappa$ and the reference curve); delete all text about
  $\widehat C_g$ and $\bar m$. Keep the high/low implications.
- Paragraph "Prospective single-run validation" (lines ~1795-1798): keep the design description
  (3 families, 3 noise levels, 3 training sizes, 8 solvers, 50 seeds, 216 configurations, 10,800
  runs, continued to $K=200$). Replace the numbers by those of Section 4 (3 and 5 splits).
- Table `tab:study-redundancy-early-stop` (lines ~1800-1823): replace the tabular body by the
  generated table (Section 4); column "Log corr." becomes "Corr. with log gain" (the redundancy can
  be zero or negative, so the correlation is between the redundancy and the log gain). Caption: say
  tertiles of $\widehat\omega^{\mathrm{study}}_{k,s}$ after $k$ splits of one seed, MSE gains.
- Paragraph after the table (line ~1825, "read asymmetrically"): keep the asymmetric reading, and
  add that the separation is between the low tertile and the other two: middle and high tertiles
  are about equally unlikely to reach large gains.
- Line ~1827: "Compute eq after two or three splits" becomes "after three or five splits".
- Paragraph "Relation with configuration-level calibration" (line ~1830): keep; optionally cite
  the configuration-level figure `omega_vs_gain_K20_k5.png` (Section 4) as supporting evidence.

**Appendix E (experimental details, lines ~1844-1858).** Mentions of "the redundancy analysis" stay;
if a sentence describes how the former score was computed, align it with the new definition.

**Appendix on repeated K-fold (line ~2096).** "dynamically estimate the study-only redundancy score":
keep, and add that the overlap factor $t$ in the redundancy is specific to MCCV.

**Appendix on hyper-parameter optimisation (lines ~2328-2332, "Interpretation through the
redundancy mechanism").** Keep the mechanism (grid search selects nearly the same configuration, so
split predictors err alike), but make sure it is phrased in terms of the loss correlation, not of
the prediction covariance. Its figure `..._r2_K20_noGB_noRidge.pdf` is an $R^2$ figure (Section 5).

**Appendix on small label budgets (lines ~2426-2428).** "The redundancy score is estimated from the
observations held out by two different splits" stays true. The reliability threshold (about 20
labels) was established with the former score: add `% TODO(author): re-verify with the new redundancy`.

**Rebuttal / response text at the end of the file (lines ~2544, 2681, 2702-2710), if it is kept in
the compiled document.** Replace "redundancy score" by "redundancy" and remove any description of
the former three-factor product.

## 4. New figures, table and numbers (simulated data, ready to use)

Files (copy into `Images/` or include from their path):
- `/data/parietal/store3/work/ceve/benchmark_regression/analysis/out_redundancy_triage_216/paper/study_redundancy_single_seed_early_stop_K20_k3.pdf`
  (main-text figure, 3 splits) and `..._k5.pdf` (5 splits).
- `/data/parietal/store3/work/ceve/benchmark_regression/analysis/out_redundancy_triage_216/paper/table_study_redundancy_early_stop.tex`
  (tabular body for `tab:study-redundancy-early-stop`; needs `booktabs` and `makecell`, as now).
- Optional supporting figures: `.../analysis/out_omega/omega_vs_gain_K20_k5.png` (configuration-level
  gain vs redundancy, regression MSE and classification accuracy, with the reference curve) and
  `.../analysis/out_omega_sweep/sweep_pcam_vs_gain_K20_k5.png` (classification sweep and PCam).

Numbers on the prospective holdout (216 configurations x 50 seeds = 10,800 single runs; MSE gains;
redundancy on the squared error; tertiles over all runs):
The gains use the paper's definition (variance-equivalent test size), with the single-split variance curve
averaged over the order of the outer chunks: same estimand as before, less estimation noise. If the paper
describes how $V^\delta_1(m)$ is estimated, add one sentence saying so.

| splits | target | corr. with log gain | rank corr. | median gain low/mid/high | large-gain probability low/mid/high |
|---|---|---|---|---|---|
| 3 | $G_{20}$ | -0.60 | -0.39 | 5.8 / 4.1 / 4.2 | $P(G_{20}\ge 10)$ = 15.2 / 0.1 / 0.0 % |
| 3 | $G_{200}$ | -0.66 | -0.41 | 6.9 / 4.8 / 4.8 | $P(G_{200}\ge 20)$ = 15.3 / 0.0 / 0.0 % |
| 5 | $G_{20}$ | -0.67 | -0.41 | 5.8 / 4.1 / 4.3 | $P(G_{20}\ge 10)$ = 15.2 / 0.1 / 0.0 % |
| 5 | $G_{200}$ | -0.74 | -0.43 | 6.9 / 4.8 / 4.8 | $P(G_{200}\ge 20)$ = 15.3 / 0.0 / 0.0 % |

Main-text figure (3 splits): $P(G_{20}\ge 7)$ = 37.9 / 1.8 / 1.9 %, $P(G_{20}\ge 8)$ = 25.9 / 0.5 / 0.0 %,
$P(G_{20}\ge 10)$ = 15.2 / 0.1 / 0.0 % (low / middle / high tertile).

Further facts you may cite (simulated data, configuration level, gain at 20 splits): across all
simulated designs the redundancy ranks configurations by gain with Spearman -0.81 (regression, MSE)
and -0.77 (classification, accuracy) after 5 splits; varying the test fraction (0.1, 0.2, 0.3, 0.5)
the fitted ratio $\rho_\delta/\rho_e$ is 0.10/0.21/0.32/0.48 (classification) and 0.14/0.20/0.36/0.46
(regression), i.e. the factor follows $t$. On PCam (20 configurations, a consistency check, not an
independent validation) the redundancy ranks the gains with Spearman -0.80 (accuracy, 5 splits)
and the PCam networks lie on the reference curve.

## 5. Consequences of the switch from R^2 to MSE (decision 4)

The following simulated figures are currently computed on $R^2$ and must be regenerated on MSE by
the author's plotting pipeline (they come from the 1000-seed simulated runs, not from the files
above). Keep their inclusions and labels; add `% TODO(author): regenerate on MSE` next to each
`\includegraphics` until the new files exist:
- `Images/aggregated_test_gain_k_1000_sim_linear_fixedFalse_r2_full.pdf` (Fig. `fig:avg_test_sample_gain_simulated`, line ~543)
- `Images/aggregated_test_gain_k_1000_sim_linear_fixedFalse_r2_K20.pdf` (Fig. `fig:all_data_test_sample_gains`, left panel, line ~558)
- `Images/aggregated_test_gain_k_1000_sim_linear_fixedFalse_r2_K20_noGB_noRidge.pdf` (HPO appendix, line ~2309)
- `Images/aggregated_test_gain_k_100_sim_linear_r2_fixedsplit_overlay_K20.pdf` (fixed vs reshuffled appendix, line ~2354)

Numbers in the text that come from these $R^2$ figures must be marked, not changed:
`% TODO(author): MSE value` next to "$G^{\text{test}}_{200}(\texttt{Ridge}) \approx 14$" and
"$G^{\text{test}}_{200} = 15$" for the MLP (line ~549-550), "$G^{\text{test}}_{20} = 4$ for ExtraTrees and
5 for GradientBoosting" (line ~582), and any other simulated gain value you find that was read off
these figures. The appendix comparing metrics (line ~2390, "$R^2$, mean squared error and ...")
reports several metrics on purpose: keep $R^2$ there, but state that the main simulated results
use MSE. Where the text says or implies that simulated gains are on $R^2$, change it to MSE.

## 6. What to return

The edited LaTeX, compiling without new errors, and a short list of every `TODO(author)` you left,
with its location. Do not invent numbers: every number you introduce must come from Section 4.
