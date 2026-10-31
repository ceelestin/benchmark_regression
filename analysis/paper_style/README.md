# Paper-style gain figures and numbers (Overleaf commit 092cfa6, 2026-10-31)

Run on Jean-Zay with `module load pytorch-gpu/py3/2.8.0` and `PYTHONPATH=analysis/paper_style/code`.
`code/` is CE's `figure_style/code` with the changes in `figure_style_code.diff` (display: thick line = estimate
on all seeds via `compute.sample_gain_bootstrap(..., plugin_estimate=True)`, thin lines = bootstrap draws, y-axis
cut 8% above the highest thick line; single-figure-sized facet panels; `--plugin-estimate`, `--save-gains`,
`--random-state`, `--hairlines` in `pcam.py` / `nlp.py`).

| Script | Produces |
|---|---|
| `paper_style_figs.py` | simulated gain figures 4, 5 (left), I.1, J.1, M.1 and their tables (`--plugin-groups` adds the estimates to stored draws) |
| `regen_real_data_gain_figures.py` | real-data figures 5 (middle, right), F.1, K.1, L.1 and their tables (data: CE's `figure_style/real_data/data`) |
| `ci_appendix.py` | the LaTeX tables of Appendix O (estimate and 95% percentile interval per figure) |
| `ci_coverage.py` | coverage check of the intervals quoted in Appendix O (groups of 50 and 200 seeds) |
| `sd_ratio_mse.py` | Appendix J standard-deviation ratios (frozen / reshuffled, MSE) |
| `redundancy_numbers.py` | Appendix I and M redundancy values (t * squared-error OOF correlation) |
| `text_numbers_v5.py` | Appendix I, L and M values quoted in the text |
| `compare_v5.py` | before / after renders (thick line = mean of draws vs estimate on all seeds) |
| `pdfcheck.py` | page sizes and fonts of the figure PDFs |
