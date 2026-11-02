# Coherence checks behind the paper's text (2026-11-02)

Scripts run on Jean-Zay (`module load pytorch-gpu/py3/2.8.0`) to verify or compute statements of the coherence
pass on the paper (items 6 to 9). Paths use `$WORK` and `$SCRATCH`.

| Script | Supports |
|---|---|
| `strat_effect.py`, `strat_table.py` | Table K.1 and Appendix K, Section 5.1: unstratified/stratified variance ratios on PCam (paired 100 seeds, n_tr = 400) |
| `strat_mech.py`, `vit_march.py` | Appendix K: ViT-B/16 close to a constant predictor (benchmark accuracy 0.58; 0.51-0.58 in the runs of Figure 5) |
| `sim_params.py`, `sim_params_all.py`, `sim_params_pilot.py` | Appendix E.1: d = 5 and sigma = 0 for the main simulated runs (and the 200-seed run of Appendix L) |
| `yelp_seeds.py`, `yelp_bench.py`, `yelp_times.py` | Appendix E.3, Tables E.2 and O.2: Yelp seeds (100 trained for BERT and XLM-RoBERTa, of which seeds 0-19 are scored on the benchmarking set in the K = 20 runs; 50 for Qwen3-0.6B) and recorded runtimes |
| `pcam24_redundancy.py`, `pcam24_gains.py` | Appendix M, On real data: PCam at n_tr = 24 (gains from `code/pcam.py --data-dir <links to the 199 August files> --n-bootstraps 200 --random-state 0 --plugin-estimate --save-gains <tables>`) |
| `vit_all.py` | Section 4: ViT-B/16 benchmark accuracy 0.51-0.59 in every run set, CNNs 0.60-0.85 |
| `fold_geometry.py` | Appendix M: share of runs in which the redundancy is defined after 3 and 5 splits |
