import glob, os
import pandas as pd
D = os.path.expandvars("$WORK/paper_style_figs/real_data/data/nlp")
d = pd.concat([pd.read_parquet(f, columns=["model_arg_choice", "study_set_size", "n_splits_cv", "seed", "fold_index", "skipped_benchmark", "benchmark_eval_accuracy", "hidden_test_accuracies"]).assign(dir=os.path.relpath(f, D).split(os.sep)[0]) for f in glob.glob(f"{D}/**/*.parquet", recursive=True)], ignore_index=True)
for (k, m, s), g in d.groupby(["n_splits_cv", "model_arg_choice", "study_set_size"]):
    hb = g[g.benchmark_eval_accuracy.notna()]
    print(f"K={k:2d} {m:11s} size {s:5d}: seeds {g.seed.nunique():3d}, with benchmark {hb.seed.nunique():3d} (range {hb.seed.min() if len(hb) else '-'}-{hb.seed.max() if len(hb) else '-'}), skipped_benchmark values {sorted(g.skipped_benchmark.dropna().unique().tolist())}")
