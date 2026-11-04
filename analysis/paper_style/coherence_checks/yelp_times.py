"""Item 11: Yelp K = 20 runs: seeds, duplicates and recorded runtimes (hours) vs Table E.2."""
import glob, os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))  # code/datapaths.py
import datapaths
import pandas as pd
D = datapaths.data_path("nlp")
rows = []
for f in glob.glob(f"{D}/**/*.parquet", recursive=True):
    d = pd.read_parquet(f, columns=["model_arg_choice", "study_set_size", "n_splits_cv", "seed", "fold_index", "train_runtime",
                                    "validation_eval_runtime", "study_test_eval_runtime", "benchmark_eval_runtime"])
    d["file"] = os.path.basename(f); d["dir"] = os.path.relpath(f, D).split(os.sep)[0]; rows.append(d)
d = pd.concat(rows, ignore_index=True)
d["study"] = d.train_runtime + d.validation_eval_runtime + d.study_test_eval_runtime
for k, lab in ((20, "K = 20"), (1, "single split")):
    g = d[d.n_splits_cv == k]
    for m, gm in g.groupby("model_arg_choice"):
        runs = gm.groupby(["study_set_size", "seed"]).file.nunique()
        dedup = gm.sort_values("file").drop_duplicates(["study_set_size", "seed", "fold_index", "dir"], keep="last")
        dd = dedup.drop_duplicates(["study_set_size", "seed", "fold_index"], keep="last")
        first50 = dd[dd.seed < 50]
        print(f"{lab:13s} {m:11s} sizes {sorted(gm.study_set_size.unique())} | seeds {gm.seed.nunique()} | runs {runs.sum()} (dup seeds {int((runs > 1).sum())})"
              f" | all rows: study {gm.study.sum()/3600:.1f} h bench {gm.benchmark_eval_runtime.sum()/3600:.1f} h"
              f" | dedup: study {dd.study.sum()/3600:.1f} bench {dd.benchmark_eval_runtime.sum()/3600:.1f}"
              f" | seeds<50: study {first50.study.sum()/3600:.1f} bench {first50.benchmark_eval_runtime.sum()/3600:.1f}")
