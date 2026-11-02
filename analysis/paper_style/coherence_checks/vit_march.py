import glob, os
import pandas as pd
D = os.path.expandvars("$WORK/paper_style_figs/real_data/data/pcam/results_output_pcam")
fs = [f for f in glob.glob(f"{D}/*.parquet") if "shuffle_split" in f]
d = pd.concat([pd.read_parquet(f, columns=["model", "study_size", "seed", "fold", "benchmark_accuracy", "benchmark_loss"]) for f in fs], ignore_index=True)
g = d.groupby(["model", "study_size"]).benchmark_accuracy
print(pd.DataFrame({"mean": g.mean(), "q10": g.quantile(0.1), "share < 0.6": g.apply(lambda x: (x < 0.6).mean())}).round(3).unstack(0).to_string())
