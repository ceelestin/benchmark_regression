"""Table E.2 rows of the Yelp re-run: recorded training and evaluation runtimes (hours, as the older Yelp rows) per
model, both sizes together, and the scheduler's GPU-hours of the two arrays (sacct, one GPU per task)."""
import glob, os, subprocess
import pandas as pd
R = os.path.expandvars("$WORK/benchmark_nlp/results_aligned")
d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{R}/yelp_results_CV_*.parquet"))], ignore_index=True)
d["eval"] = d[["val_runtime", "test_runtime", "benchmark_runtime"]].sum(axis=1) + d["hidden_runtime"].fillna(0)
t = d.groupby("model").agg(train=("train_runtime", "sum"), eval=("eval", "sum"), seeds=("seed", "nunique"),
                            runs=("seed", lambda x: len(x) // 20))
t[["train", "eval"]] /= 3600
t["total"] = t.train + t["eval"]
print(t.round(2).to_string())
print(d.groupby(["model", "study_size"]).agg(train_h=("train_runtime", lambda x: x.sum() / 3600),
      eval_h=("eval", lambda x: x.sum() / 3600)).round(2).to_string())
for job, name in (("1904013", "roberta"), ("1904208", "bert")):
    out = subprocess.run(["sacct", "-j", job, "-X", "-n", "-P", "-o", "JobID,State,ElapsedRaw"], capture_output=True, text=True).stdout
    rows = [l.split("|") for l in out.split("\n") if l]
    done = [r for r in rows if r[1] == "COMPLETED"]
    print(f"{name} ({job}): {len(done)}/{len(rows)} completed, {sum(int(r[2]) for r in done) / 3600:.1f} GPU-h "
          f"(scheduler), {sum(int(r[2]) for r in rows) / 3600:.1f} GPU-h all states")
