"""App L: Ridge (alpha = 1613) gain on R^2 and on the MSE from the same runs (July 07-24_10h46m, 1,000 seeds, K = 20),
paper estimator on all seeds, with the 95% percentile interval of 1,000 draws (CE, 2026-11-01)."""
import glob, os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style_figs as F
f = glob.glob(os.path.expandvars("$SCRATCH/paper_runs/benchopt_run_2026-07-24_10h46m*.parquet"))[0]
names = pq.read_schema(f).names
for lab, cols in (("R2", ("objective_score_test", "objective_score_bench", "objective_outer_scores")),
                  ("MSE", ("objective_neg_mse_test", "objective_neg_mse_bench", "objective_outer_neg_mse"))):
    F.T.TEST, F.T.BENCH, F.T.OUTER = cols
    keep = [c for c in F.T.COLS if c in names and c not in ("objective_neg_mse_test", "objective_neg_mse_bench", "objective_outer_neg_mse")] + list(cols)
    d = pd.read_parquet(f, columns=keep)
    d = d[d.solver_name.str.startswith("Ridge") & (d.p_obj_train_size == 1000) & ~d.p_obj_fixed_split.astype(str).str.lower().eq("true")]
    r = F.gains(F.prepare([d]), 1000, 0.20, plugin=True); r = r[r.K == 20]
    e, b = r[r.bootstrap_iter < 0].sample_gain.iloc[0], r[r.bootstrap_iter >= 0].sample_gain
    print(f"{lab}: seeds {d.p_dataset_seed.nunique()}, G20 = {e:.2f} [{b.quantile(.025):.2f}, {b.quantile(.975):.2f}], mean of draws {b.mean():.2f}", flush=True)
