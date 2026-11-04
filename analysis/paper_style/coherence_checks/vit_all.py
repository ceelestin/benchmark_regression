"""Item 10: ViT-B/16 vs the CNNs, benchmark accuracy per run set and size (mean, share of fits below 0.55 / 0.6)."""
import glob, os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))  # code/datapaths.py
import datapaths
import pandas as pd
sets = {
    "Mar 2026 (Fig 5, F.1, H.1)": glob.glob(datapaths.data_path("pcam", "results_output_pcam", "*shuffle_split*.parquet")),
    "Jul 2026 (K.1, L.1)": glob.glob(datapaths.data_path("pcam", "stratandnewmetrics", "*.parquet")),
    "Aug 2026 (App M)": glob.glob(datapaths.pcam_repo_path("results_output_pcam", "*_sizes_30_202608*.parquet")),
    "Sep 2026 round 3 (D.2)": glob.glob(datapaths.pcam_repo_path("results_output_pcam", "pcam_results_CV_shuffle_split_unstrat_*_seeds_1[0-4][0-9]-*_sizes_*.parquet")),
}
rows = []
for name, fs in sets.items():
    d = pd.concat([pd.read_parquet(f, columns=["model", "study_size", "seed", "fold", "benchmark_accuracy"]) for f in fs], ignore_index=True)
    for (m, s), g in d.groupby(["model", "study_size"]):
        b = g.benchmark_accuracy.astype(float)
        rows.append({"run set": name, "label": int(s), "model": m, "fits": len(b), "bench acc": round(b.mean(), 3),
                     "q90": round(b.quantile(0.9), 3), "< 0.55": round((b < 0.55).mean(), 2), "< 0.6": round((b < 0.6).mean(), 2)})
r = pd.DataFrame(rows)
print(r[r.model == "vit_b_16"].to_string(index=False))
print(r[r.model != "vit_b_16"].groupby(["run set", "label"])["bench acc"].agg(["min", "max"]).to_string())
