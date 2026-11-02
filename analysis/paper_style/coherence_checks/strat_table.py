"""Item 6 (CE, 2026-11-02): what stratified splitting changes on PCam (stratandnewmetrics, study set 500, n_tr 400,
100 seeds, K = 20, paired by seed: same study set, only the fold partition differs).
Per network x metric, on the benchmark-adjusted fold score delta_k = test_k - bench_k:
  V1  = Var_seeds(delta_1)            single split (first fold)
  V20 = Var_seeds(mean_k delta_k)     20-split CV
for unstratified (u) and stratified (s) splitting; ratios with 95% seed-bootstrap intervals (1,000 draws)."""
import glob, os
import numpy as np, pandas as pd
D = os.path.expandvars("$WORK/paper_style_figs/real_data/data/pcam/stratandnewmetrics")
cols = ["model", "study_size", "seed", "fold", "stratified", "benchmark_accuracy"] + [f"{a}_{m}" for a in ("test", "benchmark") for m in ("accuracy", "nll", "brier") if f"{a}_{m}" != "benchmark_accuracy"]
d = pd.concat([pd.read_parquet(f, columns=cols) for f in glob.glob(f"{D}/*.parquet")], ignore_index=True)
d = d[d.study_size == 500]
d = d.sort_values(["model", "stratified", "seed", "fold"]).drop_duplicates(["model", "stratified", "seed", "fold"], keep="last")
rng = np.random.default_rng(0)
rows = []
RAW = {}
for m in ("accuracy", "nll", "brier"):
    d["delta"] = d[f"test_{m}"] - d[f"benchmark_{m}"]
    for net, g in d.groupby("model"):
        piv = {}
        for s, gs in g.groupby("stratified"):
            w = gs.pivot(index="seed", columns="fold", values="delta")
            w = w[w.notna().sum(axis=1) == 20]
            piv[bool(s)] = w
        seeds = sorted(set(piv[False].index) & set(piv[True].index))
        u1 = piv[False].loc[seeds, 1].values; s1 = piv[True].loc[seeds, 1].values
        u20 = piv[False].loc[seeds].mean(axis=1).values; s20 = piv[True].loc[seeds].mean(axis=1).values
        def stats(idx):
            v = lambda x: np.var(x[idx], ddof=1)
            return np.array([v(u1) / v(s1), v(u20) / v(s20), v(u1) / v(u20), v(s1) / v(s20), v(u20) / v(s1)])
        est = stats(np.arange(len(seeds)))
        boots = np.array([stats(rng.integers(0, len(seeds), len(seeds))) for _ in range(1000)])
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        RAW[(m, net)] = (est, lo, hi)
        f = lambda j: f"{est[j]:.2f} [{lo[j]:.2f}, {hi[j]:.2f}]"
        rows.append({"metric": m, "network": net, "seeds": len(seeds),
                     "V1 u/s (single split)": f(0), "V20 u/s (20-split CV)": f(1),
                     "V1/V20 unstrat": f(2), "V1/V20 strat": f(3), "V20 unstrat / V1 strat": f(4),
                     "mean class-1 share of fold-1 test": None})
print(pd.DataFrame(rows).drop(columns=["mean class-1 share of fold-1 test"]).to_string(index=False))

# LaTeX rows for Table K.1 (rows: metric x score; columns: networks), 3 significant digits
def fmt(x):
    return f"{x:.2f}" if x < 10 else (f"{x:.1f}" if x < 100 else f"{x:.0f}")
order = ["densenet121", "mobilenet_v2", "wide_resnet101_2", "vit_b_16"]
names = {"accuracy": "Accuracy", "nll": "Negative log-likelihood", "brier": "Brier score"}
lines = []
for m in ("accuracy", "nll", "brier"):
    for j, lab in ((0, "single split"), (1, "$20$-split average")):
        cells = []
        for net in order:
            e, l, h = RAW[(m, net)][0][j], RAW[(m, net)][1][j], RAW[(m, net)][2][j]
            cells.append(f"${fmt(e)}$ {{\\scriptsize $[{fmt(l)},\\,{fmt(h)}]$}}")
        lines.append((names[m] if j == 0 else "") + f" & {lab} & " + " & ".join(cells) + r" \\\\")
    if m != "brier":
        lines.append(r"\\midrule")
open("strat_table_rows.tex", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
for m in ("nll", "brier"):
    e, l, h = RAW[(m, "vit_b_16")][0][4], RAW[(m, "vit_b_16")][1][4], RAW[(m, "vit_b_16")][2][4]
    print(f"ViT {m}: V20 unstrat / V1 strat = {e:.2f} [{l:.2f}, {h:.2f}]")

# summary numbers quoted in the text (pooled over both protocols)
b = d.groupby("model").benchmark_accuracy
print("bench acc mean:", b.mean().round(3).to_dict())
print("ViT share of fits with bench acc < 0.6:", round((d[d.model == "vit_b_16"].benchmark_accuracy < 0.6).mean(), 3))
print("ViT median bench NLL:", round(d[d.model == "vit_b_16"].benchmark_nll.median(), 4))
