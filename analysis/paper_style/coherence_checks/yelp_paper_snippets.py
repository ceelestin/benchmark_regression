"""LaTeX snippets of the Yelp re-run for the paper (Table G.1 rows, Table O.2 rows, Table E.2 rows) and the numbers
quoted in the text, from the final outputs of regen_v3.py fig5_yelp_aligned. Writes paper_snippets.json."""
import glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))  # code/datapaths.py
import datapaths
import numpy as np, pandas as pd
OUT = sys.argv[1] if len(sys.argv) > 1 else datapaths.figs_path("real")  # --out-dir of regen_real_data_gain_figures.py
D = datapaths.data_path("nlp", "yelp_aligned")
NEW = lambda x: "{\\NEWon " + x + "}"
snip = {}
# Table G.1
g1 = pd.read_csv(f"{OUT}/work/nlp_aligned/results/pairwise_ttest_nlp.csv")
NAMES = {"bert": "BERT", "roberta": "XLM-RoBERTa"}
rows = []
for r in g1.itertuples():
    p = "\\textbf{$<$ 0.0001}" if r.p_value < 1e-4 else (f"\\textbf{{{r.p_value:.4f}}}" if r.p_value < 0.05 else f"{r.p_value:.4f}")
    rows.append(f"{NAMES[r.highest_mean]} & {NAMES[r.lowest_mean]} & {r.n_tr:,} & " + " & ".join(NEW(x) for x in (
        f"{r.mean_diff:.5f}", f"{r.std_diff:.5f}", f"{r.t_stat:.3f}", p, f"{100 * r.single_split_retrieval:.0f}\\%",
        f"{100 * r.K20_cv_retrieval:.0f}\\%")) + " \\\\")
snip["g1_rows"] = "\n\\midrule\n".join(rows)
snip["g1"] = g1.round(5).to_dict("records")
# Table O.2 (cells as ci_appendix.py)
t = pd.read_parquet(f"{OUT}/tables/nlp_delta_aligned_with_qwen.parquet")
t = t[t.N == 1250]
o2 = {}
for m, g in t.groupby("model"):
    cells = []
    for K in (5, 10, 20):
        d = g[g.K == K]
        e, b = d[d.bootstrap_iter < 0].sample_gain, d[d.bootstrap_iter >= 0].sample_gain
        lo, hi = np.quantile(b, [.025, .975])
        flag = ("^{\\dagger}" if (b <= 1.0001).mean() >= .025 else "") + \
               ("^{\\ddagger}" if (b >= b.max() - 1e-9).mean() >= .025 and b.max() > 1.0001 else "")
        cells.append(f"${float(e.iloc[0]):.1f}{flag}$ {{\\scriptsize $[{lo:.1f},\\,{hi:.1f}]$}}")
    o2[m] = cells
snip["o2"] = o2
# Table E.2: recorded runtimes, hours
d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{D}/*.parquet"))], ignore_index=True)
d["eval"] = d[["val_runtime", "test_runtime", "benchmark_runtime"]].sum(axis=1) + d["hidden_runtime"].fillna(0)
e2 = {}
for m, g in d.groupby("model"):
    tr, ev = g.train_runtime.sum() / 3600, g["eval"].sum() / 3600
    e2[m] = {"seeds": int(g.seed.nunique()), "runs": int(len(g) // 20), "train": round(tr, 2), "eval": round(ev, 2),
             "total": round(tr + ev, 2)}
snip["e2"] = e2
# text numbers: single split + extra test data at n_tr = 3,000 (share ranking XLM-R first), max gain at K = 20
f1 = d[d.fold == 1]
share = {}
for n, g in f1.groupby("study_size"):
    for mlt in (1, 2, 5, 10, 20, 61):
        s = g.apply(lambda r: (r.test_accuracy + np.sum(r.hidden_test_accuracies[: mlt - 1])) / mlt, axis=1)
        p = g.assign(s=s).pivot(index="seed", columns="model", values="s").dropna()
        share[f"{n}_{mlt}"] = {"bert_above": round(100 * float((p.bert > p.roberta).mean()), 1),
                               "roberta_above": round(100 * float((p.roberta > p.bert).mean()), 1), "n": len(p)}
snip["single_split_share"] = share
qo = d.groupby(["study_size", "model"]).benchmark_accuracy.mean().round(4)
snip["quasi_oracle_all_models"] = {f"{n}_{m}": float(v) for (n, m), v in qo.items()}
json.dump(snip, open(f"{OUT}/paper_snippets.json", "w"), indent=1)
print(json.dumps({k: v for k, v in snip.items() if k != "g1"}, indent=1))
