"""LaTeX tables of the sample gains (estimate on all seeds) and their 95% percentile bootstrap intervals, one per
gain figure, from the tables saved by the figure scripts (CE, 2026-10-31). Prints the LaTeX to stdout."""
import os
import numpy as np, pandas as pd
V = os.path.expandvars("$WORK/paper_style_figs/v3")
S, R = f"{V}/final_sim_v5/tables", f"{V}/final_real_v5c/tables"
PCAM = {"densenet121": "DenseNet121", "mobilenet_v2": "MobileNetV2", "wide_resnet101_2": "WideResNet101\\_2", "vit_b_16": "ViT-B/16"}
NLP = {"bert": "BERT", "roberta": "XLM-RoBERTa", "qwen3-0.6b": "Qwen3-0.6B"}
tt = lambda x: "\\texttt{" + x + "}"

def cells(df, keys, Ks):
    """{(key tuple): {K: cell}} with estimate and percentile interval; flags floor / cap pile-ups."""
    out = {}
    for key, g in df.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        row = {}
        for K in Ks:
            d = g[g.K == K]
            e, b = d[d.bootstrap_iter < 0].sample_gain, d[d.bootstrap_iter >= 0].sample_gain
            if e.empty or b.empty:
                row[K] = "--"; continue
            lo, hi = np.quantile(b, [.025, .975])
            flag = ""
            if (b <= 1.0001).mean() >= .025:
                flag += "^{\\dagger}"
            if (b >= b.max() - 1e-9).mean() >= .025 and b.max() > 1.0001:
                flag += "^{\\ddagger}"
            row[K] = f"${float(e.iloc[0]):.1f}{flag}$ {{\\scriptsize $[{lo:.1f},\\,{hi:.1f}]$}}"
        out[key] = row
    return out

def table(caption, label, header, rows, Ks):
    cols = "l" * len(header) + "c" * len(Ks)
    lines = ["\\begin{table}[htbp]", "    \\centering", f"    \\caption{{{caption}}}", f"    \\label{{{label}}}",
             "    \\small", f"    \\begin{{tabular}}{{@{{}} {cols} @{{}}}}", "        \\toprule",
             "        " + " & ".join([f"\\thead{{{h}}}" for h in header] + [f"$K={K}$" for K in Ks]) + " \\\\",
             "        \\midrule"]
    prev = None
    for r in rows:
        if r == "mid":
            lines.append("        \\midrule"); prev = None; continue
        r = list(r); first = r[0]
        if len(header) > 1 and first == prev:
            r[0] = ""
        prev = first
        lines.append("        " + " & ".join(r) + " \\\\")
    lines += ["        \\bottomrule", "    \\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)

def rows_from(c, order, label_fns, Ks):
    rows = []
    for key in order:
        if key == "mid":
            rows.append("mid"); continue
        if key in c:
            rows.append([f(k) for f, k in zip(label_fns, key)] + [c[key][K] for K in Ks])
    return rows

out = []
SIM = ["Ridge", "ExtraTrees", "MLP", "RandomFeatures"]
# Figure 4 and Figure 5 (left)
f4 = pd.read_parquet(f"{S}/fig4.parquet"); K4 = [5, 10, 20, 200]
c = cells(f4, ["solver_disp"], K4)
out.append(table("Sample gains $G^{\\text{test}}_K$ of \\autoref{fig:avg_test_sample_gain_simulated} and \\autoref{fig:all_data_test_sample_gains} (left): simulated linear process, $n_{\\text{tr}} = 1{,}000$, $1{,}000$ seeds, mean squared error.",
                 "tab:ci-fig4", ["Algorithm"], rows_from(c, [(l,) for l in SIM], [tt], K4), K4))
K = [5, 10, 20]
# Figure 5 (middle, right)
pc = pd.read_parquet(f"{R}/pcam_accuracy.parquet"); pc = pc[(pc.N == 500) & (pc.cv_procedure == "MCCV")]
y = pd.read_parquet(f"{R}/nlp_delta.parquet"); y = y[y.N == 1250]
c1, c2 = cells(pc, ["model"], K), cells(y, ["model"], K)
rows = [["\\texttt{PCam}, $n_{\\text{tr}}=500$", tt(PCAM[m])] + [c1[(m,)][k] for k in K] for m in PCAM if (m,) in c1] + ["mid"] + \
       [["\\texttt{Yelp}, $n_{\\text{tr}}=1{,}000$", tt(NLP[m])] + [c2[(m,)][k] for k in K] for m in NLP if (m,) in c2]
out.append(table("Sample gains of \\autoref{fig:all_data_test_sample_gains} (middle and right): accuracy, $50$ seeds.",
                 "tab:ci-fig5", ["Dataset", "Network"], rows, K))
# Figure F.1
f1 = pd.read_parquet(f"{R}/figF1.parquet")
c = cells(f1, ["model", "cv_procedure"], K)
order = [(m, p) for m in ("densenet121", "mobilenet_v2", "wide_resnet101_2") for p in ("MCCV", "RepeatedKFold")]
out.append(table("Sample gains of \\autoref{fig:pcam_mccv_vs_kfold}: \\texttt{PCam}, $n_{\\text{tr}}=300$, accuracy, $50$ seeds.",
                 "tab:ci-figF1", ["Network", "Procedure"],
                 rows_from(c, order, [lambda m: tt(PCAM[m]), lambda p: {"MCCV": "MCCV", "RepeatedKFold": "Repeated 5-Fold"}[p]], K), K))
# Figure I.1
i1 = pd.read_parquet(f"{S}/figI1.parquet")
i1["procedure_disp"] = i1.solver_disp.map(lambda x: "Grid search" if x.endswith("_GridSearch") else
                                          "Random search" if x.endswith("_RandomSearch") else "Fixed")
i1["solver_disp"] = i1.solver_disp.str.replace("_GridSearch", "").str.replace("_RandomSearch", "")
c = cells(i1, ["solver_disp", "procedure_disp"], K)
order = [(l, p) for l in ("MLP", "ExtraTrees") for p in ("Fixed", "Grid search", "Random search")]
out.append(table("Sample gains of \\autoref{fig:hpo-gains}: simulated linear process, $n_{\\text{tr}}=1{,}000$, $1{,}000$ seeds, mean squared error.",
                 "tab:ci-figI1", ["Algorithm", "Tuning"], rows_from(c, order, [tt, str], K), K))
# Figure J.1
j1 = pd.read_parquet(f"{S}/figJ1.parquet")
c = cells(j1, ["solver_disp", "split_mode"], K)
order = [(l, s) for l in SIM for s in ("Resampled split", "Fixed split")]
out.append(table("Sample gains of \\autoref{fig:fixed-vs-resampled}: simulated linear process, $n_{\\text{tr}}=100$, $1{,}000$ seeds, mean squared error; frozen: a single split repeated $K$ times.",
                 "tab:ci-figJ1", ["Algorithm", "Split"], rows_from(c, order, [tt, lambda s: {"Resampled split": "reshuffled", "Fixed split": "frozen"}[s]], K), K))
# Figure K.1
k1 = pd.read_parquet(f"{R}/figK1.parquet")
c = cells(k1, ["model", "stratified"], K)
order = [(m, s) for m in PCAM for s in (False, True)]
out.append(table("Sample gains of \\autoref{fig:stratified-overlay}: \\texttt{PCam}, $n_{\\text{tr}}=500$, negative log-likelihood, $100$ seeds, against the stratified single-split baseline.",
                 "tab:ci-figK1", ["Network", "Resampling"], rows_from(c, order, [lambda m: tt(PCAM[m]), lambda s: "stratified" if s else "unstratified"], K), K))
# Figure L.1
l1 = pd.read_parquet(f"{R}/figL1.parquet")
c = cells(l1, ["metric"], K)
out.append(table("Sample gains of \\autoref{fig:metrics-gain}: \\texttt{PCam}, \\texttt{WideResNet101\\_2}, $n_{\\text{tr}}=500$, stratified resampling, $100$ seeds.",
                 "tab:ci-figL1", ["Metric"], rows_from(c, [("Accuracy",), ("Brier",), ("NLL",)], [lambda x: {"Accuracy": "accuracy", "Brier": "Brier score", "NLL": "negative log-likelihood"}[x]], K), K))
# Figure M.1, all budgets
m1 = pd.read_parquet(f"{S}/figM1.parquet")
c = cells(m1, ["N", "solver_disp"], K)
order = []
for n in sorted(m1.N.unique()):
    order += [(n, l) for l in SIM] + ["mid"]
rows = rows_from(c, order[:-1], [lambda n: f"${int(n)}$", tt], K)
out.append(table("Sample gains at the small budgets of \\autoref{appendix:label-scarce} (\\autoref{fig:small-label-gain} shows $n_{\\text{tr}} = 30$ and $100$): simulated linear process, $1{,}000$ seeds, mean squared error.",
                 "tab:ci-figM1", ["$n_{\\text{tr}}$", "Algorithm"], rows, K))
print("\n".join(out))
