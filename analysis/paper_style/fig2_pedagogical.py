#!/usr/bin/env python3
"""Figure 2 of the paper (pedagogical variance diagnostic: PCam MobileNetV2, study set 500 = n_tr 400, MCCV, accuracy,
50 seeds), with the single-split curve averaged exactly over the order of the outer chunks (the order-averaged
estimator of every gain of the paper): compute.v1_alpha_curve_long on the cumulative errors, the single-split side
of compute.sample_gain_bootstrap. The K-split variances are those of the first K splits of each seed.

The figure keeps the design of the previous version (x axis to 12 with ticks 1, 3, 6, 12, G^test_K marked under the
axis at each crossing), with the labels placed off the curve. Writes, to --out-dir:
  fig2_curves.csv                                          V1(alpha) (average and stored order) and V_K
  pedagogical_variance_diagnostic_delta_mobilenet_v2_N500_MCCV_paper_style.pdf  the figure (--chunk-order average)
and prints the crossings (linear interpolation, as drawn) and the paper estimator's G_5, G_10, G_20 (log-log).
--from-csv redraws from fig2_curves.csv without reading the runs.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("PAPER_STYLE_CODE") or os.path.join(HERE, "code"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import compute
import datapaths
import plot as P

NAME = "pedagogical_variance_diagnostic_delta_mobilenet_v2_N500_MCCV_paper_style.pdf"
XLABEL = "How many times more test data a\nsingle split needs to match CV variance"
TARGET_PT = (235.620667, 204.666626 - 10.5)   # previous PDF size, one label row (10.5 pt) shorter since the three
                                              # G^test labels share one row; same width, so the same page scale


def curves(data_root):
    import pcam
    import pipeline_common as C
    d = pcam.load_cv_data(os.path.join(data_root, "pcam", "results_output_pcam"))
    if "stratified" not in d.columns:
        d["stratified"] = False
    d = d[(d.model == "mobilenet_v2") & (d.study_size == 500) & d.cross_validation & (d.procedure == "MCCV")
          & (~d.stratified.fillna(False).astype(bool))].copy()
    d = d.drop_duplicates(subset=["seed", "fold"]).sort_values(["seed", "fold"])
    d = C.coerce_scalar_columns(d, columns=["test_accuracy", "benchmark_accuracy"], scalar_extractor=pcam._extract_scalar)
    d["delta"] = d.test_accuracy - d.benchmark_accuracy
    d["split"] = d.groupby("seed").cumcount()
    folds = d.pivot(index="seed", columns="split", values="delta")
    first = d[d.split == 0].set_index("seed").loc[folds.index]
    H = np.vstack([np.asarray(h, float) for h in first.hidden_test_accuracies.values]) - first.benchmark_accuracy.values[:, None]
    x0 = folds[0].values
    n = H.shape[1]
    # long table of cumulative errors on the test fold and the first k chunks, as pipeline_common's diagnostics
    cum = np.hstack([x0[:, None], x0[:, None] + np.cumsum(H, axis=1)]) / np.arange(1, n + 2)
    long = pd.DataFrame({"seed": np.repeat(folds.index.values, n + 1), "k": np.tile(np.arange(n + 1), len(x0)),
                         "delta_score": cum.ravel()}).assign(model="mobilenet_v2")
    out = pd.DataFrame({"alpha": np.arange(1, n + 2)})
    for order in ("average", "stored"):
        v = compute.v1_alpha_curve_long(long, ["model"], "delta_score", chunk_order=order)
        out[f"V1_{order}"] = v.sort_values("k").variance.values
    assert np.allclose(out.V1_average, compute.v1_alpha_curve(x0, H), rtol=1e-12)
    vk = {K: float(folds.iloc[:, :K].mean(axis=1).var(ddof=1)) for K in (5, 10, 20)}
    for K, v in vk.items():
        out[f"V_{K}"] = v
    print(f"seeds {len(x0)}, chunks {n}", flush=True)
    return out


def crossing_loglog(v1, v):
    j = np.where(v1 <= v * (1 + 1e-9))[0][0]
    if j == 0:
        return 1.0
    f = (np.log(v1[j - 1]) - np.log(v)) / (np.log(v1[j - 1]) - np.log(v1[j]))
    return float(np.exp(np.log(j) + f * (np.log(j + 1) - np.log(j))))


def draw(cv, order, path, layout):
    v1 = cv[f"V1_{order}"].to_numpy()
    vk = {K: float(cv[f"V_{K}"].iloc[0]) for K in (5, 10, 20)}
    print(order, "paper estimator G_K (log-log):", {K: round(crossing_loglog(v1, v), 2) for K, v in vk.items()})
    var_k = pd.DataFrame({"x_axis_val": cv.alpha.to_numpy(), "variance": v1})
    var_cv = pd.DataFrame({"n_splits": list(vk), "variance": list(vk.values())})
    P.set_paper_style(legend_frameon=True)
    fig = P.plot_pedagogical_variance_diagnostic_paper_style(
        var_k, var_cv, target_splits=[5, 10, 20], x_col="x_axis_val", var_col="variance", xlabel=XLABEL,
        cv_colors=["#4C78A8", "#59A14F", "#E15759"], output_path=None, **layout)
    ax = fig.axes[0]
    xs = [float(t) for t in ax.xaxis.get_minorticklocs()]
    print(order, "crossings as drawn (linear):", [round(x, 2) for x in xs], "xlim", ax.get_xlim(), "ylim", ax.get_ylim())
    # same page size as before: tight bbox, then pad symmetrically to the previous size
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    W, Hh = TARGET_PT[0] / 72, TARGET_PT[1] / 72
    print(f"tight bbox {bb.width * 72:.1f} x {bb.height * 72:.1f} pt (target {TARGET_PT[0]:.1f} x {TARGET_PT[1]:.1f})")
    from matplotlib.transforms import Bbox
    x0 = bb.x0 - (W - bb.width) / 2
    y0 = bb.y0 - (Hh - bb.height) / 2
    fig.savefig(path, bbox_inches=Bbox.from_bounds(x0, y0, W, Hh))
    plt.close(fig)


# Previous design (x axis to 12, ticks 1, 3, 6, 12); the order-averaged crossings are at 4.6, 6.2 and 7.6, so G_10
# would sit on the tick 6: the G^test labels go one row below the tick numbers.
LAYOUT = dict(x_right=12.04, major_ticks=[1, 3, 6, 12], y_lim_bottom=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-root", default=None,
                    help="run outputs: pcam/results_output_pcam under it (default: $DATA_ROOT, else data/ at the repository root)")
    ap.add_argument("--from-csv", action="store_true")
    ap.add_argument("--chunk-order", choices=compute.CHUNK_ORDERS, default=None)
    ap.add_argument("--variant", default="final")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    f = os.path.join(a.out_dir, "fig2_curves.csv")
    cv = pd.read_csv(f) if a.from_csv else curves(datapaths.data_root(a.data_root))
    if not a.from_csv:
        cv.to_csv(f, index=False)
    order = compute.resolve_chunk_order(a.chunk_order)
    import fig2_layouts
    lay = fig2_layouts.LAYOUTS[a.variant](cv)
    draw(cv, order, os.path.join(a.out_dir, NAME if a.variant == "final" else f"fig2_{a.variant}_{order}.pdf"), lay)


if __name__ == "__main__":
    main()
