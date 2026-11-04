#!/usr/bin/env python3
"""Figure 5 of the paper with one y axis for its three panels (item 32, CE 2026-11-04): the panels are redrawn from
the gain tables of the figure scripts, with exactly the plotting calls of those scripts (paper_style_figs.py
for the simulated panel, pcam.py for PCam, regen_real_data_gain_figures.py fig5_yelp_aligned for Yelp), plus a common y range (top
8% above the highest thick line of the three panels) and common y ticks (code/plot.py GAIN_DISPLAY ylim / yticks).

  --sim-tables   tables/ of paper_style_figs.py (fig4.parquet)
  --real-tables  tables/ of regen_real_data_gain_figures.py (pcam_accuracy.parquet, nlp_delta_aligned_with_qwen.parquet)
  --common-axis  off: draws each panel with its own axis (must reproduce the scripts' PDFs; a check)
Writes the three PDFs under their paper names to --out-dir.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paper_style_figs as SIMF            # puts ./code (or $PAPER_STYLE_CODE) first on sys.path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import nlp
import pcam
import pipeline_common as C
import plot as P

NAMES = {"sim": "aggregated_test_gain_k_1000_sim_linear_fixedFalse_mse_K20.pdf",
         "pcam": "aggregated_test_gain_k_400_delta_linear_accuracy_stratFalse.pdf",
         "yelp": "aggregated_test_gain_k_1000_delta_linear.pdf"}
BASE = ["ExtraTrees", "MLP", "Ridge", "RandomFeatures"]


def panels(sim_tables, real_tables):
    s = pd.read_parquet(f"{sim_tables}/fig4.parquet")
    s = s[s["K"] <= 20]
    p = pd.read_parquet(f"{real_tables}/pcam_accuracy.parquet")
    p = p[(p["stratified"] == False) & (p["cv_procedure"] == "MCCV") & (p["N"] == 500)]   # noqa: E712
    y = pd.read_parquet(f"{real_tables}/nlp_delta_aligned_with_qwen.parquet")
    y = y[y["N"] == 1250]
    return s, p, y


def common_axis(s, p, y):
    tops = [P._y_top(s, "K", "sample_gain", ["solver_disp"]), P._y_top(p, "K", "sample_gain", ["model"]),
            P._y_top(y, "K", "sample_gain", ["model"])]
    top = max(tops)
    step = 2 if top <= 14 else 4
    ticks = list(range(step, int(np.floor(top)) + 1, step))
    print("y tops (sim, PCam, Yelp):", [round(t, 3) for t in tops], "-> common top", round(top, 3), "ticks", ticks)
    return (1 - 0.05 * (top - 1), top), ticks


# Panel geometry in points (2026-11-04). The left panel keeps the margins of the previous version (y label and
# values on the left: 51.46 pt, x label below: 43.74 pt, 9.55 pt above the axes); the middle
# and right panels drop the y label and values and keep 12 pt on the left. All axes are AX_W wide, so that at the
# page scale PAGE_PER_PT the three fill 0.98 \linewidth.
AX_B, TOP, RIGHT = 43.74, 9.55, 13.5
LEFT_L, LEFT_R = 51.46, 12.0
# Page scale 0.62 (page pt per figure pt, for a 397.48 pt text width): the largest at which the direct labels still
# sit next to their curves (scanned 0.58-0.73 on 2026-11-04; at Figure 4's 0.73 they move away). Text is 1.16 times
# larger than at the old 0.53 (0.32 linewidth for 238.8 pt), and the axes are close to their original size.
PAGE_PER_PT = 0.62 / 397.48       # linewidth per point of figure
AX_W = (0.98 / PAGE_PER_PT - (LEFT_L + 2 * LEFT_R + 3 * RIGHT)) / 3
AX_H = 171.11 * AX_W / 173.84      # same aspect as the previous axes (the panels grow in height too)
FIG_H = AX_B + AX_H + TOP


def _panel(left_pt, hide):
    P.GAIN_DISPLAY["figsize"] = ((left_pt + AX_W + RIGHT) / 72, FIG_H / 72)
    P.GAIN_DISPLAY["axes_box_pt"] = (left_pt, AX_B, AX_W, AX_H)
    P.GAIN_DISPLAY["hide_ylabel"] = hide


def draw_all(s, p, y, out, ylim=None, yticks=None, shared_layout=False):
    os.makedirs(out, exist_ok=True)
    P.GAIN_DISPLAY["ylim"], P.GAIN_DISPLAY["yticks"] = ylim, yticks
    if shared_layout:
        print(f"panel axes width {AX_W:.2f} pt; figure widths {LEFT_L + AX_W + RIGHT:.2f} and "
              f"{LEFT_R + AX_W + RIGHT:.2f} pt; subfigure widths {(LEFT_L + AX_W + RIGHT) * PAGE_PER_PT:.5f} and "
              f"{(LEFT_R + AX_W + RIGHT) * PAGE_PER_PT:.5f} \\linewidth")
        _panel(LEFT_L, False)
    # simulated (paper_style_figs.main: draw(main_[K <= 20], base, K20, ..., f(1000)))
    P.set_paper_style()
    P.GAIN_DISPLAY["y_top"] = "mean"
    fig = SIMF.draw(s, BASE, SIMF.K20, f"{out}/{NAMES['sim']}", "$y = X\\beta$, $n_{\\mathrm{tr}}=1000$")
    print("sim ylim", fig.axes[0].get_ylim()); plt.close(fig)
    # PCam (pcam.py main, Plot 1, accuracy, unstratified, MCCV; hairlines 20 from regen's PIPE_ARGS)
    P.set_paper_style()
    if shared_layout:
        _panel(LEFT_R, True)
    P.GAIN_DISPLAY["hairlines"] = 20
    work = f"{out}/_work_pcam"; os.makedirs(work, exist_ok=True)
    C.plot_sample_gain_by_train_size(p, study_sizes=[500], models_to_analyze=pcam.DEFAULT_MODELS,
                                     model_display_names=pcam.MODEL_DISPLAY_NAMES, model_color_map=None,
                                     title_prefix="PCam", plots_dir=work, xticks=[1, 5, 10, 15, 20],
                                     file_suffix="_accuracy_stratFalse")
    os.replace(f"{work}/{NAMES['pcam']}", f"{out}/{NAMES['pcam']}")
    # Yelp (regen_real_data_gain_figures.fig5_yelp_aligned, the combined table with Qwen3-0.6B)
    P.set_paper_style()
    P.GAIN_DISPLAY["hairlines"] = 20
    tab10 = sns.color_palette("tab10", n_colors=10)
    work = f"{out}/_work_yelp"; os.makedirs(work, exist_ok=True)
    C.plot_sample_gain_by_train_size(y, study_sizes=[1250], models_to_analyze=nlp.DEFAULT_MODELS,
                                     model_display_names=nlp.MODEL_DISPLAY_NAMES,
                                     model_color_map={"bert": tab10[0], "roberta": tab10[1], "qwen3-0.6b": tab10[2]},
                                     title_prefix="Yelp", plots_dir=work)
    os.replace(f"{work}/{NAMES['yelp']}", f"{out}/{NAMES['yelp']}")
    P.GAIN_DISPLAY["ylim"] = P.GAIN_DISPLAY["yticks"] = None
    P.GAIN_DISPLAY["figsize"] = P.GAIN_DISPLAY["axes_box_pt"] = None
    P.GAIN_DISPLAY["hide_ylabel"] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-tables", required=True)
    ap.add_argument("--real-tables", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--common-axis", choices=["on", "off"], default="on")
    ap.add_argument("--shared-layout", choices=["on", "off"], default="on",
                    help="on: middle and right panels without y label and values, same wider axes box for all three")
    a = ap.parse_args()
    s, p, y = panels(a.sim_tables, a.real_tables)
    if a.common_axis == "on":
        ylim, ticks = common_axis(s, p, y)
        draw_all(s, p, y, a.out_dir, ylim, ticks, shared_layout=a.shared_layout == "on")
    else:
        draw_all(s, p, y, a.out_dir)


if __name__ == "__main__":
    main()
