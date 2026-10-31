"""Before (thick line = mean of draws) / after (thick line = estimate on all seeds) renders for CE (2026-10-31)."""
import os, subprocess, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
V = os.path.expandvars("$WORK/paper_style_figs/v3"); OUT = f"{V}/compare_v5"; os.makedirs(OUT, exist_ok=True)
FIGS = [  # (paper number, file, before dir, after dir)
    ("Figure 4", "aggregated_test_gain_k_1000_sim_linear_fixedFalse_mse_full", "final_sim_v4h", "final_sim_v5"),
    ("Figure 5 (left)", "aggregated_test_gain_k_1000_sim_linear_fixedFalse_mse_K20", "final_sim_v4h", "final_sim_v5"),
    ("Figure 5 (middle)", "aggregated_test_gain_k_500_delta_linear_accuracy_stratFalse", "final_real_meantop", "final_real_v5c"),
    ("Figure 5 (right)", "aggregated_test_gain_k_1000_delta_linear", "final_real_meantop", "final_real_v5c"),
    ("Figure F.1", "aggregated_test_gain_k_300_delta_linear_cv_procedures", "final_real_v5", "final_real_v5c"),
    ("Figure I.1", "aggregated_test_gain_k_1000_sim_linear_fixedFalse_mse_K20_hpo", "final_sim_v4h", "final_sim_v5"),
    ("Figure J.1", "aggregated_test_gain_k_100_sim_linear_mse_fixedsplit_overlay_K20", "final_sim_v4h", "final_sim_v5"),
    ("Figure K.1", "aggregated_test_gain_k_500_delta_linear_nll_vs_strat_baseline", "final_real_v5", "final_real_v5c"),
    ("Figure L.1", "sample_gain_by_metric_wide_resnet101_2_N500_stratTrue", "final_real_meantop", "final_real_v5c"),
    ("Figure M.1 (left)", "small_label_gain_vs_K_N30_mse", "final_sim_v4h", "final_sim_v5"),
    ("Figure M.1 (right)", "small_label_gain_vs_K_N100_mse", "final_sim_v4h", "final_sim_v5"),
]
for num, f, before, after in FIGS:
    pngs = []
    for d in (before, after):
        png = f"{OUT}/{f}__{d}.png"
        subprocess.run(["gs", "-q", "-sDEVICE=png16m", "-r110", "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4",
                        "-o", png, f"{V}/{d}/{f}.pdf"], check=True)
        pngs.append(plt.imread(png))
    wide = pngs[0].shape[1] > 1.3 * pngs[0].shape[0]
    fig, axes = plt.subplots(2 if wide else 1, 1 if wide else 2, figsize=(9, 7.5) if wide else (11, 5.4))
    for ax, im, t in zip(axes, pngs, ("before: thick line = mean of the 1,000 draws", "after: thick line = estimate on all seeds")):
        ax.imshow(im); ax.set_axis_off(); ax.set_title(t, fontsize=11)
    fig.suptitle(num, fontsize=13, fontweight="bold"); fig.tight_layout()
    fig.savefig(f"{OUT}/{num.replace(' ', '_').replace('(', '').replace(')', '')}.png", dpi=110); plt.close(fig)
    print("compared", num, flush=True)
