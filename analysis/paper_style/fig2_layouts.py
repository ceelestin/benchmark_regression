"""Label layouts tried for Figure 2 (fig2_pedagogical.py --variant NAME); 'final' is the one used in the paper.

All keep the previous design: x axis to 12 with ticks 1, 3, 6, 12, G^test_K under the axis at each crossing. The
order-averaged crossings (4.7, 6.3, 7.6 as drawn) put G^test_10 on the tick 6 and next to G^test_20, so it goes one
row lower; the y axis reaches below the curve at 12; the curve label sits right of the steep part of the curve.
"""
import numpy as np

ROW = 10.5     # points: G^test_10 one row below the tick numbers


def _base(cv, order="average"):
    v1 = cv[f"V1_{order}"].to_numpy()
    lo = float(v1[cv.alpha.to_numpy() <= 12].min())
    return dict(x_right=12.04, major_ticks=[1, 3, 10, 12], y_lim_bottom=lo * 0.92)   # no 6: it sat over G^test_10 (CE, 2026-11-04)


def auto(cv):
    return {}


def plain(cv):
    return dict(x_right=12.04, major_ticks=[1, 3, 6, 12])


def stagger(cv):
    lay = _base(cv)
    lay.update(crossing_label_row={10: ROW}, curve_label_xy=(5.6, 6.3e-4), figsize=(3.2, 2.76))
    return lay


def allrow(cv):
    lay = _base(cv)
    lay.update(crossing_label_row={5: ROW, 10: ROW, 20: ROW}, curve_label_xy=(5.6, 6.3e-4), figsize=(3.2, 2.76))
    return lay


def final_k10(cv):
    """stagger, with "CV, K=10" just above its dashed line instead of centred on it (CE, 2026-11-04, as Figure 7(a))."""
    lay = stagger(cv)
    lay.update(cv_label_xy={10: (12.04 - 0.18, float(cv["V_10"].iloc[0]) * 1.12, "right")})
    return lay


final = final_k10
LAYOUTS = {"auto": auto, "plain": plain, "stagger": stagger, "allrow": allrow, "final_k10": final_k10, "final": final}
