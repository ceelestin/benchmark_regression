#!/usr/bin/env python3
"""Figure 6 (p. 9), v7: sample gain G_20 against the redundancy after 3 splits, with the Yelp points and boxed keys.

(paper_style_figs/v3/fig6_v7/fig6_v7.py, 2026-11-04.) Item 33 redesign, which CE approved, then (item 33 follow-up):
  (1) both keys sit in a light, discreet box (white face, alpha 0.9, thin grey edge, small padding, slightly rounded
      corners), so that the key markers cannot be read as data points; the boxes are checked not to cover any point;
  (2) the Yelp points (accuracy) are drawn (--yelp yelp_points.csv).
Data unchanged:
  simulated: benchmark_regression analysis/out_omega/omega_{derivation,holdout1,holdout2,holdout3}_neg_mse.csv
             (663 configurations, gain on the MSE, omega = t * rho_e_mean_3, t = 0.2; G_paper: V1 averaged over the
             order of the outer chunks);
  PCam:      benchmark_pcam analysis/out_round3_redundancy/redundancy_vs_gain.csv (omega3) merged with
             analysis/out_round3/candidates_{accuracy,nll,brier}.csv (G_paper_20): 36 configurations;
  Yelp:      fig6_yelp_points.csv of fig6_yelp_points.py (the same two PCam functions on the 200 yelp_aligned parquets), accuracy
             rows only: BERT and XLM-RoBERTa at n_tr 1,000 and 3,000, 50 seeds; redundancy on the 0-1 loss.
Same figure size as the published PDF (MediaBox 238.79 x 224.40 pt): the page is cropped to the tight box of the figure
WITHOUT the key boxes (plus the usual 0.1 in), and the script asserts that the key boxes lie inside it.
Run in analysis/paper_style with `module load pytorch-gpu/py3/2.8.0` (code/ next to this script):
    python fig6_redundancy_vs_gain.py --yelp fig6_yelp_points.csv --out OUT/redundancy_vs_gain_all_k3.pdf
    python fig6_redundancy_vs_gain.py --out OUT/redundancy_vs_gain_all_k3.pdf          # variant without Yelp
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
from matplotlib.transforms import Bbox
from scipy.stats import spearmanr

import plot as P

K, T = 20, 0.2
REG = os.path.expandvars("$WORK/benchmark_regression/analysis")
PCAM = os.path.expandvars("$WORK/benchmark_pcam/analysis")
BINS = [(0, 100, "50–100"), (101, 750, "300–750"), (751, 3000, "1,000–3,000"), (3001, 10**9, "5,000–10,000")]
GREYS = ["#c6c6c6", "#9b9b9b", "#6f6f6f", "#444444"]          # sequential, light = small n_tr
REAL = {"PCam": dict(c="#E0A21A", ec="#8a5a00", marker="s", s=30),
        "Yelp": dict(c="#8E3FB0", ec="#4a1466", marker="D", s=26)}
INK, MUTED = "#1a1a1a", "#555555"
FS = 8.6       # key text, drawn size (the PDF is shown at 0.45\textwidth = 2.48 in, i.e. scaled by about 0.75)
SCALE = 0.45 * 5.5 / (238.79 / 72)   # printed size / drawn size (NeurIPS \textwidth = 5.5 in)
# key box, in printed points: white face at alpha 0.9, 0.5 pt edge of grey 0.75, about 2 pt of padding, 1.5 pt corners
BOX = dict(face="white", alpha=0.9, edge="0.75", lw=0.5 / SCALE, pad=2.2, round=2.0)


def bin_index(n):
    for i, (lo, hi, _) in enumerate(BINS):
        if lo <= n <= hi:
            return i
    raise ValueError(n)


def load(yelp_csv):
    d = pd.concat([pd.read_csv(f"{REG}/out_omega/omega_{s}_neg_mse.csv") for s in ("derivation", "holdout1", "holdout2", "holdout3")],
                  ignore_index=True)
    sim = pd.DataFrame({"w": T * d["rho_e_mean_3"], "g": d["G_paper"], "n": d["p_obj_train_size"].astype(int),
                        "noisy": d.solver_name.str.contains("pred_noise|coef_noise")})
    red = pd.read_csv(f"{PCAM}/out_round3_redundancy/redundancy_vs_gain.csv")
    gains = pd.concat([pd.read_csv(f"{PCAM}/out_round3/candidates_{m}.csv").assign(loss=m) for m in ("accuracy", "nll", "brier")])
    pc = red.merge(gains[["loss", "model", "train_size", "G_paper_20"]], left_on=["loss", "model", "size"],
                   right_on=["loss", "model", "train_size"], validate="1:1")
    assert len(sim) == 663 and len(pc) == 36
    real = {"PCam": pd.DataFrame({"w": pc["omega3"], "g": pc["G_paper_20"]})}
    if yelp_csv:
        y = pd.read_csv(yelp_csv)
        y = y[y["loss"] == "accuracy"]
        assert len(y) == 4 and (y.n_seeds == 50).all()
        real["Yelp"] = pd.DataFrame({"w": y["omega3"].values, "g": y["G_paper_20"].values,
                                     "label": (y["model"] + " " + y["size"].astype(str)).values})
    return sim, real


class Key:
    """A hand-placed key in axes coordinates; it records its artists so that a box can be drawn around them."""

    def __init__(self, ax):
        self.ax, self.texts, self.marks = ax, [], []      # marks: (x, y, half-width in pt, half-height in pt)

    def text(self, x, y, s, **kw):
        t = self.ax.text(x, y, s, transform=self.ax.transAxes, va="center", zorder=10, **kw)
        self.texts.append(t)
        return t

    def marker(self, x, y, **kw):
        self.ax.scatter([x], [y], transform=self.ax.transAxes, clip_on=False, zorder=10, **kw)
        r = np.sqrt(kw.get("s", 20)) / 2 * (1.25 if kw.get("marker") == "D" else 1.0) + kw.get("linewidths", 0) / 2
        self.marks.append((x, y, r, r))

    def row(self, x, y, label, handle, value=None, fs=FS, color=INK):
        """One row: handle at x, label right of it, value right-aligned at the value column."""
        if handle == "curve":
            self.ax.plot([x - 0.035, x + 0.045], [y, y], transform=self.ax.transAxes, color=INK, lw=2.0, dashes=(3.2, 1.6),
                         solid_capstyle="butt", dash_capstyle="butt", clip_on=False, zorder=10)
            self.marks += [(x - 0.035, y, 0, 1.0), (x + 0.045, y, 0, 1.0)]
            tx = x + 0.07
        else:
            self.marker(x, y, **handle)
            tx = x + 0.04
        self.text(tx, y, label, ha="left", fontsize=fs, color=color)
        if value is not None:
            self.text(VALX, y, value, ha="right", fontsize=fs, color=color)

    def extent(self, fig):
        """Bounding box of the key's artists, in display coordinates."""
        r = fig.canvas.get_renderer()
        boxes = [t.get_window_extent(r) for t in self.texts]
        pt = fig.dpi / 72
        for x, y, hw, hh in self.marks:
            cx, cy = self.ax.transAxes.transform((x, y))
            boxes.append(Bbox.from_extents(cx - hw * pt, cy - hh * pt, cx + hw * pt, cy + hh * pt))
        return Bbox.union(boxes)

    def box(self, fig):
        """Light box around the key: returns its extent in display coordinates (with padding)."""
        e = self.extent(fig)
        pt = fig.dpi / 72
        pad = BOX["pad"] / SCALE * pt
        e = Bbox.from_extents(e.x0 - pad, e.y0 - pad, e.x1 + pad, e.y1 + pad)
        (ax0, ay0), (ax1, ay1) = self.ax.transAxes.inverted().transform([[e.x0, e.y0], [e.x1, e.y1]])
        W, H = self.ax.bbox.width, self.ax.bbox.height
        rnd = BOX["round"] / SCALE * pt / W          # rounding radius in axes-x units; the aspect makes it round in y
        p = FancyBboxPatch((ax0, ay0), ax1 - ax0, ay1 - ay0, boxstyle=f"round,pad=0,rounding_size={rnd}",
                           mutation_aspect=W / H, transform=self.ax.transAxes, facecolor=BOX["face"], alpha=None,
                           edgecolor=BOX["edge"], linewidth=BOX["lw"], clip_on=False, zorder=9)
        p.set_in_layout(False)
        p.set_facecolor((1, 1, 1, BOX["alpha"]))      # alpha on the face only: the edge keeps its grey
        self.ax.add_patch(p)
        return e


VALX = 0.984   # value column (right edge), in axes coordinates
# axes position (figure fraction) of the published v6 PDF: constrained layout depends on how often the figure has been
# drawn (the keys are clip_on=False texts in axes coordinates), so v7 fixes the axes where v6 had them
AX_POS = (0.1919226996527778, 0.1998709027777778, 0.781758907493225, 0.7862390972222222)
# page (inches, figure coordinates) of the published v6 PDF, MediaBox 238.794 x 224.400 pt: the tight box of the
# constrained layout (3 pt pads) plus 0.1 in; with AX_POS it reproduces v6's PDF geometry to the pixel at 300 dpi
PAGE = Bbox.from_bounds(3 / 72 - 0.1, 3 / 72 - 0.1, 238.7936256141 / 72, 224.39952 / 72)


def covered(ax, fig, sim, real, boxes, margin_pt=0.5):
    """Data points whose marker touches a key box (display coordinates, marker radius plus a margin)."""
    pt = fig.dpi / 72
    rows = [pd.DataFrame({"w": sim.w, "g": sim.g, "src": "sim", "r": np.sqrt(17) / 2})]
    rows += [pd.DataFrame({"w": r.w, "g": r.g, "src": k, "r": np.sqrt(REAL[k]["s"]) / 2 * 1.25 + 0.35}) for k, r in real.items()]
    a = pd.concat(rows, ignore_index=True)
    xy = ax.transData.transform(a[["w", "g"]].values)
    out = []
    for name, b in boxes.items():
        rr = (a.r + margin_pt).values * pt
        hit = (xy[:, 0] + rr > b.x0) & (xy[:, 0] - rr < b.x1) & (xy[:, 1] + rr > b.y0) & (xy[:, 1] - rr < b.y1)
        # distance (pt) from each box edge to the nearest point outside it, for the report
        dx = np.maximum(np.maximum(b.x0 - xy[:, 0], xy[:, 0] - b.x1), 0)
        dy = np.maximum(np.maximum(b.y0 - xy[:, 1], xy[:, 1] - b.y1), 0)
        gap = (np.hypot(dx, dy) - a.r.values * pt) / pt * SCALE      # printed points
        near = a.assign(gap=gap, ax=ax.transAxes.inverted().transform(xy)[:, 0],
                        ay=ax.transAxes.inverted().transform(xy)[:, 1]).nsmallest(3, "gap")
        bax = ax.transAxes.inverted().transform([[b.x0, b.y0], [b.x1, b.y1]])
        out.append((name, a[hit], float(np.min(gap)), near, bax))
    return out


def draw(sim, real, out):
    P.set_paper_style()
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    # simulated cloud: neutral greys by training size, no edge, below the curve
    cols = np.array(GREYS)[sim.n.map(bin_index).values]
    for m, mk, s in ((~sim.noisy, "o", 15), (sim.noisy, "^", 17)):
        ax.scatter(sim.w[m], sim.g[m], c=cols[m.values], marker=mk, s=s, alpha=0.8, linewidths=0, zorder=2)
    # reference curve: thick, on top of the cloud, over a continuous white band; it stops at omega = t = 0.2, the
    # largest redundancy (rho <= 1)
    xs = np.linspace(-0.012, T, 300)
    ys = 1 / (1 / K + (1 - 1 / K) * xs)
    ax.plot(xs, ys, color="white", lw=4.6, alpha=0.95, solid_capstyle="butt", zorder=3.9)
    ax.plot(xs, ys, color=INK, lw=2.0, dashes=(3.2, 1.6), dash_capstyle="butt", zorder=4)
    # real data on top of the curve
    for name, r in real.items():
        st = REAL[name]
        ax.scatter(r.w, r.g, c=st["c"], marker=st["marker"], s=st["s"], edgecolors=st["ec"], linewidths=0.7, zorder=5)
    ax.set_yscale("log")
    gall = pd.concat([sim.g] + [r.g for r in real.values()])
    ticks = [t for t in (2, 5, 10, 20, 50) if gall.min() <= t <= gall.max()]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([0.00, 0.05, 0.10, 0.15, 0.20])
    ax.set_xlim(-0.035780016518425876, 0.21106687633226398)   # the published figure's limits
    ax.set_xlabel(r"Redundancy $\widehat{\omega}^{\mathrm{study}}_3$ after 3 splits", fontsize=12)
    ax.set_ylabel(r"Test sample gain ($G^{\mathrm{test}}_{20}$)", fontsize=12)
    ax.grid(True, which="major", linestyle=":", linewidth=0.7, color="#c8c8c8")
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    fig.canvas.draw()
    fig.set_layout_engine("none")
    ax.set_position(AX_POS)
    # key 1 (top right): what the marks are, with each set's rank correlation with the gain
    k1 = Key(ax)
    x0, y0, dy = 0.38, 0.945, 0.066
    k1.row(x0, y0, r"$K/(1+(K-1)\,\omega)$", "curve")
    k1.text(VALX, y0 - 0.9 * dy, "Rank corr.", ha="right", fontsize=FS - 0.9, color=MUTED)
    rows = [(f"Simulated ({len(sim)})", dict(marker="o", s=22, c=GREYS[2], linewidths=0), spearmanr(sim.w, sim.g).statistic)]
    for name, r in real.items():
        st = REAL[name]
        rows.append((f"{name} ({len(r)})", dict(marker=st["marker"], s=st["s"], c=st["c"], edgecolors=st["ec"], linewidths=0.7),
                     spearmanr(r.w, r.g).statistic))
    for i, (lab, h, rc) in enumerate(rows):
        k1.row(x0, y0 - (1.75 + i) * dy, lab, h, f"${rc:.2f}$")
    # key 2 (bottom left; no point lies below G = 3.3 left of omega = 0.087): the noise-injection triangles and the
    # training-size shading of the simulated points, light = small
    k2 = Key(ax)
    bx, fs2, yb, dy2 = 0.05, FS - 1.1, 0.055, 0.055
    k2.marker(bx, yb + dy2, marker="^", s=20, c=GREYS[2], linewidths=0)
    k2.text(bx + 0.03, yb + dy2, "Ridge + noise", ha="left", fontsize=fs2, color=MUTED)
    t = k2.text(bx - 0.012, yb, r"$n_{\mathrm{tr}}$ 50", ha="left", fontsize=fs2, color=MUTED)
    fig.canvas.draw()
    x1 = ax.transAxes.inverted().transform(t.get_window_extent().get_points())[1, 0]
    for i, c in enumerate(GREYS):
        k2.marker(x1 + 0.03 + 0.045 * i, yb, marker="o", s=18, c=c, linewidths=0)
    k2.text(x1 + 0.03 + 0.045 * 3 + 0.03, yb, "10,000", ha="left", fontsize=fs2, color=MUTED)
    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    assert PAGE.x0 <= tight.x0 and tight.x1 <= PAGE.x1 and PAGE.y0 <= tight.y0 and tight.y1 <= PAGE.y1, (tight, PAGE)
    page = PAGE
    print("axes position (figure fraction)", np.round(ax.get_position().bounds, 4))
    boxes = {"top-right": k1.box(fig), "bottom-left": k2.box(fig)}
    fig.canvas.draw()
    for name, b in boxes.items():
        bi = b.transformed(fig.dpi_scale_trans.inverted())
        assert page.x0 <= bi.x0 and bi.x1 <= page.x1 and page.y0 <= bi.y0 and bi.y1 <= page.y1, (name, bi, page)
    cov = covered(ax, fig, sim, real, boxes)       # before saving: same transforms as the boxes
    fig.savefig(out, format="pdf", bbox_inches=page)
    fig.savefig(out[:-4] + ".png", dpi=300, bbox_inches=page)
    n_hit = 0
    for name, hit, gap, near, bax in cov:
        print(f"key box {name} (axes {np.round(bax.ravel(), 3)}): {len(hit)} points covered; "
              f"nearest point {gap:.1f} pt (printed) from the box")
        print(near.round(3).to_string())
        n_hit += len(hit)
    print("page (in)", np.round(page.bounds, 4), "-> pt", round(page.width * 72, 3), "x", round(page.height * 72, 3))
    pooled = pd.concat(list(real.values()))
    print(out, {k: round(float(spearmanr(r.w, r.g).statistic), 3) for k, r in real.items()},
          "real pooled", round(float(spearmanr(pooled.w, pooled.g).statistic), 3),
          "simulated", round(float(spearmanr(sim.w, sim.g).statistic), 3), "| G range", round(gall.min(), 2), round(gall.max(), 2))
    print("ylim", ax.get_ylim())
    assert n_hit == 0, "a key box covers data points"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--yelp", default=None, help="fig6_yelp_points.csv of fig6_yelp_points.py: adds the Yelp diamonds (accuracy)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sim, real = load(a.yelp)
    draw(sim, real, a.out)


if __name__ == "__main__":
    main()
