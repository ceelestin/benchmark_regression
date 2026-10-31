"""
plot.py — Plotting helpers for the Validation Crisis experiments.

All functions return a ``plt.Figure`` and optionally save to a PDF path.
Callers are responsible for filtering DataFrames to a single (model, study_size)
slice where the docstring says "caller pre-filters".

Public API:
  set_paper_style                      apply standard rcParams
  plot_sample_gain_vs_splits           lineplot of gain / k_equiv vs K
  plot_variance_diagnostic             no-CV variance curve + CV reference lines
  plot_pedagogical_variance_diagnostic extended version with crossing annotations
  plot_retrieval_curve                 proportion-of-correct-picks vs multiplier
  plot_pareto_combined                 pareto lines with CI bands
  plot_score_distributions             boxplots of delta score per model/dataset
  plot_tau_bar                         bar plot of τ̂ per algorithm
  plot_tau_vs_gain_scatter             τ̂ vs sample-gain scatter (pooled across pipelines)
  plot_tau_vs_gain_encoded             τ̂ vs sample-gain scatter with marker/size/saturation encoding
  plot_pairplot_tau_gain               pairplot of ANOVA components vs sample gain
  plot_tau_vs_gain_studysize           scatter colored by study_size (viridis), marker by dataset
"""

from __future__ import annotations

import warnings
from typing import Literal

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, MaxNLocator, ScalarFormatter
from matplotlib.transforms import Bbox

# Uncertainty display of the sample-gain-vs-K figures (CE, 2026-10-29). With ``hairlines`` set, the line
# is the mean of all bootstrap replicates (drawn bold) and a fixed random subset of the replicates is drawn
# as thin translucent lines in the curve's colour; the y-axis is cut at a high quantile of the replicates.
# Pipelines that cannot pass arguments to the plot (pcam.py / nlp.py) set GAIN_DISPLAY["hairlines"].
GAIN_DISPLAY = {
    "hairlines": None,          # replicates drawn per curve; None = percentile band instead
    "hairline_alpha": 0.15,
    "hairline_linewidth": 0.45,
    "hairline_seed": 0,         # seed of the random choice of the drawn replicates
    "mean_linewidth": 2.2,
    "y_top_quantile": 0.99,
    # "quantile": y top at y_top_quantile of all replicates; "mean": just above the highest mean curve
    # (by y_top_margin), whatever the replicates do (CE, 2026-10-30).
    "y_top": "mean",
    "y_top_margin": 0.08,
    "replicate_col": "bootstrap_iter",
}


def _estimate_and_replicates(df):
    """Split a gain table into the thick-line rows and the bootstrap replicates (CE, 2026-10-31).

    Rows with a negative replicate index hold the estimate on the original seeds
    (``compute.sample_gain_bootstrap(..., plugin_estimate=True)``); when present they are the thick line, and the
    replicates give the thin lines. Without them, the thick line is the mean of the replicates, as before.
    """
    rcol = GAIN_DISPLAY["replicate_col"]
    if rcol in df.columns and (df[rcol] < 0).any():
        return df[df[rcol] < 0], df[df[rcol] >= 0]
    return df, df


def _y_top(df, x_col, y_col, group_cols):
    """Upper y limit of a gain figure, per GAIN_DISPLAY["y_top"]."""
    if GAIN_DISPLAY["y_top"] == "mean":
        df = _estimate_and_replicates(df)[0]
        keys = [c for c in group_cols if c is not None and c in df.columns] + [x_col]
        return float(df.groupby(keys)[y_col].mean().max()) * (1 + GAIN_DISPLAY["y_top_margin"])
    return float(np.quantile(df[y_col].dropna(), GAIN_DISPLAY["y_top_quantile"]))


def _hue_colors(levels, palette):
    """Colour of each hue level, as seaborn assigns it."""
    if isinstance(palette, dict):
        return {lv: palette[lv] for lv in levels}
    return dict(zip(levels, sns.color_palette(palette, n_colors=len(levels))))


def _draw_hairlines(ax, df, x_col, y_col, hue_col, hue_levels, colors, style_col=None, style_levels=None,
                    n_lines=20):
    """Thin translucent replicate curves, a fixed random subset of the bootstrap replicates per curve.

    Curves of the second and later style levels use a lighter shade of the hue colour, so the replicates of
    the two styles stay distinguishable. All hairlines are solid.
    """
    rcol = GAIN_DISPLAY["replicate_col"]
    df = df[df[rcol] >= 0]                     # never the full-sample estimate rows
    rng = np.random.default_rng(GAIN_DISPLAY["hairline_seed"])
    styles = style_levels if (style_col is not None and style_levels) else [None]
    for lv in hue_levels:
        g = df[df[hue_col] == lv]
        for j, st in enumerate(styles):
            gs = g[g[style_col] == st] if st is not None else g
            iters = np.sort(gs[rcol].unique())
            if n_lines and len(iters) > n_lines:
                iters = np.sort(rng.choice(iters, n_lines, replace=False))
            col = colors[lv] if j == 0 else sns.light_palette(colors[lv], 4)[max(1, 3 - j)]
            for _, c in gs[gs[rcol].isin(iters)].groupby(rcol):
                c = c.sort_values(x_col)
                ax.plot(c[x_col], c[y_col], color=col, lw=GAIN_DISPLAY["hairline_linewidth"],
                        alpha=GAIN_DISPLAY["hairline_alpha"], zorder=1, solid_capstyle="round")


# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

def set_paper_style(
    figsize: tuple[float, float] = (3.2, 3.0),
    legend_frameon: bool = False,
) -> None:
    """Apply the paper-standard rcParams used across all figures.

    Parameters
    ----------
    figsize : tuple
        Default figure size for all subsequent figures.
    legend_frameon : bool
        Set to True for the pedagogical figure which uses a framed legend.
    """
    mpl.rcParams.update({
        "figure.figsize": figsize,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
        "axes.titlepad": 8,
        "axes.labelpad": 6,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "axes.linewidth": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 1,
        "ytick.major.width": 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.grid": False,
        "figure.constrained_layout.use": True,
        "figure.autolayout": False,
        "legend.frameon": legend_frameon,
        "legend.loc": "best",
        "pdf.fonttype": 42,
    })


def save_figure(fig, output_path: str, **savefig_kwargs) -> None:
    """Save a Matplotlib/Seaborn figure and print its retrieval path."""
    print(f"Saved {output_path}")
    fig.savefig(output_path, **savefig_kwargs)


def _white_text_halo() -> list:
    """White halo used when text may overlap plotted curves."""
    return [path_effects.withStroke(linewidth=3.0, foreground="white")]


def _style_legend_halo(legend, *, title: str | None = None) -> None:
    """Use a title-free, frameless legend with a white text halo."""
    if legend is None:
        return
    legend.set_title("" if title is None else title)
    frame = legend.get_frame()
    frame.set_visible(False)
    halo = _white_text_halo()
    for text in legend.get_texts():
        text.set_path_effects(halo)
    legend.get_title().set_path_effects(halo)


def _palette_lookup(palette, label, hue_order: list | None = None):
    """Return a stable colour for a hue label when possible."""
    if isinstance(palette, dict):
        return palette.get(label)
    if palette is not None and hue_order is not None:
        try:
            idx = list(hue_order).index(label)
        except ValueError:
            return None
        if idx < len(palette):
            return palette[idx]
    return None


def _bbox_overlap_area(first: Bbox, second: Bbox) -> float:
    x0 = max(first.x0, second.x0)
    y0 = max(first.y0, second.y0)
    x1 = min(first.x1, second.x1)
    y1 = min(first.y1, second.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((x1 - x0) * (y1 - y0))


def _expanded_bbox(bbox: Bbox, pad: float) -> Bbox:
    return Bbox.from_extents(
        bbox.x0 - pad,
        bbox.y0 - pad,
        bbox.x1 + pad,
        bbox.y1 + pad,
    )


def _text_bbox(anchor, width: float, height: float, ha: str, va: str) -> Bbox:
    x, y = anchor
    if ha == "right":
        x0 = x - width
    elif ha == "center":
        x0 = x - width / 2
    else:
        x0 = x
    if va == "top":
        y0 = y - height
    elif va == "center":
        y0 = y - height / 2
    else:
        y0 = y
    return Bbox.from_bounds(x0, y0, width, height)


def _bbox_outside_axes_penalty(bbox: Bbox, axes_bbox: Bbox) -> float:
    dx = max(axes_bbox.x0 - bbox.x0, 0.0) + max(bbox.x1 - axes_bbox.x1, 0.0)
    dy = max(axes_bbox.y0 - bbox.y0, 0.0) + max(bbox.y1 - axes_bbox.y1, 0.0)
    return dx * dx + dy * dy


def _count_points_inside_bbox(points: np.ndarray, bbox: Bbox) -> int:
    if points.size == 0:
        return 0
    inside = (
        (points[:, 0] >= bbox.x0)
        & (points[:, 0] <= bbox.x1)
        & (points[:, 1] >= bbox.y0)
        & (points[:, 1] <= bbox.y1)
    )
    return int(inside.sum())


def _add_curve_labels(
    ax,
    data: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    hue_col: str,
    hue_order: list | None = None,
    palette=None,
    font_size: float | None = None,
    densify: int = 1,
    obstacle_curves: list | None = None,
) -> None:
    """Place readable labels near curves, avoiding label/curve collisions.

    ``obstacle_curves`` (optional): extra curves, DataFrames with ``x_col`` / ``y_col``, that labels must avoid
    but that get no label themselves (e.g. the other line style of a one-learner panel).

    A legend already on the axes is treated as an obstacle, like the other texts.
    ``densify`` > 1 adds that many interpolated points between consecutive curve
    points for the collision test, so a label cannot sit on the line between two
    integer K values (opt-in; 1 keeps the historical behaviour).
    """
    if hue_col not in data.columns or data.empty:
        return

    x_left, x_right = ax.get_xlim()
    y_bottom, y_top = ax.get_ylim()
    if x_right <= x_left or y_top <= y_bottom:
        return

    ordered_labels = list(hue_order) if hue_order is not None else []
    present_labels = [label for label in data[hue_col].dropna().unique()]
    for label in present_labels:
        if label not in ordered_labels:
            ordered_labels.append(label)

    curves = []
    fallback_colors = {
        line.get_label(): line.get_color()
        for line in ax.get_lines()
        if not str(line.get_label()).startswith("_")
    }
    for label in ordered_labels:
        sub = data[data[hue_col] == label]
        if sub.empty:
            continue
        curve = (
            sub[[x_col, y_col]]
            .dropna()
            .groupby(x_col, sort=True)[y_col]
            .mean()
            .reset_index()
        )
        curve = curve[(curve[x_col] >= x_left) & (curve[x_col] <= x_right)]
        if curve.empty:
            continue
        color = _palette_lookup(palette, label, hue_order) or fallback_colors.get(label)
        curves.append(
            {
                "label": label,
                "curve": curve,
                "color": color,
            }
        )
    if not curves:
        return

    font_size = (
        float(font_size)
        if font_size is not None
        else float(mpl.rcParams["legend.fontsize"])
    )
    if len(curves) > 8:
        font_size = max(7.0, font_size - 0.35 * (len(curves) - 8))
    font_props = FontProperties(
        family=mpl.rcParams["font.family"],
        size=font_size,
    )

    # Draw once so text and axes extents are available in display coordinates.
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    axes_bbox = ax.bbox
    existing_boxes = [
        _expanded_bbox(text.get_window_extent(renderer), 3.0)
        for text in ax.texts
    ]
    if ax.get_legend() is not None:
        existing_boxes.append(_expanded_bbox(ax.get_legend().get_window_extent(renderer), 3.0))

    all_points = []
    for item in curves:
        curve = item["curve"]
        points = ax.transData.transform(curve[[x_col, y_col]].to_numpy(dtype=float))
        if densify > 1 and len(points) > 1:
            t = np.linspace(0.0, 1.0, densify + 1)[:-1]
            seg = [a + (b - a) * t[:, None] for a, b in zip(points[:-1], points[1:])]
            points = np.vstack([*seg, points[-1:]])
        all_points.append(points)
    for extra in obstacle_curves or []:
        pts = ax.transData.transform(extra[[x_col, y_col]].dropna().to_numpy(dtype=float))
        if densify > 1 and len(pts) > 1:
            t = np.linspace(0.0, 1.0, densify + 1)[:-1]
            seg = [a + (b - a) * t[:, None] for a, b in zip(pts[:-1], pts[1:])]
            pts = np.vstack([*seg, pts[-1:]])
        all_points.append(pts)
    all_curve_points = np.vstack(all_points) if all_points else np.empty((0, 2))

    offset_scale = ax.figure.dpi / 72.0
    offsets = [
        (-7, 0, "right", "center"),
        (7, 0, "left", "center"),
        (0, 7, "center", "bottom"),
        (0, -7, "center", "top"),
        (-7, 7, "right", "bottom"),
        (7, 7, "left", "bottom"),
        (-7, -7, "right", "top"),
        (7, -7, "left", "top"),
    ]
    # Prefer endpoints and broad interior positions, but allow anywhere useful.
    fractions = [0.95, 0.80, 0.65, 0.50, 0.35, 0.20, 0.08]
    placed_boxes: list[Bbox] = []

    # Longer labels are harder to place; place them first.
    for item in sorted(curves, key=lambda cur: len(str(cur["label"])), reverse=True):
        label = str(item["label"])
        curve = item["curve"].reset_index(drop=True)
        width, height, _ = renderer.get_text_width_height_descent(
            label,
            font_props,
            ismath=mpl.cbook.is_math_text(label),
        )
        candidates = []
        if len(curve) == 1:
            point_indices = [0]
        else:
            point_indices = sorted(
                {
                    int(round(frac * (len(curve) - 1)))
                    for frac in fractions
                },
                reverse=True,
            )
        for rank, point_idx in enumerate(point_indices):
            point = curve.iloc[point_idx]
            point_xy = ax.transData.transform(
                [float(point[x_col]), float(point[y_col])]
            )
            for offset_rank, (dx_pt, dy_pt, ha, va) in enumerate(offsets):
                anchor = (
                    point_xy[0] + dx_pt * offset_scale,
                    point_xy[1] + dy_pt * offset_scale,
                )
                bbox = _text_bbox(anchor, width, height, ha, va)
                padded = _expanded_bbox(bbox, 3.0)
                label_overlap = sum(
                    _bbox_overlap_area(padded, other)
                    for other in [*existing_boxes, *placed_boxes]
                )
                curve_hits = _count_points_inside_bbox(
                    all_curve_points,
                    _expanded_bbox(bbox, 1.5),
                )
                outside = _bbox_outside_axes_penalty(padded, axes_bbox)
                score = (
                    80.0 * outside
                    + 12.0 * label_overlap
                    + 180.0 * curve_hits
                    + 7.0 * rank
                    + 1.5 * offset_rank
                )
                candidates.append((score, anchor, bbox, ha, va))
        if not candidates:
            continue
        _, anchor, bbox, ha, va = min(candidates, key=lambda cand: cand[0])
        placed_boxes.append(_expanded_bbox(bbox, 3.0))
        data_xy = ax.transData.inverted().transform(anchor)
        ax.text(
            data_xy[0],
            data_xy[1],
            label,
            color=item["color"],
            ha=ha,
            va=va,
            fontsize=font_size,
            fontfamily=mpl.rcParams["font.family"],
            path_effects=_white_text_halo(),
            clip_on=True,
        )


# ---------------------------------------------------------------------------
# plot_sample_gain_vs_splits
# ---------------------------------------------------------------------------

def plot_sample_gain_vs_splits(
    results_df: pd.DataFrame,
    y_col: Literal["sample_gain", "k_equiv"] = "k_equiv",
    hue_col: str = "model",
    hue_order: list | None = None,
    palette=None,
    xticks: list[int] | None = None,
    use_log_scale: bool = False,
    style_col: str | None = None,
    style_order: list | None = None,
    dashes: bool | dict = False,
    title_text: str | None = None,
    title_text_xy: tuple[float, float] = (0.25, 0.90),
    title_text_ha: str = "center",
    title_text_va: str = "top",
    xlim: tuple[float, float] | None = None,
    legend_loc: str | None = None,
    legend_title: str | None = None,
    direct_labels: bool = False,
    style_legend: bool = False,
    style_legend_loc: str = "lower right",
    style_legend_bbox: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    axis_label_fontsize: float = 12,
    tick_label_fontsize: float | None = None,
    title_fontsize: float | None = None,
    direct_label_fontsize: float | None = None,
    direct_label_densify: int = 1,
    errorbar=("pi", 95),
    output_path: str | None = None,
    hairlines: int | None = None,
    ylim_top: float | None = None,
) -> plt.Figure:
    """Line plot of test sample gain or k_equiv versus number of CV splits.

    The line is the mean over the bootstrap draws in *results_df*; the band is
    *errorbar* (default ``("pi", 95)``: the 2.5-97.5 % percentiles of the
    bootstrap draws, i.e. the 95 % bootstrap interval of the gain). The paper's
    figures before 2026-10-28 used ``("ci", 95)``, a confidence interval of the
    mean of the draws, which shrinks with the number of bootstrap draws.

    Parameters
    ----------
    results_df : DataFrame
        Output of ``compute.sample_gain_bootstrap`` (possibly pre-filtered).
        Must have columns ``['K', y_col, hue_col]``.
    y_col : str
        Column to plot on the y-axis.
    hue_col : str
        Column used for the hue (colour) of lines.
    hue_order : list, optional
        Explicit ordering of hue levels.
    palette : optional
        Seaborn-compatible colour palette.
    xticks : list of int, optional
        X-axis tick positions. Defaults to ``[1, 5, 10, 15, 20]``.
    use_log_scale : bool
        Apply log scale to the x-axis.
    style_col : str, optional
        Column to use for line style (e.g. ``'cv_procedure'``).
    dashes : bool or dict
        Passed to ``sns.lineplot`` as *dashes*.
    title_text : str, optional
        Text to annotate inside the figure (top-left region).
        Uses ``ax.text(*title_text_xy, ...)`` style.
    title_text_xy : tuple, optional
        Axes-coordinate position for ``title_text``.
    title_text_ha, title_text_va : str, optional
        Horizontal and vertical alignment for ``title_text``.
    xlim : tuple, optional
        Explicit x-axis limits.
    legend_loc : str, optional
        Explicit legend location. If omitted, keep Matplotlib's current
        location.
    legend_title : str, optional
        Legend title. Defaults to no title.
    direct_labels : bool, optional
        If True, remove the legend and write each hue label next to the
        corresponding curve inside the plot box.
    output_path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    plt.Figure
    """
    if xticks is None:
        xticks = [1, 5, 10, 15, 20]
    split_col = "K" if "K" in results_df.columns else "s"
    n_hair = hairlines if hairlines is not None else GAIN_DISPLAY["hairlines"]
    use_hair = bool(n_hair) and GAIN_DISPLAY["replicate_col"] in results_df.columns
    if use_hair:
        errorbar = None
    central_df, rep_df = _estimate_and_replicates(results_df)

    fig, ax = plt.subplots(figsize=figsize)

    sns.lineplot(
        data=central_df if use_hair else rep_df,
        x=split_col,
        y=y_col,
        hue=hue_col,
        hue_order=hue_order,
        style=style_col,
        style_order=style_order,
        dashes=dashes,
        palette=palette,
        markers=False,
        errorbar=errorbar,
        ax=ax,
    )

    if use_log_scale:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(xticks))
        ax.xaxis.set_major_formatter(ScalarFormatter())
    else:
        ax.set_xscale("linear")
        ax.xaxis.set_major_locator(FixedLocator(xticks))
    if xlim is not None:
        ax.set_xlim(*xlim)

    ax.set_xlabel("Number of Splits (K)", fontsize=axis_label_fontsize)
    ax.set_ylabel(
        r"Test Sample Gain ($G^{\mathrm{test}}_K$)",
        fontsize=axis_label_fontsize,
    )
    if tick_label_fontsize is not None:
        ax.tick_params(axis="both", which="major", labelsize=tick_label_fontsize)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, which="both", linestyle="--", alpha=0.7)
    sns.despine(ax=ax)

    if use_hair:
        for line in ax.get_lines():
            line.set_linewidth(GAIN_DISPLAY["mean_linewidth"])
            line.set_zorder(3)
        hue_levels = list(hue_order) if hue_order is not None else list(pd.unique(results_df[hue_col]))
        style_levels = None
        if style_col is not None and style_col in results_df.columns:
            style_levels = list(style_order) if style_order is not None else list(pd.unique(results_df[style_col]))
        _draw_hairlines(ax, rep_df, split_col, y_col, hue_col, hue_levels, _hue_colors(hue_levels, palette),
                        style_col=style_col, style_levels=style_levels, n_lines=n_hair)
        top = ylim_top if ylim_top is not None else _y_top(results_df, split_col, y_col, [hue_col, style_col])
        ax.set_ylim(top=top)
    elif ylim_top is not None:
        ax.set_ylim(top=ylim_top)

    if title_text is not None:
        # clip_on keeps the in-axes "fake title" from expanding the tight-bbox
        # crop: a long title is clipped to the axes instead of widening the
        # saved figure, so the plot area stays the same size regardless of title
        # length.
        ax.text(
            *title_text_xy, title_text,
            ha=title_text_ha,
            va=title_text_va,
            fontsize=(
                title_fontsize
                if title_fontsize is not None
                else mpl.rcParams["legend.fontsize"]
            ),
            fontfamily=mpl.rcParams["font.family"],
            path_effects=_white_text_halo(),
            transform=ax.transAxes,
            clip_on=True,
        )

    if direct_labels:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        # The style legend is drawn first so that the curve labels avoid it.
        if style_legend and style_col is not None and style_col in results_df.columns:
            style_labels = (
                list(style_order)
                if style_order is not None
                else list(results_df[style_col].dropna().unique())
            )
            handles = []
            for label in style_labels:
                handle = Line2D([0], [0], color="0.25", linewidth=1.6, label=str(label))
                dash_pattern = dashes.get(label) if isinstance(dashes, dict) else None
                if dash_pattern:
                    handle.set_dashes(dash_pattern)
                handles.append(handle)
            if handles:
                legend = ax.legend(
                    handles=handles,
                    loc=style_legend_loc,
                    bbox_to_anchor=style_legend_bbox,
                    title="",
                    frameon=False,
                )
                _style_legend_halo(legend, title=None)
        _add_curve_labels(
            ax,
            central_df,
            x_col=split_col,
            y_col=y_col,
            hue_col=hue_col,
            hue_order=hue_order,
            palette=palette,
            font_size=direct_label_fontsize,
            densify=direct_label_densify,
        )
    elif legend_loc is not None and ax.get_legend() is not None:
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(
            handles=handles,
            labels=labels,
            loc=legend_loc,
            title="" if legend_title is None else legend_title,
            frameon=False,
        )
        _style_legend_halo(legend, title=legend_title)
    else:
        _style_legend_halo(ax.get_legend(), title=legend_title)

    if output_path is not None:
        save_figure(fig, output_path, format="pdf", bbox_inches="tight")

    return fig


def plot_sample_gain_facets(
    results_df: pd.DataFrame,
    panel_col: str,
    panel_order: list,
    style_col: str,
    style_order: list,
    dashes: dict,
    palette: dict | None = None,
    y_col: str = "sample_gain",
    xticks: list[int] | None = None,
    xlim: tuple[float, float] | None = None,
    hairlines: int = 30,
    panel_size: tuple[float, float] = (2.0, 2.4),
    axis_label_fontsize: float = 10,
    output_path: str | None = None,
    ncols: int | None = None,
    axes_size: tuple[float, float] | None = None,
    margins: tuple[float, float, float, float] = (0.53, 0.51, 0.10, 0.10),
    gaps: tuple[float, float] = (0.20, 0.20),
) -> plt.Figure:
    """One panel per ``panel_col`` level (e.g. network), its ``style_col`` levels overlaid (CE, 2026-10-29).

    Same display as plot_sample_gain_vs_splits with hairlines: bold mean of all replicates per style, a
    fixed random subset of replicates as hairlines, shared y-axis cut at a high quantile of the replicates.
    The panel title is the panel level; the style legend sits in the first panel.
    """
    if xticks is None:
        xticks = [1, 5, 10, 15, 20]
    split_col = "K" if "K" in results_df.columns else "s"
    colors = _hue_colors(panel_order, palette)
    ncols = ncols or len(panel_order)
    nrows = int(np.ceil(len(panel_order) / ncols))
    if axes_size is None:
        fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols + 0.4, panel_size[1] * nrows),
                                 sharey=True, sharex=True, squeeze=False)
    else:
        # Every panel gets exactly ``axes_size`` inches, the axes of a single-panel gain figure, so that panels
        # print at the same size as those figures (CE, 2026-10-30). Margins (left, bottom, right, top) and gaps
        # (between columns, between rows) are in inches; inner tick labels are hidden, as with sharex/sharey.
        (aw, ah), (ml, mb, mr, mt), (gw, gh) = axes_size, margins, gaps
        fw, fh = ml + ncols * aw + (ncols - 1) * gw + mr, mb + nrows * ah + (nrows - 1) * gh + mt
        fig = plt.figure(figsize=(fw, fh), layout="none")
        axes = np.empty((nrows, ncols), dtype=object)
        for r in range(nrows):
            for c in range(ncols):
                x0, y0 = ml + c * (aw + gw), mb + (nrows - 1 - r) * (ah + gh)
                axes[r, c] = fig.add_axes([x0 / fw, y0 / fh, aw / fw, ah / fh],
                                          sharex=axes[0, 0] if (r, c) != (0, 0) else None,
                                          sharey=axes[0, 0] if (r, c) != (0, 0) else None)
                if c > 0:
                    axes[r, c].tick_params(labelleft=False)
    for ax in axes.ravel()[len(panel_order):]:
        ax.set_visible(False)
    left_col = [axes[r, 0] for r in range(nrows)]
    # lowest visible panel of each column carries the x tick labels and label (an empty last slot leaves the
    # panel above it at the bottom of its column)
    bottom = [axes[max(r for r in range(nrows) if r * ncols + c < len(panel_order)), c] for c in range(ncols)
              if c < len(panel_order)]
    axes = axes.ravel()[: len(panel_order)]
    top = _y_top(results_df, split_col, y_col, [panel_col, style_col])
    for ax, lv in zip(axes, panel_order):
        g = results_df[results_df[panel_col] == lv]
        _draw_hairlines(ax, g, split_col, y_col, panel_col, [lv], colors, style_col=style_col,
                        style_levels=style_order, n_lines=hairlines)
        g = _estimate_and_replicates(g)[0]          # thick line: full-sample estimate when present
        for st in style_order:
            mu = g[g[style_col] == st].groupby(split_col)[y_col].mean()
            line, = ax.plot(mu.index, mu.values, color=colors[lv], lw=GAIN_DISPLAY["mean_linewidth"], zorder=3)
            if dashes.get(st):
                line.set_dashes(dashes[st])
        ax.xaxis.set_major_locator(FixedLocator(xticks))
        ax.set_xlim(*(xlim if xlim is not None else (0, max(xticks))))
        ax.set_xlabel("Number of Splits (K)" if ax in bottom else "", fontsize=axis_label_fontsize)
        if ax not in bottom:
            ax.tick_params(labelbottom=False)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, which="both", linestyle="--", alpha=0.7)
        sns.despine(ax=ax)
        ax.set_ylim(top=top)
    for ax in left_col:
        ax.set_ylabel(r"Test Sample Gain ($G^{\mathrm{test}}_K$)", fontsize=axis_label_fontsize)
    handles = []
    for st in style_order:
        h = Line2D([0], [0], color="0.25", linewidth=1.6, label=str(st))
        if dashes.get(st):
            h.set_dashes(dashes[st])
        handles.append(h)
    legend = axes[0].legend(handles=handles, loc="upper left", title="", frameon=False)
    _style_legend_halo(legend, title=None)
    # Learner name inside each panel, in its colour, placed by the paper's label placer next to the first-style
    # curve and away from the other style's curve, the legend and other texts (CE, 2026-10-30).
    for ax, lv in zip(axes, panel_order):
        g = _estimate_and_replicates(results_df[results_df[panel_col] == lv])[0]
        first = g[g[style_col] == style_order[0]]
        others = [g[g[style_col] == st].groupby(split_col)[y_col].mean().reset_index() for st in style_order[1:]]
        _add_curve_labels(ax, first, x_col=split_col, y_col=y_col, hue_col=panel_col, hue_order=[lv],
                          palette={lv: colors[lv]}, densify=10, obstacle_curves=others)
    if output_path is not None:
        save_figure(fig, output_path, format="pdf", bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# plot_variance_diagnostic
# ---------------------------------------------------------------------------

def plot_variance_diagnostic(
    var_k_df: pd.DataFrame,
    var_cv_df: pd.DataFrame,
    target_splits: list[int],
    x_col: str = "x_axis_val",
    var_col: str = "variance",
    xlabel: str = "",
    cv_colors: list | None = None,
    yscale: str = "log",
    output_path: str | None = None,
) -> plt.Figure:
    """Variance diagnostic: no-CV curve (black) + horizontal CV reference lines.

    The caller must pre-filter *var_k_df* and *var_cv_df* to a single
    (model, study_size) slice before calling this function.

    Parameters
    ----------
    var_k_df : DataFrame
        Columns ``[x_col, var_col]`` — variance as a function of effective
        data multiplier for the no-CV path.
    var_cv_df : DataFrame
        Columns ``['n_splits', var_col]`` — one row per target split count.
    target_splits : list of int
        Split counts to draw as horizontal lines.
    x_col, var_col : str
        Column names.
    xlabel : str
        X-axis label.
    cv_colors : list, optional
        One colour per entry in *target_splits*. Defaults to mako palette.
    yscale : str
        Y-axis scale (``'log'`` or ``'linear'``).
    output_path : str, optional
        Save path.

    Returns
    -------
    plt.Figure
    """
    if cv_colors is None:
        cv_colors = sns.color_palette("mako", n_colors=len(target_splits))

    fig, ax = plt.subplots()

    sns.lineplot(
        data=var_k_df, x=x_col, y=var_col,
        marker="", label="Variance with hidden data", color="black", ax=ax,
    )

    for i, K_split in enumerate(target_splits):
        row = var_cv_df[var_cv_df["n_splits"] == K_split]
        if row.empty:
            continue
        val = float(row[var_col].values[0])
        color = cv_colors[i % len(cv_colors)]
        ax.axhline(y=val, color=color, linestyle="--", linewidth=2.5,
                   label=f"CV Variance (K={K_split})")

    if not var_k_df.empty:
        x_min = float(var_k_df[x_col].min())
        x_max = float(var_k_df[x_col].max())
        ax.set_xlim(left=x_min, right=x_max)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Variance")
    ax.set_yscale(yscale)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend()
    sns.despine(ax=ax)

    if output_path is not None:
        save_figure(fig, output_path, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# plot_pedagogical_variance_diagnostic
# ---------------------------------------------------------------------------

def plot_pedagogical_variance_diagnostic(
    var_k_df: pd.DataFrame,
    var_cv_df: pd.DataFrame,
    target_splits: list[int],
    x_col: str = "x_axis_val",
    var_col: str = "variance",
    xlabel: str = "Test multiplier",
    y_label_pos: float = 4.5e-4,
    x_lim: tuple[float, float] = (0.88, 20.0),
    y_lim_bottom: float = 5e-4,
    cv_colors: list | None = None,
    output_path: str | None = None,
) -> plt.Figure:
    """Pedagogical variance diagnostic with crossing-point annotations.

    Extends ``plot_variance_diagnostic`` by adding:
    - Thick CV dot at multiplier=1 for each reference line.
    - Vertical drop line from the CV variance level to the x-axis, placed at
      the exact x-value where the no-CV curve crosses that level (linearly
      interpolated).
    - Text annotation ``$G^{test}_{K}$`` below each drop line.

    The caller must pre-filter both DataFrames to a single (model, study_size).

    Parameters
    ----------
    var_k_df : DataFrame
        Columns ``[x_col, var_col]`` — no-CV variance curve.
    var_cv_df : DataFrame
        Columns ``['n_splits', var_col]`` — CV reference variances.
    target_splits : list of int
    x_col, var_col, xlabel : str
    y_label_pos : float
        Y-coordinate for the ``$G^{test}_{K}$`` text labels.
    x_lim : tuple
        (x_min, x_max) for ``ax.set_xlim``.
    y_lim_bottom : float
        Lower y-axis bound.
    cv_colors : list, optional
        One colour per split. Defaults to mako palette.
    output_path : str, optional

    Returns
    -------
    plt.Figure
    """
    if cv_colors is None:
        cv_colors = sns.color_palette("mako", n_colors=len(target_splits))

    fig, ax = plt.subplots(figsize=(3.25, 2))

    # No-CV curve
    sns.lineplot(
        data=var_k_df, x=x_col, y=var_col,
        linewidth=1.2, marker="", label="1 split", color="black", ax=ax,
    )

    curve_x = var_k_df[x_col].values
    curve_y = var_k_df[var_col].values

    for i, K_split in enumerate(target_splits):
        row = var_cv_df[var_cv_df["n_splits"] == K_split]
        if row.empty:
            continue
        val = float(row[var_col].values[0])
        color = cv_colors[i % len(cv_colors)]

        ax.axhline(y=val, color=color, linestyle="-", linewidth=1.2)
        # Thick dot at multiplier = 1
        ax.plot(1, val, marker="o", color=color, markersize=8,
                markeredgecolor="k", markeredgewidth=1.5, zorder=10,
                label=f"CV (K={K_split})")

        # Find interpolated crossing point(s)
        diff = curve_y - val
        sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
        for idx in sign_changes:
            x0, x1 = curve_x[idx], curve_x[idx + 1]
            y0, y1 = curve_y[idx], curve_y[idx + 1]
            x_cross = x0 if y1 == y0 else x0 + (val - y0) * (x1 - x0) / (y1 - y0)

            ax.vlines(
                x_cross, 0, val, color=color, linestyle="-",
                linewidth=1.2, alpha=0.9, zorder=5,
            )

            # Annotation: slight left offset for K=10 to avoid overlap
            x_ann = x_cross - 0.75 if K_split == 10 else x_cross
            label_str = f"$G^{{test}}_{{K={K_split}}}$"
            ax.text(
                x_ann, y_label_pos, label_str,
                color=color, fontsize=9, fontweight="bold",
                ha="center", va="top", zorder=20,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=x_lim[0], right=x_lim[1])
    ax.set_ylim(bottom=y_lim_bottom)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Estimation Error Variance")
    ax.grid(False)
    ax.legend(loc="upper right")
    sns.despine(ax=ax)

    if output_path is not None:
        save_figure(fig, output_path)

    return fig


def plot_pedagogical_variance_diagnostic_paper_style(
    var_k_df: pd.DataFrame,
    var_cv_df: pd.DataFrame,
    target_splits: list[int],
    x_col: str = "x_axis_val",
    var_col: str = "variance",
    xlabel: str = "Test-size multiplier for single split",
    x_lim: tuple[float, float] = (1.0, 20.0),
    y_lim_bottom: float | None = None,
    cv_colors: list | None = None,
    x_scale: Literal["linear", "log"] = "linear",
    output_path: str | None = None,
) -> plt.Figure:
    """Cleaner paper-style version of the pedagogical variance diagnostic."""
    if cv_colors is None:
        cv_colors = ["#4C78A8", "#59A14F", "#E15759"][:len(target_splits)]

    curve = var_k_df[[x_col, var_col]].dropna().sort_values(x_col)
    fig_width = 3.55 if x_scale == "log" else 3.2
    fig, ax = plt.subplots(figsize=(fig_width, 2.8))

    ax.plot(
        curve[x_col],
        curve[var_col],
        color="#5F5F5F",
        linewidth=2.0,
        zorder=3,
    )

    curve_x = curve[x_col].to_numpy(dtype=float)
    curve_y = curve[var_col].to_numpy(dtype=float)
    all_y = list(curve_y)
    crossings: list[tuple[int, float, float, str]] = []

    for i, K_split in enumerate(target_splits):
        row = var_cv_df[var_cv_df["n_splits"] == K_split]
        if row.empty:
            continue

        val = float(row[var_col].iloc[0])
        color = cv_colors[i % len(cv_colors)]
        all_y.append(val)

        ax.axhline(val, color=color, linestyle="--", linewidth=1.5, zorder=2)
        ax.plot(
            1, val, marker="o", markersize=5.8, color=color,
            markeredgecolor="white", markeredgewidth=0.7, zorder=6,
        )

        diff = curve_y - val
        sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
        if sign_changes.size == 0:
            continue
        idx = int(sign_changes[0])
        x0, x1 = curve_x[idx], curve_x[idx + 1]
        y0, y1 = curve_y[idx], curve_y[idx + 1]
        x_cross = x0 if y1 == y0 else x0 + (val - y0) * (x1 - x0) / (y1 - y0)
        crossings.append((K_split, x_cross, val, color))

    cv_values = [val for _, _, val, _ in crossings]
    if crossings:
        x_margin_factor = 1.10 if x_scale == "log" else 1.18
        x_right = min(
            x_lim[1],
            max(x_cross for _, x_cross, _, _ in crossings) * x_margin_factor,
        )
    else:
        x_right = x_lim[1]
    x_left = x_lim[0]

    curve_in_view = curve[(curve[x_col] >= x_left) & (curve[x_col] <= x_right)]
    visible_y = curve_in_view[var_col].dropna().to_list() + cv_values
    if not visible_y:
        visible_y = all_y

    y_min = (
        y_lim_bottom
        if y_lim_bottom is not None
        else max(min(cv_values or visible_y) * 0.70, 1e-12)
    )
    y_max = max(visible_y) * 1.35
    ax.set_yscale("log")
    if x_scale == "log":
        x_display_left = max(x_left / 1.12, 1e-12)
        ax.set_xscale("log")
    else:
        x_display_left = max(0.0, x_left - 0.18 * (x_right - x_left))
    ax.set_xlim(x_display_left, x_right)
    ax.set_ylim(y_min, y_max)

    halo = _white_text_halo()
    for K_split, x_cross, val, color in crossings:
        ax.vlines(
            x_cross, y_min, val, color=color, linestyle="-",
            linewidth=1.25, alpha=0.9, zorder=4,
        )

    cv_label_multiplier = {5: 1.12, 10: 1.00, 20: 0.90}
    for K_split, _, val, color in crossings:
        label_y = val * cv_label_multiplier.get(K_split, 1.0)
        label_x = x_right - 0.18
        label_ha = "right"
        if x_scale == "log":
            label_x = x_right / 1.04
            if K_split == 10:
                label_x = x_left * (x_right / x_left) ** 0.62
            elif K_split == 20:
                label_x = x_left * (x_right / x_left) ** 0.54
        if K_split == 20:
            if x_scale == "linear":
                first_crossing = min(x_cross for _, x_cross, _, _ in crossings)
                label_x = x_left + 0.34 * (first_crossing - x_left)
                label_y = val * 0.82
                label_ha = "center"
            else:
                label_ha = "right"
        ax.text(
            label_x,
            label_y,
            f"CV, K={K_split}",
            color=color, fontsize=8.5, ha=label_ha, va="center",
            path_effects=halo,
        )

    if curve_x.size:
        label_x = x_left + 0.38 * (x_right - x_left)
        label_y = np.interp(label_x, curve_x, curve_y)
        ax.text(
            label_x,
            label_y * 1.55,
            "Single split\n+ extra test data",
            color="#5F5F5F", fontsize=8.2, ha="center", va="bottom",
            path_effects=halo,
        )

    tick_candidates = [1, 2, 3, 5, 7, 10, 15, 20] if x_scale == "log" else [
        1, 3, 5, 7, 10, 15, 20
    ]
    major_tick_candidates = [
        tick for tick in tick_candidates
        if x_left <= tick <= x_right
    ]
    if crossings:
        crossing_positions = [x_cross for _, x_cross, _, _ in crossings]
        major_ticks = [
            tick for tick in major_tick_candidates
            if min(abs(tick - x_cross) for x_cross in crossing_positions) > 0.9
        ]
        if x_scale == "linear":
            ordered_crossings = sorted(crossing_positions)
            extra_ticks = []
            if len(ordered_crossings) >= 2:
                lower, upper = ordered_crossings[0], ordered_crossings[1]
                between_candidates = [
                    tick for tick in range(int(np.floor(lower)) + 1, int(np.ceil(upper)))
                    if lower + 0.9 < tick < upper - 0.9
                ]
                if between_candidates:
                    midpoint = 0.5 * (lower + upper)
                    extra_ticks.append(
                        min(between_candidates, key=lambda tick: abs(tick - midpoint))
                    )
            last = ordered_crossings[-1]
            after_candidates = [
                tick for tick in range(int(np.floor(last)) + 1, int(np.floor(x_right)) + 1)
                if tick > last + 0.9
            ]
            if after_candidates:
                extra_ticks.append(after_candidates[0])
            major_ticks = sorted({*major_ticks, *extra_ticks})
    else:
        major_ticks = major_tick_candidates
    ax.set_xticks(major_ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    if crossings:
        ax.xaxis.set_minor_locator(
            FixedLocator([x_cross for _, x_cross, _, _ in crossings])
        )
        ax.xaxis.set_minor_formatter(
            FixedFormatter([
                rf"$G^{{\mathrm{{test}}}}_{{{K_split}}}$"
                for K_split, _, _, _ in crossings
            ])
        )
        ax.tick_params(axis="x", which="major", pad=4)
        ax.tick_params(axis="x", which="minor", length=0, pad=4)
        for label, (_, _, _, color) in zip(ax.get_xticklabels(minor=True), crossings):
            label.set_color(color)
            label.set_fontsize(7.8 if x_scale == "log" else 8.5)
            label.set_path_effects(halo)
    ax.set_xlabel(xlabel, labelpad=7, fontsize=9.5)
    ax.set_ylabel("Estimation error variance")
    ax.grid(False)
    sns.despine(ax=ax)
    fig.tight_layout(pad=0.45)

    if output_path is not None:
        save_figure(fig, output_path, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# plot_retrieval_curve
# ---------------------------------------------------------------------------

def plot_retrieval_curve(
    curve_data: pd.DataFrame,
    cv_lines: dict[int, float],
    best_model: str,
    worst_model: str,
    target_splits: list[int] | None = None,
    cv_colors: list | None = None,
    curve_ci_data: pd.DataFrame | None = None,
    cv_ci_lines: dict[int, tuple[float, float]] | None = None,
    x_lim: tuple[float, float] = (0.5, 20.5),
    y_as_percent: bool = True,
    output_path: str | None = None,
) -> plt.Figure:
    """Plot the pairwise ranking retrieval proportion versus test multiplier.

    Shows the fraction of seeds that correctly identify the better model as a
    function of the effective test multiplier (for the no-CV path) plus
    horizontal reference lines and thick dots at multiplier=1 for each CV
    split count.

    Parameters
    ----------
    curve_data : DataFrame
        Columns ``['multiplier', 'correct_pick']``.
    cv_lines : dict
        Mapping split count → correct-pick proportion for that CV configuration.
    best_model, worst_model : str
        Used for the y-axis label.
    target_splits : list of int, optional
        Order of entries in *cv_lines*; used for consistent colouring.
        Defaults to ``sorted(cv_lines.keys())``.
    cv_colors : list, optional
        One colour per split. Defaults to mako palette.
    x_lim : tuple
        X-axis range.
    y_as_percent : bool
        Format y-axis ticks as percentages (True) or raw fractions (False).
    output_path : str, optional

    Returns
    -------
    plt.Figure
    """
    if target_splits is None:
        target_splits = sorted(cv_lines.keys())
    if cv_colors is None:
        cv_colors = sns.color_palette("mako", n_colors=len(target_splits))
    if cv_ci_lines is None:
        cv_ci_lines = {}

    fig, ax = plt.subplots(figsize=(3.2, 2.55))

    # Bootstrap ribbon for the no-CV curve.
    if curve_ci_data is not None and not curve_ci_data.empty:
        ci = curve_ci_data.sort_values("multiplier")
        ax.fill_between(
            ci["multiplier"].to_numpy(dtype=float),
            ci["ci_low"].to_numpy(dtype=float),
            ci["ci_high"].to_numpy(dtype=float),
            color="#6F6F6F", alpha=0.18, linewidth=0, zorder=1,
        )

    # No-CV curve
    sns.lineplot(
        data=curve_data, x="multiplier", y="correct_pick",
        marker=None, linewidth=2.0, color="#6F6F6F",
        legend=False, ax=ax, zorder=3,
    )

    # CV reference lines + dots
    all_vals = list(curve_data["correct_pick"])
    if curve_ci_data is not None and not curve_ci_data.empty:
        all_vals.extend(curve_ci_data["ci_low"].dropna().tolist())
        all_vals.extend(curve_ci_data["ci_high"].dropna().tolist())
    cv_label_data = []
    plotted_cv_splits = [split for split in target_splits if split in cv_lines]
    if len(plotted_cv_splits) == 3:
        cv_x_positions = dict(zip(plotted_cv_splits, [1.45, 2.05, 2.65]))
    elif len(plotted_cv_splits) > 1:
        cv_x_positions = dict(zip(
            plotted_cv_splits,
            np.linspace(1.45, 2.65, len(plotted_cv_splits)),
        ))
    else:
        cv_x_positions = {split: 1.45 for split in plotted_cv_splits}
    for i, K_split in enumerate(target_splits):
        if K_split not in cv_lines:
            continue
        val = cv_lines[K_split]
        color = cv_colors[i % len(cv_colors)]
        x_cv = cv_x_positions.get(K_split, 1.45)
        if K_split in cv_ci_lines:
            lo, hi = cv_ci_lines[K_split]
            ax.vlines(x_cv, lo, hi, color=color, linewidth=1.4, alpha=0.75, zorder=8)
            ax.plot(
                [x_cv - 0.030, x_cv + 0.030], [lo, lo], color=color,
                linewidth=1.2, alpha=0.75, zorder=8,
            )
            ax.plot(
                [x_cv - 0.030, x_cv + 0.030], [hi, hi], color=color,
                linewidth=1.2, alpha=0.75, zorder=8,
            )
            all_vals.extend([lo, hi])
        ax.axhline(y=val, color=color, linestyle="--", linewidth=1.7)
        ax.plot(x_cv, val, marker="o", markersize=6, color=color,
                markeredgecolor="white", markeredgewidth=0.7, zorder=10)
        all_vals.append(val)
        cv_label_data.append((K_split, val, color))

    # Dynamic y-axis range
    if all_vals:
        min_val = min(all_vals)
        max_val = max(all_vals)
        min_ref = min(min_val, 0.5)
        max_ref = max(max_val, 0.5)
        margin = max((max_ref - min_ref) * 0.18, 0.05)
        plot_min = max(0.0, min_ref - margin)
        plot_max = min(1.0, max_ref + margin)
        ax.set_ylim(plot_min, plot_max)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        if y_as_percent:
            ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))

        if plot_min < 0.5 < plot_max:
            ax.axhline(0.5, color="0.55", linestyle=":", linewidth=1.0, zorder=0)

    ax.set_xlim(left=x_lim[0], right=x_lim[1])
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xticklabels(["1", "5", "10", "15", "20"])
    best_label = str(best_model)
    worst_label = str(worst_model)
    ax.set_xlabel("Test-size multiplier for single split")
    ax.set_ylabel(f"P({best_label} ≻ {worst_label})")
    ax.grid(axis="y", color="0.88", linewidth=0.8)

    halo = _white_text_halo()
    if not curve_data.empty:
        x_label = min(x_lim[1] - 7.0, max(x_lim[0] + 2.0, curve_data["multiplier"].median()))
        y_label = np.interp(
            x_label,
            curve_data["multiplier"].to_numpy(dtype=float),
            curve_data["correct_pick"].to_numpy(dtype=float),
        )
        y_offset = 0.06 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.text(
            x_label + 1.5, y_label + y_offset,
            "Single split\n+ extra test data",
            color="#6F6F6F", ha="center", va="bottom", fontsize=8.8,
            path_effects=halo,
        )

    if cv_label_data:
        y_min, y_max = ax.get_ylim()
        gap = 0.032 * (y_max - y_min)
        label_offsets = {}
        for K_split, val, _ in cv_label_data:
            label_offsets[K_split] = gap
        if 10 in label_offsets:
            label_offsets[10] = -gap
        if 20 in label_offsets:
            label_offsets[20] = gap
        if 5 in label_offsets:
            label_offsets[5] = gap
        for K_split, val, color in cv_label_data:
            y_pos = min(max(val + label_offsets.get(K_split, gap), y_min + gap), y_max - gap)
            ax.text(
                x_lim[1] - 0.45, y_pos,
                f"CV, K={K_split}", color=color,
                ha="right", va="center", fontsize=8.8,
                path_effects=halo,
            )

    sns.despine(ax=ax)

    if output_path is not None:
        save_figure(fig, output_path, format="pdf", bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# plot_pareto_combined
# ---------------------------------------------------------------------------

def plot_pareto_combined(
    agg_dict: dict[str, pd.DataFrame],
    colors: dict[str, str],
    K_max: int = 200,
    x_ticks: list[int] | None = None,
    y_lim_top: float | None = None,
    annotation_text: str | None = None,
    output_path: str | None = None,
    **legacy_kwargs,
) -> plt.Figure:
    """Pareto-style cost–benefit plot with CI bands for multiple solver groups.

    Parameters
    ----------
    agg_dict : dict
        Mapping from solver label to CI DataFrame as returned by
        ``compute.agg_ci``. Must have columns
        ``['normalized_time', 'mean', 'lower', 'upper']``.
    colors : dict
        Mapping from the same labels to matplotlib color strings.
    K_max : int
        Maximum number of splits to plot. Controls axis limits and locators.
    x_ticks : list of int, optional
        Explicit x-axis ticks. If None: ``[1, 5, 10, 15, 20]`` for K_max ≤ 20,
        else default matplotlib ticks.
    y_lim_top : float, optional
        Upper y-axis limit. If None, use ``10`` for K_max ≤ 20, else automatic.
    annotation_text : str, optional
        Text for inset annotation at ``(0.25, 0.95)`` in axes fraction.
    output_path : str, optional

    Returns
    -------
    plt.Figure
    """
    if "s_max" in legacy_kwargs:
        K_max = legacy_kwargs.pop("s_max")
    if legacy_kwargs:
        raise TypeError(f"Unexpected keyword arguments: {sorted(legacy_kwargs)}")
    fig, ax = plt.subplots(figsize=(4, 3))

    for label, agg in agg_dict.items():
        if agg.empty:
            continue
        color = colors.get(label, "C0")
        marker = "o" if K_max <= 20 else ""
        linestyle = "-"
        ax.plot(
            agg["normalized_time"], agg["mean"],
            marker=marker, linestyle=linestyle, color=color, label=label,
        )
        ax.fill_between(
            agg["normalized_time"], agg["lower"], agg["upper"],
            color=color, alpha=0.12,
        )

    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.set_xlabel("Number of Splits (K)", fontsize=12)
    ax.set_ylabel(r"Test Sample Gain ($G^{\mathrm{test}}_K$)", fontsize=12)
    ax.grid(True, which="major", linestyle="--", alpha=0.7)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())

    if K_max <= 20:
        ticks = x_ticks or [1, 5, 10, 15, 20]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        top = y_lim_top if y_lim_top is not None else 10.0
        ax.set_ylim(top=top)
    else:
        ax.set_xlim(left=0, right=K_max)
        ax.set_ylim(bottom=0)

    ax.legend(loc="lower right", fontsize=7)
    sns.despine(ax=ax)

    if annotation_text is not None:
        ax.text(
            0.25, 0.95, annotation_text,
            ha="center", va="top", fontsize=10,
            transform=ax.transAxes,
        )

    if output_path is not None:
        save_figure(fig, output_path)

    return fig


# ---------------------------------------------------------------------------
# plot_score_distributions  (simulated only)
# ---------------------------------------------------------------------------

def plot_score_distributions(
    dist_df: pd.DataFrame,
    model_order: list[str],
    color_map: dict[str, str],
    dataset_formula_labels: dict[str, str],
    study_size: int = 1250,
    score_col: str = "delta",
    output_path: str | None = None,
) -> plt.Figure:
    """Boxplots of delta score per model per dataset.

    Parameters
    ----------
    dist_df : DataFrame
        Must have columns ``['dataset_name', 'Model', score_col]``.
        Already filtered to the desired ``study_size``.
    model_order : list of str
        Display name ordering.
    color_map : dict
        Mapping from display name to colour.
    dataset_formula_labels : dict
        Mapping from dataset_name to LaTeX formula string.
    study_size : int
        Used in the suptitle.
    score_col : str
        Column to plot on y-axis.
    output_path : str, optional

    Returns
    -------
    plt.Figure
    """
    datasets = sorted(d for d in dist_df["dataset_name"].unique() if pd.notna(d))
    if not datasets:
        return plt.figure()
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        cell = dist_df[dist_df["dataset_name"] == ds]
        sns.boxplot(
            data=cell,
            x="Model",
            hue="Model",
            y=score_col,
            order=model_order,
            palette=color_map,
            ax=ax,
            fliersize=2,
        )
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        formula = dataset_formula_labels.get(ds, ds)
        ax.set_title(formula, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel(
            r"$\delta$ (test $-$ bench score)" if ax is axes[0] else ""
        )
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        sns.despine(ax=ax)

    fig.suptitle(
        f"Score Distributions by Model and Dataset ($n_{{\\mathrm{{tr}}}}={int(study_size * 0.8)}$)",
        fontsize=12,
    )

    if output_path is not None:
        save_figure(fig, output_path, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# plot_tau_bar
# ---------------------------------------------------------------------------

def plot_tau_bar(
    tau_df: pd.DataFrame,
    x_col: str = "solver_name",
    hue_col: str | None = None,
    value_col: str = "tau",
    title: str | None = None,
    ylabel: str = r"$\hat{\tau}$",
    output_path: str | None = None,
    ci_cols: tuple[str, str] | None = None,
    overlay_B: bool = False,
) -> plt.Figure:
    """Grouped bar plot of τ̂ (and optionally B̂ = Var(Ē_K)) per algorithm.

    Parameters
    ----------
    tau_df : DataFrame
        Output of ``compute.compute_tau``.
    x_col : str
        Categorical x-axis column (e.g. algorithm name).
    hue_col : str, optional
        Column used for grouping bars (e.g. study size).
    value_col : str
        Column holding the primary bar heights. Defaults to ``'tau'``.
    title : str, optional
    ylabel : str
    output_path : str, optional
    ci_cols : (str, str), optional
        Column pair holding the lower/upper percentile-CI bounds for
        ``value_col``. If given, vertical error bars are drawn on each
        primary bar.
    overlay_B : bool
        If True, overlay hollow bars for ``B`` (= Var(Ē_K) at the group's
        current K) on top of the τ̂ bars, with CI whiskers from
        ``('B_lo', 'B_hi')`` if those columns are present. This makes the
        τ̂ floor vs total CV variance visible side by side.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots()

    sns.barplot(
        data=tau_df,
        x=x_col,
        y=value_col,
        hue=hue_col,
        ax=ax,
        errorbar=None,
    )

    if ci_cols is not None:
        _overlay_errorbars(ax, tau_df, x_col, value_col, ci_cols)

    if overlay_B:
        overlay_col = "B"
        overlay_ci_cols = ("B_lo", "B_hi")
        if value_col.endswith("_raw") and "B_raw" in tau_df.columns:
            overlay_col = "B_raw"
            overlay_ci_cols = ("B_raw_lo", "B_raw_hi")
        if overlay_col in tau_df.columns:
            _overlay_B_bars(ax, tau_df, x_col, value_col, overlay_col, overlay_ci_cols)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("")
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")
    if title is not None:
        ax.set_title(title, fontsize=11)
    sns.despine(ax=ax)
    if output_path is not None:
        save_figure(fig, output_path, bbox_inches="tight")

    return fig


def _match_bars_to_rows(ax: plt.Axes, tau_df: pd.DataFrame, x_col: str,
                        value_col: str) -> list[tuple[float, float, float, pd.Series]]:
    """Match each drawn bar to its dataframe row, robust to missing cells.

    Categorical x-levels sit at integer positions 0, 1, 2, ... so a bar belongs
    to the level nearest its center; within a level, the hue bars are told apart
    by height (= ``value_col``). This avoids relying on seaborn's patch order,
    which desynchronises from a fixed (hue, x) row list as soon as some (x, hue)
    cells have no data (e.g. a model present at only one study size).

    Returns ``[(x_center, height, width, row), ...]`` for matched bars.
    """
    x_levels = (list(tau_df[x_col].cat.categories)
                if hasattr(tau_df[x_col], "cat")
                else sorted(tau_df[x_col].unique()))
    x_pos = {lvl: i for i, lvl in enumerate(x_levels)}

    bars = []
    for patch in ax.patches:
        if not hasattr(patch, "get_x"):
            continue
        w, h = patch.get_width(), patch.get_height()
        if not w or h != h:  # skip zero-width placeholders / NaN heights
            continue
        bars.append((patch.get_x() + w / 2, h, w))

    matched, used = [], set()
    for _, row in tau_df.iterrows():
        v = row.get(value_col)
        cx = x_pos.get(row[x_col])
        if cx is None or pd.isna(v):
            continue
        best, best_d = None, None
        for j, (bx, bh, _bw) in enumerate(bars):
            if j in used or abs(bx - cx) >= 0.5:
                continue
            d = abs(bh - v)
            if best_d is None or d < best_d:
                best, best_d = j, d
        if best is None:
            continue
        used.add(best)
        matched.append((*bars[best], row))
    return matched


def _overlay_errorbars(ax, tau_df, x_col, value_col, ci_cols):
    lo_col, hi_col = ci_cols
    if lo_col not in tau_df.columns or hi_col not in tau_df.columns:
        return
    for xc, h, _w, row in _match_bars_to_rows(ax, tau_df, x_col, value_col):
        lo, hi = row.get(lo_col), row.get(hi_col)
        if pd.isna(lo) or pd.isna(hi):
            continue
        ax.errorbar(xc, h, yerr=[[max(0.0, h - lo)], [max(0.0, hi - h)]],
                    fmt="none", ecolor="black", elinewidth=0.8,
                    capsize=2.5, zorder=5)


def _overlay_B_bars(
    ax,
    tau_df,
    x_col,
    value_col,
    b_col: str = "B",
    ci_cols: tuple[str, str] = ("B_lo", "B_hi"),
):
    """Draw hollow outlines at height B on top of the matched τ̂ bars."""
    handles_added = False
    lo_col, hi_col = ci_cols
    for xc, _h, w, row in _match_bars_to_rows(ax, tau_df, x_col, value_col):
        B_val = row.get(b_col)
        if pd.isna(B_val):
            continue
        ax.bar(xc, B_val, width=w, fill=False, edgecolor="black",
               linewidth=1.1, zorder=4,
               label=r"$\hat B = \widehat{\mathrm{Var}}(\bar E_K)$"
                     if not handles_added else None)
        handles_added = True
        lo, hi = row.get(lo_col), row.get(hi_col)
        if not (pd.isna(lo) or pd.isna(hi)):
            ax.errorbar(xc, B_val,
                        yerr=[[max(0.0, B_val - lo)], [max(0.0, hi - B_val)]],
                        fmt="none", ecolor="black", elinewidth=0.7,
                        capsize=2.0, alpha=0.6, zorder=5)
    if handles_added:
        # Merge into legend without duplicating existing entries
        h, l = ax.get_legend_handles_labels()
        ax.legend(h, l, fontsize=8)


def _spearmanr_with_fallback(x, y) -> tuple[float, float]:
    """Compute Spearman correlation, falling back when scipy is unavailable."""
    try:
        from scipy.stats import spearmanr
    except ImportError:
        x_rank = pd.Series(x).rank(method="average")
        y_rank = pd.Series(y).rank(method="average")
        return float(x_rank.corr(y_rank, method="pearson")), float("nan")
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


# ---------------------------------------------------------------------------
# plot_tau_vs_gain_scatter
# ---------------------------------------------------------------------------

def plot_tau_vs_gain_scatter(
    df: pd.DataFrame,
    x_col: str = "tau",
    y_col: str = "sample_gain",
    hue_col: str | None = "pipeline",
    style_col: str | None = "cv_design",
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str = r"$\hat{\tau}$",
    ylabel: str = r"Sample gain at $K=20$",
    yscale: Literal["linear", "log"] = "linear",
    annotate_spearman: bool = True,
    ols_line: bool = True,
    output_path: str | None = None,
) -> plt.Figure:
    """Scatter of τ̂ vs sample gain, pooled across pipelines.

    Drops rows with NaN in ``x_col`` or ``y_col`` before plotting. Adds a
    Spearman-ρ annotation (plus p-value and n) and an OLS trend line
    fitted to the remaining points.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    data = df.dropna(subset=[x_col, y_col]).copy()

    if not data.empty:
        sns.scatterplot(
            data=data,
            x=x_col,
            y=y_col,
            hue=hue_col,
            style=style_col if (style_col and style_col in data.columns) else None,
            ax=ax,
            s=60,
            edgecolor="white",
            linewidth=0.5,
        )

        if ols_line and len(data) >= 2:
            xs = data[x_col].to_numpy(dtype=float)
            ys = data[y_col].to_numpy(dtype=float)
            slope, intercept = np.polyfit(xs, ys, 1)
            x_line = np.linspace(xs.min(), xs.max(), 50)
            ax.plot(x_line, slope * x_line + intercept,
                    color="black", linestyle="--", linewidth=1.0, zorder=0)

        if annotate_spearman and len(data) >= 3:
            rho, p = _spearmanr_with_fallback(data[x_col], data[y_col])
            ax.text(
                0.03, 0.97,
                rf"Spearman $\rho={rho:.2f}$, $p={p:.1g}$ ($n={len(data)}$)",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2),
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    if title is not None:
        ax.set_title(title, fontsize=11)
    sns.despine(ax=ax)

    if output_path is not None:
        save_figure(fig, output_path, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# plot_tau_vs_gain_encoded
# ---------------------------------------------------------------------------

_MARKER_POOL = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p"]


def _algorithm_family(name: str) -> str:
    """Strip bracketed hyperparameters and common HPO suffixes from an algo name."""
    import re
    s = re.sub(r"\[.*?\]$", "", str(name)).strip()
    s = re.sub(r"_(noHPO|HPO|RS\d*|GS\d*)$", "", s)
    return s


def plot_tau_vs_gain_encoded(
    df: pd.DataFrame,
    y_col: str,
    x_col: str = "tau",
    dataset_col: str = "dataset",
    algorithm_col: str = "algorithm",
    size_col: str = "study_size",
    size_label: str | None = None,
    sat_col: str | None = "mean_bench_score",
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str = r"$\hat{\tau}$",
    ylabel: str = r"Sample gain at $K=20$",
    algorithm_colors: dict | None = None,
    annotate_spearman: bool = True,
    ols_line: bool = True,
    output_path: str | None = None,
) -> plt.Figure:
    """τ̂ vs sample-gain scatter with an explicit visual encoding.

    Encodings:
      - colour hue → algorithm family (one base hue per algorithm)
      - colour saturation → ``sat_col`` (normalized within each algorithm),
        or uniform if ``sat_col is None``
      - marker  → ``dataset_col``
      - size    → ``size_col`` (log-scaled)

    Rows with NaN in any of the encoding columns are dropped.
    """
    import colorsys
    from matplotlib.colors import to_rgb
    from matplotlib.lines import Line2D
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
    else:
        fig = ax.figure

    required = [x_col, y_col, algorithm_col, size_col, dataset_col]
    if sat_col is not None:
        required.append(sat_col)
    data = df.dropna(subset=required).copy().reset_index(drop=True)
    if data.empty:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title, fontsize=11)
        return fig

    # -- colour: one hue per algorithm family --
    data["_family"] = data[algorithm_col].apply(_algorithm_family)
    families = sorted(data["_family"].unique())
    tab_palette = sns.color_palette("tab10", n_colors=max(10, len(families)))
    if algorithm_colors is None:
        algorithm_colors = {f: tab_palette[i % len(tab_palette)]
                            for i, f in enumerate(families)}
    algorithm_colors = dict(algorithm_colors)

    # -- marker: one shape per dataset --
    datasets = sorted(data[dataset_col].unique(), key=str)
    dataset_marker = {d: _MARKER_POOL[i % len(_MARKER_POOL)]
                      for i, d in enumerate(datasets)}

    # -- size: log-scaled study_size --
    sizes = data[size_col].to_numpy(dtype=float)
    log_sizes = np.log(sizes)
    s_min, s_max = 40.0, 260.0
    if log_sizes.max() > log_sizes.min():
        norm_sz = (log_sizes - log_sizes.min()) / (log_sizes.max() - log_sizes.min())
    else:
        norm_sz = np.full_like(log_sizes, 0.5)
    data["_marker_size"] = s_min + (s_max - s_min) * norm_sz

    # -- saturation: normalized within algorithm family --
    SAT_LO, SAT_HI = 0.30, 1.00

    def _norm_within(x: pd.Series) -> pd.Series:
        if x.max() > x.min():
            return (x - x.min()) / (x.max() - x.min())
        return pd.Series(1.0, index=x.index)

    if sat_col is not None:
        data["_sat"] = SAT_LO + (SAT_HI - SAT_LO) * (
            data.groupby("_family")[sat_col].transform(_norm_within)
        )
    else:
        data["_sat"] = SAT_HI

    def _apply_sat(base_rgb, sat):
        h, _, v = colorsys.rgb_to_hsv(*base_rgb)
        r, g, b = colorsys.hsv_to_rgb(h, float(sat), v)
        return (r, g, b, 1.0)

    # -- draw: group by (family, dataset) to pair colour + marker --
    draw_order = data.sort_values("_marker_size", ascending=False)
    for (fam, ds), sub in draw_order.groupby(["_family", dataset_col], sort=False):
        base_rgb = to_rgb(algorithm_colors.get(fam, "#444444"))
        colors = [_apply_sat(base_rgb, s) for s in sub["_sat"]]
        ax.scatter(
            sub[x_col], sub[y_col],
            s=sub["_marker_size"].to_numpy(),
            c=colors,
            marker=dataset_marker[ds],
            edgecolor="black",
            linewidth=0.4,
        )

    if ols_line and len(data) >= 2:
        xs = data[x_col].to_numpy(dtype=float)
        ys = data[y_col].to_numpy(dtype=float)
        slope, intercept = np.polyfit(xs, ys, 1)
        x_line = np.linspace(xs.min(), xs.max(), 50)
        ax.plot(x_line, slope * x_line + intercept,
                color="black", linestyle="--", linewidth=1.0, zorder=0)

    if annotate_spearman and len(data) >= 3:
        rho, p = _spearmanr_with_fallback(data[x_col], data[y_col])
        ax.text(
            0.03, 0.97,
            rf"Spearman $\rho={rho:.2f}$, $p={p:.1g}$ ($n={len(data)}$)",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2),
        )

    # -- legend: algorithm (colour), dataset (marker), study_size (size) --
    algo_handles = [
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor=algorithm_colors.get(f, "#444444"),
               markeredgecolor="black", markersize=8, label=f)
        for f in families
    ]
    dataset_handles = [
        Line2D([0], [0], marker=dataset_marker[d], linestyle="",
               markerfacecolor="lightgray", markeredgecolor="black",
               markersize=8, label=str(d))
        for d in datasets
    ]

    unique_sizes = np.unique(sizes)
    if len(unique_sizes) <= 3:
        size_ticks = list(unique_sizes)
    else:
        size_ticks = [unique_sizes[0],
                      unique_sizes[len(unique_sizes) // 2],
                      unique_sizes[-1]]

    def _marker_px(t):
        ln = np.log(t)
        if log_sizes.max() > log_sizes.min():
            n = (ln - log_sizes.min()) / (log_sizes.max() - log_sizes.min())
        else:
            n = 0.5
        return float(np.sqrt(s_min + (s_max - s_min) * n))

    size_handles = [
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor="lightgray", markeredgecolor="black",
               markersize=_marker_px(t), label=f"{int(t)}")
        for t in size_ticks
    ]

    leg_algo = ax.legend(handles=algo_handles, title="algorithm",
                         loc="upper left", bbox_to_anchor=(1.02, 1.0),
                         frameon=False, fontsize=8, title_fontsize=9)
    leg_dataset = ax.legend(handles=dataset_handles, title="dataset",
                            loc="upper left", bbox_to_anchor=(1.02, 0.60),
                            frameon=False, fontsize=8, title_fontsize=9)
    ax.legend(handles=size_handles, title=(size_label or size_col),
              loc="upper left", bbox_to_anchor=(1.02, 0.30),
              frameon=False, fontsize=8, title_fontsize=9)
    ax.add_artist(leg_algo)
    ax.add_artist(leg_dataset)

    if sat_col is not None:
        ax.text(
            1.02, 0.05,
            f"saturation ∝ {sat_col}\n(normalised within algorithm)",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, fontsize=11)
    sns.despine(ax=ax)

    if output_path is not None:
        save_figure(fig, output_path, bbox_inches="tight")

    return fig  # end plot_tau_vs_gain_encoded


# ---------------------------------------------------------------------------
# plot_pairplot_tau_gain
# ---------------------------------------------------------------------------

def plot_pairplot_tau_gain(
    df: pd.DataFrame,
    vars_: list[str],
    hue_col: str = "pipeline",
    output_path: str | None = None,
) -> sns.PairGrid:
    """Seaborn pairplot of ANOVA / sample-gain columns.

    Diagonal: KDE; off-diagonal: scatter.  Drops NaN rows for the
    selected variables before plotting.
    """
    data = df.dropna(subset=vars_).copy()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The figure layout has changed to tight",
            category=UserWarning,
            module="seaborn.axisgrid",
        )
        g = sns.pairplot(
            data, vars=vars_, hue=hue_col,
            diag_kind="kde", plot_kws=dict(s=30, alpha=0.6, edgecolor="none"),
            height=2.2,
        )
    g.figure.suptitle("Pairplot: ANOVA components vs sample gain", y=1.02)
    if output_path is not None:
        save_figure(g, output_path, bbox_inches="tight")
    return g


# ---------------------------------------------------------------------------
# plot_tau_vs_gain_studysize  (continuous study_size colormap)
# ---------------------------------------------------------------------------

def plot_tau_vs_gain_studysize(
    df: pd.DataFrame,
    x_col: str = "tau_raw_over_sigma2_raw",
    y_col: str = "sample_gain_K20_delta",
    size_col: str = "study_size",
    size_label: str | None = None,
    dataset_col: str = "dataset",
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = r"Sample gain at $K=20$",
    output_path: str | None = None,
) -> plt.Figure:
    """Scatter colored by study_size (viridis), marker by dataset."""
    from matplotlib.lines import Line2D

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
    else:
        fig = ax.figure

    data = df.dropna(subset=[x_col, y_col, size_col, dataset_col]).copy()
    if data.empty:
        return fig

    datasets = sorted(data[dataset_col].unique(), key=str)
    dataset_marker = {d: _MARKER_POOL[i % len(_MARKER_POOL)]
                      for i, d in enumerate(datasets)}

    sizes_val = data[size_col].to_numpy(dtype=float)
    norm = plt.Normalize(vmin=sizes_val.min(), vmax=sizes_val.max())
    cmap = plt.cm.viridis

    for ds, sub in data.groupby(dataset_col, sort=False):
        ax.scatter(
            sub[x_col], sub[y_col],
            c=sub[size_col].to_numpy(dtype=float),
            cmap=cmap, norm=norm,
            marker=dataset_marker[ds],
            s=70, edgecolor="black", linewidth=0.4,
        )

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
    cbar.set_label(size_label or size_col, fontsize=9)

    if len(data) >= 3:
        rho, p = _spearmanr_with_fallback(data[x_col], data[y_col])
        ax.text(
            0.03, 0.97,
            rf"Spearman $\rho={rho:.2f}$, $p={p:.1g}$ ($n={len(data)}$)",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2),
        )

    if len(data) >= 2:
        xs, ys = data[x_col].to_numpy(float), data[y_col].to_numpy(float)
        slope, intercept = np.polyfit(xs, ys, 1)
        x_line = np.linspace(xs.min(), xs.max(), 50)
        ax.plot(x_line, slope * x_line + intercept,
                color="black", linestyle="--", linewidth=1.0, zorder=0)

    dataset_handles = [
        Line2D([0], [0], marker=dataset_marker[d], linestyle="",
               markerfacecolor="lightgray", markeredgecolor="black",
               markersize=8, label=str(d))
        for d in datasets
    ]
    ax.legend(handles=dataset_handles, title="dataset",
              loc="upper left", bbox_to_anchor=(1.15, 1.0),
              frameon=False, fontsize=8, title_fontsize=9)

    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, fontsize=11)
    sns.despine(ax=ax)

    if output_path is not None:
        save_figure(fig, output_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# plot_predicted_vs_observed_gain
# ---------------------------------------------------------------------------

def plot_predicted_vs_observed_gain(
    df: pd.DataFrame,
    x_col: str = "predicted_sample_gain_delta_K20",
    y_col: str = "sample_gain_K20_delta",
    dataset_col: str = "dataset",
    algorithm_col: str = "algorithm",
    size_col: str = "study_size",
    sat_col: str | None = "mean_bench_score",
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str = r"Predicted gain",
    ylabel: str = r"Observed sample gain at $K=20$",
    algorithm_colors: dict | None = None,
    output_path: str | None = None,
) -> plt.Figure:
    """Predicted vs observed sample gain with a y = x diagonal.

    Uses the same visual encoding as ``plot_tau_vs_gain_encoded``
    (colour = algorithm family, marker = dataset, size = study_size,
    saturation = ``sat_col`` or uniform if ``sat_col is None``).
    """
    import colorsys
    from matplotlib.colors import to_rgb
    from matplotlib.lines import Line2D

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
    else:
        fig = ax.figure

    required = [x_col, y_col, algorithm_col, size_col, dataset_col]
    if sat_col is not None and sat_col in df.columns:
        required.append(sat_col)
    data = df.dropna(subset=required).copy().reset_index(drop=True)
    if data.empty:
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        return fig

    # -- algorithm colours --
    data["_family"] = data[algorithm_col].apply(_algorithm_family)
    families = sorted(data["_family"].unique())
    tab_palette = sns.color_palette("tab10", n_colors=max(10, len(families)))
    if algorithm_colors is None:
        algorithm_colors = {f: tab_palette[i % len(tab_palette)]
                            for i, f in enumerate(families)}
    algorithm_colors = dict(algorithm_colors)

    # -- markers per dataset --
    datasets = sorted(data[dataset_col].unique(), key=str)
    dataset_marker = {d: _MARKER_POOL[i % len(_MARKER_POOL)]
                      for i, d in enumerate(datasets)}

    # -- size --
    sizes = data[size_col].to_numpy(dtype=float)
    log_sizes = np.log(sizes)
    s_min, s_max = 40.0, 260.0
    if log_sizes.max() > log_sizes.min():
        norm_sz = (log_sizes - log_sizes.min()) / (log_sizes.max() - log_sizes.min())
    else:
        norm_sz = np.full_like(log_sizes, 0.5)
    data["_marker_size"] = s_min + (s_max - s_min) * norm_sz

    # -- saturation --
    SAT_LO, SAT_HI = 0.30, 1.00
    if sat_col is not None and sat_col in data.columns:
        def _norm_within(x):
            return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else pd.Series(1.0, index=x.index)
        data["_sat"] = SAT_LO + (SAT_HI - SAT_LO) * data.groupby("_family")[sat_col].transform(_norm_within)
    else:
        data["_sat"] = SAT_HI

    def _apply_sat(base_rgb, sat):
        h, _, v = colorsys.rgb_to_hsv(*base_rgb)
        return colorsys.hsv_to_rgb(h, float(sat), v) + (1.0,)

    # -- diagonal y = x --
    lo = min(data[x_col].min(), data[y_col].min())
    hi = max(data[x_col].max(), data[y_col].max())
    margin = (hi - lo) * 0.05
    diag = np.linspace(lo - margin, hi + margin, 50)
    ax.plot(diag, diag, color="gray", linestyle="--", linewidth=1.0, zorder=0, label="y = x")

    # -- scatter --
    draw_order = data.sort_values("_marker_size", ascending=False)
    for (fam, ds), sub in draw_order.groupby(["_family", dataset_col], sort=False):
        base_rgb = to_rgb(algorithm_colors.get(fam, "#444444"))
        colors = [_apply_sat(base_rgb, s) for s in sub["_sat"]]
        ax.scatter(
            sub[x_col], sub[y_col],
            s=sub["_marker_size"].to_numpy(),
            c=colors,
            marker=dataset_marker[ds],
            edgecolor="black", linewidth=0.4,
        )

    # -- Spearman + R² annotation --
    if len(data) >= 3:
        xs, ys = data[x_col].to_numpy(float), data[y_col].to_numpy(float)
        rho, p = _spearmanr_with_fallback(xs, ys)
        ss_res = np.sum((ys - xs) ** 2)
        ss_tot = np.sum((ys - ys.mean()) ** 2)
        r2_diag = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        ax.text(
            0.03, 0.97,
            rf"Spearman $\rho={rho:.2f}$, $p={p:.1g}$ ($n={len(data)}$)"
            f"\n$R^2_{{\\mathrm{{diag}}}} = {r2_diag:.2f}$",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2),
        )

    # -- legend --
    algo_handles = [
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor=algorithm_colors.get(f, "#444444"),
               markeredgecolor="black", markersize=8, label=f)
        for f in families
    ]
    dataset_handles = [
        Line2D([0], [0], marker=dataset_marker[d], linestyle="",
               markerfacecolor="lightgray", markeredgecolor="black",
               markersize=8, label=str(d))
        for d in datasets
    ]
    leg_algo = ax.legend(handles=algo_handles, title="algorithm",
                         loc="upper left", bbox_to_anchor=(1.02, 1.0),
                         frameon=False, fontsize=8, title_fontsize=9)
    ax.legend(handles=dataset_handles, title="dataset",
              loc="upper left", bbox_to_anchor=(1.02, 0.60),
              frameon=False, fontsize=8, title_fontsize=9)
    ax.add_artist(leg_algo)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, fontsize=11)
    sns.despine(ax=ax)

    if output_path is not None:
        save_figure(fig, output_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# plot_replicability_curves  (significance-decision replicability, rebuttal)
# ---------------------------------------------------------------------------

def plot_replicability_curves(
    rep_df: pd.DataFrame,
    power_fpr_df: pd.DataFrame,
    pair: str,
    N: int,
    *,
    alpha: float = 0.05,
    dataset_label: str = "",
    output_path: str | None = None,
) -> plt.Figure:
    """Three-panel significance-decision curves vs. the split count K.

    Panel 1: replicability Rep(K) (agreement of the K-split significance decision
    with the oracle decision). Panel 2: Power(K) if the oracle is significant,
    else the oracle-null FPR(K). Panel 3: the exact self-null FPR(K) with an
    ``alpha`` reference line. Solid = corrected resampled t-test, dashed =
    standard paired t-test.
    """
    r = rep_df[(rep_df["pair"] == pair) & (rep_df["N"] == N)]
    pf = power_fpr_df[(power_fpr_df["pair"] == pair) & (power_fpr_df["N"] == N)]
    oracle_dec = r["oracle_decision"].iloc[0] if not r.empty else "?"
    oracle_sig = bool(r["oracle_significant"].iloc[0]) if not r.empty else False

    line_color = "#4C78A8"   # Nadeau-Bengio corrected resampled t-test

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))

    def _line(ax, data, ycol):
        s = data[data["test_type"] == "corrected"].sort_values("K")
        s = s[np.isfinite(s[ycol])]
        if not s.empty:
            ax.plot(s["K"], s[ycol], color=line_color, linewidth=1.9)
        ax.set_xlabel("Number of Splits (K)")
        sns.despine(ax=ax)
        ax.grid(True, linestyle="--", alpha=0.5)

    # Panel 1 — replicability
    _line(axes[0], r, "rep")
    axes[0].set_ylabel("Replicability  Rep$(K)$")
    axes[0].set_ylim(0, 1.02)

    # Panel 2 — Power (if oracle significant) or oracle-null FPR
    if oracle_sig:
        _line(axes[1], pf, "power")
        axes[1].set_ylabel(r"Power$(K)$")
    else:
        _line(axes[1], pf, "fpr_oracle")
        axes[1].axhline(alpha, color="0.4", linestyle=":", linewidth=1.0)
        axes[1].set_ylabel(r"FPR$(K)$ (oracle non-sig.)")
    axes[1].set_ylim(0, 1.02)

    # Panel 3 — exact self-null FPR
    _line(axes[2], pf, "fpr_self")
    axes[2].axhline(alpha, color="0.4", linestyle=":", linewidth=1.0,
                    label=rf"$\alpha={alpha:g}$")
    axes[2].set_ylabel(r"FPR$(K)$ (self-null $\Delta=0$)")
    axes[2].set_ylim(0, max(0.2, float(np.nanmax(pf["fpr_self"])) * 1.1
                            if pf["fpr_self"].notna().any() else 0.2))
    axes[2].legend(fontsize=8, frameon=False, loc="upper left")

    title = f"{dataset_label + ': ' if dataset_label else ''}{pair}, $n_{{\\mathrm{{tr}}}}={N}$"
    title += f"  (oracle: {oracle_dec})"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    if output_path is not None:
        save_figure(fig, output_path, format="pdf", bbox_inches="tight")
    return fig
