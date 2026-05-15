from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from sleep2vec2.visualization.theme import (
    _FIGURE_BG,
    _PRIMARY,
    _PRIMARY_DARK,
    _PRIMARY_LIGHT,
    apply_plot_layout,
    style_axes,
    style_plot_text,
    use_openai_like_theme,
)


def render_prediction_scatter(
    targets: np.ndarray,
    preds: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: tuple[float, float] = (7.1, 6.7),
    subplots_adjust: dict[str, float] | None = None,
) -> plt.Figure:
    use_openai_like_theme()

    targets = np.asarray(targets, dtype=np.float32).reshape(-1)
    preds = np.asarray(preds, dtype=np.float32).reshape(-1)

    bounds = np.concatenate([targets, preds], axis=0)
    lower = float(bounds.min())
    upper = float(bounds.max())
    if lower == upper:
        lower -= 1.0
        upper += 1.0
    padding = max(0.04 * (upper - lower), 1e-6)
    lower -= padding
    upper += padding

    fig, ax = plt.subplots(figsize=figsize, facecolor=_FIGURE_BG)
    style_axes(ax, show_grid=True)

    marker_size = 92 if len(targets) <= 24 else 58
    ax.scatter(
        targets,
        preds,
        s=marker_size,
        facecolors=_PRIMARY_LIGHT,
        edgecolors=_PRIMARY_DARK,
        linewidth=1.4,
        alpha=0.98,
        zorder=3,
    )
    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle=(0, (3, 5)),
        color=_PRIMARY,
        linewidth=2.2,
        alpha=0.95,
        zorder=2,
    )

    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_box_aspect(1)
    style_plot_text(ax, title=title, xlabel=xlabel, ylabel=ylabel)

    use_integer_ticks = bool(np.allclose(bounds, np.round(bounds), atol=1e-6))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=use_integer_ticks))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=use_integer_ticks))

    apply_plot_layout(
        fig,
        defaults={"left": 0.15, "right": 0.97, "bottom": 0.15, "top": 0.90},
        subplots_adjust=subplots_adjust,
    )
    return fig


__all__ = ["render_prediction_scatter"]
