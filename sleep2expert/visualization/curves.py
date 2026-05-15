from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from sleep2expert.visualization.theme import (
    _FIGURE_BG,
    _PRIMARY,
    _PRIMARY_DARK,
    _PRIMARY_PALE,
    add_openai_legend,
    apply_plot_layout,
    style_axes,
    style_plot_text,
    use_openai_like_theme,
)


def render_binary_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    *,
    roc_auc: float,
    title: str,
    xlabel: str = "False Positive Rate",
    ylabel: str = "True Positive Rate",
    figsize: tuple[float, float] = (7.2, 6.7),
    subplots_adjust: dict[str, float] | None = None,
) -> plt.Figure:
    use_openai_like_theme()

    fig, ax = plt.subplots(figsize=figsize, facecolor=_FIGURE_BG)
    style_axes(ax, show_grid=True)

    ax.plot(
        fpr,
        tpr,
        color=_PRIMARY_DARK,
        linewidth=2.6,
        label=f"ROC Curve (AUC = {roc_auc:.3f})",
        zorder=3,
    )
    ax.fill_between(fpr, tpr, 0.0, color=_PRIMARY_PALE, alpha=0.4, zorder=2)
    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle=(0, (3, 5)),
        color=_PRIMARY,
        linewidth=1.8,
        alpha=0.95,
        label="Chance",
        zorder=1,
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_box_aspect(1)
    style_plot_text(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    add_openai_legend(
        ax,
        loc="lower right",
        bbox_to_anchor=(0.97, 0.05),
    )

    apply_plot_layout(
        fig,
        defaults={"left": 0.16, "right": 0.96, "bottom": 0.15, "top": 0.90},
        subplots_adjust=subplots_adjust,
    )
    return fig


__all__ = ["render_binary_roc_curve"]
