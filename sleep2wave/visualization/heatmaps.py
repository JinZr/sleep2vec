from __future__ import annotations

import typing as t

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from sleep2wave.visualization.theme import (
    _FIGURE_BG,
    _OPENAI_BLUE_CMAP,
    _TEXT_COLOR,
    apply_plot_layout,
    pick_mono_font_family,
    style_plot_text,
    use_openai_like_theme,
)


def _style_heatmap_annotations(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    formatter: t.Callable[[float], str],
    annotation_texts: np.ndarray | None,
    threshold_ratio: float,
    fontsize: int,
    fontweight: str,
) -> None:
    threshold = threshold_ratio * float(matrix.max()) if matrix.size and matrix.max() > 0 else 0.0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = float(matrix[row_idx, col_idx])
            text = str(annotation_texts[row_idx, col_idx]) if annotation_texts is not None else formatter(value)
            ax.text(
                col_idx,
                row_idx,
                text,
                ha="center",
                va="center",
                fontfamily=pick_mono_font_family(),
                fontsize=fontsize,
                fontweight=fontweight,
                color="#FFFFFF" if value > threshold else _TEXT_COLOR,
            )


def _style_heatmap_colorbar(
    cbar: mpl.colorbar.Colorbar,
    *,
    max_value: float,
    title: str,
    integer_ticks: bool,
) -> None:
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, colors=_TEXT_COLOR, labelsize=11, pad=6)
    if integer_ticks:
        integer_max = int(max_value)
        if integer_max <= 8:
            cbar.set_ticks(np.arange(0, integer_max + 1, 1))
    cbar.ax.set_title(title, color=_TEXT_COLOR, fontsize=12, pad=12, loc="left")


def render_matrix_heatmap(
    matrix: np.ndarray,
    x_labels: t.Sequence[str],
    y_labels: t.Sequence[str],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    cmap: str | mpl.colors.Colormap = _OPENAI_BLUE_CMAP,
    norm: mpl.colors.Normalize | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float],
    annotation_formatter: t.Callable[[float], str],
    annotation_texts: np.ndarray | None = None,
    annotation_threshold_ratio: float = 0.58,
    annotation_fontsize: int = 12,
    annotation_fontweight: str = "bold",
    colorbar_title: str = "Value",
    integer_colorbar_ticks: bool = False,
    colorbar_fraction: float = 0.05,
    colorbar_pad: float = 0.025,
    colorbar_shrink: float = 0.9,
    minor_grid_color: str = "#F3F5F9",
    minor_grid_linewidth: float = 1.6,
    subplots_adjust: dict[str, float] | None = None,
) -> plt.Figure:
    use_openai_like_theme()

    mat = np.array(matrix, copy=True)
    fig, ax = plt.subplots(figsize=figsize, facecolor=_FIGURE_BG)

    image_kwargs: dict[str, t.Any] = {
        "cmap": cmap,
        "interpolation": "nearest",
        "aspect": "equal",
    }
    if norm is not None:
        image_kwargs["norm"] = norm
    else:
        image_kwargs["vmin"] = vmin
        image_kwargs["vmax"] = vmax
    image = ax.imshow(mat, **image_kwargs)

    num_rows, num_cols = mat.shape
    style_plot_text(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    ax.set_xticks(np.arange(num_cols), labels=list(x_labels))
    ax.set_yticks(np.arange(num_rows), labels=list(y_labels))
    ax.tick_params(axis="x", rotation=0, length=0, pad=8, labelsize=11)
    ax.tick_params(axis="y", rotation=0, length=0, pad=8, labelsize=11)

    ax.set_xticks(np.arange(-0.5, num_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_rows, 1), minor=True)
    ax.grid(which="minor", color=minor_grid_color, linestyle="-", linewidth=minor_grid_linewidth)
    ax.tick_params(which="minor", bottom=False, left=False)

    _style_heatmap_annotations(
        ax,
        mat,
        formatter=annotation_formatter,
        annotation_texts=annotation_texts,
        threshold_ratio=annotation_threshold_ratio,
        fontsize=annotation_fontsize,
        fontweight=annotation_fontweight,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    max_value = float(np.max(mat)) if mat.size else 0.0
    cbar = fig.colorbar(image, ax=ax, fraction=colorbar_fraction, pad=colorbar_pad, shrink=colorbar_shrink)
    _style_heatmap_colorbar(
        cbar,
        max_value=max_value,
        title=colorbar_title,
        integer_ticks=integer_colorbar_ticks,
    )

    apply_plot_layout(
        fig,
        defaults={"left": 0.16, "right": 0.89, "bottom": 0.16, "top": 0.89},
        subplots_adjust=subplots_adjust,
    )
    return fig


__all__ = ["render_matrix_heatmap"]
