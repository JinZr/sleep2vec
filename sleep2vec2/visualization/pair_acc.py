from __future__ import annotations

import typing as t

from matplotlib.colors import PowerNorm
import matplotlib.pyplot as plt
import numpy as np

from sleep2vec2.visualization.heatmaps import add_axis_title_boxes, render_matrix_heatmap
from sleep2vec2.visualization.theme import _OPENAI_BLUE_CMAP


def render_pair_acc_heatmap(
    matrix: np.ndarray,
    modality_names: t.Sequence[str],
    *,
    title: str = "Alignment Accuracy (Top-1)",
) -> plt.Figure:
    mat = np.array(matrix, dtype=np.float32, copy=True)
    size = len(modality_names)
    if mat.shape != (size, size):
        raise ValueError(f"pair acc matrix shape {mat.shape} does not match modalities {size}x{size}")
    for i in range(size):
        mat[i, i] = 1.0

    fig = render_matrix_heatmap(
        mat,
        modality_names,
        modality_names,
        title=title,
        xlabel="Gallery Modality",
        ylabel="Query Modality",
        cmap=_OPENAI_BLUE_CMAP,
        norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=1.0),
        figsize=(8.2, 8.0),
        annotation_formatter=lambda value: f"{value:.3f}",
        annotation_fontsize=8,
        annotation_fontweight="normal",
        colorbar_title="Accuracy",
        colorbar_fraction=0.045,
        colorbar_pad=0.02,
        subplots_adjust={"left": 0.18, "right": 0.94, "bottom": 0.24, "top": 0.90},
    )
    ax = fig.axes[0]
    ax.tick_params(axis="x", pad=6, labelsize=8)
    ax.tick_params(axis="y", pad=7, labelsize=8)
    ax.xaxis.labelpad = 34
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
        label.set_rotation_mode("anchor")
    add_axis_title_boxes(
        fig,
        ax,
        xlabel="Gallery Modality",
        ylabel="Query Modality",
        x_box_height=0.035,
        y_box_width=0.035,
        label_gap=0.012,
        figure_margin=0.012,
    )
    return fig
