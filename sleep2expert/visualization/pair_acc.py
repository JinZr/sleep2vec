from __future__ import annotations

import typing as t

from matplotlib.colors import PowerNorm
import matplotlib.pyplot as plt
import numpy as np

from sleep2expert.visualization.heatmaps import render_matrix_heatmap
from sleep2expert.visualization.theme import _OPENAI_BLUE_CMAP


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

    return render_matrix_heatmap(
        mat,
        modality_names,
        modality_names,
        title=title,
        xlabel="Gallery Modality",
        ylabel="Query Modality",
        cmap=_OPENAI_BLUE_CMAP,
        norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=1.0),
        figsize=(12.0, 9.0),
        annotation_formatter=lambda value: f"{value:.3f}",
        colorbar_title="Accuracy",
        subplots_adjust={"left": 0.16, "right": 0.91, "bottom": 0.16, "top": 0.90},
    )
