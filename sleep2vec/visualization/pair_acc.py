from __future__ import annotations

import typing as t

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import PowerNorm


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
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        mat,
        annot=True,
        fmt=".3f",
        xticklabels=modality_names,
        yticklabels=modality_names,
        cmap="Blues",
        cbar=True,
        square=True,
        norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=1.0),
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Gallery Modality")
    ax.set_ylabel("Query Modality")
    plt.tight_layout()
    return fig
