from __future__ import annotations

import typing as t

import matplotlib.pyplot as plt
import numpy as np

from sleep2vec_hires.visualization.theme import (
    _ROUTING_BG,
    _ROUTING_TEXT,
    build_routing_usage_cmap,
    pick_mono_font_family,
    style_plot_text,
    use_openai_like_theme,
)


def render_routing_usage_heatmap(
    matrix: np.ndarray,
    layer_labels: t.Sequence[str],
    expert_labels: t.Sequence[str],
    *,
    title: str,
    colorbar_label: str = "Expert usage share",
) -> plt.Figure:
    mat = np.asarray(matrix, dtype=np.float32)
    expected_shape = (len(layer_labels), len(expert_labels))
    if mat.shape != expected_shape:
        raise ValueError(f"routing heatmap shape {mat.shape} does not match labels {expected_shape}")

    use_openai_like_theme()
    masked = np.ma.masked_invalid(mat)
    fig_width = max(7.2, 0.48 * len(expert_labels) + 3.2)
    fig_height = max(4.8, 0.42 * len(layer_labels) + 2.7)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor=_ROUTING_BG)
    ax.set_facecolor(_ROUTING_BG)

    image = ax.imshow(
        masked,
        cmap=build_routing_usage_cmap(),
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="auto",
    )

    style_plot_text(ax, title=title, xlabel="Expert", ylabel="MoE layer")
    ax.set_xticks(np.arange(len(expert_labels)), labels=list(expert_labels))
    ax.set_yticks(np.arange(len(layer_labels)), labels=list(layer_labels))
    ax.tick_params(axis="x", colors=_ROUTING_TEXT, labelsize=8, length=0, pad=6)
    ax.tick_params(axis="y", colors=_ROUTING_TEXT, labelsize=9, length=0, pad=8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(pick_mono_font_family())
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.10)
    cbar.outline.set_visible(False)
    cbar.set_ticks(np.linspace(0.0, 1.0, 11))
    cbar.ax.tick_params(length=0, colors=_ROUTING_TEXT, labelsize=8, pad=7)
    cbar.ax.set_ylabel(colorbar_label, color=_ROUTING_TEXT, fontsize=9, rotation=270, labelpad=22)
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily(pick_mono_font_family())

    fig.subplots_adjust(left=0.16, right=0.86, bottom=0.18, top=0.89)
    return fig


__all__ = ["render_routing_usage_heatmap"]
