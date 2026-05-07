from __future__ import annotations

import typing as t

from matplotlib.patches import FancyBboxPatch, Patch
import matplotlib.pyplot as plt
import numpy as np

from sleep2expert.visualization.theme import (
    _FIGURE_BG,
    _PRIMARY,
    _PRIMARY_DARK,
    _PRIMARY_LIGHT,
    _PRIMARY_MID,
    _PRIMARY_PALE,
    _TEXT_COLOR,
    apply_plot_layout,
    pick_mono_font_family,
    style_axes,
    style_plot_text,
    use_openai_like_theme,
)


def render_routing_entropy_boxplot(
    rows: t.Sequence[dict[str, t.Any]],
    *,
    entropy_normalizer_by_modality: dict[str, float],
    title: str = "Router entropy by layer and modality",
) -> plt.Figure:
    values_by_group: dict[tuple[str, str], list[float]] = {}
    all_values: list[float] = []
    for row in rows:
        modality = str(row["modality"])
        if modality not in entropy_normalizer_by_modality:
            raise ValueError(f"missing entropy normalizer for modality '{modality}'")
        normalizer = float(entropy_normalizer_by_modality[modality])
        if normalizer <= 0:
            raise ValueError(f"entropy normalizer for modality '{modality}' must be positive")

        layer_label = _format_layer_label(row["layer_idx"])
        value = float(row["router_entropy"]) / normalizer
        values_by_group.setdefault((layer_label, modality), []).append(value)
        all_values.append(value)

    if not all_values:
        raise ValueError("routing entropy boxplot requires at least one row")

    layer_labels = sorted({layer for layer, _ in values_by_group}, key=_layer_sort_key)
    modalities = sorted({modality for _, modality in values_by_group})
    palette = [_PRIMARY_LIGHT, _PRIMARY, _PRIMARY_MID, _PRIMARY_PALE]

    use_openai_like_theme()
    fig_width = max(7.2, 1.15 * len(layer_labels) + 3.1)
    fig, ax = plt.subplots(figsize=(fig_width, 6.4), facecolor=_FIGURE_BG)
    style_axes(ax, show_grid=True)
    ax.xaxis.grid(False)

    group_width = 0.72
    box_width = min(0.24, group_width / (len(modalities) + 1))
    offsets = (np.arange(len(modalities), dtype=np.float32) - (len(modalities) - 1) / 2.0) * (
        group_width / max(len(modalities), 1)
    )
    x_centers = np.arange(len(layer_labels), dtype=np.float32)

    for modality_idx, modality in enumerate(modalities):
        series: list[list[float]] = []
        positions: list[float] = []
        for layer_idx, layer_label in enumerate(layer_labels):
            values = values_by_group.get((layer_label, modality), [])
            if values:
                series.append(values)
                positions.append(float(x_centers[layer_idx] + offsets[modality_idx]))
        if not series:
            continue

        color = palette[modality_idx % len(palette)]
        artists = ax.boxplot(
            series,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            manage_ticks=False,
            showfliers=True,
            boxprops={"edgecolor": _PRIMARY_DARK, "linewidth": 1.35},
            medianprops={"color": _TEXT_COLOR, "linewidth": 1.45},
            whiskerprops={"color": _TEXT_COLOR, "linewidth": 1.15},
            capprops={"color": _TEXT_COLOR, "linewidth": 1.15},
            flierprops={
                "marker": "o",
                "markersize": 2.0,
                "markerfacecolor": _TEXT_COLOR,
                "markeredgecolor": _TEXT_COLOR,
                "markeredgewidth": 0.35,
                "alpha": 0.28,
            },
        )
        for box in artists["boxes"]:
            _replace_with_rounded_box(ax, box, facecolor=color)

    overall_mean = float(np.mean(np.asarray(all_values, dtype=np.float32)))
    ax.axhline(overall_mean, color=_TEXT_COLOR, linestyle="--", linewidth=1.15, alpha=0.9, zorder=1)

    ax.set_xlim(-0.6, len(layer_labels) - 0.4)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x_centers, labels=layer_labels)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(pick_mono_font_family())
    style_plot_text(ax, title=title, xlabel="MoE layer", ylabel="Normalized router entropy")

    legend = ax.legend(
        handles=[
            Patch(
                facecolor=palette[idx % len(palette)],
                edgecolor=_PRIMARY_DARK,
                linewidth=1.2,
                label=modality,
                alpha=0.82,
            )
            for idx, modality in enumerate(modalities)
        ],
        title="modality",
        loc="upper right",
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        facecolor=_FIGURE_BG,
        edgecolor=_TEXT_COLOR,
        fontsize=10,
        title_fontsize=10,
    )
    for text in legend.get_texts():
        text.set_color(_TEXT_COLOR)
        text.set_fontfamily(pick_mono_font_family())
    legend.get_title().set_color(_TEXT_COLOR)
    legend.get_title().set_fontfamily(pick_mono_font_family())

    apply_plot_layout(
        fig,
        defaults={"left": 0.13, "right": 0.96, "bottom": 0.15, "top": 0.90},
    )
    return fig


def _format_layer_label(value: t.Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric.is_integer():
        return str(int(numeric))
    return str(value)


def _layer_sort_key(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def _replace_with_rounded_box(ax: plt.Axes, box, *, facecolor: str) -> None:
    vertices = box.get_path().vertices
    data_vertices = ax.transData.inverted().transform(box.get_transform().transform(vertices))
    x0, y0 = data_vertices.min(axis=0)
    x1, y1 = data_vertices.max(axis=0)
    width = float(x1 - x0)
    height = float(y1 - y0)
    rounding_size = min(0.012, width * 0.08, max(height, 1e-6) * 0.25)
    rounded = FancyBboxPatch(
        (float(x0), float(y0)),
        width,
        height,
        boxstyle=f"round,pad=0,rounding_size={rounding_size}",
        facecolor=facecolor,
        edgecolor=_PRIMARY_DARK,
        linewidth=1.35,
        alpha=0.82,
        transform=ax.transData,
        zorder=box.get_zorder() - 0.1,
    )
    box.remove()
    ax.add_patch(rounded)


__all__ = ["render_routing_entropy_boxplot"]
