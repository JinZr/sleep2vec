from __future__ import annotations

from collections.abc import Mapping
import typing as t

import matplotlib.pyplot as plt
import numpy as np

from sleep2vec.visualization.heatmaps import render_matrix_heatmap
from sleep2vec.visualization.theme import _OPENAI_BLUE_CMAP


def render_layer_mix_heatmap(
    matrix: np.ndarray,
    modality_names: t.Sequence[str],
    layer_ids: t.Sequence[int],
    *,
    title: str = "Layer-Mix Weights",
) -> plt.Figure:
    mat = np.array(matrix, dtype=np.float32, copy=True)
    expected_shape = (len(modality_names), len(layer_ids))
    if mat.shape != expected_shape:
        raise ValueError(f"layer mix matrix shape {mat.shape} does not match expected {expected_shape}")

    fig_width = max(8.0, 0.8 * len(layer_ids) + 3.0)
    fig_height = max(4.0, 0.7 * len(modality_names) + 2.0)
    return render_matrix_heatmap(
        mat,
        [str(v) for v in layer_ids],
        list(modality_names),
        title=title,
        xlabel="Layer",
        ylabel="Modality",
        cmap=_OPENAI_BLUE_CMAP,
        vmin=0.0,
        vmax=1.0,
        figsize=(fig_width, fig_height),
        annotation_formatter=lambda value: f"{value:.3f}",
        colorbar_title="Weight",
        subplots_adjust={"left": 0.16, "right": 0.90, "bottom": 0.18, "top": 0.88},
    )


def build_layer_mix_rows(
    stage: str,
    epoch: int,
    shared: bool,
    layer_ids: t.Sequence[int],
    effective_by_modality: Mapping[str, t.Any],
) -> list[dict[str, t.Any]]:
    rows: list[dict[str, t.Any]] = []
    for modality, mod_info in effective_by_modality.items():
        if not isinstance(mod_info, Mapping):
            continue
        row_name = str(mod_info.get("row_name", ""))
        row_index = int(mod_info.get("row_index", -1))
        weights = mod_info.get("layer_weights", [])
        if not isinstance(weights, (list, tuple)):
            continue
        for idx, layer_id in enumerate(layer_ids):
            if idx >= len(weights):
                break
            rows.append(
                {
                    "stage": stage,
                    "epoch": int(epoch),
                    "modality": str(modality),
                    "layer_id": int(layer_id),
                    "weight": float(weights[idx]),
                    "shared_across_modalities": bool(shared),
                    "row_name": row_name,
                    "row_index": row_index,
                }
            )
    return rows


__all__ = ["build_layer_mix_rows", "render_layer_mix_heatmap"]
