from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np
import pytest

from sleep2expert.visualization.pair_acc import render_pair_acc_heatmap as render_sleep2expert_pair_acc_heatmap
from sleep2vec2.visualization.pair_acc import render_pair_acc_heatmap as render_sleep2vec2_pair_acc_heatmap
from sleep2vec.visualization.pair_acc import render_pair_acc_heatmap as render_sleep_pair_acc_heatmap
from wrist2vec.visualization.pair_acc import render_pair_acc_heatmap as render_wrist_pair_acc_heatmap


@pytest.mark.parametrize(
    "render_heatmap",
    [
        render_sleep_pair_acc_heatmap,
        render_sleep2vec2_pair_acc_heatmap,
        render_sleep2expert_pair_acc_heatmap,
        render_wrist_pair_acc_heatmap,
    ],
)
def test_render_pair_acc_heatmap_tilts_x_labels(render_heatmap):
    matrix = np.zeros((3, 3), dtype=np.float32)
    fig = render_heatmap(matrix, ["heartbeat", "eeg_original", "resp_nasal_original"])
    try:
        labels = fig.axes[0].get_xticklabels()
        assert [label.get_rotation() for label in labels] == [45.0, 45.0, 45.0]
        assert [label.get_ha() for label in labels] == ["right", "right", "right"]
        assert [label.get_rotation_mode() for label in labels] == ["anchor", "anchor", "anchor"]
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "render_heatmap",
    [
        render_sleep_pair_acc_heatmap,
        render_sleep2vec2_pair_acc_heatmap,
        render_sleep2expert_pair_acc_heatmap,
        render_wrist_pair_acc_heatmap,
    ],
)
def test_render_pair_acc_heatmap_draws_full_axis_title_boxes_clear_of_long_labels(render_heatmap):
    matrix = np.zeros((2, 2), dtype=np.float32)
    fig = render_heatmap(matrix, ["very-long-heartbeat-label", "very-long-respiration-label"])

    try:
        ax = fig.axes[0]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        axis_box = ax.get_position()
        title_boxes = [patch for patch in fig.patches if isinstance(patch, FancyBboxPatch)]
        assert len(title_boxes) == 2
        x_title_box = next(box for box in title_boxes if np.isclose(box.get_width(), axis_box.width))
        y_title_box = next(box for box in title_boxes if np.isclose(box.get_height(), axis_box.height))
        x_tick_bottom = min(
            label.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted()).y0
            for label in ax.get_xticklabels()
        )
        y_tick_left = min(
            label.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted()).x0
            for label in ax.get_yticklabels()
        )

        assert x_title_box.get_y() + x_title_box.get_height() <= x_tick_bottom - 0.011
        assert y_title_box.get_x() + y_title_box.get_width() <= y_tick_left - 0.011
        assert x_title_box.get_y() >= 0.012
        assert y_title_box.get_x() >= 0.012
        assert {text.get_text() for text in fig.texts} >= {"Gallery Modality", "Query Modality"}
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "render_heatmap",
    [render_pair_acc_heatmap, render_sleep2vec2_pair_acc_heatmap, render_sleep2expert_pair_acc_heatmap],
)
def test_render_pair_acc_heatmap_uses_compact_large_matrix_layout(render_heatmap):
    labels = [
        "heartbeat",
        "breath",
        "eeg_original",
        "ecg_original",
        "eog_original",
        "emg_original",
        "spo2",
        "resp_original",
        "resp_nasal_original",
        "ppg",
        "actigraphy",
    ]
    matrix = np.zeros((len(labels), len(labels)), dtype=np.float32)
    fig = render_heatmap(matrix, labels)

    try:
        ax = fig.axes[0]
        fig.canvas.draw()
        axis_box = ax.get_position()

        assert fig.get_size_inches()[0] <= 8.3
        assert axis_box.x0 <= 0.27
        assert axis_box.width >= 0.62
        assert {text.get_fontsize() for text in ax.texts} == {8.0}
        assert {text.get_fontweight() for text in ax.texts} == {"normal"}
        assert {label.get_fontsize() for label in ax.get_xticklabels()} == {8.0}
        assert {label.get_fontsize() for label in ax.get_yticklabels()} == {8.0}
    finally:
        plt.close(fig)
