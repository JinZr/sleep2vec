from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np
import pytest

from sleep2vec.visualization.pair_acc import render_pair_acc_heatmap as render_sleep_pair_acc_heatmap
from wrist2vec.visualization.pair_acc import render_pair_acc_heatmap as render_wrist_pair_acc_heatmap


@pytest.mark.parametrize("render_heatmap", [render_sleep_pair_acc_heatmap, render_wrist_pair_acc_heatmap])
def test_render_pair_acc_heatmap_tilts_x_labels(render_heatmap):
    matrix = np.zeros((3, 3), dtype=np.float32)
    fig = render_heatmap(matrix, ["heartbeat", "eeg_original", "resp_nasal_original"])
    try:
        labels = fig.axes[0].get_xticklabels()
        assert [label.get_rotation() for label in labels] == [45.0, 45.0, 45.0]
        assert [label.get_ha() for label in labels] == ["right", "right", "right"]
    finally:
        plt.close(fig)


def test_render_pair_acc_heatmap_draws_full_axis_title_boxes_clear_of_long_labels():
    matrix = np.zeros((2, 2), dtype=np.float32)
    fig = render_pair_acc_heatmap(matrix, ["very-long-heartbeat-label", "very-long-respiration-label"])

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

        assert x_title_box.get_y() + x_title_box.get_height() <= x_tick_bottom - 0.017
        assert y_title_box.get_x() + y_title_box.get_width() <= y_tick_left - 0.017
        assert x_title_box.get_y() >= 0.018
        assert y_title_box.get_x() >= 0.018
        assert {text.get_text() for text in fig.texts} >= {"Gallery Modality", "Query Modality"}
    finally:
        plt.close(fig)
