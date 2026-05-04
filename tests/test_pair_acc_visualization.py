import matplotlib.pyplot as plt
import numpy as np

from sleep2expert.visualization.pair_acc import render_pair_acc_heatmap as render_sleep2expert_pair_acc_heatmap
from sleep2vec2.visualization.pair_acc import render_pair_acc_heatmap as render_sleep2vec2_pair_acc_heatmap
from sleep2vec.visualization.pair_acc import render_pair_acc_heatmap


def test_render_pair_acc_heatmap_uses_non_overlapping_vertical_x_labels():
    matrix = np.zeros((3, 3), dtype=np.float32)
    fig = render_pair_acc_heatmap(matrix, ["heartbeat", "eeg_original", "resp_nasal_original"])
    try:
        labels = fig.axes[0].get_xticklabels()
        assert [label.get_rotation() for label in labels] == [90.0, 90.0, 90.0]
        assert [label.get_ha() for label in labels] == ["center", "center", "center"]
        assert [label.get_va() for label in labels] == ["top", "top", "top"]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tick_boxes = [label.get_window_extent(renderer=renderer) for label in labels]
        heatmap_bottom = fig.axes[0].get_window_extent(renderer=renderer).y0
        xlabel_top = fig.axes[0].xaxis.label.get_window_extent(renderer=renderer).y1
        assert max(box.y1 for box in tick_boxes) <= heatmap_bottom
        assert xlabel_top <= min(box.y0 for box in tick_boxes)
    finally:
        plt.close(fig)


def test_render_sleep2vec2_pair_acc_heatmap_uses_non_overlapping_vertical_x_labels():
    matrix = np.zeros((3, 3), dtype=np.float32)
    fig = render_sleep2vec2_pair_acc_heatmap(matrix, ["heartbeat", "eeg_original", "resp_nasal_original"])
    try:
        labels = fig.axes[0].get_xticklabels()
        assert [label.get_rotation() for label in labels] == [90.0, 90.0, 90.0]
        assert [label.get_ha() for label in labels] == ["center", "center", "center"]
        assert [label.get_va() for label in labels] == ["top", "top", "top"]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tick_boxes = [label.get_window_extent(renderer=renderer) for label in labels]
        heatmap_bottom = fig.axes[0].get_window_extent(renderer=renderer).y0
        xlabel_top = fig.axes[0].xaxis.label.get_window_extent(renderer=renderer).y1
        assert max(box.y1 for box in tick_boxes) <= heatmap_bottom
        assert xlabel_top <= min(box.y0 for box in tick_boxes)
    finally:
        plt.close(fig)


def test_render_sleep2expert_pair_acc_heatmap_uses_non_overlapping_vertical_x_labels():
    matrix = np.zeros((3, 3), dtype=np.float32)
    fig = render_sleep2expert_pair_acc_heatmap(matrix, ["heartbeat", "eeg_original", "resp_nasal_original"])
    try:
        labels = fig.axes[0].get_xticklabels()
        assert [label.get_rotation() for label in labels] == [90.0, 90.0, 90.0]
        assert [label.get_ha() for label in labels] == ["center", "center", "center"]
        assert [label.get_va() for label in labels] == ["top", "top", "top"]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tick_boxes = [label.get_window_extent(renderer=renderer) for label in labels]
        heatmap_bottom = fig.axes[0].get_window_extent(renderer=renderer).y0
        xlabel_top = fig.axes[0].xaxis.label.get_window_extent(renderer=renderer).y1
        assert max(box.y1 for box in tick_boxes) <= heatmap_bottom
        assert xlabel_top <= min(box.y0 for box in tick_boxes)
    finally:
        plt.close(fig)
