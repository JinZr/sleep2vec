import matplotlib.pyplot as plt
import numpy as np

from sleep2expert.visualization.pair_acc import render_pair_acc_heatmap as render_sleep2expert_pair_acc_heatmap
from sleep2vec2.visualization.pair_acc import render_pair_acc_heatmap as render_sleep2vec2_pair_acc_heatmap
from sleep2vec.visualization.pair_acc import render_pair_acc_heatmap


def test_render_pair_acc_heatmap_tilts_x_labels():
    matrix = np.zeros((3, 3), dtype=np.float32)
    fig = render_pair_acc_heatmap(matrix, ["heartbeat", "eeg_original", "resp_nasal_original"])
    try:
        labels = fig.axes[0].get_xticklabels()
        assert [label.get_rotation() for label in labels] == [45.0, 45.0, 45.0]
        assert [label.get_ha() for label in labels] == ["right", "right", "right"]
    finally:
        plt.close(fig)


def test_render_sleep2vec2_pair_acc_heatmap_tilts_x_labels():
    matrix = np.zeros((3, 3), dtype=np.float32)
    fig = render_sleep2vec2_pair_acc_heatmap(matrix, ["heartbeat", "eeg_original", "resp_nasal_original"])
    try:
        labels = fig.axes[0].get_xticklabels()
        assert [label.get_rotation() for label in labels] == [45.0, 45.0, 45.0]
        assert [label.get_ha() for label in labels] == ["right", "right", "right"]
    finally:
        plt.close(fig)


def test_render_sleep2expert_pair_acc_heatmap_tilts_x_labels():
    matrix = np.zeros((3, 3), dtype=np.float32)
    fig = render_sleep2expert_pair_acc_heatmap(matrix, ["heartbeat", "eeg_original", "resp_nasal_original"])
    try:
        labels = fig.axes[0].get_xticklabels()
        assert [label.get_rotation() for label in labels] == [45.0, 45.0, 45.0]
        assert [label.get_ha() for label in labels] == ["right", "right", "right"]
    finally:
        plt.close(fig)
