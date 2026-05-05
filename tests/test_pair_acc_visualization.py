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
