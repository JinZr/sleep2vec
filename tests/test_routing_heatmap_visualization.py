from __future__ import annotations

from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest

from sleep2expert.visualization.routing_heatmap import render_routing_usage_heatmap as render_sleep2expert_heatmap
from sleep2vec2.visualization.routing_heatmap import render_routing_usage_heatmap as render_sleep2vec2_heatmap
from sleep2vec.visualization.routing_heatmap import render_routing_usage_heatmap


@pytest.mark.parametrize(
    "renderer",
    [
        render_routing_usage_heatmap,
        render_sleep2vec2_heatmap,
        render_sleep2expert_heatmap,
    ],
)
def test_render_routing_usage_heatmap_uses_white_masked_style(renderer):
    matrix = np.array(
        [
            [0.0, 0.25, np.nan],
            [1.0, 0.5, 0.0],
        ],
        dtype=np.float32,
    )

    fig = renderer(matrix, ["4", "6"], ["0", "1", "2"], title="demo")

    ax = fig.axes[0]
    image = ax.images[0]
    assert image.get_clim() == (0.0, 1.0)
    assert np.allclose(image.cmap.get_bad(), mcolors.to_rgba("#9EA19A"))
    assert ax.texts == []
    assert len(fig.axes) == 2
    assert np.allclose(fig.get_facecolor(), mcolors.to_rgba("#FFFFFF"))
    assert ax.get_title(loc="center") == "demo"
    assert ax.get_title(loc="left") == ""
    assert ax.title.get_fontsize() == pytest.approx(18.0)
    expected_text_color = mcolors.to_rgba("#171717")
    cbar_ax = fig.axes[1]
    text_artists = (
        [ax.title, ax.xaxis.label, ax.yaxis.label, cbar_ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
        + cbar_ax.get_yticklabels()
    )
    for artist in text_artists:
        assert np.allclose(mcolors.to_rgba(artist.get_color()), expected_text_color)
    plt.close(fig)


def test_render_routing_usage_heatmap_rejects_shape_mismatch():
    matrix = np.zeros((2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="routing heatmap shape"):
        render_routing_usage_heatmap(matrix, ["4"], ["0", "1"], title="bad")
