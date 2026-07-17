from __future__ import annotations

from matplotlib import colors as mcolors
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np
import pytest

from sleep2expert.visualization.boxplot import render_routing_entropy_boxplot as render_sleep2expert_boxplot
from sleep2vec2.visualization.boxplot import render_routing_entropy_boxplot as render_sleep2vec2_boxplot
from sleep2vec.visualization.boxplot import render_routing_entropy_boxplot


def _rows() -> list[dict[str, float | int | str]]:
    normalized_values = {
        (4, "heartbeat"): [0.20, 0.21, 0.22, 0.95],
        (4, "breath"): [0.40, 0.42, 0.43, 0.44],
        (6, "heartbeat"): [0.30, 0.31, 0.32, 0.33],
        (6, "breath"): [0.60, 0.61, 0.62, 0.63],
    }
    normalizers = {"heartbeat": 2.0, "breath": 1.0}
    return [
        {
            "layer_idx": layer_idx,
            "modality": modality,
            "router_entropy": value * normalizers[modality],
        }
        for (layer_idx, modality), values in normalized_values.items()
        for value in values
    ]


@pytest.mark.parametrize(
    "renderer",
    [
        render_routing_entropy_boxplot,
        render_sleep2vec2_boxplot,
        render_sleep2expert_boxplot,
    ],
)
def test_render_routing_entropy_boxplot_uses_vertical_openai_style(renderer):
    fig = renderer(
        _rows(),
        entropy_normalizer_by_modality={"heartbeat": 2.0, "breath": 1.0},
        title="demo",
    )

    ax = fig.axes[0]
    assert ax.get_title(loc="center") == "demo"
    assert ax.title.get_fontsize() == pytest.approx(18.0)
    assert ax.get_xlabel() == "MoE layer"
    assert ax.get_ylabel() == "Normalized router entropy"
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["4", "6"]
    assert ax.get_ylim() == (0.0, 1.0)
    rounded_boxes = [patch for patch in ax.patches if isinstance(patch, FancyBboxPatch)]
    assert len(rounded_boxes) == 4
    assert all("Round" in type(patch.get_boxstyle()).__name__ for patch in rounded_boxes)

    normalized = np.asarray(
        [row["router_entropy"] / {"heartbeat": 2.0, "breath": 1.0}[str(row["modality"])] for row in _rows()],
        dtype=np.float32,
    )
    mean_lines = [line for line in ax.lines if line.get_linestyle() == "--"]
    assert len(mean_lines) == 1
    assert np.allclose(mean_lines[0].get_ydata(), np.full(2, float(normalized.mean())))

    fliers = [line for line in ax.lines if line.get_marker() == "o"]
    assert fliers
    assert all(line.get_markersize() <= 2.0 for line in fliers)
    assert all(line.get_alpha() == pytest.approx(0.28) for line in fliers)

    expected_text_color = mcolors.to_rgba("#171717")
    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_title().get_text() == "modality"
    assert {text.get_text() for text in legend.get_texts()} == {"breath", "heartbeat"}
    text_artists = (
        [ax.title, ax.xaxis.label, ax.yaxis.label, legend.get_title()]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
        + legend.get_texts()
    )
    for artist in text_artists:
        assert np.allclose(mcolors.to_rgba(artist.get_color()), expected_text_color)
    plt.close(fig)


def test_render_routing_entropy_boxplot_rejects_missing_normalizer():
    with pytest.raises(ValueError, match="missing entropy normalizer"):
        render_routing_entropy_boxplot(
            [{"layer_idx": 4, "modality": "heartbeat", "router_entropy": 1.0}],
            entropy_normalizer_by_modality={},
        )
