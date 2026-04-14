import matplotlib.pyplot as plt
import numpy as np
import pytest

from sleep2vec.visualization.layer_mix import build_layer_mix_rows, render_layer_mix_heatmap
from sleep2vec.visualization.theme import _OPENAI_BLUE_CMAP


def test_render_layer_mix_heatmap_accepts_valid_shape():
    matrix = np.array([[0.10, 0.90], [0.65, 0.35]], dtype=np.float32)
    fig = render_layer_mix_heatmap(matrix, ["eeg", "ecg"], [1, 2], title="demo")
    try:
        assert fig.axes[0].get_xlabel() == "Layer"
        assert fig.axes[0].get_ylabel() == "Modality"
        assert fig.axes[0].get_title() == "demo"
        assert fig.axes[0].images[0].cmap.name == _OPENAI_BLUE_CMAP.name
    finally:
        plt.close(fig)


def test_render_layer_mix_heatmap_rejects_shape_mismatch():
    matrix = np.array([[0.20, 0.80]], dtype=np.float32)
    with pytest.raises(ValueError, match="layer mix matrix shape"):
        render_layer_mix_heatmap(matrix, ["eeg", "ecg"], [1, 2])


def test_build_layer_mix_rows_expands_modality_layer_pairs():
    rows = build_layer_mix_rows(
        stage="val",
        epoch=3,
        shared=True,
        layer_ids=[1, 2, 3],
        effective_by_modality={
            "eeg": {"row_name": "shared", "row_index": 0, "layer_weights": [0.1, 0.2, 0.7]},
            "ecg": {"row_name": "shared", "row_index": 0, "layer_weights": [0.3, 0.3, 0.4]},
        },
    )

    assert len(rows) == 6
    assert {row["modality"] for row in rows} == {"eeg", "ecg"}
    assert {row["layer_id"] for row in rows} == {1, 2, 3}
    assert all(row["shared_across_modalities"] is True for row in rows)
    assert rows[0]["stage"] == "val"
    assert rows[0]["epoch"] == 3
