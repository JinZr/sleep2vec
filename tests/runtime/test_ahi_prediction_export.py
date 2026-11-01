import importlib

import numpy as np
import pytest


@pytest.mark.parametrize("package_name", ("sleep2vec", "sleep2vec2", "sleep2expert"))
def test_build_ahi_prediction_row_deduplicates_window_metadata(package_name: str):
    inference_mod = importlib.import_module(f"{package_name}.sleep2vec_inference")
    first = {
        "path": "night.npz",
        "token_start": 0,
        "truth": np.zeros(30, dtype=np.int64),
        "score": np.full(30, 0.1, dtype=np.float32),
        "true_ahi": 10.0,
        "tst_hours": 3.0,
        "stage5": np.array([2], dtype=np.int64),
        "second_valid_mask": np.ones(30, dtype=np.bool_),
    }
    second = {
        **first,
        "token_start": 1,
        "truth": np.ones(30, dtype=np.int64),
        "score": np.full(30, 0.9, dtype=np.float32),
    }

    rows = inference_mod.build_ahi_prediction_rows([second, first, second, first], threshold=0.5)

    assert len(rows) == 1
    assert rows[0]["n_windows"] == 2
    assert rows[0]["token_starts"] == [0, 1]
    assert rows[0]["n_predictions"] == 60
    assert rows[0]["groundtruth"] == rows[0]["prediction"] == [0] * 30 + [1] * 30
    assert rows[0]["prob"] == pytest.approx([0.1] * 30 + [0.9] * 30)
    assert rows[0]["pred_ahi"] == pytest.approx(1 / 3.0)
