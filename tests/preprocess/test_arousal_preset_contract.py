from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.utils import AROUSAL_INDEX_KEYS
from preprocess.save_dataset_presets import (
    _build_preset_job,
    _filter_index_df_for_required_channels,
    _mask_column_for_channel,
    _resolve_effective_min_channels,
    _resolve_validation_channels,
)


def test_arousal_preset_is_builtin_and_auto_requires_stage5() -> None:
    channels, dims = _resolve_validation_channels(
        model_channels=["ppg"],
        channel_input_dims={"ppg": 8},
        preset_required_channels=None,
        selected_channels=["ppg", "arousal"],
    )

    assert channels == ["ppg", "arousal", "stage5"]
    assert dims == {"ppg": 8, "arousal": 120, "stage5": 1}
    assert _mask_column_for_channel("arousal") == "arousal_event_mask"
    assert (
        _resolve_effective_min_channels(
            channel_names=channels,
            cli_min_channels=1,
            preset_min_channels=1,
        )
        == 3
    )


def test_arousal_preset_rejects_declared_input_dim_other_than_120() -> None:
    with pytest.raises(ValueError, match="input_dim=120"):
        _resolve_validation_channels(
            model_channels=["arousal"],
            channel_input_dims={"arousal": 4},
            preset_required_channels=None,
            selected_channels=None,
        )


def test_arousal_strict_preset_requires_and_applies_both_masks() -> None:
    with pytest.raises(ValueError, match="arousal_event_mask"):
        _filter_index_df_for_required_channels(
            pd.DataFrame([{"path": "missing.npz", "stage_mask": 1}]),
            ["arousal"],
        )

    df = pd.DataFrame(
        [
            {"path": "keep.npz", "arousal_event_mask": 1, "stage_mask": 1},
            {"path": "no-arousal.npz", "arousal_event_mask": 0, "stage_mask": 1},
            {"path": "no-stage.npz", "arousal_event_mask": 1, "stage_mask": 0},
        ]
    )

    filtered = _filter_index_df_for_required_channels(df, ["arousal"])

    assert filtered["path"].tolist() == ["keep.npz"]


def test_arousal_preset_applies_strict_masks_even_when_missing_channels_are_allowed(tmp_path: Path) -> None:
    paths = [tmp_path / "keep.npz", tmp_path / "masked.npz"]
    for path in paths:
        payload = {
            "arousal_event": np.zeros((30, 4), dtype=np.float32),
            "stage5": np.ones(1, dtype=np.float32),
            "tst": np.asarray(1.0, dtype=np.float32),
        }
        payload.update({key: np.asarray(0.0, dtype=np.float32) for key in AROUSAL_INDEX_KEYS})
        np.savez(path, **payload)
    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(paths[0]),
                "split": "train",
                "duration": 30,
                "arousal_event_mask": 1,
                "stage_mask": 1,
            },
            {
                "path": str(paths[1]),
                "split": "train",
                "duration": 30,
                "arousal_event_mask": 0,
                "stage_mask": 1,
            },
        ]
    ).to_csv(index_path, index=False)

    _, sample_count = _build_preset_job(
        output_path=tmp_path / "preset.pickle",
        index_paths=[str(index_path)],
        channel_names=["arousal", "stage5"],
        channel_input_dims={"arousal": 120, "stage5": 1},
        split="train",
        meta_data_name=None,
        n_tokens=1,
        stride_tokens=0,
        mask_rate=0.0,
        allow_missing_channels=True,
        min_channels=2,
        batch_size=1,
        shuffle=False,
        filter_max_workers=1,
    )

    assert sample_count == 1
