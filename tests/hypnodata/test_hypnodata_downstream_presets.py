from pathlib import Path

import pandas as pd

from preprocess.save_dataset_presets import (
    _filter_index_df_for_required_channels,
    _load_config_mapping,
    _load_model_channels,
    _resolve_validation_channels,
)
from tests.hypnodata_test_helpers import run_tiny_hypnodata


def test_preset_mask_filter_accepts_hypnodata_record_manifest(tmp_path: Path):
    output_dir = run_tiny_hypnodata(tmp_path)
    manifest_path = output_dir / "manifest" / "record_manifest.csv"
    config_path = tmp_path / "preset_config.yaml"
    config_path.write_text("model:\n  channels:\n    - name: eeg\n      input_dim: 30\n")

    config_data = _load_config_mapping(config_path)
    model_channels, dims = _load_model_channels(config_data)
    channel_names, channel_dims = _resolve_validation_channels(
        model_channels=model_channels,
        channel_input_dims=dims,
        preset_required_channels=None,
        selected_channels=["eeg"],
    )
    filtered = _filter_index_df_for_required_channels(pd.read_csv(manifest_path), channel_names)

    assert channel_names == ["eeg"]
    assert channel_dims == {"eeg": 30}
    assert filtered["record_id"].tolist() == ["night1"]
    assert filtered["path"].str.endswith("night1.npz").all()
