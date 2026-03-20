from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from preprocess.save_dataset_presets import _resolve_channels_and_dims


def _write_yaml(tmp_path: Path, payload: dict, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload))
    return path


def _model_payload() -> dict:
    return {
        "model": {
            "backbone": {
                "name": "roformer",
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "vocab_size": 1,
            },
            "projection": {
                "name": "simclr",
                "enabled": True,
                "hidden_dim": 8,
                "out_dim": 4,
            },
            "cls": {
                "embedding_type": None,
                "downstream": "tokens",
            },
            "channels": [
                {"name": "eeg", "input_dim": 4, "tokenizer": {"name": "linear", "out_dim": 8}},
                {"name": "ecg", "input_dim": 4, "tokenizer": {"name": "linear", "out_dim": 8}},
                {"name": "ppg", "input_dim": 8, "tokenizer": {"name": "linear", "out_dim": 8}},
            ],
        }
    }


def _channels_only_payload() -> dict:
    return {
        "model": {
            "channels": [
                {"name": "eeg", "input_dim": 4},
                {"name": "ecg", "input_dim": 4},
                {"name": "ppg", "input_dim": 8},
            ]
        }
    }


def test_resolve_channels_and_dims_defaults_to_all_yaml_channels(tmp_path: Path):
    config_path = _write_yaml(tmp_path, _model_payload())

    channels, dims = _resolve_channels_and_dims(config_path, None)

    assert channels == ["eeg", "ecg", "ppg"]
    assert dims == {"eeg": 4, "ecg": 4, "ppg": 8}


def test_resolve_channels_and_dims_allows_yaml_declared_subset_in_requested_order(tmp_path: Path):
    config_path = _write_yaml(tmp_path, _model_payload())

    channels, dims = _resolve_channels_and_dims(config_path, ["ppg", "eeg"])

    assert channels == ["ppg", "eeg"]
    assert dims == {"ppg": 8, "eeg": 4}


def test_resolve_channels_and_dims_rejects_unknown_subset_channels(tmp_path: Path):
    config_path = _write_yaml(tmp_path, _model_payload())

    with pytest.raises(ValueError, match="Channels must be declared in YAML model.channels"):
        _resolve_channels_and_dims(config_path, ["unknown"])


def test_resolve_channels_and_dims_accepts_channels_only_yaml(tmp_path: Path):
    config_path = _write_yaml(tmp_path, _channels_only_payload())

    channels, dims = _resolve_channels_and_dims(config_path, None)

    assert channels == ["eeg", "ecg", "ppg"]
    assert dims == {"eeg": 4, "ecg": 4, "ppg": 8}
