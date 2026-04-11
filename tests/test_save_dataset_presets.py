from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest
import yaml

from preprocess.save_dataset_presets import (
    _build_preset_job,
    _resolve_channels_and_dims,
)


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


def test_build_preset_job_passes_filter_workers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class FakeDataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __len__(self) -> int:
            return 7

    fake_module = types.SimpleNamespace(PSGPretrainDataset=FakeDataset)
    monkeypatch.setitem(sys.modules, "data.psg_pretrain_dataset", fake_module)

    output_path, sample_count = _build_preset_job(
        output_path=tmp_path / "preset.pkl",
        index_paths=["/tmp/index.csv"],
        channel_names=["eeg", "ecg"],
        channel_input_dims={"eeg": 4, "ecg": 4},
        split="train",
        meta_data_name="ahi",
        n_tokens=128,
        stride_tokens=64,
        mask_rate=0.0,
        allow_missing_channels=True,
        min_channels=2,
        batch_size=8,
        shuffle=False,
        filter_max_workers=3,
    )

    assert output_path == tmp_path / "preset.pkl"
    assert sample_count == 7
    assert captured["filter_max_workers"] == 3
    assert captured["meta_data_names"] == ["ahi"]
