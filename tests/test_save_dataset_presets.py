from __future__ import annotations

from pathlib import Path
import pickle
import sys
import types

import pandas as pd
import pytest
import yaml

from preprocess.save_dataset_presets import (
    _build_preset_job,
    _filter_index_df_for_required_channels,
    _load_index_df,
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


def test_resolve_channels_and_dims_accepts_builtin_stage5_subset(tmp_path: Path):
    config_path = _write_yaml(tmp_path, _model_payload())

    channels, dims = _resolve_channels_and_dims(config_path, ["ppg", "stage5"])

    assert channels == ["ppg", "stage5"]
    assert dims == {"ppg": 8, "stage5": 1}


def test_filter_index_df_for_required_channels_uses_generic_and_builtin_masks():
    df = pd.DataFrame(
        [
            {"path": "a.npz", "ppg_mask": "1", "stage_mask": 1},
            {"path": "b.npz", "ppg_mask": "1", "stage_mask": 0},
            {"path": "c.npz", "ppg_mask": "0", "stage_mask": 1},
            {"path": "d.npz", "ppg_mask": "True", "stage_mask": 1},
        ]
    )

    filtered = _filter_index_df_for_required_channels(df, ["ppg", "stage5"])

    assert filtered["path"].tolist() == ["a.npz", "d.npz"]


def test_filter_index_df_for_required_channels_falls_back_when_some_masks_are_missing():
    df = pd.DataFrame(
        [
            {"path": "a.npz", "ppg_mask": 1},
            {"path": "b.npz", "ppg_mask": 0},
        ]
    )

    filtered = _filter_index_df_for_required_channels(df, ["ppg", "stage5"])

    assert filtered["path"].tolist() == ["a.npz"]


def test_filter_index_df_for_required_channels_falls_back_when_all_masks_are_missing():
    df = pd.DataFrame([{"path": "a.npz"}, {"path": "b.npz"}])

    filtered = _filter_index_df_for_required_channels(df, ["ppg", "stage5"])

    assert filtered["path"].tolist() == ["a.npz", "b.npz"]


def test_filter_index_df_for_required_channels_rejects_empty_filtered_frame():
    df = pd.DataFrame(
        [
            {"path": "a.npz", "ppg_mask": 0, "stage_mask": 0},
            {"path": "b.npz", "ppg_mask": "False", "stage_mask": 0},
        ]
    )

    with pytest.raises(ValueError, match="No rows satisfy required mask columns"):
        _filter_index_df_for_required_channels(df, ["ppg", "stage5"])


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


def test_build_preset_job_prefilters_index_with_required_masks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class FakeDataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            index_paths = kwargs["index"]
            assert isinstance(index_paths, list)
            assert len(index_paths) == 1
            filtered_df = pd.read_csv(index_paths[0], low_memory=False)
            captured["filtered_paths"] = filtered_df["path"].tolist()

        def __len__(self) -> int:
            return 2

    fake_module = types.SimpleNamespace(PSGPretrainDataset=FakeDataset)
    monkeypatch.setitem(sys.modules, "data.psg_pretrain_dataset", fake_module)

    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {"path": "a.npz", "split": "train", "duration": 2, "age": 40, "sex": 1, "ppg_mask": 1, "stage_mask": 1},
            {"path": "b.npz", "split": "train", "duration": 2, "age": 40, "sex": 1, "ppg_mask": 1, "stage_mask": 0},
            {"path": "c.npz", "split": "train", "duration": 2, "age": 40, "sex": 1, "ppg_mask": 0, "stage_mask": 1},
        ]
    ).to_csv(index_path, index=False)

    output_path, sample_count = _build_preset_job(
        output_path=tmp_path / "preset.pkl",
        index_paths=[str(index_path)],
        channel_names=["ppg", "stage5"],
        channel_input_dims={"ppg": 8, "stage5": 1},
        split="train",
        meta_data_name=None,
        n_tokens=128,
        stride_tokens=64,
        mask_rate=0.0,
        allow_missing_channels=False,
        min_channels=2,
        batch_size=8,
        shuffle=False,
        filter_max_workers=1,
    )

    assert output_path == tmp_path / "preset.pkl"
    assert sample_count == 2
    assert captured["filtered_paths"] == ["a.npz"]


def test_build_preset_job_restores_original_source_after_strict_prefilter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class FakeDataset:
        def __init__(self, **kwargs):
            output_path = Path(kwargs["save_preset_path"])
            samples = [types.SimpleNamespace(metadata={"source": "temp.csv"})]
            with open(output_path, "wb") as f:
                pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)

        def __len__(self) -> int:
            return 1

    fake_module = types.SimpleNamespace(PSGPretrainDataset=FakeDataset)
    monkeypatch.setitem(sys.modules, "data.psg_pretrain_dataset", fake_module)

    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {"path": "a.npz", "split": "train", "duration": 2, "age": 40, "sex": 1, "ppg_mask": 1, "stage_mask": 1},
        ]
    ).to_csv(index_path, index=False)

    output_path, sample_count = _build_preset_job(
        output_path=tmp_path / "preset.pkl",
        index_paths=[str(index_path)],
        channel_names=["ppg", "stage5"],
        channel_input_dims={"ppg": 8, "stage5": 1},
        split="train",
        meta_data_name=None,
        n_tokens=128,
        stride_tokens=64,
        mask_rate=0.0,
        allow_missing_channels=False,
        min_channels=2,
        batch_size=8,
        shuffle=False,
        filter_max_workers=1,
    )

    assert output_path == tmp_path / "preset.pkl"
    assert sample_count == 1
    with open(output_path, "rb") as f:
        saved = pickle.load(f)
    assert saved[0].metadata["source"] == str(index_path)


def test_load_index_df_rejects_multiple_index_paths(tmp_path: Path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    pd.DataFrame([{"path": "a.npz"}]).to_csv(first, index=False)
    pd.DataFrame([{"path": "b.npz"}]).to_csv(second, index=False)

    with pytest.raises(ValueError, match="accepts exactly one index CSV"):
        _load_index_df([str(first), str(second)])
