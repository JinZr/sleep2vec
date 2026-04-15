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
    _load_config_mapping,
    _load_index_df,
    _load_model_channels,
    _load_preset_build_block,
    _resolve_channels_and_dims,
    _resolve_effective_min_channels,
    _resolve_validation_channels,
    main as save_dataset_presets_main,
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


def _preset_build_payload(*, required_channels=None, min_channels=None) -> dict:
    payload = _model_payload()
    preset_build = {}
    if required_channels is not None:
        preset_build["required_channels"] = required_channels
    if min_channels is not None:
        preset_build["min_channels"] = min_channels
    payload["preset_build"] = preset_build
    return payload


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

    with pytest.raises(ValueError, match="Unknown: \\['unknown'\\]"):
        _resolve_channels_and_dims(config_path, ["unknown"])


def test_resolve_channels_and_dims_accepts_builtin_stage5_subset(tmp_path: Path):
    config_path = _write_yaml(tmp_path, _model_payload())

    channels, dims = _resolve_channels_and_dims(config_path, ["ppg", "stage5"])

    assert channels == ["ppg", "stage5"]
    assert dims == {"ppg": 8, "stage5": 1}


def test_resolve_channels_and_dims_accepts_builtin_ahi_subset(tmp_path: Path):
    config_path = _write_yaml(tmp_path, _model_payload())

    channels, dims = _resolve_channels_and_dims(config_path, ["ppg", "ahi"])

    assert channels == ["ppg", "ahi", "stage5"]
    assert dims == {"ppg": 8, "ahi": 30, "stage5": 1}


def test_load_preset_build_block_parses_explicit_contract(tmp_path: Path):
    config_path = _write_yaml(tmp_path, _preset_build_payload(required_channels=["ppg", "stage5"], min_channels=2))
    config_data = _load_config_mapping(config_path)

    required_channels, min_channels = _load_preset_build_block(config_data)

    assert required_channels == ["ppg", "stage5"]
    assert min_channels == 2


def test_load_preset_build_block_rejects_duplicate_required_channels(tmp_path: Path):
    config_path = _write_yaml(tmp_path, _preset_build_payload(required_channels=["ppg", "ppg"]))
    config_data = _load_config_mapping(config_path)

    with pytest.raises(ValueError, match="must not contain duplicates"):
        _load_preset_build_block(config_data)


def test_load_preset_build_block_rejects_partial_contract(tmp_path: Path):
    config_path = _write_yaml(tmp_path, _preset_build_payload(required_channels=["ppg"]))
    config_data = _load_config_mapping(config_path)

    with pytest.raises(
        ValueError, match="must define both preset_build.required_channels and preset_build.min_channels"
    ):
        _load_preset_build_block(config_data)


def test_resolve_validation_channels_uses_preset_build_required_channels(tmp_path: Path):
    config_path = _write_yaml(tmp_path, _preset_build_payload(required_channels=["ppg", "stage5"], min_channels=2))
    config_data = _load_config_mapping(config_path)
    model_channels, channel_input_dims = _load_model_channels(config_data)
    preset_required_channels, _ = _load_preset_build_block(config_data)

    channels, dims = _resolve_validation_channels(
        model_channels=model_channels,
        channel_input_dims=channel_input_dims,
        preset_required_channels=preset_required_channels,
        selected_channels=None,
    )

    assert channels == ["ppg", "stage5"]
    assert dims == {"ppg": 8, "stage5": 1}


def test_resolve_validation_channels_rejects_cli_channels_when_preset_build_required_channels_exist(tmp_path: Path):
    config_path = _write_yaml(tmp_path, _preset_build_payload(required_channels=["ppg", "stage5"], min_channels=2))
    config_data = _load_config_mapping(config_path)
    model_channels, channel_input_dims = _load_model_channels(config_data)
    preset_required_channels, _ = _load_preset_build_block(config_data)

    with pytest.raises(ValueError, match="--channels cannot be used when preset_build.required_channels is set"):
        _resolve_validation_channels(
            model_channels=model_channels,
            channel_input_dims=channel_input_dims,
            preset_required_channels=preset_required_channels,
            selected_channels=["ppg"],
        )


def test_resolve_effective_min_channels_prefers_preset_build_override():
    effective_min_channels = _resolve_effective_min_channels(
        channel_names=["ppg"],
        cli_min_channels=2,
        preset_min_channels=1,
    )

    assert effective_min_channels == 1


def test_resolve_effective_min_channels_enforces_full_channels_for_ahi():
    effective_min_channels = _resolve_effective_min_channels(
        channel_names=["ppg", "ahi", "stage5"],
        cli_min_channels=2,
        preset_min_channels=2,
    )

    assert effective_min_channels == 3


def test_resolve_effective_min_channels_rejects_value_above_channel_count():
    with pytest.raises(ValueError, match="exceeds the number of validation channels"):
        _resolve_effective_min_channels(
            channel_names=["ppg"],
            cli_min_channels=2,
            preset_min_channels=2,
        )


def test_strict_mode_single_channel_subset_does_not_need_min_channel_validation(tmp_path: Path, monkeypatch):
    config_path = _write_yaml(tmp_path, _model_payload())
    index_path = tmp_path / "index.csv"
    pd.DataFrame([{"path": "a.npz"}]).to_csv(index_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "save_dataset_presets.py",
            "--config",
            str(config_path),
            "--index",
            str(index_path),
            "--channels",
            "ppg",
            "--no-allow-missing-channels",
            "--dry-run",
        ],
    )

    save_dataset_presets_main()


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


def test_filter_index_df_for_required_channels_uses_ahi_mask():
    df = pd.DataFrame(
        [
            {"path": "a.npz", "ppg_mask": "1", "ahi_mask": 1},
            {"path": "b.npz", "ppg_mask": "1", "ahi_mask": 0},
            {"path": "c.npz", "ppg_mask": "0", "ahi_mask": 1},
        ]
    )

    filtered = _filter_index_df_for_required_channels(df, ["ppg", "ahi"])

    assert filtered["path"].tolist() == ["a.npz"]


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


def test_build_preset_job_prefilters_index_with_ahi_mask(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class FakeDataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            index_paths = kwargs["index"]
            filtered_df = pd.read_csv(index_paths[0], low_memory=False)
            captured["filtered_paths"] = filtered_df["path"].tolist()

        def __len__(self) -> int:
            return 1

    fake_module = types.SimpleNamespace(PSGPretrainDataset=FakeDataset)
    monkeypatch.setitem(sys.modules, "data.psg_pretrain_dataset", fake_module)

    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {"path": "a.npz", "split": "train", "duration": 60, "age": 40, "sex": 1, "ppg_mask": 1, "ahi_mask": 1},
            {"path": "b.npz", "split": "train", "duration": 60, "age": 40, "sex": 1, "ppg_mask": 1, "ahi_mask": 0},
        ]
    ).to_csv(index_path, index=False)

    output_path, sample_count = _build_preset_job(
        output_path=tmp_path / "preset.pkl",
        index_paths=[str(index_path)],
        channel_names=["ppg", "ahi"],
        channel_input_dims={"ppg": 8, "ahi": 30},
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
