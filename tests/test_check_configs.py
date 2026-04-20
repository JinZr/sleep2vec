from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from utils.check_configs import check_config_file


def _write_yaml(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload))
    return path


def _ppg_finetune_payload(*, is_seq: bool, preset_build: dict | None, task_overrides: dict | None = None) -> dict:
    payload = {
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
                {"name": "ppg", "input_dim": 8, "tokenizer": {"name": "linear", "out_dim": 8}},
            ],
            "head": {
                "name": "classification" if is_seq else "regression",
                "dropout": 0.1,
                "hidden_dim": None,
                "channel_agg": {"name": "mean", "kwargs": {}},
                "temporal_agg": {"name": "mean", "kwargs": {}},
            },
        },
        "data": {
            "max_tokens": 4,
            "data_channel_names": ["ppg"],
            "finetune_data_index": "index.csv",
            "finetune_preset_path": "preset.pkl",
            "train_dataset_names": ["train_ds"],
            "test_dataset_names": ["test_ds"],
            "n_few_shot": 16,
        },
        "finetune": {
            "freeze_tokenizer": True,
            "lora": {
                "freeze_backbone_and_insert_lora": False,
                "insert_lora": True,
                "separate_adapters": False,
            },
            "task": {
                "type": "classification" if is_seq else "regression",
                "output_dim": 3 if is_seq else 1,
                "is_seq": is_seq,
                "monitor": "val_accuracy" if is_seq else "val_mae",
                "monitor_mod": "max" if is_seq else "min",
            },
        },
    }
    if task_overrides is not None:
        payload["finetune"]["task"].update(task_overrides)
    if preset_build is not None:
        payload["preset_build"] = preset_build
    return payload


def test_check_config_file_accepts_repo_ppg_stage3_config():
    path = Path(__file__).resolve().parents[1] / "configs" / "ppg_stage3_finetune.yaml"
    check_config_file(path)


def test_check_config_file_accepts_repo_ppg_age_config():
    path = Path(__file__).resolve().parents[1] / "configs" / "ppg_age_finetune_large.yaml"
    check_config_file(path)


def test_check_config_file_accepts_repo_ppg_ahi_config():
    path = Path(__file__).resolve().parents[1] / "configs" / "ppg_ahi_finetune.yaml"
    check_config_file(path)


def test_check_config_file_accepts_repo_ppg_ahi_large_config():
    path = Path(__file__).resolve().parents[1] / "configs" / "ppg_ahi_finetune_large.yaml"
    check_config_file(path)


def test_check_config_file_accepts_repo_ppg_ahi_large_temporal_conv_config():
    path = Path(__file__).resolve().parents[1] / "configs" / "ppg_ahi_finetune_large_temporal_conv.yaml"
    check_config_file(path)


def test_check_config_file_accepts_repo_ppg_ahi_large_temporal_unet_aux_config():
    path = Path(__file__).resolve().parents[1] / "configs" / "ppg_ahi_finetune_large_temporal_unet_aux.yaml"
    check_config_file(path)


def test_check_config_file_rejects_missing_preset_build_for_ppg_finetune(tmp_path: Path):
    path = tmp_path / "configs" / "ppg_stage3_finetune.yaml"
    _write_yaml(path, _ppg_finetune_payload(is_seq=True, preset_build=None))

    with pytest.raises(
        ValueError, match="must define both preset_build.required_channels and preset_build.min_channels"
    ):
        check_config_file(path)


def test_check_config_file_rejects_wrong_required_channels_for_ppg_stage_config(tmp_path: Path):
    path = tmp_path / "configs" / "ppg_stage3_finetune.yaml"
    payload = _ppg_finetune_payload(
        is_seq=True,
        preset_build={"required_channels": ["ppg"], "min_channels": 1},
    )
    _write_yaml(path, payload)

    with pytest.raises(ValueError, match="must set preset_build.required_channels to \\[ppg, stage5\\]"):
        check_config_file(path)


def test_check_config_file_rejects_wrong_required_channels_for_ppg_ahi_config(tmp_path: Path):
    path = tmp_path / "configs" / "ppg_ahi_finetune.yaml"
    payload = _ppg_finetune_payload(
        is_seq=True,
        preset_build={"required_channels": ["ppg", "stage5"], "min_channels": 2},
        task_overrides={"output_dim": 30, "monitor": "val_ahi_pearson"},
    )
    _write_yaml(path, payload)

    with pytest.raises(ValueError, match="must set preset_build.required_channels to \\[ppg, ahi\\]"):
        check_config_file(path)


def test_check_config_file_rejects_wrong_min_channels_for_ppg_ahi_config(tmp_path: Path):
    path = tmp_path / "configs" / "ppg_ahi_finetune.yaml"
    payload = _ppg_finetune_payload(
        is_seq=True,
        preset_build={"required_channels": ["ppg", "ahi"], "min_channels": 1},
        task_overrides={"output_dim": 30, "monitor": "val_ahi_pearson"},
    )
    _write_yaml(path, payload)

    with pytest.raises(ValueError, match="must set preset_build.min_channels to 2"):
        check_config_file(path)


def test_check_config_file_rejects_wrong_min_channels_for_ppg_age_config(tmp_path: Path):
    path = tmp_path / "configs" / "ppg_age_finetune_large.yaml"
    payload = _ppg_finetune_payload(
        is_seq=False,
        preset_build={"required_channels": ["ppg"], "min_channels": 2},
    )
    _write_yaml(path, payload)

    with pytest.raises(ValueError, match="must set preset_build.min_channels to 1"):
        check_config_file(path)


def test_check_config_file_rejects_partial_preset_build_block(tmp_path: Path):
    path = tmp_path / "configs" / "ppg_age_finetune_large.yaml"
    payload = _ppg_finetune_payload(
        is_seq=False,
        preset_build={"required_channels": ["ppg"]},
    )
    _write_yaml(path, payload)

    with pytest.raises(
        ValueError, match="must define both preset_build.required_channels and preset_build.min_channels"
    ):
        check_config_file(path)
