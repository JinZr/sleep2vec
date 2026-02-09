from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import yaml

from sleep2vec.common import apply_finetune_config, apply_task_flags, dump_cli_args_yaml
from sleep2vec.config import TaskConfig


def _write_yaml(tmp_path: Path, payload: dict, name: str = "finetune.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload))
    return path


def _finetune_payload() -> dict:
    return {
        "model": {
            "backbone": {
                "name": "roformer",
                "hidden_size": 8,
                "num_hidden_layers": 3,
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
            ],
            "head": {
                "name": "classification",
                "dropout": 0.1,
                "hidden_dim": None,
                "channel_agg": {"name": "mean", "kwargs": {}},
                "temporal_agg": {"name": "mean", "kwargs": {}},
            },
        },
        "data": {
            "max_tokens": 4,
            "data_channel_names": ["ecg", "eeg"],
            "finetune_data_index": "index/custom.csv",
            "finetune_preset_path": "preset/custom.pkl",
            "train_dataset_names": ["train_a"],
            "test_dataset_names": ["test_a"],
            "n_few_shot": 32,
        },
        "finetune": {
            "freeze_tokenizer": False,
            "lora": {
                "freeze_backbone_and_insert_lora": True,
                "insert_lora": True,
                "separate_adapters": True,
            },
            "task": {
                "type": "classification",
                "output_dim": 2,
                "is_seq": False,
                "monitor": "val_accuracy",
                "monitor_mod": "max",
            },
        },
    }


@pytest.mark.parametrize(
    ("label_name", "expected"),
    [
        (
            "stage5",
            {
                "output_dim": 5,
                "is_classification": True,
                "is_seq": True,
                "monitor": "val_accuracy",
                "monitor_mod": "max",
            },
        ),
        (
            "sex",
            {
                "output_dim": 2,
                "is_classification": True,
                "is_seq": False,
                "monitor": "val_accuracy",
                "monitor_mod": "max",
            },
        ),
        (
            "age",
            {
                "output_dim": 1,
                "is_classification": False,
                "is_seq": False,
                "monitor": "val_mae",
                "monitor_mod": "min",
            },
        ),
    ],
)
def test_apply_task_flags_builtin_labels(label_name: str, expected: dict):
    args = argparse.Namespace(label_name=label_name)
    apply_task_flags(args)

    for key, expected_value in expected.items():
        assert getattr(args, key) == expected_value


def test_apply_task_flags_unknown_label_requires_task_config():
    args = argparse.Namespace(label_name="custom_target")

    with pytest.raises(ValueError, match="Unknown label_name 'custom_target'"):
        apply_task_flags(args)


def test_apply_task_flags_rejects_builtin_conflict_from_yaml_task():
    args = argparse.Namespace(label_name="stage5")
    task_cfg = TaskConfig(
        type="classification",
        output_dim=2,
        is_seq=True,
        monitor="val_accuracy",
        monitor_mod="max",
    )

    with pytest.raises(ValueError, match="output_dim must be 5 when --label-name is 'stage5'"):
        apply_task_flags(args, task_cfg)


def test_apply_finetune_config_populates_namespace(tmp_path: Path):
    config_path = _write_yaml(tmp_path, _finetune_payload())
    args = argparse.Namespace(config=config_path, label_name="custom_target")

    config_bundle, model_cfg = apply_finetune_config(args)

    assert [c.name for c in model_cfg.channels] == ["eeg", "ecg"]
    assert args.channel_names == ["eeg", "ecg"]
    assert set(args.data_channel_names) == {"eeg", "ecg"}
    assert args.max_tokens == 4
    assert args.finetune_data_index == Path("index/custom.csv")
    assert args.finetune_preset_path == Path("preset/custom.pkl")
    assert args.train_dataset_names == ["train_a"]
    assert args.test_dataset_names == ["test_a"]
    assert args.n_few_shot == 32
    assert args.freeze_backbone_and_insert_lora is True
    assert args.insert_lora is True
    assert args.separate_adapters is True
    assert args.freeze_tokenizer is False
    assert args.output_dim == 2
    assert args.is_classification is True
    assert args.is_seq is False
    assert config_bundle.finetune.task is not None


def test_apply_finetune_config_rejects_data_channel_mismatch(tmp_path: Path):
    payload = _finetune_payload()
    payload["data"]["data_channel_names"] = ["eeg"]
    config_path = _write_yaml(tmp_path, payload)
    args = argparse.Namespace(config=config_path, label_name="custom_target")

    with pytest.raises(ValueError, match="data.data_channel_names in YAML must match model.channels"):
        apply_finetune_config(args)


def test_dump_cli_args_yaml_converts_namespace_and_paths(tmp_path: Path):
    args = argparse.Namespace(
        alpha=1,
        output_path=Path("outputs/run_a"),
        nested={"artifact": Path("artifacts/model.ckpt")},
        values=[Path("x.txt"), 5],
    )
    dest = tmp_path / "logs" / "cli_args.yaml"

    written = dump_cli_args_yaml(args, dest)

    assert written == dest
    loaded = yaml.safe_load(dest.read_text())
    assert loaded["alpha"] == 1
    assert loaded["output_path"] == "outputs/run_a"
    assert loaded["nested"]["artifact"] == "artifacts/model.ckpt"
    assert loaded["values"] == ["x.txt", 5]
