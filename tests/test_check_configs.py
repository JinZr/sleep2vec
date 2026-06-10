from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from sleep2vec.common import apply_finetune_config
from utils.check_configs import check_config_file

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_FINETUNE_CONFIGS = [
    ("stage3", REPO_ROOT / "configs" / "examples" / "stage3" / "FINETUNE_EXAMPLE.yaml"),
    ("stage4", REPO_ROOT / "configs" / "examples" / "stage4" / "FINETUNE_EXAMPLE.yaml"),
    ("stage5", REPO_ROOT / "configs" / "examples" / "stage5" / "FINETUNE_EXAMPLE.yaml"),
    ("ahi", REPO_ROOT / "configs" / "examples" / "ahi" / "FINETUNE_EXAMPLE.yaml"),
    ("sex", REPO_ROOT / "configs" / "examples" / "sex" / "FINETUNE_EXAMPLE.yaml"),
    ("age", REPO_ROOT / "configs" / "examples" / "age" / "FINETUNE_EXAMPLE.yaml"),
]


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
    path = REPO_ROOT / "configs" / "ppg_stage3_finetune.yaml"
    check_config_file(path)


def test_check_config_file_accepts_repo_ppg_age_config():
    path = REPO_ROOT / "configs" / "ppg_age_finetune_large.yaml"
    check_config_file(path)


def test_check_config_file_accepts_repo_ppg_ahi_config():
    path = REPO_ROOT / "configs" / "ppg_ahi_finetune.yaml"
    check_config_file(path)


def test_check_config_file_accepts_repo_ppg_ahi_large_config():
    path = REPO_ROOT / "configs" / "ppg_ahi_finetune_large.yaml"
    check_config_file(path)


def test_check_config_file_accepts_repo_ppg_ahi_large_temporal_conv_config():
    path = REPO_ROOT / "configs" / "ppg_ahi_finetune_large_temporal_conv.yaml"
    check_config_file(path)


def test_repo_template_finetune_configs_do_not_bind_dataset_inputs():
    offenders = []
    for path in sorted((REPO_ROOT / "configs").rglob("*.yaml")):
        if "examples" in path.parts:
            continue
        payload = yaml.safe_load(path.read_text()) or {}
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        has_finetune_input_fields = "finetune_data_index" in data or "finetune_preset_path" in data
        if has_finetune_input_fields and (
            data.get("finetune_data_index") is not None or data.get("finetune_preset_path") is not None
        ):
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []


@pytest.mark.parametrize(
    "path",
    [REPO_ROOT / "configs" / "examples" / "PRETRAIN_EXAMPLE.yaml"] + [path for _, path in EXAMPLE_FINETUNE_CONFIGS],
)
def test_check_config_file_accepts_example_configs(path: Path):
    check_config_file(path)


@pytest.mark.parametrize(("label_name", "path"), EXAMPLE_FINETUNE_CONFIGS)
def test_apply_finetune_config_accepts_builtin_task_examples(label_name: str, path: Path):
    args = argparse.Namespace(config=str(path), label_name=label_name)

    apply_finetune_config(args)


def test_check_config_file_accepts_sleep2expert_moe_pretrain_config():
    path = REPO_ROOT / "configs" / "sleep2expert" / "moe" / "sleep2expert_phase_moe_pretrain.yaml"
    check_config_file(path)


def test_check_config_file_accepts_sleep2expert_moe_finetune_cls_conservative_config():
    path = REPO_ROOT / "configs" / "sleep2expert" / "moe" / "sleep2expert_phase_moe_finetune_cls_conservative.yaml"
    check_config_file(path)


def test_check_config_file_accepts_sleep2expert_moe_finetune_reg_conservative_tokens_config():
    path = (
        REPO_ROOT / "configs" / "sleep2expert" / "moe" / "sleep2expert_phase_moe_finetune_reg_conservative_tokens.yaml"
    )
    check_config_file(path)


def test_check_config_file_accepts_sleep2expert_moe_finetune_cls_head_only_fewshot_config():
    path = REPO_ROOT / "configs" / "sleep2expert" / "moe" / "sleep2expert_phase_moe_finetune_cls_head_only_fewshot.yaml"
    check_config_file(path)


def test_check_config_file_accepts_sleep2expert_moe_router_trainable_ablation_config():
    path = REPO_ROOT / "configs" / "sleep2expert" / "moe" / "finetune_ablations" / "router_trainable.yaml"
    check_config_file(path)


def test_check_config_file_accepts_sleep2expert_moe_top_layer_expert_only_ablation_config():
    path = REPO_ROOT / "configs" / "sleep2expert" / "moe" / "finetune_ablations" / "top_moe_layer_expert_only.yaml"
    check_config_file(path)


def test_check_config_file_does_not_use_base_loader_for_sleep2expert_moe(monkeypatch):
    import sleep2vec.config as base_config

    def fail_base_loader(*args, **kwargs):
        raise AssertionError("base sleep2vec loader should not validate sleep2expert configs")

    monkeypatch.setattr(base_config, "load_pretrain_config", fail_base_loader)

    path = REPO_ROOT / "configs" / "sleep2expert" / "moe" / "sleep2expert_phase_moe_pretrain.yaml"
    check_config_file(path)


def test_check_config_file_accepts_out_of_tree_sleep2expert_moe_config(monkeypatch, tmp_path: Path):
    import sleep2vec.config as base_config

    def fail_base_loader(*args, **kwargs):
        raise AssertionError("base sleep2vec loader should not validate copied sleep2expert configs")

    monkeypatch.setattr(base_config, "load_pretrain_config", fail_base_loader)
    source = REPO_ROOT / "configs" / "sleep2expert" / "moe" / "sleep2expert_phase_moe_pretrain.yaml"
    path = tmp_path / "copied.yaml"
    path.write_text(source.read_text())

    check_config_file(path)


def test_check_config_file_accepts_out_of_tree_sleep2vec2_config_with_path_hint(
    monkeypatch,
    tmp_path: Path,
):
    import sleep2vec.config as base_config

    def fail_base_loader(*args, **kwargs):
        raise AssertionError("base sleep2vec loader should not validate copied sleep2vec2 configs")

    monkeypatch.setattr(base_config, "load_pretrain_config", fail_base_loader)
    source = REPO_ROOT / "configs" / "sleep2vec2" / "sleep2vec_dense_pretrain.yaml"
    path = tmp_path / "sleep2vec2" / "sleep2vec_dense_pretrain.yaml"
    path.parent.mkdir()
    path.write_text(source.read_text())

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

    with pytest.raises(ValueError, match="must set preset_build.required_channels to \\[ppg, ahi, stage5\\]"):
        check_config_file(path)


def test_check_config_file_rejects_wrong_min_channels_for_ppg_ahi_config(tmp_path: Path):
    path = tmp_path / "configs" / "ppg_ahi_finetune.yaml"
    payload = _ppg_finetune_payload(
        is_seq=True,
        preset_build={"required_channels": ["ppg", "ahi", "stage5"], "min_channels": 2},
        task_overrides={"output_dim": 30, "monitor": "val_ahi_pearson"},
    )
    _write_yaml(path, payload)

    with pytest.raises(ValueError, match="must set preset_build.min_channels to 3"):
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
