from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from pathlib import Path

import pytest
import yaml

from sleep2vec.config import (
    BackboneConfig,
    ChannelAggConfig,
    ChannelConfig,
    ClsConfig,
    HeadConfig,
    ModelConfig,
    ProjectionConfig,
    TemporalAggConfig,
    TokenizerConfig,
    load_finetune_config,
    load_pretrain_config,
    validate_model_config,
)
from wrist2vec_flex.config import (
    load_finetune_config as load_wrist2vec_finetune_config,
    load_pretrain_config as load_wrist2vec_pretrain_config,
    validate_model_config as validate_wrist2vec_model_config,
)


def _write_yaml(tmp_path: Path, payload: dict, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload))
    return path


def _base_model_block() -> dict:
    return {
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
            {
                "name": "eeg",
                "input_dim": 4,
                "tokenizer": {"name": "linear", "out_dim": 8},
            },
            {
                "name": "ecg",
                "input_dim": 4,
                "tokenizer": {"name": "linear", "out_dim": 8},
            },
        ],
    }


def _pretrain_payload() -> dict:
    return {
        "model": _base_model_block(),
        "loss": {"name": "info_nce", "temperature": 0.2},
        "data": {"mask_rate": 0.1, "max_tokens": 4},
    }


def _finetune_payload() -> dict:
    payload = {
        "model": _base_model_block(),
        "data": {
            "max_tokens": 4,
            "data_channel_names": ["ecg", "eeg"],
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
            "layer_mix": {
                "enabled": False,
                "shared_across_modalities": False,
                "layer_indices": None,
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
    payload["model"]["head"] = {
        "name": "classification",
        "dropout": 0.1,
        "hidden_dim": None,
        "channel_agg": {"name": "mean", "kwargs": {}},
        "temporal_agg": {"name": "mean", "kwargs": {}},
    }
    return payload


def _valid_model_config() -> ModelConfig:
    channels = [
        ChannelConfig(
            name="eeg",
            input_dim=4,
            tokenizer=TokenizerConfig(name="linear", out_dim=8),
        ),
        ChannelConfig(
            name="ecg",
            input_dim=4,
            tokenizer=TokenizerConfig(name="linear", out_dim=8),
        ),
    ]
    return ModelConfig(
        channels=channels,
        backbone=BackboneConfig(
            name="roformer",
            hidden_size=8,
            num_hidden_layers=3,
            num_attention_heads=2,
            vocab_size=1,
        ),
        projection=ProjectionConfig(name="simclr", enabled=True, hidden_dim=8, out_dim=4),
        cls=ClsConfig(downstream="tokens", embedding_type=None),
        head=HeadConfig(
            channel_agg=ChannelAggConfig(name="mean"),
            temporal_agg=TemporalAggConfig(name="mean"),
            name="classification",
        ),
    )


def test_load_pretrain_config_parses_valid_yaml(tmp_path: Path):
    config_path = _write_yaml(tmp_path, _pretrain_payload())
    bundle = load_pretrain_config(config_path)

    assert [c.name for c in bundle.model.channels] == ["eeg", "ecg"]
    assert bundle.model.backbone.hidden_size == 8
    assert bundle.loss.name == "info_nce"
    assert bundle.data.backend == "npz"
    assert bundle.data.kaldi_data_root is None
    assert bundle.data.kaldi_manifest is None
    assert bundle.data.max_tokens == 4


def test_load_wrist2vec_pretrain_config_parses_token_sec(tmp_path: Path):
    payload = _pretrain_payload()
    payload["data"]["token_sec"] = 2
    config_path = _write_yaml(tmp_path, payload)

    bundle = load_wrist2vec_pretrain_config(config_path)

    assert bundle.data.token_sec == 2


def test_load_wrist2vec_pretrain_config_rejects_non_positive_token_sec(tmp_path: Path):
    payload = _pretrain_payload()
    payload["data"]["token_sec"] = 0
    config_path = _write_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="data.token_sec must be a positive integer"):
        load_wrist2vec_pretrain_config(config_path)


def test_load_wrist2vec_pretrain_config_parses_source_dropout_fields(tmp_path: Path):
    payload = _pretrain_payload()
    payload["data"].update(
        {
            "source_dropout_rate": 0.25,
            "min_sources_after_dropout": 2,
        }
    )
    config_path = _write_yaml(tmp_path, payload)

    bundle = load_wrist2vec_pretrain_config(config_path)

    assert bundle.data.source_dropout_rate == pytest.approx(0.25)
    assert bundle.data.min_sources_after_dropout == 2


@pytest.mark.parametrize(
    ("field", "value", "pattern"),
    [
        ("source_dropout_rate", 1.1, r"data\.source_dropout_rate must be in \[0\.0, 1\.0\]"),
        ("channel_dropout_rate", -0.1, r"data\.channel_dropout_rate must be in \[0\.0, 1\.0\]"),
        ("channel_dropout_rate", 0.1, "data.channel_dropout_rate is not supported for wrist2vec_flex pretrain"),
        ("min_sources_after_dropout", 0, "data.min_sources_after_dropout must be a positive integer"),
        ("min_channels_after_dropout", False, "data.min_channels_after_dropout must be a positive integer"),
        (
            "min_channels_after_dropout",
            3,
            "data.min_channels_after_dropout must remain 2 because pretrain channel dropout is unsupported",
        ),
    ],
)
def test_load_wrist2vec_pretrain_config_rejects_invalid_dropout_fields(
    tmp_path: Path,
    field: str,
    value,
    pattern: str,
):
    payload = _pretrain_payload()
    payload["data"][field] = value
    config_path = _write_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match=pattern):
        load_wrist2vec_pretrain_config(config_path)


def test_load_pretrain_config_parses_kaldi_data_fields(tmp_path: Path):
    payload = _pretrain_payload()
    payload["data"].update(
        {
            "backend": "kaldi",
            "kaldi_data_root": "/tmp/kaldi_root",
            "kaldi_manifest": "/tmp/kaldi_root/manifest.json",
        }
    )
    config_path = _write_yaml(tmp_path, payload)

    bundle = load_pretrain_config(config_path)

    assert bundle.data.backend == "kaldi"
    assert bundle.data.kaldi_data_root == "/tmp/kaldi_root"
    assert bundle.data.kaldi_manifest == "/tmp/kaldi_root/manifest.json"


def test_load_pretrain_config_parses_adapt_block(tmp_path: Path):
    payload = _pretrain_payload()
    payload["adapt"] = {
        "new_channels": ["eeg"],
        "stage1": {"train_shared_projection": True},
        "stage2": {
            "lr_scales": {"encoder": 0.2, "shared_legacy": 0.6, "new_modalities": 1.0},
            "pair_schedule": [
                {"until": 0.5, "new_pair_ratio": 1.0},
                {"until": 1.0, "new_pair_ratio": 0.0},
            ],
        },
    }
    config_path = _write_yaml(tmp_path, payload)

    bundle = load_pretrain_config(config_path)

    assert bundle.adapt is not None
    assert bundle.adapt.new_channels == ["eeg"]
    assert bundle.adapt.stage1.train_shared_projection is True
    assert bundle.adapt.stage2.lr_scales.encoder == pytest.approx(0.2)
    assert bundle.adapt.stage2.pair_schedule[-1].until == pytest.approx(1.0)


def test_load_pretrain_config_requires_adapt_new_channels(tmp_path: Path):
    payload = _pretrain_payload()
    payload["adapt"] = {"stage1": {"train_shared_projection": False}}
    config_path = _write_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="adapt.new_channels is required"):
        load_pretrain_config(config_path)


def test_load_pretrain_config_validates_adapt_schedule_endpoint(tmp_path: Path):
    payload = _pretrain_payload()
    payload["adapt"] = {
        "new_channels": ["eeg"],
        "stage2": {
            "pair_schedule": [
                {"until": 0.5, "new_pair_ratio": 1.0},
                {"until": 0.8, "new_pair_ratio": 0.0},
            ]
        },
    }
    config_path = _write_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="must end with until=1.0"):
        load_pretrain_config(config_path)


@pytest.mark.parametrize(
    "config_name",
    [
        "sleep2vec_dense_adapt_ppg_actigraphy.yaml",
        "sleep2vec_dense_adapt_ppg_actigraphy_cls.yaml",
    ],
)
def test_ppg_actigraphy_adapt_configs_keep_uniform_final_stage_sampling(config_name: str):
    config_path = Path(__file__).resolve().parents[1] / "configs" / config_name
    bundle = load_pretrain_config(config_path)

    assert bundle.adapt is not None
    channel_names = [channel.name for channel in bundle.model.channels]
    new_channels = set(bundle.adapt.new_channels)
    all_pairs = list(combinations(channel_names, 2))
    new_pair_count = sum(1 for left, right in all_pairs if left in new_channels or right in new_channels)
    final_ratio = bundle.adapt.stage2.pair_schedule[-1].new_pair_ratio

    assert final_ratio > 0.0
    assert final_ratio == pytest.approx(new_pair_count / len(all_pairs))


@pytest.mark.parametrize(
    "config_name",
    [
        "wrist2vec_flex_multilight_ppg_accgyro_pretrain_resnet1d.yaml",
    ],
)
def test_wrist2vec_repo_pretrain_configs_load(config_name: str):
    config_path = Path(__file__).resolve().parents[1] / "configs" / "wrist2vec_flex" / config_name
    bundle = load_wrist2vec_pretrain_config(config_path)

    validate_wrist2vec_model_config(bundle.model)
    assert bundle.model.channels


def test_wrist2vec_resnet1d_example_pretrain_config_loads():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "wrist2vec_flex"
        / "wrist2vec_flex_multilight_ppg_accgyro_pretrain_resnet1d.yaml"
    )
    bundle = load_wrist2vec_pretrain_config(config_path)

    validate_wrist2vec_model_config(bundle.model)
    hidden_size = bundle.model.backbone.hidden_size
    assert hidden_size == 384
    assert bundle.model.backbone.num_hidden_layers == 12
    assert bundle.model.backbone.num_attention_heads == 16
    assert bundle.model.projection.hidden_dim == hidden_size
    assert [channel.name for channel in bundle.model.channels] == [
        "ppg_green",
        "ppg_red",
        "ppg_infrared",
        "gyro_vm",
        "acc_vm",
    ]
    assert [channel.tokenizer.name for channel in bundle.model.channels] == [
        "resnet1d",
        "resnet1d",
        "resnet1d",
        "sundial2",
        "sundial2",
    ]
    assert all(channel.tokenizer.out_dim == hidden_size for channel in bundle.model.channels)
    assert bundle.model.channels[0].tokenizer.kwargs["block_counts"] == [2, 2, 2]
    assert bundle.loss.name == "info_nce"
    assert bundle.data.token_sec == 2


def test_load_finetune_config_parses_valid_yaml(tmp_path: Path):
    config_path = _write_yaml(tmp_path, _finetune_payload())
    bundle = load_finetune_config(config_path)

    assert bundle.model.head is not None
    assert bundle.model.head.temporal_agg.name == "mean"
    assert bundle.data.backend == "npz"
    assert bundle.data.kaldi_data_root is None
    assert bundle.data.kaldi_manifest is None
    assert bundle.finetune.task is not None
    assert bundle.finetune.task.output_dim == 2


def test_load_finetune_config_parses_kaldi_data_fields(tmp_path: Path):
    payload = _finetune_payload()
    payload["data"].update(
        {
            "backend": "kaldi",
            "kaldi_data_root": "/tmp/kaldi_root",
            "kaldi_manifest": "/tmp/kaldi_root/manifest.json",
        }
    )
    config_path = _write_yaml(tmp_path, payload)

    bundle = load_finetune_config(config_path)

    assert bundle.data.backend == "kaldi"
    assert bundle.data.kaldi_data_root == "/tmp/kaldi_root"
    assert bundle.data.kaldi_manifest == "/tmp/kaldi_root/manifest.json"


def test_load_wrist2vec_finetune_config_parses_dropout_fields(tmp_path: Path):
    payload = _finetune_payload()
    payload["data"].update(
        {
            "channel_dropout_rate": 0.3,
            "min_channels_after_dropout": 2,
            "source_dropout_rate": 0.4,
            "min_sources_after_dropout": 3,
        }
    )
    config_path = _write_yaml(tmp_path, payload)

    bundle = load_wrist2vec_finetune_config(config_path)

    assert bundle.data.channel_dropout_rate == pytest.approx(0.3)
    assert bundle.data.min_channels_after_dropout == 2
    assert bundle.data.source_dropout_rate == pytest.approx(0.4)
    assert bundle.data.min_sources_after_dropout == 3


@pytest.mark.parametrize(
    ("field", "value", "pattern"),
    [
        ("channel_dropout_rate", 1.5, r"data\.channel_dropout_rate must be in \[0\.0, 1\.0\]"),
        ("min_channels_after_dropout", 0, "data.min_channels_after_dropout must be a positive integer"),
        ("source_dropout_rate", -0.1, r"data\.source_dropout_rate must be in \[0\.0, 1\.0\]"),
        ("min_sources_after_dropout", 0, "data.min_sources_after_dropout must be a positive integer"),
    ],
)
def test_load_wrist2vec_finetune_config_rejects_invalid_dropout_fields(
    tmp_path: Path,
    field: str,
    value,
    pattern: str,
):
    payload = _finetune_payload()
    payload["data"][field] = value
    config_path = _write_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match=pattern):
        load_wrist2vec_finetune_config(config_path)


@pytest.mark.parametrize(
    ("loader", "payload_factory"),
    [
        (load_pretrain_config, _pretrain_payload),
        (load_finetune_config, _finetune_payload),
    ],
)
def test_load_config_rejects_invalid_data_backend(tmp_path: Path, loader, payload_factory):
    payload = payload_factory()
    payload["data"]["backend"] = "hdf5"
    config_path = _write_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="data.backend must be one of"):
        loader(config_path)


def test_load_finetune_config_parses_eval_visualizations(tmp_path: Path):
    payload = _finetune_payload()
    payload["finetune"]["eval_visualizations"] = {
        "enabled": True,
        "stages": ["val", "test"],
        "confusion_matrix": {"enabled": True, "show_raw_counts": True},
        "roc_curve": {"enabled": True},
        "regression_scatter": {"enabled": False},
    }
    config_path = _write_yaml(tmp_path, payload)

    bundle = load_finetune_config(config_path)

    assert bundle.finetune.eval_visualizations is not None
    assert bundle.finetune.eval_visualizations.enabled is True
    assert bundle.finetune.eval_visualizations.stages == ["val", "test"]
    assert bundle.finetune.eval_visualizations.confusion_matrix.enabled is True
    assert bundle.finetune.eval_visualizations.confusion_matrix.show_raw_counts is True
    assert bundle.finetune.eval_visualizations.roc_curve.enabled is True
    assert bundle.finetune.eval_visualizations.regression_scatter.enabled is False


@pytest.mark.parametrize("missing_key", ["backbone", "projection", "cls"])
def test_load_pretrain_config_requires_model_blocks(tmp_path: Path, missing_key: str):
    payload = _pretrain_payload()
    payload["model"].pop(missing_key)
    config_path = _write_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match=rf"model\.{missing_key} is required"):
        load_pretrain_config(config_path)


@pytest.mark.parametrize("missing_key", ["backbone", "projection", "cls"])
def test_load_finetune_config_requires_model_blocks(tmp_path: Path, missing_key: str):
    payload = _finetune_payload()
    payload["model"].pop(missing_key)
    config_path = _write_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match=rf"model\.{missing_key} is required"):
        load_finetune_config(config_path)


def test_load_finetune_config_requires_finetune_block(tmp_path: Path):
    payload = _finetune_payload()
    payload.pop("finetune")
    config_path = _write_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="top-level 'finetune' block"):
        load_finetune_config(config_path)


def test_load_pretrain_config_rejects_non_list_channels(tmp_path: Path):
    payload = _pretrain_payload()
    payload["model"]["channels"] = {"name": "eeg"}
    config_path = _write_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="model.channels must be a list"):
        load_pretrain_config(config_path)


def test_load_pretrain_config_rejects_non_mapping_tokenizer(tmp_path: Path):
    payload = _pretrain_payload()
    payload["model"]["channels"][0]["tokenizer"] = "linear"
    config_path = _write_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="channel.tokenizer must be a mapping"):
        load_pretrain_config(config_path)


def test_load_pretrain_config_requires_tokenizer_name(tmp_path: Path):
    payload = _pretrain_payload()
    payload["model"]["channels"][0]["tokenizer"] = {"out_dim": 8}
    config_path = _write_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="must set tokenizer.name"):
        load_pretrain_config(config_path)


def test_load_pretrain_config_requires_tokenizer_out_dim(tmp_path: Path):
    payload = _pretrain_payload()
    payload["model"]["channels"][0]["tokenizer"] = {"name": "linear"}
    config_path = _write_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="must set tokenizer.out_dim"):
        load_pretrain_config(config_path)


def test_load_finetune_config_requires_head_block(tmp_path: Path):
    payload = _finetune_payload()
    payload["model"].pop("head")
    config_path = _write_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="model.head must be a mapping and is required"):
        load_finetune_config(config_path)


def test_load_finetune_config_requires_temporal_agg(tmp_path: Path):
    payload = _finetune_payload()
    payload["model"]["head"].pop("temporal_agg")
    config_path = _write_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="model.head.temporal_agg is required"):
        load_finetune_config(config_path)


def test_load_finetune_config_requires_channel_agg(tmp_path: Path):
    payload = _finetune_payload()
    payload["model"]["head"].pop("channel_agg")
    config_path = _write_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="model.head.channel_agg is required"):
        load_finetune_config(config_path)


def test_load_finetune_config_rejects_task_missing_fields(tmp_path: Path):
    payload = _finetune_payload()
    payload["finetune"]["task"] = {"type": "classification"}
    config_path = _write_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="missing required fields"):
        load_finetune_config(config_path)


def test_load_finetune_config_rejects_task_extra_fields(tmp_path: Path):
    payload = _finetune_payload()
    payload["finetune"]["task"]["extra"] = "x"
    config_path = _write_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="unsupported fields"):
        load_finetune_config(config_path)


@pytest.mark.parametrize(
    ("task_patch", "pattern"),
    [
        ({"type": "invalid"}, "must be 'classification' or 'regression'"),
        ({"output_dim": 0}, "must be a positive integer"),
        ({"is_seq": "yes"}, "must be a boolean"),
        ({"monitor_mod": "up"}, "must be 'min' or 'max'"),
        ({"type": "classification", "output_dim": 1}, "must be >= 2 for classification"),
        ({"type": "regression", "output_dim": 2}, "must be 1 for regression"),
    ],
)
def test_load_finetune_config_rejects_invalid_task_semantics(tmp_path: Path, task_patch: dict, pattern: str):
    payload = _finetune_payload()
    payload["finetune"]["task"].update(task_patch)
    config_path = _write_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match=pattern):
        load_finetune_config(config_path)


@pytest.mark.parametrize(
    ("layer_indices", "pattern"),
    [
        ("not-a-list", "must be a non-empty list"),
        ([], "must be a non-empty list"),
        ([1, "2"], "must be a list of integers"),
        ([0, 1], "values must be >= 1"),
        ([1, 1], "must not contain duplicates"),
    ],
)
def test_load_finetune_config_rejects_invalid_layer_mix_indices(tmp_path: Path, layer_indices, pattern: str):
    payload = _finetune_payload()
    payload["finetune"]["layer_mix"] = {
        "enabled": True,
        "shared_across_modalities": False,
        "layer_indices": layer_indices,
    }
    config_path = _write_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match=pattern):
        load_finetune_config(config_path)


def test_load_finetune_config_rejects_layer_index_above_backbone_depth(tmp_path: Path):
    payload = _finetune_payload()
    payload["model"]["backbone"]["num_hidden_layers"] = 2
    payload["finetune"]["layer_mix"] = {
        "enabled": True,
        "shared_across_modalities": False,
        "layer_indices": [3],
    }
    config_path = _write_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match="must be <= num_hidden_layers"):
        load_finetune_config(config_path)


@pytest.mark.parametrize(
    ("eval_visualizations", "pattern"),
    [
        ("bad", "must be a mapping"),
        ({"enabled": "yes"}, "enabled must be a boolean"),
        ({"stages": "val"}, "stages must be a non-empty list"),
        ({"stages": []}, "stages must be a non-empty list"),
        ({"stages": ["val", "val"]}, "must not contain duplicates"),
        ({"stages": ["train"]}, "only supports 'val' and 'test'"),
        ({"unknown": {}}, "unsupported fields"),
        ({"confusion_matrix": {"extra": True}}, "confusion_matrix has unsupported fields"),
        (
            {"confusion_matrix": {"show_raw_counts": "yes"}},
            "confusion_matrix.show_raw_counts must be a boolean",
        ),
        ({"roc_curve": {"enabled": "yes"}}, "roc_curve.enabled must be a boolean"),
        ({"regression_scatter": {"enabled": "yes"}}, "regression_scatter.enabled must be a boolean"),
    ],
)
def test_load_finetune_config_rejects_invalid_eval_visualizations(
    tmp_path: Path,
    eval_visualizations,
    pattern: str,
):
    payload = _finetune_payload()
    payload["finetune"]["eval_visualizations"] = eval_visualizations
    config_path = _write_yaml(tmp_path, payload)

    with pytest.raises(ValueError, match=pattern):
        load_finetune_config(config_path)


def test_validate_model_config_accepts_valid_config():
    model_cfg = _valid_model_config()
    feature_dim = validate_model_config(model_cfg)
    assert feature_dim == 8


def test_validate_model_config_rejects_mismatched_tokenizer_dims():
    model_cfg = _valid_model_config()
    model_cfg.channels[1].tokenizer.out_dim = 16

    with pytest.raises(ValueError, match="must share the same out_dim"):
        validate_model_config(model_cfg)


def test_validate_model_config_rejects_invalid_cls_embedding_type():
    model_cfg = _valid_model_config()
    model_cfg.cls = ClsConfig(downstream="tokens", embedding_type="foo")

    with pytest.raises(ValueError, match="embedding_type must be null/none or 'bert'"):
        validate_model_config(model_cfg)


def test_validate_model_config_requires_cls_embedding_for_cls_downstream():
    model_cfg = _valid_model_config()
    model_cfg.cls = ClsConfig(downstream="cls", embedding_type=None)

    with pytest.raises(ValueError, match="must be set when model.cls.downstream is 'cls'"):
        validate_model_config(model_cfg)


def test_validate_model_config_rejects_invalid_temporal_aggregator():
    model_cfg = _valid_model_config()
    model_cfg.head = deepcopy(model_cfg.head)
    model_cfg.head.temporal_agg.name = "invalid"

    with pytest.raises(ValueError, match="temporal_agg.name must be 'mean' or 'attn'"):
        validate_model_config(model_cfg)


def test_validate_model_config_rejects_invalid_channel_aggregator():
    model_cfg = _valid_model_config()
    model_cfg.head = deepcopy(model_cfg.head)
    model_cfg.head.channel_agg.name = "invalid"

    with pytest.raises(ValueError, match="channel_agg.name must be 'mean', 'concat', or 'gated_scalar'"):
        validate_model_config(model_cfg)
