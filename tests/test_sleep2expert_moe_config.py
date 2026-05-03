from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import typing as t

import pytest
import yaml

from sleep2expert.config import FinetuneMoeTuningConfig, MoeConfig, load_finetune_config, load_pretrain_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _base_payload() -> dict[str, t.Any]:
    return {
        "model": {
            "backbone": {
                "name": "roformer",
                "hidden_size": 16,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "vocab_size": 1,
                "config_overrides": {
                    "intermediate_size": 32,
                    "max_position_embeddings": 16,
                },
            },
            "projection": {
                "name": "simclr",
                "enabled": True,
                "hidden_dim": 16,
                "out_dim": 8,
            },
            "cls": {
                "embedding_type": None,
                "downstream": "tokens",
            },
            "channels": [
                {"name": "eeg", "input_dim": 8, "tokenizer": {"name": "linear", "out_dim": 16}},
            ],
        },
        "loss": {"name": "info_nce"},
        "data": {"max_tokens": 8},
    }


def _valid_moe_payload() -> dict[str, t.Any]:
    payload = _base_payload()
    payload["model"]["backbone"]["moe"] = {
        "enabled": True,
        "layer_indices": [1, 3],
        "num_experts": 4,
        "top_k": 2,
        "expert_hidden_size": 32,
        "router_type": "learned",
        "expert_groups": {
            "shared": [0, 1],
            "neuro": [2, 3],
        },
        "modality_to_groups": {
            "eeg": ["shared", "neuro"],
        },
        "route_consistency_layers": [3],
    }
    return payload


def _valid_finetune_payload() -> dict[str, t.Any]:
    payload = _valid_moe_payload()
    payload.pop("loss")
    payload["model"]["head"] = {
        "name": "classification",
        "channel_agg": {"name": "mean"},
        "temporal_agg": {"name": "mean"},
    }
    payload["finetune"] = {"freeze_tokenizer": True}
    return payload


def _write_config(tmp_path: Path, payload: dict[str, t.Any]) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def test_sleep2expert_dense_yaml_has_no_moe_config():
    path = REPO_ROOT / "configs" / "sleep2expert" / "sleep2vec_dense_pretrain.yaml"

    bundle = load_pretrain_config(path)

    assert bundle.model.backbone.moe is None


def test_sleep2expert_moe_yaml_parses_into_moe_config(tmp_path: Path):
    path = _write_config(tmp_path, _valid_moe_payload())

    bundle = load_pretrain_config(path)

    moe_cfg = bundle.model.backbone.moe
    assert isinstance(moe_cfg, MoeConfig)
    assert moe_cfg.enabled is True
    assert moe_cfg.layer_indices == [1, 3]
    assert moe_cfg.route_consistency_layers == [3]


def test_sleep2expert_roformer_builder_passes_moe_config(tmp_path: Path):
    pytest.importorskip("torch")
    from sleep2expert.backbones.encoder_factory import build_roformer

    path = _write_config(tmp_path, _valid_moe_payload())
    bundle = load_pretrain_config(path)
    moe_cfg = bundle.model.backbone.moe

    encoder, _ = build_roformer(bundle.model.backbone).build()

    assert encoder.config.moe is moe_cfg


def test_sleep2expert_moe_config_rejects_non_roformer_backbone(tmp_path: Path):
    payload = _valid_moe_payload()
    payload["model"]["backbone"]["name"] = "hf_bert"
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="only supported for backbone.name='roformer'"):
        load_pretrain_config(path)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"enabled": "false"}, "enabled must be a boolean"),
        ({"enabled": None}, "enabled must be a boolean"),
        ({"use_modality_group_mask": "false"}, "use_modality_group_mask must be a boolean"),
        ({"use_modality_group_mask": None}, "use_modality_group_mask must be a boolean"),
        ({"num_experts": True}, "num_experts must be an integer"),
        ({"top_k": True}, "top_k must be an integer"),
        ({"top_k": 5}, "top_k must be <= backbone.moe.num_experts"),
        ({"expert_hidden_size": True}, "expert_hidden_size must be an integer"),
        ({"router_noise": "bad"}, "router_noise must be a number"),
        ({"router_noise": -0.1}, "router_noise must be >= 0"),
        ({"load_balance_coef": "bad"}, "load_balance_coef must be a number"),
        ({"load_balance_coef": -1.0}, "load_balance_coef must be >= 0"),
        ({"expert_dropout_prob": "bad"}, "expert_dropout_prob must be a number"),
        ({"expert_dropout_prob": -0.1}, "expert_dropout_prob must be >= 0"),
        ({"expert_dropout_prob": 1.1}, "expert_dropout_prob must be <= 1"),
        ({"layer_indices": [True]}, "layer_indices must contain only integers"),
        ({"layer_indices": [0, 3]}, "layer_indices values must be within"),
        (
            {"route_consistency_coef": 0.1, "route_consistency_layers": None},
            "route_consistency_layers is required",
        ),
        ({"route_consistency_layers": [2]}, "route_consistency_layers must be a subset"),
        ({"expert_diversity_coef": 0.1}, "expert_diversity_coef is not supported yet"),
        ({"expert_groups": ["shared"]}, "expert_groups must be a mapping"),
        ({"modality_to_groups": ["eeg"]}, "modality_to_groups must be a mapping"),
        ({"modality_to_groups": {"eeg": ["missing"]}}, "references unknown groups"),
        ({"expert_groups": {"shared": [False, 1]}}, "expert_groups.shared must contain only integer expert ids"),
        ({"expert_groups": {"shared": [0, 4]}}, "expert ids must be within"),
        (
            {
                "expert_groups": {"shared": [0]},
                "modality_to_groups": {"eeg": ["shared"]},
            },
            "must expose at least top_k experts",
        ),
    ],
)
def test_sleep2expert_moe_config_rejects_invalid_settings(
    tmp_path: Path,
    update: dict[str, t.Any],
    message: str,
):
    payload = _valid_moe_payload()
    payload["model"]["backbone"]["moe"].update(deepcopy(update))
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match=message):
        load_pretrain_config(path)


def test_sleep2expert_moe_group_mask_requires_every_configured_channel(tmp_path: Path):
    payload = _valid_moe_payload()
    payload["model"]["channels"].append({"name": "ppg", "input_dim": 8, "tokenizer": {"name": "linear", "out_dim": 16}})
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="must include every configured channel"):
        load_pretrain_config(path)


def test_sleep2expert_moe_config_rejects_config_overrides_moe(tmp_path: Path):
    payload = _base_payload()
    payload["model"]["backbone"]["config_overrides"]["moe"] = {"enabled": True}
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="config_overrides.moe is not supported"):
        load_pretrain_config(path)


@pytest.mark.parametrize(
    "relative_path",
    [
        "configs/sleep2expert/moe/ablations/random_router.yaml",
        "configs/sleep2expert/moe/ablations/hard_modality_router.yaml",
        "configs/sleep2expert/moe/ablations/hard_physiology_group_router.yaml",
    ],
)
def test_sleep2expert_non_learned_router_ablations_keep_aux_losses_disabled(relative_path: str):
    bundle = load_pretrain_config(REPO_ROOT / relative_path)
    moe_cfg = bundle.model.backbone.moe

    assert moe_cfg.router_type in {"random", "hard_modality", "hard_group"}
    assert moe_cfg.router_z_loss_coef == 0.0
    assert moe_cfg.load_balance_coef == 0.0
    assert moe_cfg.modality_balance_coef == 0.0
    assert moe_cfg.route_consistency_coef == 0.0


@pytest.mark.parametrize(
    "relative_path",
    [
        "configs/sleep2expert/sleep2vec_dense_finetune_cls.yaml",
        "configs/sleep2expert/moe/sleep2expert_phase_moe_finetune_cls.yaml",
    ],
)
def test_sleep2expert_existing_finetune_yamls_do_not_enable_moe_tuning(relative_path: str):
    bundle = load_finetune_config(REPO_ROOT / relative_path)

    assert bundle.finetune.moe_tuning is None


def test_sleep2expert_finetune_moe_tuning_conservative_mode_parses(tmp_path: Path):
    payload = _valid_finetune_payload()
    payload["finetune"]["moe_tuning"] = {"mode": "conservative_full_router_frozen"}
    path = _write_config(tmp_path, payload)

    bundle = load_finetune_config(path)

    cfg = bundle.finetune.moe_tuning
    assert isinstance(cfg, FinetuneMoeTuningConfig)
    assert cfg.mode == "conservative_full_router_frozen"
    assert cfg.freeze_router is True
    assert cfg.freeze_experts is False
    assert cfg.lr_scales.head == pytest.approx(1.0)
    assert cfg.lr_scales.backbone == pytest.approx(0.1)
    assert cfg.lr_scales.experts == pytest.approx(0.1)
    assert cfg.lr_scales.routers == pytest.approx(0.0)
    assert cfg.lr_scales.tokenizers == pytest.approx(0.0)
    assert cfg.lr_scales.projection == pytest.approx(0.0)
    assert cfg.moe_regularization.enabled is False


def test_sleep2expert_finetune_moe_tuning_head_only_allows_dense_config(tmp_path: Path):
    payload = _valid_finetune_payload()
    payload["model"]["backbone"].pop("moe")
    payload["finetune"]["moe_tuning"] = {"mode": "head_only"}
    path = _write_config(tmp_path, payload)

    bundle = load_finetune_config(path)

    cfg = bundle.finetune.moe_tuning
    assert cfg is not None
    assert cfg.mode == "head_only"
    assert cfg.lr_scales.head == pytest.approx(1.0)
    assert cfg.lr_scales.backbone == pytest.approx(0.0)
    assert cfg.lr_scales.experts == pytest.approx(0.0)


def test_sleep2expert_finetune_moe_tuning_rejects_dense_moe_regularization(tmp_path: Path):
    payload = _valid_finetune_payload()
    payload["model"]["backbone"].pop("moe")
    payload["finetune"]["moe_tuning"] = {
        "mode": "head_only",
        "moe_regularization": {
            "enabled": True,
            "collect_train_moe_aux": True,
            "router_z_loss_coef": 0.1,
        },
    }
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="moe_regularization.enabled requires model.backbone.moe.enabled=true"):
        load_finetune_config(path)


@pytest.mark.parametrize(
    ("moe_tuning", "message"),
    [
        ({"mode": "bad"}, "mode must be one of"),
        ({"mode": "head_only", "lr_scales": {"backbone": -0.1}}, "lr_scales.backbone must be >= 0"),
        ({"mode": "head_only", "lr_scales": {"head": True}}, "lr_scales.head must be a number"),
        ({"mode": "custom", "freeze_router": True}, "custom requires explicit"),
        ({"mode": "head_only", "freeze_router": True}, "only supported in custom mode"),
        (
            {"mode": "conservative_full_router_frozen", "train_moe_layer_indices": [3]},
            "train_moe_layer_indices is only supported",
        ),
    ],
)
def test_sleep2expert_finetune_moe_tuning_rejects_invalid_settings(
    tmp_path: Path,
    moe_tuning: dict[str, t.Any],
    message: str,
):
    payload = _valid_finetune_payload()
    payload["finetune"]["moe_tuning"] = deepcopy(moe_tuning)
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match=message):
        load_finetune_config(path)


def test_sleep2expert_finetune_moe_tuning_rejects_top_layer_mode_without_moe(tmp_path: Path):
    payload = _valid_finetune_payload()
    payload["model"]["backbone"].pop("moe")
    payload["finetune"]["moe_tuning"] = {"mode": "top_moe_layer_expert_only"}
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="requires model.backbone.moe.enabled=true"):
        load_finetune_config(path)


def test_sleep2expert_finetune_moe_tuning_rejects_invalid_top_layer_index(tmp_path: Path):
    payload = _valid_finetune_payload()
    payload["finetune"]["moe_tuning"] = {
        "mode": "top_moe_layer_expert_only",
        "train_moe_layer_indices": [4],
    }
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="must be a subset"):
        load_finetune_config(path)


def test_sleep2expert_finetune_moe_tuning_defaults_to_last_moe_layer(tmp_path: Path):
    payload = _valid_finetune_payload()
    payload["finetune"]["moe_tuning"] = {"mode": "top_moe_layer_expert_only"}
    path = _write_config(tmp_path, payload)

    bundle = load_finetune_config(path)

    cfg = bundle.finetune.moe_tuning
    assert cfg is not None
    assert cfg.train_moe_layer_indices == [3]
    assert cfg.lr_scales.head == pytest.approx(1.0)
    assert cfg.lr_scales.experts == pytest.approx(0.1)
    assert cfg.lr_scales.backbone == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("field_name", "message"),
    [
        ("route_consistency_coef", "downstream route consistency is not supported yet"),
        ("load_balance_coef", "downstream load balancing is not supported yet"),
        ("modality_balance_coef", "downstream modality balancing is not supported yet"),
        ("entropy_coef", "downstream entropy regularization is not supported yet"),
    ],
)
def test_sleep2expert_finetune_moe_tuning_rejects_unsupported_downstream_regularizers(
    tmp_path: Path,
    field_name: str,
    message: str,
):
    payload = _valid_finetune_payload()
    payload["finetune"]["moe_tuning"] = {
        "mode": "conservative_full_router_frozen",
        "moe_regularization": {field_name: 0.1},
    }
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match=message):
        load_finetune_config(path)


def test_sleep2expert_finetune_moe_tuning_requires_train_aux_when_regularization_enabled(tmp_path: Path):
    payload = _valid_finetune_payload()
    payload["finetune"]["moe_tuning"] = {
        "mode": "conservative_full_router_frozen",
        "moe_regularization": {"enabled": True},
    }
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="collect_train_moe_aux must be true"):
        load_finetune_config(path)
