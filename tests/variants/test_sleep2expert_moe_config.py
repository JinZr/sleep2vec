from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import typing as t

import pytest
import yaml

from sleep2expert.config import FinetuneTuningConfig, MoeConfig, load_finetune_config, load_pretrain_config

REPO_ROOT = Path(__file__).resolve().parents[2]


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
    payload["finetune"] = {"tuning": {"preset": "full", "groups": {"tokenizers": {"train": False}}}}
    return payload


def _write_config(tmp_path: Path, payload: dict[str, t.Any]) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def test_sleep2expert_checked_pretrain_yaml_has_moe_config():
    path = REPO_ROOT / "configs" / "sleep2expert" / "moe" / "sleep2expert_phase_moe_pretrain.yaml"

    bundle = load_pretrain_config(path)

    moe_cfg = bundle.model.backbone.moe
    assert isinstance(moe_cfg, MoeConfig)
    assert moe_cfg.enabled is True


def test_sleep2expert_required_shared_expert_pretrain_yaml_has_moe_config():
    path = REPO_ROOT / "configs" / "sleep2expert" / "moe" / "sleep2expert_phase_moe_pretrain_with_shared_expert.yaml"

    bundle = load_pretrain_config(path)

    moe_cfg = bundle.model.backbone.moe
    assert isinstance(moe_cfg, MoeConfig)
    assert moe_cfg.top_k == 3
    assert moe_cfg.required_expert_ids == [0]
    assert moe_cfg.required_expert_weight_mode == "router"
    assert moe_cfg.required_expert_weight is None
    assert moe_cfg.expert_groups["shared"] == [0]
    assert moe_cfg.expert_groups["cardiac"] == [1, 6, 7, 8]
    assert moe_cfg.expert_groups["respiratory"] == [1, 9, 10, 11]


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
        ({"required_expert_ids": [0, 0]}, "required_expert_ids must not contain duplicates"),
        ({"required_expert_ids": [True]}, "required_expert_ids must contain only integers"),
        (
            {
                "required_expert_ids": [4],
                "required_expert_weight_mode": "fixed",
                "required_expert_weight": 0.2,
            },
            "required_expert_ids values must be within",
        ),
        (
            {
                "required_expert_ids": [0, 1, 2],
                "required_expert_weight_mode": "fixed",
                "required_expert_weight": 0.2,
            },
            "required_expert_ids must not contain more experts than",
        ),
        ({"required_expert_weight_mode": "fixed"}, "required_expert_weight_mode requires"),
        ({"required_expert_weight": 0.2}, "required_expert_weight requires"),
        ({"required_expert_ids": [0]}, "required_expert_weight_mode must be one of"),
        (
            {
                "required_expert_ids": [0],
                "required_expert_weight_mode": "bad",
            },
            "required_expert_weight_mode must be one of",
        ),
        (
            {
                "required_expert_ids": [0],
                "required_expert_weight_mode": "fixed",
            },
            "required_expert_weight must be a number",
        ),
        (
            {
                "required_expert_ids": [0],
                "required_expert_weight_mode": "fixed",
                "required_expert_weight": "bad",
            },
            "required_expert_weight must be a number",
        ),
        (
            {
                "required_expert_ids": [0],
                "required_expert_weight_mode": "fixed",
                "required_expert_weight": 0.0,
            },
            "required_expert_weight must be > 0 and < 1",
        ),
        (
            {
                "required_expert_ids": [0, 1],
                "required_expert_weight_mode": "fixed",
                "required_expert_weight": 0.6,
            },
            "fixed weighting must leave a routed expert slot",
        ),
        (
            {
                "top_k": 3,
                "required_expert_ids": [0, 1],
                "required_expert_weight_mode": "fixed",
                "required_expert_weight": 0.5,
            },
            r"required_expert_weight \* len",
        ),
        (
            {
                "required_expert_ids": [0],
                "required_expert_weight_mode": "router",
                "required_expert_weight": 0.2,
            },
            "required_expert_weight must not be set",
        ),
        (
            {
                "router_type": "hard_group",
                "required_expert_ids": [0],
                "required_expert_weight_mode": "router",
            },
            "required_expert_weight_mode=router requires router_type='learned'",
        ),
        (
            {
                "router_type": "random",
                "required_expert_ids": [0],
                "required_expert_weight_mode": "router",
            },
            "required_expert_weight_mode=router requires router_type='learned'",
        ),
        ({"expert_hidden_size": True}, "expert_hidden_size must be an integer"),
        ({"router_noise": "bad"}, "router_noise must be a number"),
        ({"router_noise": float("nan")}, "router_noise must be finite"),
        ({"router_noise": -0.1}, "router_noise must be >= 0"),
        ({"load_balance_coef": "bad"}, "load_balance_coef must be a number"),
        ({"load_balance_coef": float("inf")}, "load_balance_coef must be finite"),
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


def test_sleep2expert_moe_config_rejects_required_expert_outside_modality_groups(tmp_path: Path):
    payload = _valid_moe_payload()
    payload["model"]["backbone"]["moe"].update(
        {
            "required_expert_ids": [0],
            "required_expert_weight_mode": "fixed",
            "required_expert_weight": 0.25,
            "expert_groups": {
                "shared": [1],
                "neuro": [2, 3],
            },
        }
    )
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="must expose required_expert_ids"):
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
        "configs/sleep2expert/moe/sleep2expert_phase_moe_finetune_cls.yaml",
    ],
)
def test_sleep2expert_plain_moe_finetune_yaml_trains_everything(relative_path: str):
    bundle = load_finetune_config(REPO_ROOT / relative_path)

    tuning = bundle.finetune.tuning
    assert tuning.preset == "full"
    assert tuning.trains("experts") is True
    assert tuning.trains("routers") is True
    assert bundle.finetune.moe_regularization.enabled is False


def _tuning(payload: dict[str, t.Any], **tuning_block: t.Any) -> None:
    payload["finetune"]["tuning"] = tuning_block


def test_sleep2expert_finetune_tuning_moe_conservative_preset_parses(tmp_path: Path):
    payload = _valid_finetune_payload()
    _tuning(payload, preset="moe_conservative")
    path = _write_config(tmp_path, payload)

    bundle = load_finetune_config(path)

    cfg = bundle.finetune.tuning
    assert isinstance(cfg, FinetuneTuningConfig)
    assert cfg.preset == "moe_conservative"
    assert cfg.trains("routers") is False
    assert cfg.trains("experts") is True
    assert cfg.lr_scale("head") == pytest.approx(1.0)
    assert cfg.lr_scale("encoder") == pytest.approx(0.1)
    assert cfg.lr_scale("experts") == pytest.approx(0.1)
    assert cfg.trains("tokenizers") is False
    assert cfg.trains("projection") is False
    assert bundle.finetune.moe_regularization.enabled is False


def test_sleep2expert_finetune_tuning_freezing_a_scaled_group_drops_its_scale(tmp_path: Path):
    """Freezing a group must drop the preset's scale, not inherit it.

    `moe_conservative` scales the encoder by 0.1. An override of `{train: false}` used to
    keep that 0.1, so `finetune_status` reported `train: false, lr_scale: 0.1` -- the pair
    the parser rejects when a config writes it out, and one no preset table contains.
    """
    payload = _valid_finetune_payload()
    _tuning(payload, preset="moe_conservative", groups={"encoder": {"train": False}, "experts": {"train": False}})
    path = _write_config(tmp_path, payload)

    cfg = load_finetune_config(path).finetune.tuning

    assert cfg.trains("encoder") is False
    assert cfg.lr_scale("encoder") == pytest.approx(1.0)
    assert cfg.trains("experts") is False
    assert cfg.lr_scale("experts") == pytest.approx(1.0)


@pytest.mark.parametrize("router_type", ["random", "hard_modality", "hard_group"])
def test_sleep2expert_finetune_tuning_rejects_training_a_parameterless_router(tmp_path: Path, router_type: str):
    """Only the learned router has parameters, so training any other trains nothing.

    The ablation grid is where this bites: `finetune_ablations/router_trainable.yaml` asks
    whether router adaptation helps, and pointing it at a `random` router backbone would
    answer with a run whose routers group holds zero trainable parameters.
    """
    payload = _valid_finetune_payload()
    payload["model"]["backbone"]["moe"]["router_type"] = router_type
    payload["model"]["backbone"]["moe"].pop("route_consistency_layers", None)
    _tuning(payload, preset="moe_conservative", groups={"routers": {"train": True}})
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="requires model.backbone.moe.router_type='learned'"):
        load_finetune_config(path)


@pytest.mark.parametrize("router_type", ["random", "hard_modality", "hard_group"])
def test_sleep2expert_moe_conservative_routers_preset_needs_a_learned_router(tmp_path: Path, router_type: str):
    payload = _valid_finetune_payload()
    payload["model"]["backbone"]["moe"]["router_type"] = router_type
    payload["model"]["backbone"]["moe"].pop("route_consistency_layers", None)
    _tuning(payload, preset="moe_conservative_routers")
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="requires model.backbone.moe.router_type='learned'"):
        load_finetune_config(path)


def test_sleep2expert_finetune_tuning_head_only_allows_dense_config(tmp_path: Path):
    payload = _valid_finetune_payload()
    payload["model"]["backbone"].pop("moe")
    _tuning(payload, preset="head_only")
    path = _write_config(tmp_path, payload)

    bundle = load_finetune_config(path)

    cfg = bundle.finetune.tuning
    assert cfg.preset == "head_only"
    assert cfg.trains("head") is True
    assert cfg.trains("encoder") is False
    assert cfg.trains("experts") is False


def test_sleep2expert_finetune_rejects_dense_moe_regularization(tmp_path: Path):
    payload = _valid_finetune_payload()
    payload["model"]["backbone"].pop("moe")
    _tuning(payload, preset="head_only")
    payload["finetune"]["moe_regularization"] = {
        "enabled": True,
        "collect_train_moe_aux": True,
        "router_z_loss_coef": 0.1,
    }
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="moe_regularization.enabled requires model.backbone.moe.enabled=true"):
        load_finetune_config(path)


@pytest.mark.parametrize(
    ("tuning", "message"),
    [
        ({"preset": "bad"}, "preset must be one of"),
        (
            {"preset": "head_only", "groups": {"encoder": {"train": True, "lr_scale": -0.1}}},
            "lr_scale must be > 0",
        ),
        (
            {"preset": "head_only", "groups": {"encoder": {"train": True, "lr_scale": float("nan")}}},
            "lr_scale must be finite",
        ),
        (
            {"preset": "head_only", "groups": {"head": {"train": True, "lr_scale": True}}},
            "lr_scale must be a number",
        ),
        ({"preset": "head_only", "groups": {"encoder": {"lr_scale": 0.1}}}, "train is required"),
        ({"preset": "head_only", "groups": {"nonsense": {"train": True}}}, "unknown group"),
        ({"preset": "custom"}, "requires every group to be explicit"),
        (
            {"preset": "moe_conservative", "moe": {"layer_indices": [3]}},
            "only supported for preset 'moe_top_experts'",
        ),
        ({"preset": "full", "groups": {"lora": {"train": True}}}, "cannot train the encoder and insert LoRA"),
    ],
)
def test_sleep2expert_finetune_tuning_rejects_invalid_settings(
    tmp_path: Path,
    tuning: dict[str, t.Any],
    message: str,
):
    payload = _valid_finetune_payload()
    payload["finetune"]["tuning"] = deepcopy(tuning)
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match=message):
        load_finetune_config(path)


@pytest.mark.parametrize(
    ("field_name", "message"),
    [
        ("router_z_loss_coef", "moe_regularization.router_z_loss_coef must be finite"),
    ],
)
def test_sleep2expert_finetune_moe_regularization_rejects_non_finite_coefficients(
    tmp_path: Path,
    field_name: str,
    message: str,
):
    payload = _valid_finetune_payload()
    payload["finetune"]["moe_regularization"] = {field_name: float("inf")}
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match=message):
        load_finetune_config(path)


def test_sleep2expert_finetune_tuning_rejects_top_experts_preset_without_moe(tmp_path: Path):
    payload = _valid_finetune_payload()
    payload["model"]["backbone"].pop("moe")
    _tuning(payload, preset="moe_top_experts")
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="requires model.backbone.moe.enabled=true"):
        load_finetune_config(path)


def test_sleep2expert_finetune_tuning_rejects_invalid_top_layer_index(tmp_path: Path):
    payload = _valid_finetune_payload()
    _tuning(payload, preset="moe_top_experts", moe={"layer_indices": [4]})
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="must be a subset"):
        load_finetune_config(path)


def test_sleep2expert_finetune_tuning_defaults_to_last_moe_layer(tmp_path: Path):
    payload = _valid_finetune_payload()
    _tuning(payload, preset="moe_top_experts")
    path = _write_config(tmp_path, payload)

    bundle = load_finetune_config(path)

    cfg = bundle.finetune.tuning
    assert cfg.moe.layer_indices == [3]
    assert cfg.lr_scale("head") == pytest.approx(1.0)
    assert cfg.lr_scale("experts") == pytest.approx(0.1)
    assert cfg.trains("encoder") is False


@pytest.mark.parametrize(
    ("field_name", "message"),
    [
        ("route_consistency_coef", "downstream route consistency is not supported yet"),
        ("load_balance_coef", "downstream load balancing is not supported yet"),
        ("modality_balance_coef", "downstream modality balancing is not supported yet"),
        ("entropy_coef", "downstream entropy regularization is not supported yet"),
    ],
)
def test_sleep2expert_finetune_rejects_unsupported_downstream_regularizers(
    tmp_path: Path,
    field_name: str,
    message: str,
):
    payload = _valid_finetune_payload()
    _tuning(payload, preset="moe_conservative")
    payload["finetune"]["moe_regularization"] = {field_name: 0.1}
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match=message):
        load_finetune_config(path)


def test_sleep2expert_finetune_requires_train_aux_when_regularization_enabled(tmp_path: Path):
    payload = _valid_finetune_payload()
    _tuning(payload, preset="moe_conservative")
    payload["finetune"]["moe_regularization"] = {"enabled": True}
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="collect_train_moe_aux must be true"):
        load_finetune_config(path)


def test_sleep2expert_finetune_tuning_rejects_the_legacy_keys(tmp_path: Path):
    payload = _valid_finetune_payload()
    payload["finetune"].pop("tuning")
    payload["finetune"]["moe_tuning"] = {"mode": "conservative_full_router_frozen"}
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="was replaced by finetune.tuning"):
        load_finetune_config(path)


@pytest.mark.parametrize("preset", ["moe_conservative", "moe_conservative_routers", "moe_top_experts"])
def test_sleep2expert_finetune_tuning_rejects_every_moe_preset_without_moe(preset: str, tmp_path: Path):
    """A dense backbone has no expert or router group to act on.

    Every moe_* preset would collapse to the same head-plus-encoder policy while still
    carrying a name that claims otherwise, so the run report and the logged group table
    would describe a MoE policy that never existed.
    """
    payload = _valid_finetune_payload()
    payload["model"]["backbone"].pop("moe")
    _tuning(payload, preset=preset)
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="requires model.backbone.moe.enabled=true"):
        load_finetune_config(path)
