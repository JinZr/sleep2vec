from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")

import torch.nn as nn

from sleep2expert import downstream_model as downstream_module
from sleep2expert.backbones.roformer.moe import MoERoutingOutput
from sleep2expert.config import (
    BackboneConfig,
    ChannelAggConfig,
    ChannelConfig,
    ClsConfig,
    FinetuneConfig,
    FinetuneGroupConfig,
    FinetuneMoeRegularizationConfig,
    FinetuneTuningConfig,
    FinetuneTuningLoraConfig,
    FinetuneTuningMoeConfig,
    HeadConfig,
    ModelConfig,
    MoeConfig,
    ProjectionConfig,
    TemporalAggConfig,
    TokenizerConfig,
)
import sleep2expert.downstreams.heads  # noqa: F401
from sleep2expert.losses.base import LossOutput
from sleep2expert.sleep2vec_finetuning import Sleep2vecFinetuning


def _moe_config() -> MoeConfig:
    return MoeConfig(
        enabled=True,
        layer_indices=[1, 3],
        num_experts=4,
        top_k=2,
        expert_hidden_size=16,
        router_type="learned",
        use_modality_group_mask=False,
    )


def _model_config() -> ModelConfig:
    return ModelConfig(
        channels=[
            ChannelConfig(name="eeg", input_dim=8, tokenizer=TokenizerConfig(name="linear", out_dim=16)),
            ChannelConfig(name="ppg", input_dim=8, tokenizer=TokenizerConfig(name="linear", out_dim=16)),
        ],
        backbone=BackboneConfig(
            name="roformer",
            hidden_size=16,
            num_hidden_layers=3,
            num_attention_heads=4,
            vocab_size=1,
            config_overrides={
                "intermediate_size": 32,
                "hidden_dropout_prob": 0.0,
                "attention_probs_dropout_prob": 0.0,
                "max_position_embeddings": 16,
            },
            moe=_moe_config(),
        ),
        projection=ProjectionConfig(name="simclr", enabled=True, hidden_dim=16, out_dim=8),
        cls=ClsConfig(downstream="tokens", embedding_type=None),
        head=HeadConfig(
            name="classification",
            channel_agg=ChannelAggConfig(name="mean"),
            temporal_agg=TemporalAggConfig(name="mean"),
            hidden_dim=8,
            dropout=0.0,
        ),
    )


class _FakePeftEncoder(nn.Module):
    def __init__(self, base_encoder: nn.Module, cfg):
        super().__init__()
        self.base_encoder = base_encoder
        self.peft_config = {"default": cfg}
        self.lora_A = nn.ModuleDict({"default": nn.Linear(1, 1, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(1, 1, bias=False)})


def _fake_get_peft_model(encoder, cfg):
    return _FakePeftEncoder(encoder, cfg)


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        device="cpu",
        label_name="stage5",
        output_dim=5,
        is_classification=True,
        is_seq=True,
        pretrained_backbone_path=None,
        print_diagnostics=False,
        diagnostics_steps=5,
        is_multilabel=False,
        lr=1e-3,
        weight_decay=0.01,
        warmup_steps=0,
    )


def _groups(**overrides: tuple[bool, float]) -> dict[str, FinetuneGroupConfig]:
    """Spell out a full group table the way `preset: custom` requires."""
    table = {
        "head": (True, 1.0),
        "encoder": (True, 1.0),
        "tokenizers": (True, 1.0),
        "experts": (True, 1.0),
        "routers": (True, 1.0),
        "projection": (True, 1.0),
        "lora": (False, 1.0),
    }
    table.update(overrides)
    return {group: FinetuneGroupConfig(train=train, lr_scale=lr_scale) for group, (train, lr_scale) in table.items()}


def _module(
    preset: str,
    *,
    groups: dict[str, FinetuneGroupConfig] | None = None,
    layer_indices: list[int] | None = None,
    moe_regularization: FinetuneMoeRegularizationConfig | None = None,
    lora_target_modules: list[str] | None = None,
) -> Sleep2vecFinetuning:
    """Build a finetuning module from a `finetune.tuning` policy.

    `groups` is the resolved table the config loader would have produced; the tests pass
    it directly so they exercise the runtime pass, not the preset lookup.
    """
    tuning = FinetuneTuningConfig(
        preset=preset,
        groups=groups if groups is not None else _groups(),
        lora=FinetuneTuningLoraConfig(
            r=4,
            alpha=12,
            dropout=0.15,
            target_modules=lora_target_modules or ["query", "key", "value"],
            use_dora=False,
        ),
        moe=FinetuneTuningMoeConfig(layer_indices=layer_indices),
    )
    kwargs = {}
    if moe_regularization is not None:
        kwargs["moe_regularization"] = moe_regularization
    finetune_config = FinetuneConfig(tuning=tuning, **kwargs)
    return Sleep2vecFinetuning(_args(), _model_config(), finetune_config=finetune_config)


def _named_params(module: Sleep2vecFinetuning, fragment: str) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, param) for name, param in module.model.named_parameters() if fragment in name]


def _set_fake_trainer(module: Sleep2vecFinetuning) -> None:
    module._trainer = SimpleNamespace(estimated_stepping_batches=100)


def _routing_aux(router_probs: torch.Tensor, *, layer_idx: int = 1) -> MoERoutingOutput:
    topk_probs, topk_indices = torch.topk(router_probs, k=2, dim=-1)
    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(router_probs.dtype).eps)
    expert_mask = torch.zeros_like(router_probs, dtype=torch.bool)
    expert_mask.scatter_(-1, topk_indices, True)
    load = expert_mask.float().sum(dim=(0, 1))
    importance = router_probs.sum(dim=(0, 1))
    entropy = -(router_probs * router_probs.clamp_min(torch.finfo(router_probs.dtype).eps).log()).sum(dim=-1)
    return MoERoutingOutput(
        router_logits=router_probs.clamp_min(torch.finfo(router_probs.dtype).eps).log() + 0.5,
        router_probs=router_probs,
        topk_indices=topk_indices,
        topk_probs=topk_probs,
        expert_mask=expert_mask,
        load=load,
        importance=importance,
        z_loss=torch.tensor(0.25),
        entropy=entropy.mean(),
        modality_name="eeg",
        layer_idx=layer_idx,
    )


def test_head_only_freezes_entire_backbone():
    module = _module(
        "head_only",
        groups=_groups(
            encoder=(False, 1.0),
            tokenizers=(False, 1.0),
            experts=(False, 1.0),
            routers=(False, 1.0),
            projection=(False, 1.0),
        ),
    )

    assert all(not param.requires_grad for _, param in _named_params(module, "backbone."))
    assert any(param.requires_grad for _, param in _named_params(module, "head."))


def _conservative_groups(*, routers: tuple[bool, float] = (False, 1.0)) -> dict[str, FinetuneGroupConfig]:
    return _groups(
        encoder=(True, 0.1),
        tokenizers=(False, 1.0),
        experts=(True, 0.1),
        routers=routers,
        projection=(False, 1.0),
    )


def test_moe_conservative_freezes_tokenizers_and_routers_but_trains_experts():
    module = _module("moe_conservative", groups=_conservative_groups())

    tokenizers = _named_params(module, "tokenizer_mapping.")
    routers = _named_params(module, "moe_ffn.router.")
    experts = _named_params(module, "moe_ffn.experts.")
    assert tokenizers
    assert routers
    assert experts
    assert all(not param.requires_grad for _, param in tokenizers)
    assert all(not param.requires_grad for _, param in routers)
    assert all(param.requires_grad for _, param in experts)
    assert any(param.requires_grad for _, param in _named_params(module, "embedding_projection."))


def test_moe_conservative_routers_trains_router_with_experts():
    module = _module("moe_conservative_routers", groups=_conservative_groups(routers=(True, 0.01)))

    routers = _named_params(module, "moe_ffn.router.")
    experts = _named_params(module, "moe_ffn.experts.")
    assert routers
    assert experts
    assert all(param.requires_grad for _, param in routers)
    assert all(param.requires_grad for _, param in experts)


def test_moe_top_experts_trains_only_selected_layer_experts():
    module = _module(
        "moe_top_experts",
        groups=_groups(
            encoder=(False, 1.0),
            tokenizers=(False, 1.0),
            experts=(True, 0.1),
            routers=(False, 1.0),
            projection=(False, 1.0),
        ),
        layer_indices=[3],
    )

    selected = _named_params(module, "backbone.encoder.encoder.layer.2.moe_ffn.experts.")
    unselected = _named_params(module, "backbone.encoder.encoder.layer.0.moe_ffn.experts.")
    assert selected
    assert unselected
    assert all(param.requires_grad for _, param in selected)
    assert all(not param.requires_grad for _, param in unselected)
    assert all(not param.requires_grad for _, param in _named_params(module, "moe_ffn.router."))
    assert all(not param.requires_grad for _, param in _named_params(module, "embedding_projection."))
    assert any(param.requires_grad for _, param in _named_params(module, "head."))


def test_moe_top_experts_rejects_layer_indices_that_match_no_expert():
    with pytest.raises(ValueError, match="matched no expert parameters"):
        _module(
            "moe_top_experts",
            groups=_groups(
                encoder=(False, 1.0),
                tokenizers=(False, 1.0),
                experts=(True, 0.1),
                routers=(False, 1.0),
                projection=(False, 1.0),
            ),
            layer_indices=[99],
        )


def test_a_preset_that_trains_no_head_parameter_is_rejected():
    with pytest.raises(ValueError, match="left no trainable head parameters"):
        _module("custom", groups=_groups(head=(False, 1.0)))


def test_projection_and_mask_embeddings_are_frozen_under_moe_conservative():
    module = _module("moe_conservative", groups=_conservative_groups())

    projection = _named_params(module, "backbone.proj_head.")
    mask_embeddings = _named_params(module, "backbone.mask_embed.")
    assert projection
    assert mask_embeddings
    assert all(not param.requires_grad for _, param in projection)
    assert all(not param.requires_grad for _, param in mask_embeddings)


def test_full_preset_trains_everything_except_the_tokenizers_when_they_are_overridden():
    module = _module("full", groups=_groups(tokenizers=(False, 1.0)))

    assert all(not param.requires_grad for _, param in _named_params(module, "tokenizer_mapping."))
    assert all(param.requires_grad for _, param in _named_params(module, "moe_ffn.router."))
    assert all(param.requires_grad for _, param in _named_params(module, "moe_ffn.experts."))
    assert all(param.requires_grad for _, param in _named_params(module, "backbone.proj_head."))


def test_every_parameter_is_classified_into_a_declared_group():
    module = _module("full")

    names = {name for name, _ in module.model.named_parameters()}
    assert names
    assert set(module._finetune_param_to_group) == names
    assert set(module._finetune_param_to_group.values()) <= set(module.tuning_config.groups)


def test_configure_optimizers_uses_semantic_lr_multipliers():
    module = _module("moe_conservative_routers", groups=_conservative_groups(routers=(True, 0.01)))
    _set_fake_trainer(module)

    optimizers, _ = module.configure_optimizers()
    groups = {group["name"]: group["lr"] for group in optimizers[0].param_groups}

    assert groups["head/decay"] == pytest.approx(module.args.lr)
    assert groups["encoder/decay"] == pytest.approx(module.args.lr * 0.1)
    assert groups["experts/decay"] == pytest.approx(module.args.lr * 0.1)
    assert groups["routers/decay"] == pytest.approx(module.args.lr * 0.01)


def test_sleep2expert_finetune_cli_default_lr_is_sleep_staging_scale():
    tree = ast.parse((Path(__file__).parents[2] / "sleep2expert" / "finetune.py").read_text())

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or getattr(node.func, "attr", None) != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant) or node.args[0].value != "--lr":
            continue
        default = next(keyword.value for keyword in node.keywords if keyword.arg == "default")
        assert isinstance(default, ast.Constant)
        assert default.value == pytest.approx(1e-4)
        return

    pytest.fail("sleep2expert.finetune --lr argument not found")


def test_frozen_groups_are_not_in_optimizer():
    module = _module("moe_conservative", groups=_conservative_groups())
    _set_fake_trainer(module)

    optimizers, _ = module.configure_optimizers()
    semantic_groups = {group["name"].split("/")[0] for group in optimizers[0].param_groups}

    assert {"head", "encoder", "experts"}.issubset(semantic_groups)
    assert "routers" not in semantic_groups
    assert "tokenizers" not in semantic_groups
    assert "projection" not in semantic_groups


def test_lora_uses_independent_optimizer_group(monkeypatch):
    monkeypatch.setattr(downstream_module, "get_peft_model", _fake_get_peft_model)
    module = _module(
        "custom",
        groups=_groups(
            encoder=(False, 1.0),
            tokenizers=(False, 1.0),
            experts=(True, 0.1),
            routers=(False, 1.0),
            projection=(False, 1.0),
            lora=(True, 0.25),
        ),
        lora_target_modules=["query", "key", "value", "dense_in", "dense_out"],
    )
    _set_fake_trainer(module)

    lora_params = _named_params(module, "lora_")
    routers = _named_params(module, "moe_ffn.router.")
    experts = _named_params(module, "moe_ffn.experts.")
    assert lora_params
    assert routers
    assert experts
    assert all(param.requires_grad for _, param in lora_params)
    assert all(not param.requires_grad for _, param in routers)
    assert all(param.requires_grad for _, param in experts)

    optimizers, _ = module.configure_optimizers()
    groups = {group["name"]: group["lr"] for group in optimizers[0].param_groups}
    assert groups["lora/decay"] == pytest.approx(module.args.lr * 0.25)
    assert groups["experts/decay"] == pytest.approx(module.args.lr * 0.1)
    assert "routers/decay" not in groups
    assert module.finetune_status["param_groups"]["lora"]["trainable_params"] > 0


def test_an_untrained_lora_group_inserts_no_adapter_and_gets_no_optimizer_group(monkeypatch):
    """`train: false` is the only freeze switch; a zero lr_scale is rejected by the loader."""
    monkeypatch.setattr(downstream_module, "get_peft_model", _fake_get_peft_model)
    module = _module("moe_conservative_routers", groups=_conservative_groups(routers=(True, 0.01)))
    _set_fake_trainer(module)

    assert _named_params(module, "lora_") == []

    optimizers, _ = module.configure_optimizers()
    semantic_groups = {group["name"].split("/")[0] for group in optimizers[0].param_groups}
    assert "lora" not in semantic_groups
    assert module.finetune_status["param_groups"]["lora"]["trainable_params"] == 0


def test_status_snapshot_records_moe_conservative_trainability():
    module = _module("moe_conservative", groups=_conservative_groups())
    status = module.finetune_status
    groups = status["param_groups"]

    assert status["preset"] == "moe_conservative"
    assert status["moe_enabled"] is True
    assert status["moe_layer_indices"] == [1, 3]
    assert status["num_experts"] == 4
    assert status["top_k"] == 2
    assert status["router_type"] == "learned"
    assert status["collect_train_moe_aux"] is False
    assert groups["routers"]["train"] is False
    assert groups["routers"]["trainable_params"] == 0
    assert groups["routers"]["total_params"] > 0
    assert groups["experts"]["trainable_params"] > 0
    assert groups["encoder"]["trainable_params"] > 0
    assert groups["encoder"]["lr_scale"] == pytest.approx(0.1)

    hparams = module.finetune_hparams()
    assert hparams["finetune/preset"] == "moe_conservative"
    assert hparams["finetune/moe_layer_indices"] == "1,3"
    assert any(row[0] == "routers" for row in module.finetune_param_group_rows())


def test_status_snapshot_records_router_trainable_regularized_mode():
    reg_cfg = FinetuneMoeRegularizationConfig(
        enabled=True,
        collect_train_moe_aux=True,
        router_z_loss_coef=0.1,
    )
    module = _module(
        "moe_conservative_routers",
        groups=_conservative_groups(routers=(True, 0.01)),
        moe_regularization=reg_cfg,
    )
    status = module.finetune_status
    groups = status["param_groups"]

    assert status["preset"] == "moe_conservative_routers"
    assert status["collect_train_moe_aux"] is True
    assert status["moe_regularization"]["enabled"] is True
    assert status["moe_regularization"]["collect_train_moe_aux"] is True
    assert groups["routers"]["train"] is True
    assert groups["routers"]["trainable_params"] > 0
    assert groups["routers"]["lr_scale"] == pytest.approx(0.01)


def test_status_snapshot_covers_every_group_under_the_full_preset():
    module = _module("full")
    status = module.finetune_status

    assert set(status["param_groups"]) == set(module.tuning_config.groups)
    assert status["lr_scales"] == {group: 1.0 for group in module.tuning_config.groups}
    assert status["moe_regularization"]["enabled"] is False
    assert status["collect_train_moe_aux"] is False
    assert status["trainable_params"] == sum(group["trainable_params"] for group in status["param_groups"].values())


def test_shared_step_adds_downstream_moe_zloss_only_when_enabled(monkeypatch):
    reg_cfg = FinetuneMoeRegularizationConfig(
        enabled=True,
        collect_train_moe_aux=True,
        router_z_loss_coef=0.1,
    )
    module = _module(
        "moe_conservative_routers",
        groups=_conservative_groups(routers=(True, 0.01)),
        moe_regularization=reg_cfg,
    )
    assert module.model.collect_train_moe_aux is True

    class DummyModel:
        def __init__(self):
            self.backbone = SimpleNamespace(last_moe_aux=["aux"])

        def __call__(self, batch):
            return torch.zeros(2, 5)

    def fake_downstream_moe_regularization(moe_aux, cfg, batch, *, prefix=None):
        assert moe_aux == ["aux"]
        assert cfg is reg_cfg
        assert prefix == "train"
        return LossOutput(
            loss=torch.tensor(0.5),
            metrics={"train_downstream_moe_router_z_loss": torch.tensor(1.0)},
        )

    logged = {}
    monkeypatch.setattr(
        "sleep2expert.sleep2vec_finetuning.compute_downstream_moe_regularization",
        fake_downstream_moe_regularization,
    )
    monkeypatch.setattr(module, "log", lambda name, value, **kwargs: logged.setdefault(name, value.detach()))
    monkeypatch.setattr(module, "_compute_loss", lambda logits, batch: (torch.tensor(2.0), 4))
    monkeypatch.setattr(module, "_extract_valid_predictions", lambda batch, logits: None)

    loss = module._shared_step({}, stage="train", model=DummyModel())

    assert loss.item() == pytest.approx(2.5)
    assert logged["train_supervised_loss"].item() == pytest.approx(2.0)
    assert logged["train_downstream_moe_router_z_loss"].item() == pytest.approx(1.0)
    assert logged["train_loss"].item() == pytest.approx(2.5)


def test_eval_steps_log_downstream_moe_metrics_when_aux_is_available(monkeypatch):
    module = _module("moe_conservative", groups=_conservative_groups())
    router_probs = torch.tensor(
        [
            [[0.70, 0.20, 0.05, 0.05], [0.10, 0.70, 0.10, 0.10], [0.25, 0.25, 0.25, 0.25]],
            [[0.05, 0.05, 0.80, 0.10], [0.40, 0.20, 0.20, 0.20], [0.10, 0.20, 0.30, 0.40]],
        ],
        dtype=torch.float32,
    )
    moe_aux = [{"modality": "eeg", "aux": (_routing_aux(router_probs),), "attention_mask": None}]
    batch = {
        "tokens": {"eeg": torch.zeros(2, 3, 1)},
        "length": torch.tensor([3, 3]),
    }

    class DummyEvalModel:
        def __init__(self):
            self.backbone = SimpleNamespace(last_moe_aux=None)

        def __call__(self, batch):
            self.backbone.last_moe_aux = moe_aux
            return torch.zeros(2, 5)

    logged = {}
    monkeypatch.setattr(module, "log", lambda name, value, **kwargs: logged.setdefault(name, value.detach()))
    monkeypatch.setattr(module, "_compute_loss", lambda logits, batch: (torch.tensor(2.0), 6))
    monkeypatch.setattr(module, "_extract_valid_predictions", lambda batch, logits: None)

    assert module._shared_step(batch, stage="val", model=DummyEvalModel()) is None
    assert module._shared_step(batch, stage="test", model=DummyEvalModel()) is None

    assert logged["val_downstream_moe_router_z_loss"].item() == pytest.approx(0.25)
    assert "val_downstream_moe_entropy" in logged
    assert "val_downstream_moe_expert_usage_entropy" in logged
    assert "val_downstream_moe_active_experts_per_token" in logged
    assert logged["test_downstream_moe_router_z_loss"].item() == pytest.approx(0.25)


def test_main_conservative_mode_does_not_collect_train_aux_or_add_moe_loss(monkeypatch):
    module = _module("moe_conservative", groups=_conservative_groups())
    assert module.model.collect_train_moe_aux is False

    class DummyModel:
        def __init__(self):
            self.backbone = SimpleNamespace(last_moe_aux=None)

        def __call__(self, batch):
            return torch.zeros(2, 5)

    def fail_downstream_moe_regularization(*args, **kwargs):
        raise AssertionError("downstream MoE regularization should not run")

    logged = {}
    monkeypatch.setattr(
        "sleep2expert.sleep2vec_finetuning.compute_downstream_moe_regularization",
        fail_downstream_moe_regularization,
    )
    monkeypatch.setattr(module, "log", lambda name, value, **kwargs: logged.setdefault(name, value.detach()))
    monkeypatch.setattr(module, "_compute_loss", lambda logits, batch: (torch.tensor(2.0), 4))
    monkeypatch.setattr(module, "_extract_valid_predictions", lambda batch, logits: None)

    loss = module._shared_step({}, stage="train", model=DummyModel())

    assert loss.item() == pytest.approx(2.0)
    assert "train_supervised_loss" not in logged
    assert logged["train_loss"].item() == pytest.approx(2.0)
