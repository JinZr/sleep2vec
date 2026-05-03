from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")

from sleep2expert.config import (
    BackboneConfig,
    ChannelAggConfig,
    ChannelConfig,
    ClsConfig,
    FinetuneConfig,
    FinetuneLrScalesConfig,
    FinetuneMoeRegularizationConfig,
    FinetuneMoeTuningConfig,
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


def _args(*, freeze_tokenizer: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        device="cpu",
        label_name="stage5",
        output_dim=5,
        is_classification=True,
        is_seq=True,
        pretrained_backbone_path=None,
        freeze_backbone_and_insert_lora=False,
        insert_lora=False,
        separate_adapters=False,
        freeze_tokenizer=freeze_tokenizer,
        print_diagnostics=False,
        diagnostics_steps=5,
        is_multilabel=False,
        lr=1e-3,
        weight_decay=0.01,
        warmup_steps=0,
    )


def _tuning(
    mode: str,
    *,
    train_moe_layer_indices: list[int] | None = None,
    moe_regularization: FinetuneMoeRegularizationConfig | None = None,
) -> FinetuneMoeTuningConfig:
    lr_scales = FinetuneLrScalesConfig()
    if mode == "head_only":
        lr_scales = FinetuneLrScalesConfig(
            head=1.0,
            backbone=0.0,
            experts=0.0,
            routers=0.0,
            tokenizers=0.0,
            projection=0.0,
        )
    elif mode == "conservative_full_router_trainable":
        lr_scales = FinetuneLrScalesConfig(
            head=1.0,
            backbone=0.1,
            experts=0.1,
            routers=0.01,
            tokenizers=0.0,
            projection=0.0,
        )
    elif mode == "top_moe_layer_expert_only":
        lr_scales = FinetuneLrScalesConfig(
            head=1.0,
            backbone=0.0,
            experts=0.1,
            routers=0.0,
            tokenizers=0.0,
            projection=0.0,
        )
    kwargs = {}
    if moe_regularization is not None:
        kwargs["moe_regularization"] = moe_regularization
    return FinetuneMoeTuningConfig(
        mode=mode,
        train_moe_layer_indices=train_moe_layer_indices,
        lr_scales=lr_scales,
        **kwargs,
    )


def _module(
    mode: str | None,
    *,
    freeze_tokenizer: bool = True,
    train_moe_layer_indices: list[int] | None = None,
    moe_regularization: FinetuneMoeRegularizationConfig | None = None,
) -> Sleep2vecFinetuning:
    finetune_config = FinetuneConfig(
        freeze_tokenizer=freeze_tokenizer,
        moe_tuning=(
            _tuning(
                mode,
                train_moe_layer_indices=train_moe_layer_indices,
                moe_regularization=moe_regularization,
            )
            if mode is not None
            else None
        ),
    )
    return Sleep2vecFinetuning(
        _args(freeze_tokenizer=freeze_tokenizer),
        _model_config(),
        finetune_config=finetune_config,
    )


def _named_params(module: Sleep2vecFinetuning, fragment: str) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, param) for name, param in module.model.named_parameters() if fragment in name]


def _set_fake_trainer(module: Sleep2vecFinetuning) -> None:
    module._trainer = SimpleNamespace(estimated_stepping_batches=100)


def test_head_only_freezes_entire_backbone():
    module = _module("head_only")

    assert all(not param.requires_grad for _, param in _named_params(module, "backbone."))
    assert any(param.requires_grad for _, param in _named_params(module, "head."))


def test_conservative_router_frozen_freezes_tokenizers_and_routers_but_trains_experts():
    module = _module("conservative_full_router_frozen")

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


def test_router_trainable_mode_trains_router_with_experts():
    module = _module("conservative_full_router_trainable")

    routers = _named_params(module, "moe_ffn.router.")
    experts = _named_params(module, "moe_ffn.experts.")
    assert routers
    assert experts
    assert all(param.requires_grad for _, param in routers)
    assert all(param.requires_grad for _, param in experts)


def test_top_moe_layer_expert_only_trains_only_selected_layer_experts():
    module = _module("top_moe_layer_expert_only", train_moe_layer_indices=[3])

    selected = _named_params(module, "backbone.encoder.encoder.layer.2.moe_ffn.experts.")
    unselected = _named_params(module, "backbone.encoder.encoder.layer.0.moe_ffn.experts.")
    assert selected
    assert unselected
    assert all(param.requires_grad for _, param in selected)
    assert all(not param.requires_grad for _, param in unselected)
    assert all(not param.requires_grad for _, param in _named_params(module, "moe_ffn.router."))
    assert all(not param.requires_grad for _, param in _named_params(module, "embedding_projection."))
    assert any(param.requires_grad for _, param in _named_params(module, "head."))


def test_projection_and_mask_embeddings_are_frozen_under_moe_tuning():
    module = _module("conservative_full_router_frozen")

    projection = _named_params(module, "backbone.proj_head.")
    mask_embeddings = _named_params(module, "backbone.mask_embed.")
    assert projection
    assert mask_embeddings
    assert all(not param.requires_grad for _, param in projection)
    assert all(not param.requires_grad for _, param in mask_embeddings)


def test_absent_moe_tuning_preserves_existing_trainability_except_freeze_tokenizer():
    module = _module(None, freeze_tokenizer=True)

    assert module._finetune_param_to_group == {}
    assert all(not param.requires_grad for _, param in _named_params(module, "tokenizer_mapping."))
    assert all(param.requires_grad for _, param in _named_params(module, "moe_ffn.router."))
    assert all(param.requires_grad for _, param in _named_params(module, "moe_ffn.experts."))
    assert all(param.requires_grad for _, param in _named_params(module, "backbone.proj_head."))


def test_configure_optimizers_uses_semantic_lr_multipliers():
    module = _module("conservative_full_router_trainable")
    _set_fake_trainer(module)

    optimizers, _ = module.configure_optimizers()
    groups = {group["name"]: group["lr"] for group in optimizers[0].param_groups}

    assert groups["head/decay"] == pytest.approx(module.args.lr)
    assert groups["backbone/decay"] == pytest.approx(module.args.lr * 0.1)
    assert groups["experts/decay"] == pytest.approx(module.args.lr * 0.1)
    assert groups["routers/decay"] == pytest.approx(module.args.lr * 0.01)


def test_zero_lr_groups_are_not_in_optimizer():
    module = _module("conservative_full_router_frozen")
    _set_fake_trainer(module)

    optimizers, _ = module.configure_optimizers()
    semantic_groups = {group["name"].split("/")[0] for group in optimizers[0].param_groups}

    assert {"head", "backbone", "experts"}.issubset(semantic_groups)
    assert "routers" not in semantic_groups
    assert "tokenizers" not in semantic_groups
    assert "projection" not in semantic_groups


def test_no_moe_tuning_uses_legacy_two_groups():
    module = _module(None, freeze_tokenizer=False)
    _set_fake_trainer(module)

    optimizers, _ = module.configure_optimizers()
    optimizer_groups = optimizers[0].param_groups

    assert len(optimizer_groups) == 2
    assert all("name" not in group for group in optimizer_groups)
    assert all(group["lr"] == pytest.approx(module.args.lr) for group in optimizer_groups)


def test_shared_step_adds_downstream_moe_zloss_only_when_enabled(monkeypatch):
    reg_cfg = FinetuneMoeRegularizationConfig(
        enabled=True,
        collect_train_moe_aux=True,
        router_z_loss_coef=0.1,
    )
    module = _module("conservative_full_router_trainable", moe_regularization=reg_cfg)
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


def test_main_conservative_mode_does_not_collect_train_aux_or_add_moe_loss(monkeypatch):
    module = _module("conservative_full_router_frozen")
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
