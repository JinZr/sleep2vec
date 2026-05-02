from __future__ import annotations

from argparse import Namespace

import pytest

torch = pytest.importorskip("torch")

from sleep2expert.backbones.roformer.moe import MoERoutingOutput
from sleep2expert.config import (
    BackboneConfig,
    ChannelConfig,
    ClsConfig,
    LossConfig,
    ModelConfig,
    MoeConfig,
    ProjectionConfig,
    TokenizerConfig,
)
from sleep2expert.losses.base import LossOutput
from sleep2expert.losses.moe_regularization import compute_moe_regularization


def _moe_config(**updates) -> MoeConfig:
    cfg = MoeConfig(
        enabled=True,
        layer_indices=[1, 3],
        num_experts=4,
        top_k=2,
        expert_hidden_size=8,
        route_consistency_layers=[3],
        load_balance_coef=0.1,
        modality_balance_coef=0.2,
        router_z_loss_coef=0.3,
        router_entropy_coef=0.0,
        route_consistency_coef=0.4,
        expert_groups={
            "neuro": [0, 1],
            "cardiac": [2, 3],
        },
        modality_to_groups={
            "eeg": ["neuro"],
            "ppg": ["cardiac"],
        },
    )
    for key, value in updates.items():
        setattr(cfg, key, value)
    return cfg


def _shared_support_moe_config(**updates) -> MoeConfig:
    return _moe_config(
        expert_groups={"shared": [0, 1, 2, 3]},
        modality_to_groups={"eeg": ["shared"], "ppg": ["shared"]},
        **updates,
    )


def _batch(*, batch_size: int = 2, seq_len: int = 3) -> dict:
    return {
        "tokens": {
            "eeg": torch.zeros(batch_size, seq_len, 1),
            "ppg": torch.zeros(batch_size, seq_len, 1),
        },
        "length": torch.full((batch_size,), seq_len, dtype=torch.long),
    }


def _routing_aux(
    router_probs: torch.Tensor,
    *,
    layer_idx: int = 3,
    modality_name: str = "eeg",
    attention_mask: torch.Tensor | None = None,
    z_loss: float = 0.25,
) -> MoERoutingOutput:
    topk_probs, topk_indices = torch.topk(router_probs, k=2, dim=-1)
    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(router_probs.dtype).eps)
    expert_mask = torch.zeros_like(router_probs, dtype=torch.bool)
    expert_mask.scatter_(-1, topk_indices, True)

    if attention_mask is None:
        valid_mask = torch.ones(router_probs.shape[:2], dtype=torch.bool, device=router_probs.device)
    else:
        valid_mask = attention_mask.to(device=router_probs.device, dtype=torch.bool)
    valid_weight = valid_mask.to(dtype=router_probs.dtype).unsqueeze(-1)
    load = (expert_mask.to(dtype=router_probs.dtype) * valid_weight).sum(dim=(0, 1))
    importance = (router_probs * valid_weight).sum(dim=(0, 1))
    token_count = valid_mask.to(dtype=router_probs.dtype).sum().clamp_min(1.0)
    entropy_per_token = -(router_probs * router_probs.clamp_min(torch.finfo(router_probs.dtype).eps).log()).sum(dim=-1)

    return MoERoutingOutput(
        router_logits=router_probs.clamp_min(torch.finfo(router_probs.dtype).eps).log(),
        router_probs=router_probs,
        topk_indices=topk_indices,
        topk_probs=topk_probs,
        expert_mask=expert_mask,
        load=load,
        importance=importance,
        z_loss=torch.tensor(z_loss, dtype=router_probs.dtype, device=router_probs.device),
        entropy=(entropy_per_token * valid_mask.to(dtype=router_probs.dtype)).sum() / token_count,
        modality_name=modality_name,
        layer_idx=layer_idx,
    )


def _records(
    first_probs: torch.Tensor | None = None,
    second_probs: torch.Tensor | None = None,
    *,
    attention_mask: torch.Tensor | None = None,
) -> list[dict]:
    if first_probs is None:
        first_probs = torch.tensor(
            [
                [[0.70, 0.20, 0.10, 0.00], [0.15, 0.70, 0.10, 0.05], [0.20, 0.20, 0.50, 0.10]],
                [[0.60, 0.25, 0.10, 0.05], [0.10, 0.20, 0.60, 0.10], [0.25, 0.25, 0.25, 0.25]],
            ],
            dtype=torch.float32,
        )
    if second_probs is None:
        second_probs = torch.tensor(
            [
                [[0.20, 0.60, 0.15, 0.05], [0.50, 0.20, 0.20, 0.10], [0.10, 0.20, 0.60, 0.10]],
                [[0.25, 0.25, 0.40, 0.10], [0.45, 0.15, 0.30, 0.10], [0.10, 0.65, 0.20, 0.05]],
            ],
            dtype=torch.float32,
        )
    first = _routing_aux(first_probs, layer_idx=3, modality_name="eeg", attention_mask=attention_mask)
    second = _routing_aux(second_probs, layer_idx=3, modality_name="ppg", attention_mask=attention_mask)
    return [
        {"modality": "eeg", "aux": (first,), "attention_mask": attention_mask},
        {"modality": "ppg", "aux": (second,), "attention_mask": attention_mask},
    ]


def test_sleep2expert_moe_regularization_disabled_returns_zero_but_enabled_requires_aux():
    disabled = compute_moe_regularization(None, _moe_config(enabled=False), {}, prefix=None)

    assert disabled.loss.item() == 0.0
    assert disabled.metrics == {}
    with pytest.raises(ValueError, match="requires model.last_moe_aux"):
        compute_moe_regularization(None, _moe_config(), _batch(), prefix=None)


def test_sleep2expert_moe_regularization_reports_finite_components():
    out = compute_moe_regularization(_records(), _moe_config(router_entropy_coef=0.1), _batch())

    assert torch.isfinite(out.loss)
    for key in [
        "moe_load_balance_loss",
        "moe_modality_balance_loss",
        "moe_route_consistency_loss",
        "moe_router_z_loss",
        "moe_entropy",
        "moe_expert_diversity_loss",
        "moe_expert_usage_entropy",
        "moe_active_experts_per_token",
    ]:
        assert key in out.metrics
        assert torch.isfinite(out.metrics[key])
    assert out.metrics["moe_active_experts_per_token"].item() == pytest.approx(2.0)


def test_sleep2expert_moe_modality_balance_uses_each_modality_allowed_experts():
    first_probs = torch.tensor([[[0.50, 0.50, 0.00, 0.00], [0.50, 0.50, 0.00, 0.00]]])
    second_probs = torch.tensor([[[0.00, 0.00, 0.50, 0.50], [0.00, 0.00, 0.50, 0.50]]])
    out = compute_moe_regularization(
        _records(first_probs, second_probs),
        _moe_config(
            load_balance_coef=0.0,
            modality_balance_coef=1.0,
            router_z_loss_coef=0.0,
            route_consistency_coef=0.0,
        ),
        _batch(batch_size=1, seq_len=2),
    )

    assert out.metrics["moe_modality_balance_loss"].item() == pytest.approx(1.0)


def test_sleep2expert_moe_route_consistency_loss_is_finite_for_two_views():
    out = compute_moe_regularization(_records(), _shared_support_moe_config(route_consistency_coef=1.0), _batch())

    assert torch.isfinite(out.metrics["moe_route_consistency_loss"])
    assert out.metrics["moe_route_consistency_loss"].item() > 0


def test_sleep2expert_moe_route_consistency_skips_disjoint_group_support():
    out = compute_moe_regularization(_records(), _moe_config(route_consistency_coef=1.0), _batch())

    assert out.metrics["moe_route_consistency_loss"].item() == pytest.approx(0.0)


def test_sleep2expert_moe_route_consistency_ignores_padding_tokens():
    attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    first_probs = torch.tensor([[[0.80, 0.20, 0.00, 0.00], [0.10, 0.80, 0.10, 0.00], [1.00, 0.00, 0.00, 0.00]]])
    second_probs = torch.tensor([[[0.80, 0.20, 0.00, 0.00], [0.10, 0.80, 0.10, 0.00], [0.00, 1.00, 0.00, 0.00]]])
    batch = _batch(batch_size=1, seq_len=3)

    out = compute_moe_regularization(
        _records(first_probs, second_probs, attention_mask=attention_mask),
        _shared_support_moe_config(route_consistency_coef=1.0),
        batch,
    )

    assert out.metrics["moe_route_consistency_loss"].item() == pytest.approx(0.0)


def test_sleep2expert_moe_route_consistency_excludes_cls_when_present():
    attention_mask = torch.ones(1, 4, dtype=torch.bool)
    first_probs = torch.tensor(
        [[[1.00, 0.00, 0.00, 0.00], [0.60, 0.40, 0.00, 0.00], [0.25, 0.25, 0.25, 0.25], [0.20, 0.20, 0.50, 0.10]]]
    )
    second_probs = torch.tensor(
        [[[0.00, 1.00, 0.00, 0.00], [0.60, 0.40, 0.00, 0.00], [0.25, 0.25, 0.25, 0.25], [0.20, 0.20, 0.50, 0.10]]]
    )
    batch = _batch(batch_size=1, seq_len=3)

    out = compute_moe_regularization(
        _records(first_probs, second_probs, attention_mask=attention_mask),
        _shared_support_moe_config(route_consistency_coef=1.0),
        batch,
    )

    assert out.metrics["moe_route_consistency_loss"].item() == pytest.approx(0.0)


def test_sleep2expert_moe_regularization_excludes_cls_from_entropy():
    attention_mask = torch.ones(1, 4, dtype=torch.bool)
    router_probs = torch.tensor(
        [[[1.00, 0.00, 0.00, 0.00], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]]
    )
    out = compute_moe_regularization(
        _records(router_probs, router_probs, attention_mask=attention_mask),
        _moe_config(
            use_modality_group_mask=False,
            load_balance_coef=0.0,
            modality_balance_coef=0.0,
            router_z_loss_coef=0.0,
            route_consistency_coef=0.0,
        ),
        _batch(batch_size=1, seq_len=3),
    )

    assert out.metrics["moe_entropy"].item() == pytest.approx(torch.log(torch.tensor(4.0)).item())


def test_sleep2expert_moe_route_consistency_requires_configured_layers():
    with pytest.raises(ValueError, match="requires route_consistency_layers"):
        compute_moe_regularization(
            _records(),
            _moe_config(route_consistency_coef=1.0, route_consistency_layers=None),
            _batch(),
        )


def test_sleep2expert_moe_route_consistency_rejects_missing_aux_layer():
    records = _records()
    records[0]["aux"] = (_routing_aux(records[0]["aux"][0].router_probs, layer_idx=1, modality_name="eeg"),)

    with pytest.raises(ValueError, match="layer 3 is missing"):
        compute_moe_regularization(records, _moe_config(route_consistency_coef=1.0), _batch())


def test_sleep2expert_moe_expert_diversity_nonzero_fails_fast():
    with pytest.raises(ValueError, match="expert_diversity_coef is not supported yet"):
        compute_moe_regularization(_records(), _moe_config(expert_diversity_coef=0.1), _batch())


def test_sleep2expert_pretraining_adds_moe_regularization_and_logs_metrics(monkeypatch):
    pytest.importorskip("pytorch_lightning")
    from sleep2expert.sleep2vec_modelling import Sleep2vecPretraining

    moe_cfg = _moe_config(load_balance_coef=0.5, modality_balance_coef=0.0, route_consistency_coef=0.0)
    model_config = ModelConfig(
        channels=[
            ChannelConfig(name="eeg", input_dim=4, tokenizer=TokenizerConfig(name="linear", out_dim=8)),
            ChannelConfig(name="ppg", input_dim=4, tokenizer=TokenizerConfig(name="linear", out_dim=8)),
        ],
        backbone=BackboneConfig(
            name="roformer",
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            config_overrides={"intermediate_size": 16, "max_position_embeddings": 8},
            moe=moe_cfg,
        ),
        projection=ProjectionConfig(name="simclr", enabled=False, hidden_dim=8, out_dim=4),
        cls=ClsConfig(downstream="tokens", embedding_type=None),
    )
    module = Sleep2vecPretraining(
        Namespace(print_diagnostics=False, diagnostics_steps=5, weight_decay=0.0, lr=1e-3),
        model_config,
        LossConfig(name="info_nce", temperature=0.2),
    )

    class FixedLoss(torch.nn.Module):
        def forward(self, first_hidden, second_hidden, batch):
            loss = torch.tensor(2.0)
            return LossOutput(
                loss=loss,
                metrics={"contrastive_loss": loss.detach(), "contrastive_acc": torch.tensor(0.5)},
            )

    class DummyModel:
        def __init__(self, last_moe_aux):
            self.last_moe_aux = last_moe_aux

        def __call__(self, batch, apply_mask):
            return torch.zeros(2, 3, 8), torch.zeros(2, 3, 8)

    records = _records()
    dummy = DummyModel(records)
    batch = _batch()
    expected_moe = compute_moe_regularization(records, moe_cfg, batch).loss
    logged = {}
    monkeypatch.setattr(module, "log", lambda name, value, **kwargs: logged.setdefault(name, value.detach()))
    module.loss_fn = FixedLoss()

    loss, acc = module._contrastive_step(batch, log_prefix="train", model=dummy)

    assert torch.allclose(loss, torch.tensor(2.0) + expected_moe)
    assert acc.item() == pytest.approx(0.5)
    assert logged["train_contrastive_loss"].item() == pytest.approx(2.0)
    assert "train_moe_load_balance_loss" in logged
