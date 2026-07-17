from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sleep2expert.backbones.roformer import RoFormerConfig, RoFormerEncoderModel
from sleep2expert.backbones.roformer.moe import TopKRouter, apply_route_expert_filter, resolve_route_expert_ids
from sleep2expert.config import (
    BackboneConfig,
    ChannelAggConfig,
    ChannelConfig,
    ClsConfig,
    HeadConfig,
    ModelConfig,
    MoeConfig,
    ProjectionConfig,
    TemporalAggConfig,
    TokenizerConfig,
)
import sleep2expert.downstreams.heads  # noqa: F401
from sleep2expert.pretrain_model import Sleep2vecPretrainModel


def _downstream_model_cls():
    from sleep2expert.downstream_model import Sleep2vecDownstreamModel

    return Sleep2vecDownstreamModel


def _moe_config(
    *,
    router_type: str = "learned",
    top_k: int = 2,
    use_modality_group_mask: bool = True,
    required_expert_ids: list[int] | None = None,
    required_expert_weight_mode: str | None = None,
    required_expert_weight: float | None = None,
) -> MoeConfig:
    return MoeConfig(
        enabled=True,
        layer_indices=[1, 3],
        num_experts=4,
        top_k=top_k,
        expert_hidden_size=16,
        router_type=router_type,
        use_modality_group_mask=use_modality_group_mask,
        required_expert_ids=required_expert_ids,
        required_expert_weight_mode=required_expert_weight_mode,
        required_expert_weight=required_expert_weight,
        expert_groups={
            "shared": [0],
            "neuro": [2, 3],
            "cardiac": [1, 3],
        },
        modality_to_groups={
            "eeg": ["shared", "neuro"],
            "ppg": ["shared", "cardiac"],
        },
        route_consistency_layers=[3],
    )


def _model(moe: MoeConfig) -> RoFormerEncoderModel:
    config = RoFormerConfig(
        vocab_size=11,
        hidden_size=16,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=32,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=16,
        moe=moe,
    )
    return RoFormerEncoderModel(config)


def _sleep2expert_model_config(*, moe: MoeConfig | None = None, with_head: bool = False) -> ModelConfig:
    head = None
    if with_head:
        head = HeadConfig(
            name="classification",
            channel_agg=ChannelAggConfig(name="mean"),
            temporal_agg=TemporalAggConfig(name="mean"),
            hidden_dim=8,
            dropout=0.0,
        )

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
            moe=moe,
        ),
        projection=ProjectionConfig(name="simclr", enabled=False, hidden_dim=16, out_dim=8),
        cls=ClsConfig(downstream="tokens", embedding_type=None),
        head=head,
    )


def _sleep2expert_batch() -> dict:
    return {
        "tokens": {
            "eeg": torch.randn(2, 4, 8),
            "ppg": torch.randn(2, 4, 8),
        },
        "mlm_mask": {
            "eeg": torch.zeros(2, 4, dtype=torch.long),
            "ppg": torch.zeros(2, 4, dtype=torch.long),
        },
        "length": torch.tensor([4, 3], dtype=torch.long),
    }


def test_sleep2expert_moe_forward_preserves_shape_and_hidden_states():
    torch.manual_seed(0)
    model = _model(_moe_config()).eval()
    inputs = torch.randn(2, 5, 16)
    attention_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.float32)

    with torch.no_grad():
        output = model(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
            output_hidden_states=True,
            modality_name="eeg",
            collect_moe_aux=True,
        )

    assert output.last_hidden_state.shape == inputs.shape
    assert output.hidden_states is not None
    assert len(output.hidden_states) == 4
    assert output.moe_aux is not None
    valid_tokens = attention_mask.sum()
    assert [aux.layer_idx for aux in output.moe_aux] == [1, 3]
    for aux in output.moe_aux:
        assert aux.modality_name == "eeg"
        assert aux.required_expert_ids == ()
        assert aux.router_logits.shape == (2, 5, 4)
        assert aux.topk_indices.shape == (2, 5, 2)
        assert aux.topk_probs.shape == (2, 5, 2)
        assert torch.allclose(aux.topk_probs.sum(dim=-1), torch.ones(2, 5))
        assert torch.allclose(aux.load.sum(), valid_tokens * aux.topk_indices.size(-1))
        assert torch.allclose(aux.importance.sum(), valid_tokens, atol=1e-6)


def test_sleep2expert_moe_return_dict_false_appends_aux_only_when_requested():
    torch.manual_seed(0)
    model = _model(_moe_config()).eval()
    inputs = torch.randn(2, 4, 16)

    with torch.no_grad():
        default_output = model(inputs_embeds=inputs, modality_name="eeg", return_dict=False)
        aux_output = model(
            inputs_embeds=inputs,
            modality_name="eeg",
            collect_moe_aux=True,
            return_dict=False,
        )

    assert len(default_output) == 1
    assert default_output[0].shape == inputs.shape
    assert len(aux_output) == 2
    assert aux_output[0].shape == inputs.shape
    assert aux_output[1] is not None
    assert [aux.layer_idx for aux in aux_output[1]] == [1, 3]


@pytest.mark.parametrize("router_type", ["learned", "random", "hard_modality", "hard_group"])
@pytest.mark.parametrize("top_k", [1, 2])
def test_sleep2expert_moe_router_types_forward(router_type: str, top_k: int):
    torch.manual_seed(0)
    model = _model(_moe_config(router_type=router_type, top_k=top_k)).eval()
    inputs = torch.randn(2, 4, 16)

    with torch.no_grad():
        output = model(inputs_embeds=inputs, modality_name="ppg", collect_moe_aux=True)

    assert output.last_hidden_state.shape == inputs.shape
    assert output.moe_aux is not None
    assert output.moe_aux[0].topk_indices.shape[-1] == top_k
    router_params = list(model.encoder.layer[0].moe_ffn.router.parameters())
    if router_type == "learned":
        assert router_params
    else:
        assert router_params == []


@pytest.mark.parametrize("router_type", ["learned", "random", "hard_modality", "hard_group"])
def test_sleep2expert_moe_required_expert_fixed_weight(router_type: str):
    torch.manual_seed(0)
    model = _model(
        _moe_config(
            router_type=router_type,
            top_k=3,
            required_expert_ids=[0],
            required_expert_weight_mode="fixed",
            required_expert_weight=1 / 3,
        )
    ).eval()
    inputs = torch.randn(2, 4, 16)

    with torch.no_grad():
        output = model(inputs_embeds=inputs, modality_name="ppg", collect_moe_aux=True)

    assert output.moe_aux is not None
    for aux in output.moe_aux:
        assert aux.required_expert_ids == (0,)
        assert aux.topk_indices.shape[-1] == 3
        assert (aux.topk_indices[..., 0] == 0).all()
        assert torch.allclose(aux.topk_probs[..., 0], torch.full((2, 4), 1 / 3))
        assert torch.allclose(aux.topk_probs.sum(dim=-1), torch.ones(2, 4))


def test_sleep2expert_moe_required_expert_fixed_weight_normalizes_from_routed_logits():
    torch.manual_seed(0)
    model = (
        _model(
            _moe_config(
                router_type="learned",
                top_k=3,
                required_expert_ids=[0],
                required_expert_weight_mode="fixed",
                required_expert_weight=1 / 3,
            )
        )
        .eval()
        .half()
    )
    for module in model.modules():
        if isinstance(module, TopKRouter):
            module.router.weight.data.zero_()
            module.router.bias.data.copy_(torch.tensor([100.0, 0.0, -1.0, -2.0], dtype=torch.float16))
    inputs = torch.randn(2, 4, 16).half()

    with torch.no_grad():
        output = model(inputs_embeds=inputs, modality_name="ppg", collect_moe_aux=True)

    assert output.moe_aux is not None
    for aux in output.moe_aux:
        assert aux.required_expert_ids == (0,)
        assert (aux.topk_indices[..., 0] == 0).all()
        assert torch.allclose(aux.topk_probs[..., 0].float(), torch.full((2, 4), 1 / 3), atol=1e-3)
        assert torch.allclose(aux.topk_probs.sum(dim=-1).float(), torch.ones(2, 4), atol=1e-3)
        assert (aux.topk_probs[..., 1:].sum(dim=-1) > 0).all()


def test_sleep2expert_moe_required_expert_router_weight_uses_router_probability():
    torch.manual_seed(0)
    model = _model(
        _moe_config(
            router_type="learned",
            top_k=3,
            required_expert_ids=[0],
            required_expert_weight_mode="router",
        )
    ).eval()
    for module in model.modules():
        if isinstance(module, TopKRouter):
            module.router.weight.data.zero_()
            module.router.bias.data.copy_(torch.tensor([-2.0, 0.0, -10.0, 2.0]))
    inputs = torch.randn(2, 4, 16)

    with torch.no_grad():
        output = model(inputs_embeds=inputs, modality_name="ppg", collect_moe_aux=True)

    assert output.moe_aux is not None
    for aux in output.moe_aux:
        assert aux.required_expert_ids == (0,)
        assert (aux.topk_indices[..., 0] == 0).all()
        assert (aux.topk_probs[..., 0] < 0.1).all()
        assert not torch.allclose(aux.topk_probs[..., 0], torch.full((2, 4), 1 / 3))
        assert torch.allclose(aux.topk_probs.sum(dim=-1), torch.ones(2, 4))


def test_sleep2expert_moe_group_mask_excludes_disallowed_experts():
    torch.manual_seed(0)
    model = _model(_moe_config(router_type="learned")).eval()
    inputs = torch.randn(2, 4, 16)

    with torch.no_grad():
        output = model(inputs_embeds=inputs, modality_name="eeg", collect_moe_aux=True)

    assert output.moe_aux is not None
    allowed = torch.tensor([0, 2, 3])
    for aux in output.moe_aux:
        assert (aux.topk_indices.unsqueeze(-1) == allowed).any(dim=-1).all()
        assert not (aux.topk_indices == 1).any()


def test_sleep2expert_route_expert_filter_excludes_disallowed_experts():
    torch.manual_seed(0)
    model = _model(_moe_config(router_type="learned")).eval()
    apply_route_expert_filter(model, model.config.moe, ["neuro"])
    inputs = torch.randn(2, 4, 16)

    with torch.no_grad():
        output = model(inputs_embeds=inputs, modality_name="eeg", collect_moe_aux=True)

    assert output.moe_aux is not None
    allowed = torch.tensor([2, 3])
    for aux in output.moe_aux:
        assert (aux.topk_indices.unsqueeze(-1) == allowed).any(dim=-1).all()
        assert not (aux.topk_indices == 0).any()
        assert not (aux.topk_indices == 1).any()


def test_sleep2expert_route_expert_filter_intersects_modality_group_mask():
    model = _model(_moe_config(router_type="hard_group", top_k=1)).eval()
    apply_route_expert_filter(model, model.config.moe, ["neuro"])
    inputs = torch.randn(2, 4, 16)

    with torch.no_grad():
        output = model(inputs_embeds=inputs, modality_name="ppg", collect_moe_aux=True)

    assert output.moe_aux is not None
    for aux in output.moe_aux:
        assert (aux.topk_indices == 3).all()


def test_sleep2expert_route_expert_filter_rejects_unknown_group():
    moe_cfg = _moe_config()

    with pytest.raises(ValueError, match="Unknown route expert group"):
        resolve_route_expert_ids(moe_cfg, ["unknown"])


def test_sleep2expert_route_expert_filter_requires_enough_candidates():
    model = _model(_moe_config(router_type="learned", top_k=2)).eval()
    apply_route_expert_filter(model, model.config.moe, ["cardiac"])
    inputs = torch.randn(2, 4, 16)

    with pytest.raises(ValueError, match="after route expert group filtering"):
        model(inputs_embeds=inputs, modality_name="eeg", collect_moe_aux=True)


def test_sleep2expert_route_expert_filter_rejects_missing_required_expert():
    moe_cfg = _moe_config(
        router_type="learned",
        top_k=2,
        required_expert_ids=[0],
        required_expert_weight_mode="fixed",
        required_expert_weight=0.5,
    )

    with pytest.raises(ValueError, match="exclude required_expert_ids"):
        resolve_route_expert_ids(moe_cfg, ["cardiac"])


def test_sleep2expert_moe_entropy_uses_full_router_distribution():
    torch.manual_seed(0)
    model = _model(_moe_config(router_type="learned", top_k=1)).eval()
    inputs = torch.randn(2, 4, 16)

    with torch.no_grad():
        output = model(inputs_embeds=inputs, modality_name="eeg", collect_moe_aux=True)

    assert output.moe_aux is not None
    for aux in output.moe_aux:
        assert aux.entropy > 0


def test_sleep2expert_moe_top1_importance_keeps_router_gradient():
    torch.manual_seed(0)
    model = _model(_moe_config(router_type="learned", top_k=1, use_modality_group_mask=False)).train()
    inputs = torch.randn(2, 4, 16)

    output = model(inputs_embeds=inputs, modality_name="eeg", collect_moe_aux=True)

    assert output.moe_aux is not None
    router = model.encoder.layer[0].moe_ffn.router.router
    router.weight.grad = None
    output.moe_aux[0].importance[0].backward()
    assert router.weight.grad is not None
    assert router.weight.grad.abs().sum() > 0


def test_sleep2expert_moe_group_mask_requires_known_modality():
    model = _model(_moe_config()).eval()
    inputs = torch.randn(2, 4, 16)

    with pytest.raises(ValueError, match="modality_name must reference"):
        model(inputs_embeds=inputs, collect_moe_aux=True)


def test_sleep2expert_pretrain_forward_records_two_moe_aux_views():
    torch.manual_seed(0)
    model = Sleep2vecPretrainModel(
        model_config=_sleep2expert_model_config(moe=_moe_config()),
        device="cpu",
        specified_two_mods=["eeg", "ppg"],
    ).eval()
    batch = _sleep2expert_batch()

    with torch.no_grad():
        output = model(batch, apply_mask=False)

    assert len(output) == 2
    assert output[0].shape == (2, 4, 16)
    assert output[1].shape == (2, 4, 16)
    assert model.last_moe_aux is not None
    assert [record["modality"] for record in model.last_moe_aux] == ["eeg", "ppg"]
    for record in model.last_moe_aux:
        assert record["attention_mask"].shape == (2, 4)
        assert record["aux"] is not None
        assert [aux.layer_idx for aux in record["aux"]] == [1, 3]
        assert all(aux.modality_name == record["modality"] for aux in record["aux"])


def test_sleep2expert_encode_records_single_moe_aux_view():
    torch.manual_seed(0)
    model = Sleep2vecPretrainModel(
        model_config=_sleep2expert_model_config(moe=_moe_config()),
        device="cpu",
    ).eval()
    batch = _sleep2expert_batch()

    with torch.no_grad():
        hidden = model.encode(batch, "ppg")

    assert hidden.shape == (2, 4, 16)
    assert model.last_moe_aux is not None
    assert [record["modality"] for record in model.last_moe_aux] == ["ppg"]
    assert [aux.layer_idx for aux in model.last_moe_aux[0]["aux"]] == [1, 3]


def test_sleep2expert_dense_pretrain_forward_does_not_collect_moe_aux():
    torch.manual_seed(0)
    model = Sleep2vecPretrainModel(
        model_config=_sleep2expert_model_config(moe=None),
        device="cpu",
        specified_two_mods=["eeg", "ppg"],
    ).eval()
    batch = _sleep2expert_batch()

    with torch.no_grad():
        output = model(batch, apply_mask=False)

    assert len(output) == 2
    assert model.last_moe_aux is None


def test_sleep2expert_downstream_eval_records_moe_aux_without_changing_output():
    torch.manual_seed(0)
    Sleep2vecDownstreamModel = _downstream_model_cls()
    model_config = _sleep2expert_model_config(moe=_moe_config(), with_head=True)
    backbone = Sleep2vecPretrainModel(model_config=model_config, device="cpu")
    downstream = Sleep2vecDownstreamModel(
        target="stage",
        backbone=backbone,
        channel_names=["eeg", "ppg"],
        output_dim=2,
        is_classification=True,
        is_seq=False,
        device="cpu",
        model_config=model_config,
        head_config=model_config.head,
    ).eval()
    batch = _sleep2expert_batch()

    with torch.no_grad():
        output = downstream(batch)

    assert output.shape == (2, 2)
    assert backbone.last_moe_aux is not None
    assert [record["modality"] for record in backbone.last_moe_aux] == ["eeg", "ppg"]
    for record in backbone.last_moe_aux:
        assert record["aux"] is not None
        assert [aux.layer_idx for aux in record["aux"]] == [1, 3]


def test_sleep2expert_downstream_train_passes_modality_but_does_not_collect_moe_aux():
    torch.manual_seed(0)
    Sleep2vecDownstreamModel = _downstream_model_cls()
    model_config = _sleep2expert_model_config(moe=_moe_config(), with_head=True)
    backbone = Sleep2vecPretrainModel(model_config=model_config, device="cpu")
    downstream = Sleep2vecDownstreamModel(
        target="stage",
        backbone=backbone,
        channel_names=["eeg", "ppg"],
        output_dim=2,
        is_classification=True,
        is_seq=False,
        device="cpu",
        model_config=model_config,
        head_config=model_config.head,
    ).train()
    batch = _sleep2expert_batch()

    output = downstream(batch)

    assert output.shape == (2, 2)
    assert backbone.last_moe_aux is None


def test_sleep2expert_downstream_train_collects_moe_aux_when_enabled():
    torch.manual_seed(0)
    Sleep2vecDownstreamModel = _downstream_model_cls()
    model_config = _sleep2expert_model_config(moe=_moe_config(), with_head=True)
    backbone = Sleep2vecPretrainModel(model_config=model_config, device="cpu")
    downstream = Sleep2vecDownstreamModel(
        target="stage",
        backbone=backbone,
        channel_names=["eeg", "ppg"],
        output_dim=2,
        is_classification=True,
        is_seq=False,
        device="cpu",
        model_config=model_config,
        head_config=model_config.head,
    ).train()
    downstream.collect_train_moe_aux = True
    batch = _sleep2expert_batch()

    output = downstream(batch)

    assert output.shape == (2, 2)
    assert backbone.last_moe_aux is not None
    assert [record["modality"] for record in backbone.last_moe_aux] == ["eeg", "ppg"]
    for record in backbone.last_moe_aux:
        assert record["aux"] is not None
        assert [aux.layer_idx for aux in record["aux"]] == [1, 3]


def test_sleep2expert_expert_lora_real_peft_forward_backward_smoke():
    pytest.importorskip("peft")
    torch.manual_seed(0)
    Sleep2vecDownstreamModel = _downstream_model_cls()
    model_config = _sleep2expert_model_config(moe=_moe_config(), with_head=True)
    backbone = Sleep2vecPretrainModel(model_config=model_config, device="cpu")
    downstream = Sleep2vecDownstreamModel(
        target="stage",
        backbone=backbone,
        channel_names=["eeg", "ppg"],
        output_dim=2,
        is_classification=True,
        is_seq=False,
        device="cpu",
        model_config=model_config,
        head_config=model_config.head,
    ).train()
    downstream.collect_train_moe_aux = True
    downstream.freeze_backbone_and_insert_lora(
        insert_lora=True,
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        target_modules=["query", "key", "value", "dense_in", "dense_out"],
    )

    output = downstream(_sleep2expert_batch())
    loss = output.square().mean()
    loss.backward()

    lora_params = [(name, param) for name, param in downstream.named_parameters() if "lora_" in name]
    expert_lora_params = [
        (name, param)
        for name, param in lora_params
        if ".moe_ffn.experts." in name and (".dense_in." in name or ".dense_out." in name)
    ]
    assert output.shape == (2, 2)
    assert backbone.last_moe_aux is not None
    assert expert_lora_params
    assert not any("moe_ffn.router" in name for name, _ in lora_params)
    assert any(param.grad is not None and param.grad.abs().sum() > 0 for _, param in expert_lora_params)
