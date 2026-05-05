from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import transformers.utils as transformers_utils

from sleep2wave.backbones.encoder_factory import build_roformer
from sleep2wave.backbones.roformer import RoFormerConfig, RoFormerEncoderModel
from sleep2wave.config import BackboneConfig


def _import_hf_roformer():
    transformers_utils.is_sklearn_available = lambda: False
    from transformers.models.roformer.modeling_roformer import RoFormerConfig as HFRoFormerConfig, RoFormerModel

    return HFRoFormerConfig, RoFormerModel


def _copy_param(target: torch.Tensor, source: torch.Tensor, name: str) -> None:
    if tuple(target.shape) != tuple(source.shape):
        raise ValueError(f"Shape mismatch for {name}: target={tuple(target.shape)}, source={tuple(source.shape)}")
    with torch.no_grad():
        target.copy_(source)


def _copy_hf_weights_to_standalone(hf_model, standalone_model: RoFormerEncoderModel) -> None:
    hf_state = hf_model.state_dict()

    _copy_param(
        standalone_model.embeddings.word_embeddings.weight,
        hf_state["embeddings.word_embeddings.weight"],
        "embeddings.word_embeddings.weight",
    )
    _copy_param(
        standalone_model.embeddings.token_type_embeddings.weight,
        hf_state["embeddings.token_type_embeddings.weight"],
        "embeddings.token_type_embeddings.weight",
    )
    _copy_param(
        standalone_model.embeddings.layer_norm.weight,
        hf_state["embeddings.LayerNorm.weight"],
        "embeddings.layer_norm.weight",
    )
    _copy_param(
        standalone_model.embeddings.layer_norm.bias,
        hf_state["embeddings.LayerNorm.bias"],
        "embeddings.layer_norm.bias",
    )
    _copy_param(
        standalone_model.encoder.embed_positions.weight,
        hf_state["encoder.embed_positions.weight"],
        "encoder.embed_positions.weight",
    )

    if hasattr(standalone_model, "embeddings_project"):
        _copy_param(
            standalone_model.embeddings_project.weight,
            hf_state["embeddings_project.weight"],
            "embeddings_project.weight",
        )
        _copy_param(
            standalone_model.embeddings_project.bias,
            hf_state["embeddings_project.bias"],
            "embeddings_project.bias",
        )

    for index in range(standalone_model.config.num_hidden_layers):
        s_layer = standalone_model.encoder.layer[index]
        prefix = f"encoder.layer.{index}"

        _copy_param(
            s_layer.attention.self_attention.query.weight,
            hf_state[f"{prefix}.attention.self.query.weight"],
            f"{prefix}.attention.self.query.weight",
        )
        _copy_param(
            s_layer.attention.self_attention.query.bias,
            hf_state[f"{prefix}.attention.self.query.bias"],
            f"{prefix}.attention.self.query.bias",
        )
        _copy_param(
            s_layer.attention.self_attention.key.weight,
            hf_state[f"{prefix}.attention.self.key.weight"],
            f"{prefix}.attention.self.key.weight",
        )
        _copy_param(
            s_layer.attention.self_attention.key.bias,
            hf_state[f"{prefix}.attention.self.key.bias"],
            f"{prefix}.attention.self.key.bias",
        )
        _copy_param(
            s_layer.attention.self_attention.value.weight,
            hf_state[f"{prefix}.attention.self.value.weight"],
            f"{prefix}.attention.self.value.weight",
        )
        _copy_param(
            s_layer.attention.self_attention.value.bias,
            hf_state[f"{prefix}.attention.self.value.bias"],
            f"{prefix}.attention.self.value.bias",
        )

        _copy_param(
            s_layer.attention.output.dense.weight,
            hf_state[f"{prefix}.attention.output.dense.weight"],
            f"{prefix}.attention.output.dense.weight",
        )
        _copy_param(
            s_layer.attention.output.dense.bias,
            hf_state[f"{prefix}.attention.output.dense.bias"],
            f"{prefix}.attention.output.dense.bias",
        )
        _copy_param(
            s_layer.attention.output.layer_norm.weight,
            hf_state[f"{prefix}.attention.output.LayerNorm.weight"],
            f"{prefix}.attention.output.LayerNorm.weight",
        )
        _copy_param(
            s_layer.attention.output.layer_norm.bias,
            hf_state[f"{prefix}.attention.output.LayerNorm.bias"],
            f"{prefix}.attention.output.LayerNorm.bias",
        )

        _copy_param(
            s_layer.intermediate.dense.weight,
            hf_state[f"{prefix}.intermediate.dense.weight"],
            f"{prefix}.intermediate.dense.weight",
        )
        _copy_param(
            s_layer.intermediate.dense.bias,
            hf_state[f"{prefix}.intermediate.dense.bias"],
            f"{prefix}.intermediate.dense.bias",
        )
        _copy_param(
            s_layer.output.dense.weight,
            hf_state[f"{prefix}.output.dense.weight"],
            f"{prefix}.output.dense.weight",
        )
        _copy_param(
            s_layer.output.dense.bias,
            hf_state[f"{prefix}.output.dense.bias"],
            f"{prefix}.output.dense.bias",
        )
        _copy_param(
            s_layer.output.layer_norm.weight,
            hf_state[f"{prefix}.output.LayerNorm.weight"],
            f"{prefix}.output.LayerNorm.weight",
        )
        _copy_param(
            s_layer.output.layer_norm.bias,
            hf_state[f"{prefix}.output.LayerNorm.bias"],
            f"{prefix}.output.LayerNorm.bias",
        )


def _copy_sleep2vec_forward_weights(base_model, sleep2wave_model) -> None:
    base_state = base_model.state_dict()
    sleep2wave_state = sleep2wave_model.state_dict()
    missing: list[str] = []

    for name, target in sleep2wave_state.items():
        if name.startswith("encoder."):
            continue
        source = base_state.get(name)
        if source is None:
            missing.append(name)
            continue
        _copy_param(target, source, name)

    if missing:
        raise AssertionError(f"sleep2wave forward state has no sleep2vec source for: {missing}")

    _copy_hf_weights_to_standalone(base_model.encoder, sleep2wave_model.encoder)


def test_sleep2wave_roformer_builder_uses_standalone_model():
    factory = build_roformer(
        BackboneConfig(
            name="roformer",
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            vocab_size=31,
            config_overrides={"intermediate_size": 64, "max_position_embeddings": 64},
        )
    )

    encoder, hidden_size = factory.build()

    assert isinstance(encoder, RoFormerEncoderModel)
    assert hidden_size == 32


def test_sleep2wave_roformer_matches_hf_forward_with_copied_weights():
    HFRoFormerConfig, HFRoFormerModel = _import_hf_roformer()

    torch.manual_seed(0)
    hf_config = HFRoFormerConfig(
        vocab_size=37,
        embedding_size=32,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=64,
        type_vocab_size=2,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        rotary_value=False,
        is_decoder=False,
        add_cross_attention=False,
    )
    standalone_config = RoFormerConfig(
        vocab_size=37,
        embedding_size=32,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=64,
        type_vocab_size=2,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        rotary_value=False,
    )
    hf_model = HFRoFormerModel(hf_config).eval()
    standalone_model = RoFormerEncoderModel(standalone_config).eval()
    _copy_hf_weights_to_standalone(hf_model, standalone_model)

    inputs_embeds = torch.randn(2, 8, standalone_config.embedding_size)
    attention_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]],
        dtype=torch.float32,
    )

    with torch.no_grad():
        hf_output = hf_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )
        standalone_output = standalone_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )

    assert torch.allclose(standalone_output.last_hidden_state, hf_output.last_hidden_state, atol=1e-6, rtol=1e-5)
    assert standalone_output.hidden_states is not None
    assert hf_output.hidden_states is not None
    assert len(standalone_output.hidden_states) == len(hf_output.hidden_states)
    for standalone_state, hf_state in zip(standalone_output.hidden_states, hf_output.hidden_states):
        assert torch.allclose(standalone_state, hf_state, atol=1e-6, rtol=1e-5)

    assert standalone_output.attentions is not None
    assert hf_output.attentions is not None
    assert len(standalone_output.attentions) == len(hf_output.attentions)
    for standalone_attention, hf_attention in zip(standalone_output.attentions, hf_output.attentions):
        assert torch.allclose(standalone_attention, hf_attention, atol=1e-6, rtol=1e-5)


def test_sleep2wave_pretrain_forward_matches_sleep2vec_with_identical_config():
    from sleep2vec.config import (
        BackboneConfig as BaseBackboneConfig,
        ChannelConfig as BaseChannelConfig,
        ClsConfig as BaseClsConfig,
        ModelConfig as BaseModelConfig,
        ProjectionConfig as BaseProjectionConfig,
        TokenizerConfig as BaseTokenizerConfig,
    )
    from sleep2vec.pretrain_model import Sleep2vecPretrainModel as BaseSleep2vecPretrainModel
    from sleep2wave.config import (
        BackboneConfig as VariantBackboneConfig,
        ChannelConfig as VariantChannelConfig,
        ClsConfig as VariantClsConfig,
        ModelConfig as VariantModelConfig,
        ProjectionConfig as VariantProjectionConfig,
        TokenizerConfig as VariantTokenizerConfig,
    )
    from sleep2wave.pretrain_model import Sleep2vecPretrainModel as VariantSleep2vecPretrainModel

    backbone_kwargs = {
        "name": "roformer",
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "vocab_size": 37,
        "config_overrides": {
            "intermediate_size": 64,
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "max_position_embeddings": 64,
        },
    }
    channel_specs = [
        ("eeg", 4),
        ("ecg", 4),
    ]

    base_config = BaseModelConfig(
        channels=[
            BaseChannelConfig(
                name=name,
                input_dim=input_dim,
                tokenizer=BaseTokenizerConfig(name="linear", out_dim=16),
            )
            for name, input_dim in channel_specs
        ],
        backbone=BaseBackboneConfig(**backbone_kwargs),
        projection=BaseProjectionConfig(name="simclr", enabled=True, hidden_dim=32, out_dim=8),
        cls=BaseClsConfig(downstream="tokens", embedding_type=None),
    )
    variant_config = VariantModelConfig(
        channels=[
            VariantChannelConfig(
                name=name,
                input_dim=input_dim,
                tokenizer=VariantTokenizerConfig(name="linear", out_dim=16),
            )
            for name, input_dim in channel_specs
        ],
        backbone=VariantBackboneConfig(**backbone_kwargs),
        projection=VariantProjectionConfig(name="simclr", enabled=True, hidden_dim=32, out_dim=8),
        cls=VariantClsConfig(downstream="tokens", embedding_type=None),
    )

    torch.manual_seed(123)
    base_model = BaseSleep2vecPretrainModel(
        model_config=base_config,
        device="cpu",
        specified_two_mods=["eeg", "ecg"],
    )
    variant_model = VariantSleep2vecPretrainModel(
        model_config=variant_config,
        device="cpu",
        specified_two_mods=["eeg", "ecg"],
    )
    _copy_sleep2vec_forward_weights(base_model, variant_model)
    base_model.eval()
    variant_model.eval()

    batch = {
        "tokens": {
            "eeg": torch.randn(2, 5, 4),
            "ecg": torch.randn(2, 5, 4),
        },
        "mlm_mask": {
            "eeg": torch.tensor([[0, 1, 0, 0, 1], [0, 0, 1, 0, 0]], dtype=torch.long),
            "ecg": torch.tensor([[0, 0, 1, 0, 0], [1, 0, 0, 1, 0]], dtype=torch.long),
        },
        "length": torch.tensor([5, 3], dtype=torch.long),
    }

    with torch.no_grad():
        base_output = base_model(batch, apply_mask=True)
        variant_output = variant_model(batch, apply_mask=True)

    assert len(base_output) == len(variant_output) == 2
    for base_tensor, variant_tensor in zip(base_output, variant_output):
        assert torch.allclose(variant_tensor, base_tensor, atol=1e-5, rtol=1e-5)
