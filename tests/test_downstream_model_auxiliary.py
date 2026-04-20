from __future__ import annotations

import torch
import torch.nn as nn

from sleep2vec.config import (
    AuxiliaryHeadConfig,
    AuxiliaryTaskConfig,
    BackboneConfig,
    ChannelAggConfig,
    ChannelConfig,
    ClsConfig,
    HeadConfig,
    ModelConfig,
    ProjectionConfig,
    TemporalAggConfig,
    TokenizerConfig,
)
from sleep2vec.downstream_model import DownstreamOutput, Sleep2vecDownstreamModel


class _FakeBackbone(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.transformer_hidden_size = hidden_size
        self.cls_embedding = None

    def _tokenize_all(self, tokens):
        return tokens

    def _token_embeddings_to_hidden(self, single_mod_token_embeddings, batch, *, return_hidden_states: bool = False):
        hidden = single_mod_token_embeddings
        attn_mask = torch.ones(hidden.size(0), hidden.size(1), dtype=torch.bool, device=hidden.device)
        if return_hidden_states:
            return hidden, attn_mask, None
        return hidden, attn_mask, None


def _model_config(*, head_name: str) -> ModelConfig:
    return ModelConfig(
        channels=[
            ChannelConfig(name="ppg", input_dim=4, tokenizer=TokenizerConfig(name="linear", out_dim=8)),
        ],
        backbone=BackboneConfig(
            name="roformer",
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            vocab_size=1,
        ),
        projection=ProjectionConfig(name="simclr", enabled=True, hidden_dim=8, out_dim=4),
        cls=ClsConfig(downstream="tokens", embedding_type=None),
        head=HeadConfig(
            name=head_name,
            channel_agg=ChannelAggConfig(name="gated_scalar"),
            temporal_agg=TemporalAggConfig(name="mean"),
        ),
    )


def test_downstream_model_returns_plain_tensor_without_auxiliary_task():
    cfg = _model_config(head_name="classification")
    model = Sleep2vecDownstreamModel(
        "stage5",
        _FakeBackbone(hidden_size=8),
        channel_names=["ppg"],
        output_dim=5,
        is_classification=True,
        is_seq=True,
        device="cpu",
        model_config=cfg,
        head_config=cfg.head,
    )
    batch = {
        "tokens": {"ppg": torch.randn(2, 4, 8)},
        "length": torch.tensor([4, 4], dtype=torch.long),
    }

    output = model(batch)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 4, 5)


def test_downstream_model_returns_main_and_aux_outputs_with_temporal_unet():
    cfg = _model_config(head_name="temporal_unet")
    aux_cfg = AuxiliaryTaskConfig(
        enabled=True,
        target="ahi",
        type="regression",
        output_dim=1,
        loss_weight=0.1,
        temporal_agg=TemporalAggConfig(name="mean"),
        head=AuxiliaryHeadConfig(
            name="regression",
            channel_agg=ChannelAggConfig(name="gated_scalar"),
        ),
    )
    model = Sleep2vecDownstreamModel(
        "ahi",
        _FakeBackbone(hidden_size=8),
        channel_names=["ppg"],
        output_dim=30,
        is_classification=True,
        is_seq=True,
        device="cpu",
        model_config=cfg,
        head_config=cfg.head,
        auxiliary_task_cfg=aux_cfg,
    )
    batch = {
        "tokens": {"ppg": torch.randn(2, 8, 8)},
        "length": torch.tensor([8, 8], dtype=torch.long),
    }

    output = model(batch)

    assert isinstance(output, DownstreamOutput)
    assert output.main_logits.shape == (2, 8, 30)
    assert output.aux_logits is not None
    assert output.aux_logits.shape == (2, 1)
