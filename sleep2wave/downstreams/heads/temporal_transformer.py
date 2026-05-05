import math
import typing as t

import torch
import torch.nn as nn

from sleep2wave.downstreams.head_registry import register_head

from .base import FeatureFusion


def _sinusoidal_position_embedding(length: int, dim: int, device, dtype) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    if dim % 2 == 0:
        pe[:, 1::2] = torch.cos(position * div_term)
    else:
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe


class TemporalTransformerHead(nn.Module):
    """
    Transformer head for sequence labeling.
    Applies channel fusion -> positional encoding -> TransformerEncoder -> linear classifier.
    """

    # NOTE: This flag tells the downstream runner to pass padding masks into the head,
    #       ensuring attention ignores padded tokens in long sequence labeling tasks.
    supports_token_mask = True

    def __init__(
        self,
        feature_dim: int,
        n_mods: int,
        out_dim: int,
        *,
        agg: str = "gated_scalar",
        hidden_dim: t.Optional[int] = None,
        dropout: float = 0.1,
        act: t.Type[nn.Module] = nn.GELU,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: t.Optional[int] = None,
        attn_dropout: t.Optional[float] = None,
        max_len: int = 2048,
        use_learned_pos_emb: bool = True,
        norm_first: bool = True,
    ):
        super().__init__()
        self.fusion = FeatureFusion(feature_dim, n_mods, agg)
        in_dim = self.fusion.output_dim
        model_dim = hidden_dim or in_dim
        self.proj_in = nn.Linear(in_dim, model_dim) if model_dim != in_dim else nn.Identity()
        self.max_len = int(max_len)
        self.use_learned_pos_emb = bool(use_learned_pos_emb)

        if self.use_learned_pos_emb:
            self.pos_emb = nn.Embedding(self.max_len, model_dim)
        else:
            self.pos_emb = None

        attn_dropout = dropout if attn_dropout is None else attn_dropout
        ff_dim = ff_dim or model_dim * 4
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation=act(),
            batch_first=True,
            norm_first=bool(norm_first),
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers), norm=nn.LayerNorm(model_dim))
        self.drop = nn.Dropout(attn_dropout) if attn_dropout and attn_dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(model_dim, out_dim)

    def _position_embedding(self, length: int, device, dtype, dim: int) -> torch.Tensor:
        if self.pos_emb is not None and length <= self.max_len:
            positions = torch.arange(length, device=device)
            return self.pos_emb(positions).to(dtype)
        return _sinusoidal_position_embedding(length, dim, device, dtype)

    def forward(
        self, feature_of_different_mods: t.List[torch.Tensor], *, token_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        fused, has_L = self.fusion.aggregator(feature_of_different_mods)
        if not has_L:
            fused = fused.unsqueeze(1)

        x = self.proj_in(fused)
        pos = self._position_embedding(x.size(1), x.device, x.dtype, x.size(-1))
        x = x + pos.unsqueeze(0)
        x = self.drop(x)

        key_padding_mask = None
        if token_mask is not None:
            if token_mask.dim() == 3 and token_mask.size(1) == 1:
                token_mask = token_mask.squeeze(1)
            token_mask = token_mask.to(torch.bool)
            key_padding_mask = ~token_mask
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        logits = self.classifier(x)

        if not has_L:
            logits = logits.squeeze(1)
        return logits


@register_head("temporal_transformer")
def build_temporal_transformer_head(
    *,
    target,
    feature_dim,
    n_mods,
    output_dim,
    agg: str = "gated_scalar",
    hidden_dim: t.Optional[int] = None,
    dropout: float = 0.1,
    act: t.Type[nn.Module] = nn.GELU,
    num_layers: int = 4,
    num_heads: int = 8,
    ff_dim: t.Optional[int] = None,
    attn_dropout: t.Optional[float] = None,
    max_len: int = 2048,
    use_learned_pos_emb: bool = True,
    norm_first: bool = True,
    **_,
) -> nn.Module:
    return TemporalTransformerHead(
        feature_dim,
        n_mods,
        output_dim,
        agg=agg,
        hidden_dim=hidden_dim,
        dropout=dropout,
        act=act,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        attn_dropout=attn_dropout,
        max_len=max_len,
        use_learned_pos_emb=use_learned_pos_emb,
        norm_first=norm_first,
    )


__all__ = ["TemporalTransformerHead", "build_temporal_transformer_head"]
