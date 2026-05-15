import typing as t

import torch
import torch.nn as nn

from sleep2vec2.downstreams.head_registry import register_head

from .base import FeatureFusion


class TemporalConvBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
        act: t.Type[nn.Module],
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("TemporalConvBlock expects an odd kernel_size for length-preserving padding.")
        padding = (kernel_size - 1) // 2 * dilation
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            groups=dim,
        )
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)
        self.act = act()
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = x.transpose(1, 2)
        y = self.depthwise(y)
        y = self.pointwise(y)
        y = y.transpose(1, 2)
        y = self.norm(y)
        y = self.act(y)
        y = self.drop(y)
        return residual + y


class TemporalConvHead(nn.Module):
    """
    Temporal convolutional head for sequence labeling.
    Applies channel fusion -> temporal conv stack -> linear classifier.
    """

    def __init__(
        self,
        feature_dim: int,
        n_mods: int,
        out_dim: int,
        *,
        agg: str = "gated_scalar",
        hidden_dim: t.Optional[int] = None,
        dropout: float = 0.1,
        act: t.Type[nn.Module] = nn.ELU,
        temporal_layers: int = 4,
        temporal_kernel: int = 7,
        temporal_dilation_base: int = 2,
        temporal_dropout: t.Optional[float] = None,
    ):
        super().__init__()
        self.fusion = FeatureFusion(feature_dim, n_mods, agg)
        in_dim = self.fusion.output_dim
        model_dim = hidden_dim or in_dim

        self.proj_in = nn.Linear(in_dim, model_dim) if model_dim != in_dim else nn.Identity()
        block_dropout = dropout if temporal_dropout is None else temporal_dropout
        temporal_layers = max(1, int(temporal_layers))
        temporal_dilation_base = max(1, int(temporal_dilation_base))
        self.blocks = nn.ModuleList(
            [
                TemporalConvBlock(
                    model_dim,
                    kernel_size=int(temporal_kernel),
                    dilation=temporal_dilation_base**i,
                    dropout=block_dropout,
                    act=act,
                )
                for i in range(temporal_layers)
            ]
        )
        self.norm = nn.LayerNorm(model_dim)
        self.classifier = nn.Linear(model_dim, out_dim)

    def forward(self, feature_of_different_mods: t.List[torch.Tensor]) -> torch.Tensor:
        fused, has_L = self.fusion.aggregator(feature_of_different_mods)
        if not has_L:
            fused = fused.unsqueeze(1)

        x = self.proj_in(fused)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.classifier(x)

        if not has_L:
            logits = logits.squeeze(1)
        return logits


@register_head("temporal_conv")
def build_temporal_conv_head(
    *,
    target,
    feature_dim,
    n_mods,
    output_dim,
    agg: str = "gated_scalar",
    hidden_dim: t.Optional[int] = None,
    dropout: float = 0.1,
    act: t.Type[nn.Module] = nn.ELU,
    temporal_layers: int = 4,
    temporal_kernel: int = 7,
    temporal_dilation_base: int = 2,
    temporal_dropout: t.Optional[float] = None,
    **_,
) -> nn.Module:
    return TemporalConvHead(
        feature_dim,
        n_mods,
        output_dim,
        agg=agg,
        hidden_dim=hidden_dim,
        dropout=dropout,
        act=act,
        temporal_layers=temporal_layers,
        temporal_kernel=temporal_kernel,
        temporal_dilation_base=temporal_dilation_base,
        temporal_dropout=temporal_dropout,
    )


__all__ = ["TemporalConvHead", "build_temporal_conv_head"]
