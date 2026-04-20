import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F

from sleep2vec.downstreams.head_registry import register_head

from .base import FeatureFusion
from .temporal_conv import TemporalConvBlock


def _apply_mask(hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return hidden
    return hidden * mask.unsqueeze(-1).to(hidden.dtype)


def _downsample_mask(mask: torch.Tensor, stride: int) -> torch.Tensor:
    pooled = F.max_pool1d(mask.to(torch.float32).unsqueeze(1), kernel_size=stride, stride=stride, ceil_mode=True)
    return pooled.squeeze(1) > 0


class _DownsampleBlock(nn.Module):
    def __init__(self, dim: int, *, stride: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=stride, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class _UpsampleBlock(nn.Module):
    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        return F.interpolate(x.transpose(1, 2), size=target_len, mode="linear", align_corners=False).transpose(1, 2)


class TemporalUNetHead(nn.Module):
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
        num_levels: int = 4,
        blocks_per_level: int = 2,
        kernel_size: int = 5,
        downsample_stride: int = 2,
    ):
        super().__init__()
        self.fusion = FeatureFusion(feature_dim, n_mods, agg)
        in_dim = self.fusion.output_dim
        model_dim = hidden_dim or in_dim
        self.proj_in = nn.Linear(in_dim, model_dim) if model_dim != in_dim else nn.Identity()
        self.num_levels = max(2, int(num_levels))
        self.downsample_stride = max(2, int(downsample_stride))

        def make_stage() -> nn.Sequential:
            return nn.Sequential(
                *[
                    TemporalConvBlock(
                        model_dim,
                        kernel_size=int(kernel_size),
                        dilation=1,
                        dropout=dropout,
                        act=act,
                    )
                    for _ in range(max(1, int(blocks_per_level)))
                ]
            )

        self.encoder_stages = nn.ModuleList([make_stage() for _ in range(self.num_levels)])
        self.downsamples = nn.ModuleList(
            [_DownsampleBlock(model_dim, stride=self.downsample_stride) for _ in range(self.num_levels - 1)]
        )
        self.upsamples = nn.ModuleList([_UpsampleBlock() for _ in range(self.num_levels - 1)])
        self.decoder_stages = nn.ModuleList([make_stage() for _ in range(self.num_levels - 1)])
        self.norm = nn.LayerNorm(model_dim)
        self.classifier = nn.Linear(model_dim, out_dim)

    def forward(
        self, feature_of_different_mods: t.List[torch.Tensor], *, token_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        fused, has_L = self.fusion.aggregator(feature_of_different_mods)
        if not has_L:
            fused = fused.unsqueeze(1)

        x = self.proj_in(fused)
        if token_mask is None:
            token_mask = torch.ones(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
        elif token_mask.dim() == 3 and token_mask.size(1) == 1:
            token_mask = token_mask.squeeze(1)
        token_mask = token_mask.to(torch.bool)

        x = _apply_mask(x, token_mask)
        skip_hidden: list[torch.Tensor] = []
        skip_masks: list[torch.Tensor] = []

        current_mask = token_mask
        for stage_idx, stage in enumerate(self.encoder_stages):
            x = stage(_apply_mask(x, current_mask))
            x = _apply_mask(x, current_mask)
            if stage_idx == self.num_levels - 1:
                continue
            skip_hidden.append(x)
            skip_masks.append(current_mask)
            x = self.downsamples[stage_idx](x)
            current_mask = _downsample_mask(current_mask, self.downsample_stride)
            x = _apply_mask(x, current_mask)

        for upsample, stage in zip(self.upsamples, self.decoder_stages):
            skip = skip_hidden.pop()
            current_mask = skip_masks.pop()
            x = upsample(x, skip.size(1))
            x = _apply_mask(x, current_mask)
            x = x + skip
            x = stage(_apply_mask(x, current_mask))
            x = _apply_mask(x, current_mask)

        logits = self.classifier(self.norm(_apply_mask(x, current_mask)))
        logits = _apply_mask(logits, current_mask)

        if not has_L:
            logits = logits.squeeze(1)
        return logits


@register_head("temporal_unet")
def build_temporal_unet_head(
    *,
    target,
    feature_dim,
    n_mods,
    output_dim,
    agg: str = "gated_scalar",
    hidden_dim: t.Optional[int] = None,
    dropout: float = 0.1,
    act: t.Type[nn.Module] = nn.GELU,
    num_levels: int = 4,
    blocks_per_level: int = 2,
    kernel_size: int = 5,
    downsample_stride: int = 2,
    **_,
) -> nn.Module:
    return TemporalUNetHead(
        feature_dim,
        n_mods,
        output_dim,
        agg=agg,
        hidden_dim=hidden_dim,
        dropout=dropout,
        act=act,
        num_levels=num_levels,
        blocks_per_level=blocks_per_level,
        kernel_size=kernel_size,
        downsample_stride=downsample_stride,
    )


__all__ = ["TemporalUNetHead", "build_temporal_unet_head"]
