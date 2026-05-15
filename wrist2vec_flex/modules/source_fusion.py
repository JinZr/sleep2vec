from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from wrist2vec_flex.config import SourceFusionConfig
from wrist2vec_flex.registry import get_source_fusion_builder, register_source_fusion


def _validate_source_tokens(x: torch.Tensor) -> tuple[int, int, int, int]:
    if x.dim() != 4:
        raise ValueError(f"source fusion expects [B, L, S, D], got shape {tuple(x.shape)}.")
    batch_size, num_tokens, num_sources, feature_dim = x.shape
    if num_sources <= 0:
        raise ValueError("source fusion requires at least one source.")
    return batch_size, num_tokens, num_sources, feature_dim


def _normalize_source_mask(
    source_mask: torch.Tensor | None,
    *,
    batch_size: int,
    num_tokens: int,
    num_sources: int,
    device: torch.device,
) -> torch.Tensor:
    if source_mask is None:
        return torch.ones(batch_size, num_tokens, num_sources, dtype=torch.bool, device=device)

    mask = source_mask.to(device=device, dtype=torch.bool)
    if mask.dim() == 2:
        if mask.shape != (batch_size, num_sources):
            raise ValueError(
                f"source_mask shape must be [B, S], got {tuple(mask.shape)} for B={batch_size}, S={num_sources}."
            )
        mask = mask[:, None, :].expand(batch_size, num_tokens, num_sources)
    elif mask.dim() == 3:
        if mask.shape != (batch_size, num_tokens, num_sources):
            raise ValueError(
                "source_mask shape must be [B, L, S], "
                f"got {tuple(mask.shape)} for B={batch_size}, L={num_tokens}, S={num_sources}."
            )
    else:
        raise ValueError(f"source_mask must be [B, S] or [B, L, S], got shape {tuple(mask.shape)}.")

    if not mask.any(dim=-1).all():
        raise ValueError("source_mask has no available source for at least one sample/token.")
    return mask


class SourceFusion(nn.Module):
    def __init__(self, *, feature_dim: int, num_sources: int):
        super().__init__()
        if int(feature_dim) <= 0:
            raise ValueError("feature_dim must be > 0.")
        if int(num_sources) <= 0:
            raise ValueError("num_sources must be > 0.")
        self.feature_dim = int(feature_dim)
        self.num_sources = int(num_sources)

    def _mask(self, x: torch.Tensor, source_mask: torch.Tensor | None) -> torch.Tensor:
        batch_size, num_tokens, num_sources, _ = _validate_source_tokens(x)
        if num_sources != self.num_sources:
            raise ValueError(f"expected {self.num_sources} sources, got {num_sources}.")
        return _normalize_source_mask(
            source_mask,
            batch_size=batch_size,
            num_tokens=num_tokens,
            num_sources=num_sources,
            device=x.device,
        )


@register_source_fusion("identity")
class IdentitySourceFusion(SourceFusion):
    def __init__(self, *, feature_dim: int, num_sources: int):
        super().__init__(feature_dim=feature_dim, num_sources=num_sources)
        if self.num_sources != 1:
            raise ValueError("identity source fusion is only valid for a single source.")

    def forward(self, x: torch.Tensor, source_mask: torch.Tensor | None = None) -> torch.Tensor:
        self._mask(x, source_mask)
        return x.squeeze(2)


@register_source_fusion("masked_mean")
class MaskedMeanSourceFusion(SourceFusion):
    def forward(self, x: torch.Tensor, source_mask: torch.Tensor | None = None) -> torch.Tensor:
        mask = self._mask(x, source_mask)
        weights = mask.to(dtype=x.dtype).unsqueeze(-1)
        denom = weights.sum(dim=2).clamp_min(1.0)
        return (x * weights).sum(dim=2) / denom


@register_source_fusion("masked_gated_scalar")
class MaskedGatedScalarSourceFusion(SourceFusion):
    def __init__(self, *, feature_dim: int, num_sources: int):
        super().__init__(feature_dim=feature_dim, num_sources=num_sources)
        self.gates = nn.Parameter(torch.zeros(self.num_sources))

    def forward(self, x: torch.Tensor, source_mask: torch.Tensor | None = None) -> torch.Tensor:
        mask = self._mask(x, source_mask)
        logits = self.gates.view(1, 1, self.num_sources).expand_as(mask).to(dtype=x.dtype)
        weights = F.softmax(logits.masked_fill(~mask, torch.finfo(x.dtype).min), dim=-1)
        return (x * weights.unsqueeze(-1)).sum(dim=2)


@register_source_fusion("learned_query")
class LearnedQuerySourceFusion(SourceFusion):
    def __init__(self, *, feature_dim: int, num_sources: int, dropout: float = 0.0):
        super().__init__(feature_dim=feature_dim, num_sources=num_sources)
        self.query = nn.Parameter(torch.empty(self.feature_dim))
        self.dropout = nn.Dropout(float(dropout))
        nn.init.normal_(self.query, std=self.feature_dim**-0.5)

    def forward(self, x: torch.Tensor, source_mask: torch.Tensor | None = None) -> torch.Tensor:
        mask = self._mask(x, source_mask)
        scores = (x * self.query.view(1, 1, 1, self.feature_dim)).sum(dim=-1) / math.sqrt(self.feature_dim)
        weights = F.softmax(scores.masked_fill(~mask, torch.finfo(x.dtype).min), dim=-1)
        return self.dropout((x * weights.unsqueeze(-1)).sum(dim=2))


def build_source_fusion(config: SourceFusionConfig, *, feature_dim: int, num_sources: int) -> SourceFusion:
    kwargs = dict(config.kwargs or {})
    builder = get_source_fusion_builder(config.name)
    return builder(feature_dim=feature_dim, num_sources=num_sources, **kwargs)


__all__ = [
    "IdentitySourceFusion",
    "LearnedQuerySourceFusion",
    "MaskedGatedScalarSourceFusion",
    "MaskedMeanSourceFusion",
    "SourceFusion",
    "build_source_fusion",
]
