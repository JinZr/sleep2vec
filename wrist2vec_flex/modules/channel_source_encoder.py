from __future__ import annotations

import torch
import torch.nn as nn

from wrist2vec_flex.config import ChannelConfig, SourceEmbeddingConfig, SourceFusionConfig
from wrist2vec_flex.modules.mask_dropout import apply_mask_dropout
from wrist2vec_flex.modules.source_fusion import build_source_fusion


class ChannelSourceEncoder(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        num_sources: int,
        source_fusion: SourceFusionConfig,
        source_embedding: SourceEmbeddingConfig,
        source_dropout_rate: float = 0.0,
        min_sources_after_dropout: int = 1,
    ):
        super().__init__()
        if int(num_sources) <= 0:
            raise ValueError("num_sources must be > 0.")
        if not 0.0 <= float(source_dropout_rate) <= 1.0:
            raise ValueError("source_dropout_rate must be in [0.0, 1.0].")
        if int(min_sources_after_dropout) <= 0:
            raise ValueError("min_sources_after_dropout must be > 0.")
        self.feature_dim = int(feature_dim)
        self.num_sources = int(num_sources)
        self.source_dropout_rate = float(source_dropout_rate)
        self.min_sources_after_dropout = int(min_sources_after_dropout)
        self.source_embedding = nn.Embedding(self.num_sources, self.feature_dim) if source_embedding.enabled else None
        self.source_fusion = build_source_fusion(
            source_fusion,
            feature_dim=self.feature_dim,
            num_sources=self.num_sources,
        )

    @classmethod
    def from_channel(
        cls,
        channel: ChannelConfig,
        *,
        feature_dim: int,
        source_dropout_rate: float = 0.0,
        min_sources_after_dropout: int = 1,
    ) -> "ChannelSourceEncoder":
        source_names = list(channel.source_names) or [channel.name]
        source_fusion = channel.source_fusion or SourceFusionConfig(
            name="learned_query" if len(source_names) > 1 else "identity"
        )
        source_embedding = channel.source_embedding or SourceEmbeddingConfig(enabled=len(source_names) > 1)
        return cls(
            feature_dim=feature_dim,
            num_sources=len(source_names),
            source_fusion=source_fusion,
            source_embedding=source_embedding,
            source_dropout_rate=source_dropout_rate,
            min_sources_after_dropout=min_sources_after_dropout,
        )

    @classmethod
    def single_source(
        cls,
        *,
        feature_dim: int,
        source_dropout_rate: float = 0.0,
        min_sources_after_dropout: int = 1,
    ) -> "ChannelSourceEncoder":
        return cls(
            feature_dim=feature_dim,
            num_sources=1,
            source_fusion=SourceFusionConfig(name="identity"),
            source_embedding=SourceEmbeddingConfig(enabled=False),
            source_dropout_rate=source_dropout_rate,
            min_sources_after_dropout=min_sources_after_dropout,
        )

    def _normalize_tokens(self, tokens: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        if tokens.dim() == 3:
            tokens = tokens.unsqueeze(2)
        elif tokens.dim() != 4:
            raise ValueError(f"channel source encoder expects [B, L, T] or [B, L, S, T], got {tuple(tokens.shape)}.")

        batch_size, num_tokens, num_sources, token_width = tokens.shape
        if num_sources != self.num_sources:
            raise ValueError(f"expected {self.num_sources} sources, got {num_sources}.")
        return tokens, (batch_size, num_tokens, token_width)

    def _normalize_source_mask(
        self,
        source_mask: torch.Tensor | None,
        *,
        batch_size: int,
        num_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        if source_mask is None:
            return torch.ones(batch_size, self.num_sources, dtype=torch.bool, device=device)

        mask = source_mask.to(device=device, dtype=torch.bool)
        if mask.dim() == 2:
            if mask.shape != (batch_size, self.num_sources):
                raise ValueError(
                    f"source_mask shape must be [B, S], got {tuple(mask.shape)} "
                    f"for B={batch_size}, S={self.num_sources}."
                )
        elif mask.dim() == 3:
            if mask.shape != (batch_size, num_tokens, self.num_sources):
                raise ValueError(
                    "source_mask shape must be [B, L, S], "
                    f"got {tuple(mask.shape)} for B={batch_size}, L={num_tokens}, S={self.num_sources}."
                )
        else:
            raise ValueError(f"source_mask must be [B, S] or [B, L, S], got shape {tuple(mask.shape)}.")

        if not mask.any(dim=-1).all():
            raise ValueError("source_mask has no available source for at least one sample/token.")
        return mask

    def _apply_source_dropout(self, source_mask: torch.Tensor) -> torch.Tensor:
        if not self.training or self.source_dropout_rate <= 0.0:
            return source_mask

        return apply_mask_dropout(
            source_mask,
            self.source_dropout_rate,
            self.min_sources_after_dropout,
        )

    def _add_source_embedding(
        self,
        token_embeddings: torch.Tensor,
        source_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.source_embedding is None:
            return token_embeddings

        if source_ids is None:
            source_ids = torch.arange(self.num_sources, device=token_embeddings.device)
        else:
            source_ids = source_ids.to(device=token_embeddings.device, dtype=torch.long)

        if source_ids.dim() == 1:
            if source_ids.shape[0] != self.num_sources:
                raise ValueError(f"source_ids shape must be [S], got {tuple(source_ids.shape)}.")
            embeddings = self.source_embedding(source_ids).view(1, 1, self.num_sources, self.feature_dim)
        elif source_ids.dim() == 2:
            if source_ids.shape[1] != self.num_sources:
                raise ValueError(f"source_ids shape must be [B, S], got {tuple(source_ids.shape)}.")
            if source_ids.shape[0] != token_embeddings.shape[0]:
                raise ValueError(
                    f"source_ids batch size {source_ids.shape[0]} does not match tokens {token_embeddings.shape[0]}."
                )
            embeddings = self.source_embedding(source_ids).unsqueeze(1)
        else:
            raise ValueError(f"source_ids must be [S] or [B, S], got shape {tuple(source_ids.shape)}.")
        return token_embeddings + embeddings

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        tokenizer: nn.Module,
        source_mask: torch.Tensor | None = None,
        source_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens, (batch_size, num_tokens, token_width) = self._normalize_tokens(tokens)
        source_mask = self._normalize_source_mask(
            source_mask,
            batch_size=batch_size,
            num_tokens=num_tokens,
            device=tokens.device,
        )

        flattened = tokens.reshape(batch_size * num_tokens * self.num_sources, token_width)
        token_embeddings = tokenizer(flattened).reshape(batch_size, num_tokens, self.num_sources, self.feature_dim)
        token_embeddings = self._add_source_embedding(token_embeddings, source_ids)
        source_mask = self._apply_source_dropout(source_mask)
        return self.source_fusion(token_embeddings, source_mask)


__all__ = ["ChannelSourceEncoder"]
