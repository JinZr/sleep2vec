# coding=utf-8

"""Rotary position embedding utilities for the standalone RoFormer encoder.

RoFormer uses RoPE (Rotary Positional Embeddings) implemented by generating a
sinusoidal table (sin in first half, cos in second half) and rotating the Q/K
(and optionally V) projections.

This module is kept local (like Icefall keeps scaling.py next to zipformer.py)
so you can customize the RoPE formulation easily.

Only dependency: torch.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """Sinusoidal position embeddings used for RoPE.

    Note: This follows the HF RoFormer convention where sin/cos features are not
    interleaved; instead, sin features occupy the first half of the vector and
    cos features occupy the second half.

    The embedding dimension should be the per-head attention dimension.
    """

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
        super().__init__(num_positions, embedding_dim, _freeze=True)
        with torch.no_grad():
            self.weight.copy_(self.create_weight())

    def create_weight(self) -> Tensor:
        n_pos, dim = self.weight.shape
        if dim % 2 != 0:
            raise ValueError(f"RoFormer sinusoidal embedding dim must be even. Got dim={dim}")

        half_dim = dim // 2
        positions = torch.arange(n_pos, dtype=self.weight.dtype, device=self.weight.device).unsqueeze(1)  # (n_pos, 1)

        # Compute the frequency terms
        div_term = torch.exp(
            torch.arange(0, half_dim, dtype=self.weight.dtype, device=self.weight.device)
            * (-math.log(10000.0) / half_dim)
        )  # (half_dim,)

        angles = positions * div_term  # (n_pos, half_dim)

        out = torch.empty(n_pos, dim, dtype=self.weight.dtype, device=self.weight.device, requires_grad=False)
        out[:, :half_dim] = torch.sin(angles)
        out[:, half_dim:] = torch.cos(angles)
        return out

    @torch.no_grad()
    def forward(
        self,
        input_shape: Tuple[int, int],
        past_key_values_length: int = 0,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Return positional embeddings.

        Args:
          input_shape:
            Usually (batch_size, seq_len). Only seq_len is used to create positions.
          past_key_values_length:
            Offset for positions.
          position_ids:
            Optional 1D or 2D position ids.

        Returns:
          Tensor of shape (seq_len, head_dim) if position_ids is 1D.
        """

        if position_ids is None:
            _, seq_len = input_shape[:2]
            position_ids = torch.arange(
                past_key_values_length,
                past_key_values_length + seq_len,
                dtype=torch.long,
                device=self.weight.device,
            )
        return super().forward(position_ids)


def apply_rotary_position_embeddings(
    sinusoidal_pos: Tensor,
    query_layer: Tensor,
    key_layer: Tensor,
    value_layer: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Apply RoPE rotation to (Q, K) and optionally V.

    This implementation follows the HF RoFormer reference implementation.

    Args:
      sinusoidal_pos:
        Tensor of shape (batch, num_heads, seq_len, head_dim) containing sin/cos.
      query_layer/key_layer/value_layer:
        Tensors of shape (batch, num_heads, seq_len, head_dim).

    Returns:
      Rotated (query_layer, key_layer, value_layer (maybe None)).
    """

    sin, cos = sinusoidal_pos.chunk(2, dim=-1)

    # Expand sin/cos to match head_dim by duplicating each element.
    sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
    cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)

    def rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.stack([-x2, x1], dim=-1).reshape_as(x)

    q = (query_layer * cos_pos) + (rotate_half(query_layer) * sin_pos)
    k = (key_layer * cos_pos) + (rotate_half(key_layer) * sin_pos)

    if value_layer is None:
        return q, k, None

    v = (value_layer * cos_pos) + (rotate_half(value_layer) * sin_pos)
    return q, k, v


__all__ = ["RoFormerSinusoidalPositionalEmbedding", "apply_rotary_position_embeddings"]
