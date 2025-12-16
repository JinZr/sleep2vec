# coding=utf-8

"""Utilities used by the standalone RoFormer encoder.

Icefall's Zipformer recipe keeps small helper layers/utilities next to the model
(e.g., scaling.py). This file plays that role for RoFormer.

Only dependency: torch.
"""

from __future__ import annotations

import math
from typing import Callable, Union

import torch
from torch import Tensor
from torch.nn import functional as F


def get_activation_fn(act: Union[str, Callable[[Tensor], Tensor]]) -> Callable[[Tensor], Tensor]:
    """Resolve an activation specified by string or callable."""

    if callable(act):
        return act

    name = act.lower()
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    if name in ("silu", "swish"):
        return F.silu
    if name == "tanh":
        return torch.tanh
    if name in ("gelu_new", "gelu-fast"):

        def gelu_new(x: Tensor) -> Tensor:
            # Approximate GELU used in e.g. GPT.
            return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        return gelu_new

    raise ValueError(f"Unsupported activation: {act}")


def make_extended_attention_mask(attention_mask: Tensor, dtype: torch.dtype) -> Tensor:
    """Convert a 2D attention mask (bsz, seq) with 1/0 values to an additive mask.

    Returns shape (bsz, 1, 1, seq) where masked positions are a large negative value.

    This matches the convention in HF RoFormerModel: additive mask is added to
    attention scores before softmax.
    """

    if attention_mask.dim() != 2:
        raise ValueError(f"attention_mask must be 2D (bsz, seq). Got shape: {tuple(attention_mask.shape)}")

    # Ensure float
    mask = attention_mask.to(dtype=dtype)

    # 1 -> keep (add 0), 0 -> mask (add -inf)
    extended = (1.0 - mask)[:, None, None, :] * torch.finfo(dtype).min
    return extended


def apply_chunking_to_forward(
    forward_fn: Callable[..., Tensor],
    chunk_size: int,
    chunk_dim: int,
    *input_tensors: Tensor,
) -> Tensor:
    """A minimal `apply_chunking_to_forward` equivalent.

    Splits input tensors into chunks along `chunk_dim`, applies `forward_fn` to each chunk,
    then concatenates outputs.

    Args:
      forward_fn:
        The function to apply.
      chunk_size:
        If <=0, chunking is disabled.
      chunk_dim:
        Dimension along which to chunk.
      input_tensors:
        Tensors to be chunked.

    Returns:
      The concatenated output tensor.
    """

    if chunk_size <= 0:
        return forward_fn(*input_tensors)

    # Validate shapes are chunkable and aligned
    tensor_shape = input_tensors[0].shape
    for t in input_tensors:
        if t.shape[chunk_dim] != tensor_shape[chunk_dim]:
            raise ValueError("All input tensors must have the same shape in the chunk dimension")

    dim_size = tensor_shape[chunk_dim]
    if dim_size % chunk_size != 0:
        raise ValueError(f"The dimension to be chunked ({dim_size}) must be a multiple of chunk_size ({chunk_size}).")

    num_chunks = dim_size // chunk_size
    input_chunks = tuple(t.chunk(num_chunks, dim=chunk_dim) for t in input_tensors)
    output_chunks = []
    for i in range(num_chunks):
        chunk_inputs = [chunks[i] for chunks in input_chunks]
        output_chunks.append(forward_fn(*chunk_inputs))
    return torch.cat(output_chunks, dim=chunk_dim)


__all__ = ["get_activation_fn", "make_extended_attention_mask", "apply_chunking_to_forward"]
