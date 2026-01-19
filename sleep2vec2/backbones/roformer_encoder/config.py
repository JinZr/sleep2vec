# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone RoFormer encoder configuration.

This file is intentionally *not* derived from `transformers.PreTrainedConfig`.
It is a lightweight dataclass meant to be used the same way Icefall uses
explicit params/args to construct Zipformer.

You can either:
  1) Pass explicit kwargs to RoFormerEncoderModel(...)
  2) Build a RoFormerEncoderConfig and pass it to RoFormerEncoderModel(config=...)

Only dependency: torch (for type hints).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Callable, Dict, Optional, Union

import torch
from torch import Tensor


@dataclass
class RoFormerEncoderConfig:
    # Architecture
    embedding_size: Optional[int] = None
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072

    # Activations / dropout
    hidden_act: Union[str, Callable[[Tensor], Tensor]] = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    # Positions / segments
    max_position_embeddings: int = 1536
    # Number of token types (a.k.a. segment IDs). Set to 0 to disable token-type embeddings.
    num_token_types: int = 2

    # Numerics
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12

    # RoFormer specifics
    rotary_value: bool = False  # whether to apply RoPE to value as well

    # Feed-forward chunking (0 disables)
    chunk_size_feed_forward: int = 0

    # MoE (optional)
    use_moe: bool = False
    moe_num_experts: int = 0
    moe_top_k: int = 1
    # "all" or comma-separated layer indices (0-based)
    moe_layers: str = "all"
    moe_router_context_dim: int = 0
    moe_router_type: str = "linear"
    moe_router_hidden_dim: int = 256
    moe_router_dropout: float = 0.0
    moe_capacity_factor_train: float = 1.25
    moe_capacity_factor_eval: float = 2.0
    moe_use_z_loss: bool = True
    moe_z_loss_coef: float = 1e-3
    moe_aux_loss_coef: float = 1e-2
    moe_return_routing: bool = True

    # Output style
    use_return_dict: bool = True

    def __post_init__(self) -> None:
        if self.embedding_size is None:
            self.embedding_size = self.hidden_size

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
            )

        head_dim = self.hidden_size // self.num_attention_heads
        if head_dim % 2 != 0:
            # RoFormer rotary implementation splits sin/cos along the last dim.
            raise ValueError(f"Per-head dim (hidden_size/num_attention_heads = {head_dim}) must be even for RoPE.")

    # --- (optional) JSON helpers for easy reuse with HF-style config.json files ---

    @classmethod
    def from_json_file(cls, path: str) -> "RoFormerEncoderConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        field_names = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered)

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json_string(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


__all__ = ["RoFormerEncoderConfig"]
