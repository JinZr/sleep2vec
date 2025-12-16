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

"""Standalone RoFormer Encoder.

Design goals:
  - Core encoder architecture lives in a single local module (this file).
  - Helper utilities are kept alongside (see config.py, rotary.py, utils.py).
  - No dependency on Hugging Face `transformers`.
  - Recipe code can import the encoder locally, e.g.:
        from roformer_encoder import RoFormerEncoderModel

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from .config import RoFormerEncoderConfig
from .rotary import RoFormerSinusoidalPositionalEmbedding, apply_rotary_position_embeddings
from .utils import apply_chunking_to_forward, get_activation_fn, make_extended_attention_mask

# -------------------------
# Outputs (HF-like but lightweight)
# -------------------------


@dataclass
class RoFormerEncoderOutput:
    last_hidden_state: Tensor
    hidden_states: Optional[Tuple[Tensor, ...]] = None
    attentions: Optional[Tuple[Tensor, ...]] = None


@dataclass
class RoFormerModelOutput:
    last_hidden_state: Tensor
    hidden_states: Optional[Tuple[Tensor, ...]] = None
    attentions: Optional[Tuple[Tensor, ...]] = None


# -------------------------
# Blocks
# -------------------------


class RoFormerEmbeddings(nn.Module):
    """Input (+ token_type) embeddings, layernorm, dropout.

    This encoder-only extraction is designed to work on **pre-computed input
    embeddings** (e.g., acoustic features projected to `embedding_size`). It
    intentionally does **not** contain a word embedding table.
    """

    def __init__(self, config: RoFormerEncoderConfig):
        super().__init__()
        self.token_type_embeddings: Optional[nn.Embedding]
        if config.num_token_types > 0:
            self.token_type_embeddings = nn.Embedding(config.num_token_types, config.embedding_size)
        else:
            self.token_type_embeddings = None

        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        token_type_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
    ) -> Tensor:
        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided (this encoder does not take token ids)")

        input_shape = inputs_embeds.size()[:-1]

        expected_dim = (
            self.token_type_embeddings.embedding_dim
            if self.token_type_embeddings is not None
            else self.LayerNorm.normalized_shape[0]
        )
        if inputs_embeds.size(-1) != expected_dim:
            raise ValueError(
                f"inputs_embeds last dim ({inputs_embeds.size(-1)}) must equal embedding_size " f"({expected_dim})."
            )

        embeddings = inputs_embeds
        if self.token_type_embeddings is not None:
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=inputs_embeds.device)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class RoFormerSelfAttention(nn.Module):
    def __init__(self, config: RoFormerEncoderConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.rotary_value = config.rotary_value

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        sinusoidal_pos: Optional[Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # (bsz, seq, hidden)
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Apply RoPE
        if sinusoidal_pos is not None:
            if self.rotary_value:
                query_layer, key_layer, value_layer = apply_rotary_position_embeddings(
                    sinusoidal_pos, query_layer, key_layer, value_layer
                )
                assert value_layer is not None
            else:
                query_layer, key_layer, _ = apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer)

        # attention_scores: (bsz, heads, seq, seq)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size**0.5)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if output_attentions:
            return context_layer, attention_probs
        return context_layer, None


class RoFormerSelfOutput(nn.Module):
    def __init__(self, config: RoFormerEncoderConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class RoFormerAttention(nn.Module):
    def __init__(self, config: RoFormerEncoderConfig):
        super().__init__()
        self.self = RoFormerSelfAttention(config)
        self.output = RoFormerSelfOutput(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        sinusoidal_pos: Optional[Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        self_output, attn = self.self(
            hidden_states,
            attention_mask=attention_mask,
            sinusoidal_pos=sinusoidal_pos,
            output_attentions=output_attentions,
        )
        attention_output = self.output(self_output, hidden_states)
        return attention_output, attn


class RoFormerIntermediate(nn.Module):
    def __init__(self, config: RoFormerEncoderConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = get_activation_fn(config.hidden_act)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class RoFormerOutput(nn.Module):
    def __init__(self, config: RoFormerEncoderConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class RoFormerLayer(nn.Module):
    def __init__(self, config: RoFormerEncoderConfig):
        super().__init__()
        self.config = config
        self.attention = RoFormerAttention(config)
        self.intermediate = RoFormerIntermediate(config)
        self.output = RoFormerOutput(config)

        # HF-style knob; 0 disables
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        sinusoidal_pos: Optional[Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        attention_output, attn = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            sinusoidal_pos=sinusoidal_pos,
            output_attentions=output_attentions,
        )

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        return layer_output, attn

    def feed_forward_chunk(self, attention_output: Tensor) -> Tensor:
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class RoFormerEncoder(nn.Module):
    def __init__(self, config: RoFormerEncoderConfig):
        super().__init__()
        self.config = config

        head_dim = config.hidden_size // config.num_attention_heads
        self.embed_positions = RoFormerSinusoidalPositionalEmbedding(config.max_position_embeddings, head_dim)

        self.layer = nn.ModuleList([RoFormerLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[RoFormerEncoderOutput, Tuple[Tensor, ...]]:
        all_hidden_states: Optional[Tuple[Tensor, ...]] = () if output_hidden_states else None
        all_attentions: Optional[Tuple[Tensor, ...]] = () if output_attentions else None

        # (seq, head_dim) -> (1, 1, seq, head_dim)
        sinusoidal_pos = self.embed_positions(hidden_states.shape[:-1])[None, None, :, :]

        for layer_module in self.layer:
            if output_hidden_states:
                assert all_hidden_states is not None
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states, attn = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                sinusoidal_pos=sinusoidal_pos,
                output_attentions=output_attentions,
            )

            if output_attentions:
                assert all_attentions is not None
                all_attentions = all_attentions + (attn,)

        if output_hidden_states:
            assert all_hidden_states is not None
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            out = (hidden_states,)
            if output_hidden_states:
                out = out + (all_hidden_states,)
            if output_attentions:
                out = out + (all_attentions,)
            return out

        return RoFormerEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


# -------------------------
# Top-level encoder-only model
# -------------------------


class RoFormerEncoderModel(nn.Module):
    """Encoder-only RoFormer model: embeddings (+ optional projection) + encoder stack.

    Constructor style intentionally mirrors Icefall Zipformer usage:
      - Pass explicit kwargs (usually from args/params) to build the model.
      - Internally stores a lightweight RoFormerEncoderConfig for reference.

    Example (like Icefall's get_encoder_model(params)):

        encoder = RoFormerEncoderModel(
            hidden_size=params.hidden_size,
            num_hidden_layers=params.num_hidden_layers,
            num_attention_heads=params.num_attention_heads,
            intermediate_size=params.intermediate_size,
            max_position_embeddings=params.max_position_embeddings,
            rotary_value=params.rotary_value,
        )
    """

    def __init__(
        self,
        embedding_size: Optional[int] = None,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: Union[str, Callable[[Tensor], Tensor]] = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 1536,
        num_token_types: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        rotary_value: bool = False,
        chunk_size_feed_forward: int = 0,
        use_return_dict: bool = True,
    ):
        super().__init__()

        self.config = RoFormerEncoderConfig(
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            num_token_types=num_token_types,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            rotary_value=rotary_value,
            chunk_size_feed_forward=chunk_size_feed_forward,
            use_return_dict=use_return_dict,
        )

        self.embeddings = RoFormerEmbeddings(self.config)
        if self.config.embedding_size != self.config.hidden_size:
            self.embeddings_project = nn.Linear(self.config.embedding_size, self.config.hidden_size)

        self.encoder = RoFormerEncoder(self.config)

        self.apply(self._init_weights)

        # Overwrite sinusoidal weights (init above may have touched them)
        with torch.no_grad():
            self.encoder.embed_positions.weight.copy_(self.encoder.embed_positions.create_weight())

    @classmethod
    def from_config(cls, config: RoFormerEncoderConfig) -> "RoFormerEncoderModel":
        return cls(**config.to_dict())

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[RoFormerModelOutput, Tuple[Tensor, ...]]:
        if return_dict is None:
            return_dict = self.config.use_return_dict

        if inputs_embeds is None:
            raise ValueError("You must specify inputs_embeds (this encoder does not take token ids)")

        input_shape = inputs_embeds.size()[:-1]
        device = inputs_embeds.device

        batch_size, seq_len = input_shape

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = make_extended_attention_mask(attention_mask, dtype=torch.float32)

        embedding_output = self.embeddings(
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        if hasattr(self, "embeddings_project"):
            embedding_output = self.embeddings_project(embedding_output)

        # Cast mask to model dtype (important for bf16/fp16)
        extended_attention_mask = extended_attention_mask.to(dtype=embedding_output.dtype)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            # encoder returns tuples when return_dict=False
            if isinstance(encoder_outputs, tuple):
                return encoder_outputs
            return (encoder_outputs.last_hidden_state,)

        assert isinstance(encoder_outputs, RoFormerEncoderOutput)
        return RoFormerModelOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


__all__ = [
    "RoFormerEncoderConfig",
    "RoFormerEmbeddings",
    "RoFormerSelfAttention",
    "RoFormerLayer",
    "RoFormerEncoder",
    "RoFormerEncoderModel",
    "RoFormerEncoderOutput",
    "RoFormerModelOutput",
]
