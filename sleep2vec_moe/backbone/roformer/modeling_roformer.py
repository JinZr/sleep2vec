from __future__ import annotations

"""Core RoFormer encoder modules with no transformers dependency."""

import math

import torch
from torch import nn

from .configuration import RoFormerConfig
from .moe import SparseMoE
from .outputs import RoFormerModelOutput


def get_activation_fn(name: str):
    """Return the activation function for the configured hidden act."""

    name = name.lower()
    if name == "gelu":
        return nn.functional.gelu
    if name == "relu":
        return nn.functional.relu
    if name == "selu":
        return nn.functional.selu
    if name == "gelu_new":

        def gelu_new(x: torch.Tensor) -> torch.Tensor:
            return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

        return gelu_new
    raise ValueError(f"Unsupported hidden_act='{name}'. Supported: gelu, relu, selu, gelu_new")


class RoFormerSinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding used by RoFormer rotary attention."""

    def __init__(self, num_positions: int, embedding_dim: int) -> None:
        super().__init__()
        weight = self.create_weight(num_positions, embedding_dim)
        self.register_buffer("weight", weight, persistent=False)

    @staticmethod
    def create_weight(num_positions: int, embedding_dim: int) -> torch.Tensor:
        position_ids = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)
        index_ids = torch.arange(embedding_dim, dtype=torch.float32).unsqueeze(0)
        position_enc = position_ids / torch.pow(10000, 2 * torch.floor_divide(index_ids, 2) / embedding_dim)

        sentinel = embedding_dim // 2 if embedding_dim % 2 == 0 else (embedding_dim // 2) + 1
        out = torch.empty(num_positions, embedding_dim, dtype=torch.float32)
        out[:, :sentinel] = torch.sin(position_enc[:, 0::2])
        out[:, sentinel:] = torch.cos(position_enc[:, 1::2])
        return out

    def forward(
        self, seq_len: int, device: torch.device, offset: int = 0, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        position_ids = torch.arange(offset, offset + seq_len, dtype=torch.long, device=device)
        values = self.weight.index_select(0, position_ids)
        if dtype is not None:
            values = values.to(dtype=dtype)
        return values


class RoFormerEmbeddings(nn.Module):
    """Embedding preprocessor used before encoder stack."""

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)
        self.layer_norm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=inputs_embeds.device)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings)


class RoFormerSelfAttention(nn.Module):
    """Multi-head self-attention with rotary position embedding."""

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.rotary_value = config.rotary_value

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        rotary_components: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size = hidden_states.size(0)

        query_layer = (
            self.query(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        key_layer = (
            self.key(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        value_layer = (
            self.value(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )

        if rotary_components is not None:
            if self.rotary_value:
                query_layer, key_layer, value_layer = self.apply_rotary_position_embeddings(
                    rotary_components, query_layer, key_layer, value_layer
                )
            else:
                query_layer, key_layer = self.apply_rotary_position_embeddings(
                    rotary_components, query_layer, key_layer
                )

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(batch_size, -1, self.all_head_size)

        if output_attentions:
            return context_layer, attention_probs
        return context_layer, None

    @staticmethod
    def build_rotary_components(sinusoidal_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)
        sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
        cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
        return sin_pos, cos_pos

    @staticmethod
    def apply_rotary_position_embeddings(
        rotary_components: tuple[torch.Tensor, torch.Tensor],
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor | None = None,
    ):
        sin_pos, cos_pos = rotary_components

        rotate_half_query_layer = torch.stack([-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1).reshape_as(
            query_layer
        )
        rotate_half_key_layer = torch.stack([-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1).reshape_as(key_layer)

        query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
        key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos

        if value_layer is None:
            return query_layer, key_layer

        rotate_half_value_layer = torch.stack([-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1).reshape_as(
            value_layer
        )
        value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
        return query_layer, key_layer, value_layer


class RoFormerSelfOutput(nn.Module):
    """Post-attention projection and residual connection."""

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.layer_norm(hidden_states + input_tensor)


class RoFormerAttention(nn.Module):
    """RoFormer self-attention block."""

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.self_attention = RoFormerSelfAttention(config)
        self.output = RoFormerSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        rotary_components: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self_output, attention_probs = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            rotary_components=rotary_components,
            output_attentions=output_attentions,
        )
        attention_output = self.output(self_output, hidden_states)
        return attention_output, attention_probs


class RoFormerIntermediate(nn.Module):
    """Feed-forward intermediate projection."""

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = get_activation_fn(config.hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.intermediate_act_fn(self.dense(hidden_states))


class RoFormerOutput(nn.Module):
    """Feed-forward output projection with residual."""

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.layer_norm(hidden_states + input_tensor)


class RoFormerMoEBlock(nn.Module):
    """MoE FFN block with residual, dropout, and LayerNorm."""

    def __init__(self, config: RoFormerConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.moe = SparseMoE(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_ctx: torch.Tensor | None = None,
        router_group_ids: dict[str, torch.Tensor] | None = None,
        collect_moe_stats: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        moe_output, moe_aux = self.moe(
            hidden_states,
            ctx=router_ctx,
            group_ids=router_group_ids,
            collect_stats=collect_moe_stats,
        )
        moe_output = self.dropout(moe_output)
        layer_output = self.layer_norm(moe_output + hidden_states)
        return layer_output, moe_aux


class RoFormerLayer(nn.Module):
    """Single transformer encoder layer used by RoFormer."""

    def __init__(self, config: RoFormerConfig, layer_idx: int, moe_layer_indices: set[int]) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.use_moe = bool(config.moe_enabled and layer_idx in moe_layer_indices)
        self.attention = RoFormerAttention(config)
        if self.use_moe:
            self.moe_ffn = RoFormerMoEBlock(config, layer_idx=layer_idx)
        else:
            self.intermediate = RoFormerIntermediate(config)
            self.output = RoFormerOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        rotary_components: tuple[torch.Tensor, torch.Tensor] | None = None,
        router_ctx: torch.Tensor | None = None,
        router_group_ids: dict[str, torch.Tensor] | None = None,
        collect_moe_stats: bool = False,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor] | None]:
        attention_output, attention_probs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            rotary_components=rotary_components,
            output_attentions=output_attentions,
        )
        if self.use_moe:
            layer_output, moe_aux = self.moe_ffn(
                attention_output,
                router_ctx=router_ctx,
                router_group_ids=router_group_ids,
                collect_moe_stats=collect_moe_stats,
            )
            return layer_output, attention_probs, moe_aux

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs, None


class RoFormerEncoder(nn.Module):
    """RoFormer encoder stack."""

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_positions = RoFormerSinusoidalPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size // config.num_attention_heads
        )
        self.moe_layer_indices = config.resolve_moe_layer_indices() if config.moe_enabled else set()
        self.layer = nn.ModuleList(
            [
                RoFormerLayer(config, layer_idx=idx, moe_layer_indices=self.moe_layer_indices)
                for idx in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        router_ctx: torch.Tensor | None = None,
        router_group_ids: dict[str, torch.Tensor] | None = None,
        collect_moe_stats: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        total_aux_loss = hidden_states.new_zeros((), dtype=torch.float32)
        moe_layer_count = 0
        moe_sums: dict[str, torch.Tensor] = {}
        moe_last: dict[str, torch.Tensor] = {}
        moe_lists: dict[str, list[torch.Tensor | int]] = {}

        seq_len = hidden_states.size(1)
        sinusoidal_pos = self.embed_positions(
            seq_len=seq_len,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )[None, None, :, :]
        rotary_components = RoFormerSelfAttention.build_rotary_components(sinusoidal_pos)

        for layer_module in self.layer:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states, attention_probs, moe_aux = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                rotary_components=rotary_components,
                router_ctx=router_ctx,
                router_group_ids=router_group_ids,
                collect_moe_stats=collect_moe_stats,
                output_attentions=output_attentions,
            )

            if output_attentions:
                all_attentions = all_attentions + (attention_probs,)
            if moe_aux is not None:
                total_aux_loss = total_aux_loss + moe_aux["aux_loss"]
                moe_layer_count += 1
                for key, value in moe_aux.items():
                    if key == "aux_loss":
                        continue
                    if key in {
                        "router_logits",
                        "router_probs",
                        "expert_indices",
                        "dispatch_mask",
                        "dropped_mask",
                        "capacity",
                    }:
                        if collect_moe_stats:
                            moe_lists.setdefault(key, []).append(value)
                        continue
                    if key == "num_experts":
                        if collect_moe_stats:
                            moe_lists.setdefault(key, []).append(value)
                        continue
                    value_detached = value.detach()
                    if key not in moe_sums:
                        moe_sums[key] = value_detached.clone()
                    else:
                        moe_sums[key] = moe_sums[key] + value_detached
                    moe_last[key] = value_detached

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if moe_layer_count > 0:
            inv_layers = 1.0 / float(moe_layer_count)
            moe_metrics = {f"mean/{key}": value * inv_layers for key, value in moe_sums.items()}
            moe_metrics.update({f"last/{key}": value for key, value in moe_last.items()})
            if collect_moe_stats:
                for key, value in moe_lists.items():
                    moe_metrics[key] = list(value)
            moe_loss = total_aux_loss
        else:
            moe_metrics = {}
            moe_loss = hidden_states.new_zeros((), dtype=torch.float32)

        if not return_dict:
            outputs = (hidden_states,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_attentions,)
            outputs = outputs + (moe_loss, moe_metrics)
            return outputs

        return RoFormerModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            moe_loss=moe_loss,
            moe_metrics=moe_metrics,
        )
