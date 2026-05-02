from __future__ import annotations

"""Sparse MoE feed-forward block for standalone RoFormer."""

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


@dataclass
class MoERoutingOutput:
    router_logits: torch.Tensor
    router_probs: torch.Tensor
    topk_indices: torch.Tensor
    topk_probs: torch.Tensor
    expert_mask: torch.Tensor
    load: torch.Tensor
    importance: torch.Tensor
    z_loss: torch.Tensor
    entropy: torch.Tensor
    modality_name: str | None
    layer_idx: int


class TopKRouter(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.moe = config.moe
        self.layer_idx = layer_idx
        self.num_experts = self.moe.num_experts
        self.top_k = self.moe.top_k
        self.router_type = self.moe.router_type
        if self.router_type == "learned":
            self.router = nn.Linear(config.hidden_size, self.num_experts)

    def forward(
        self,
        hidden_states: torch.Tensor,
        modality_name: str | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> MoERoutingOutput:
        allowed_experts = self._allowed_experts(hidden_states.device, modality_name)
        if allowed_experts.numel() < self.top_k:
            raise ValueError(
                f"MoE modality '{modality_name}' exposes {allowed_experts.numel()} experts, " f"but top_k={self.top_k}."
            )

        if self.router_type in {"hard_modality", "hard_group"}:
            return self._hard_route(hidden_states, allowed_experts, modality_name, token_mask)

        if self.router_type == "random":
            router_logits = torch.randn(
                *hidden_states.shape[:-1],
                self.num_experts,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        else:
            router_logits = self.router(hidden_states)

        if self.training and self.moe.router_noise > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.moe.router_noise

        router_logits = self._mask_logits(router_logits, allowed_experts)
        router_probs = torch.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(topk_probs.dtype).eps)
        return self._build_output(router_logits, router_probs, topk_indices, topk_probs, modality_name, token_mask)

    def _allowed_experts(self, device: torch.device, modality_name: str | None) -> torch.Tensor:
        if not self.moe.use_modality_group_mask:
            return torch.arange(self.num_experts, device=device)
        if modality_name is None or modality_name not in self.moe.modality_to_groups:
            raise ValueError(
                "modality_name must reference model.backbone.moe.modality_to_groups " "when MoE group mask is enabled."
            )

        expert_ids: set[int] = set()
        for group_name in self.moe.modality_to_groups[modality_name]:
            expert_ids.update(self.moe.expert_groups[group_name])
        return torch.tensor(sorted(expert_ids), device=device, dtype=torch.long)

    def _mask_logits(self, router_logits: torch.Tensor, allowed_experts: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(self.num_experts, device=router_logits.device, dtype=torch.bool)
        mask[allowed_experts] = True
        fill_value = torch.finfo(router_logits.dtype).min
        return router_logits.masked_fill(~mask.view(*([1] * (router_logits.dim() - 1)), -1), fill_value)

    def _hard_route(
        self,
        hidden_states: torch.Tensor,
        allowed_experts: torch.Tensor,
        modality_name: str | None,
        token_mask: torch.Tensor | None,
    ) -> MoERoutingOutput:
        batch_size, seq_len = hidden_states.shape[:2]
        flat_count = batch_size * seq_len
        offset = self._hard_route_offset(modality_name, allowed_experts.numel())
        slots = torch.arange(flat_count * self.top_k, device=hidden_states.device).view(flat_count, self.top_k)
        topk_indices = allowed_experts[(slots + offset) % allowed_experts.numel()].view(batch_size, seq_len, self.top_k)
        topk_probs = torch.full(
            (batch_size, seq_len, self.top_k),
            1.0 / self.top_k,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        router_logits = hidden_states.new_full(
            (batch_size, seq_len, self.num_experts),
            torch.finfo(hidden_states.dtype).min,
        )
        router_probs = torch.zeros_like(router_logits)
        router_probs.scatter_add_(-1, topk_indices, topk_probs)
        router_logits = router_logits.masked_fill(router_probs > 0, 0.0)
        return self._build_output(router_logits, router_probs, topk_indices, topk_probs, modality_name, token_mask)

    def _hard_route_offset(self, modality_name: str | None, allowed_count: int) -> int:
        if modality_name is None:
            return 0
        if self.router_type == "hard_group" and self.moe.use_modality_group_mask:
            key = "|".join(self.moe.modality_to_groups[modality_name])
        else:
            key = modality_name
        return sum(ord(ch) for ch in key) % allowed_count

    def _build_output(
        self,
        router_logits: torch.Tensor,
        router_probs: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_probs: torch.Tensor,
        modality_name: str | None,
        token_mask: torch.Tensor | None,
    ) -> MoERoutingOutput:
        expert_mask = torch.zeros_like(router_probs, dtype=torch.bool)
        expert_mask.scatter_(-1, topk_indices, True)
        z_loss_per_token = torch.logsumexp(router_logits, dim=-1).pow(2)
        entropy_per_token = -(router_probs * router_probs.clamp_min(torch.finfo(router_probs.dtype).eps).log()).sum(
            dim=-1
        )
        if token_mask is None:
            load = expert_mask.float().sum(dim=(0, 1))
            importance = router_probs.sum(dim=(0, 1))
            z_loss = z_loss_per_token.mean()
            entropy = entropy_per_token.mean()
        else:
            token_weight = token_mask.to(device=router_probs.device, dtype=router_probs.dtype)
            token_weight_expanded = token_weight.unsqueeze(-1)
            load = (expert_mask.to(dtype=router_probs.dtype) * token_weight_expanded).sum(dim=(0, 1))
            importance = (router_probs * token_weight_expanded).sum(dim=(0, 1))
            valid_tokens = token_weight.sum()
            z_loss = (z_loss_per_token * token_weight).sum() / valid_tokens
            entropy = (entropy_per_token * token_weight).sum() / valid_tokens
        return MoERoutingOutput(
            router_logits=router_logits,
            router_probs=router_probs,
            topk_indices=topk_indices,
            topk_probs=topk_probs,
            expert_mask=expert_mask,
            load=load,
            importance=importance,
            z_loss=z_loss,
            entropy=entropy,
            modality_name=modality_name,
            layer_idx=self.layer_idx,
        )


class SparseExpertMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        expert_hidden_size: int,
        activation_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()
        self.dense_in = nn.Linear(hidden_size, expert_hidden_size)
        self.dense_out = nn.Linear(expert_hidden_size, hidden_size)
        self.activation_fn = activation_fn

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dense_out(self.activation_fn(self.dense_in(hidden_states)))


class SparseMoEFFN(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        activation_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.router = TopKRouter(config, layer_idx=layer_idx)
        expert_hidden_size = config.moe.expert_hidden_size or config.intermediate_size
        self.experts = nn.ModuleList(
            [
                SparseExpertMLP(config.hidden_size, expert_hidden_size, activation_fn)
                for _ in range(config.moe.num_experts)
            ]
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.moe.expert_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
        *,
        modality_name: str | None = None,
        attention_mask: torch.Tensor | None = None,
        collect_aux: bool = False,
    ) -> tuple[torch.Tensor, MoERoutingOutput | None]:
        token_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                token_mask = attention_mask[:, 0, 0, :].eq(0)
            elif attention_mask.dim() == 3:
                token_mask = attention_mask[:, 0, :].eq(0)
            elif attention_mask.dim() == 2:
                token_mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
            else:
                raise ValueError(f"attention_mask should have 2, 3, or 4 dimensions; got {tuple(attention_mask.shape)}")
        routing = self.router(hidden_states, modality_name=modality_name, token_mask=token_mask)
        flat_hidden = hidden_states.reshape(-1, hidden_states.size(-1))
        flat_indices = routing.topk_indices.reshape(-1, routing.topk_indices.size(-1))
        flat_probs = routing.topk_probs.reshape(-1, routing.topk_probs.size(-1))
        flat_output = torch.zeros_like(flat_hidden)

        for expert_id, expert in enumerate(self.experts):
            token_idx, slot_idx = (flat_indices == expert_id).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            expert_input = flat_hidden.index_select(0, token_idx)
            expert_output = expert(expert_input)
            flat_output.index_add_(0, token_idx, expert_output * flat_probs[token_idx, slot_idx].unsqueeze(-1))

        hidden_states = flat_output.view_as(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states, routing if collect_aux else None
