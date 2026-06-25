from __future__ import annotations

"""Sparse MoE feed-forward block for standalone RoFormer."""

from dataclasses import dataclass
import typing as t
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
        self.required_expert_ids = tuple(int(expert_id) for expert_id in (self.moe.required_expert_ids or ()))
        # Runtime-only restriction; keep checkpoint/state_dict keys unchanged.
        self.route_filter_expert_ids: tuple[int, ...] | None = None
        if self.router_type == "learned":
            self.router = nn.Linear(config.hidden_size, self.num_experts)

    def set_route_filter_expert_ids(self, expert_ids: t.Iterable[int] | None) -> None:
        if expert_ids is None:
            self.route_filter_expert_ids = None
            return
        self.route_filter_expert_ids = tuple(sorted({int(expert_id) for expert_id in expert_ids}))

    def forward(
        self,
        hidden_states: torch.Tensor,
        modality_name: str | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> MoERoutingOutput:
        allowed_experts = self._allowed_experts(hidden_states.device, modality_name)
        if allowed_experts.numel() < self.top_k:
            raise ValueError(
                f"MoE modality '{modality_name}' exposes {allowed_experts.numel()} experts after route expert "
                f"group filtering, but top_k={self.top_k}."
            )

        required_experts = self._required_experts(hidden_states.device, allowed_experts, modality_name)
        routed_allowed_experts = self._routed_allowed_experts(allowed_experts, required_experts)
        routed_slots = self.top_k - required_experts.numel()
        if routed_allowed_experts.numel() < routed_slots:
            raise ValueError(
                f"MoE modality '{modality_name}' exposes {routed_allowed_experts.numel()} non-required experts "
                f"after route expert group filtering, but top_k={self.top_k} and "
                f"required_expert_ids={list(self.required_expert_ids)}."
            )

        if self.router_type in {"hard_modality", "hard_group"}:
            return self._hard_route(
                hidden_states,
                allowed_experts,
                required_experts,
                routed_allowed_experts,
                modality_name,
                token_mask,
            )

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
        if required_experts.numel() == 0:
            topk_probs, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(topk_probs.dtype).eps)
        else:
            topk_indices, topk_probs = self._select_required_router_topk(
                router_logits,
                router_probs,
                required_experts,
                routed_allowed_experts,
            )
        return self._build_output(router_logits, router_probs, topk_indices, topk_probs, modality_name, token_mask)

    def _allowed_experts(self, device: torch.device, modality_name: str | None) -> torch.Tensor:
        if not self.moe.use_modality_group_mask:
            expert_ids = set(range(self.num_experts))
        else:
            if modality_name is None or modality_name not in self.moe.modality_to_groups:
                raise ValueError(
                    "modality_name must reference model.backbone.moe.modality_to_groups "
                    "when MoE group mask is enabled."
                )

            expert_ids = set()
            for group_name in self.moe.modality_to_groups[modality_name]:
                expert_ids.update(self.moe.expert_groups[group_name])
        if self.route_filter_expert_ids is not None:
            # Keep modality eligibility first, then narrow it for specialist-style evaluation.
            expert_ids &= set(self.route_filter_expert_ids)
        return torch.tensor(sorted(expert_ids), device=device, dtype=torch.long)

    def _mask_logits(self, router_logits: torch.Tensor, allowed_experts: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(self.num_experts, device=router_logits.device, dtype=torch.bool)
        mask[allowed_experts] = True
        fill_value = torch.finfo(router_logits.dtype).min
        return router_logits.masked_fill(~mask.view(*([1] * (router_logits.dim() - 1)), -1), fill_value)

    def _required_experts(
        self,
        device: torch.device,
        allowed_experts: torch.Tensor,
        modality_name: str | None,
    ) -> torch.Tensor:
        if not self.required_expert_ids:
            return torch.empty(0, device=device, dtype=torch.long)
        allowed_ids = {int(expert_id) for expert_id in allowed_experts.tolist()}
        missing_required = [expert_id for expert_id in self.required_expert_ids if expert_id not in allowed_ids]
        if missing_required:
            raise ValueError(
                f"MoE modality '{modality_name}' does not expose required_expert_ids {missing_required} "
                "after route expert group filtering."
            )
        return torch.tensor(self.required_expert_ids, device=device, dtype=torch.long)

    def _routed_allowed_experts(
        self,
        allowed_experts: torch.Tensor,
        required_experts: torch.Tensor,
    ) -> torch.Tensor:
        if required_experts.numel() == 0:
            return allowed_experts
        required_ids = {int(expert_id) for expert_id in required_experts.tolist()}
        routed_ids = [int(expert_id) for expert_id in allowed_experts.tolist() if int(expert_id) not in required_ids]
        return torch.tensor(routed_ids, device=allowed_experts.device, dtype=torch.long)

    def _expand_required_indices(
        self,
        required_experts: torch.Tensor,
        batch_shape: torch.Size,
    ) -> torch.Tensor:
        view_shape = [1] * len(batch_shape) + [required_experts.numel()]
        return required_experts.view(*view_shape).expand(*batch_shape, required_experts.numel())

    def _select_required_router_topk(
        self,
        router_logits: torch.Tensor,
        router_probs: torch.Tensor,
        required_experts: torch.Tensor,
        routed_allowed_experts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_shape = router_probs.shape[:-1]
        required_indices = self._expand_required_indices(required_experts, batch_shape)
        routed_slots = self.top_k - required_experts.numel()
        if routed_slots > 0:
            routed_logits = self._mask_logits(router_logits, routed_allowed_experts)
            _, routed_indices = torch.topk(routed_logits, k=routed_slots, dim=-1)
        else:
            routed_indices = required_indices.new_empty(*batch_shape, 0)

        topk_indices = torch.cat((required_indices, routed_indices), dim=-1)
        if self.moe.required_expert_weight_mode == "fixed":
            required_weight = float(self.moe.required_expert_weight)
            required_probs = router_probs.new_full(required_indices.shape, required_weight)
            if routed_slots > 0:
                routed_raw_probs = router_probs.gather(-1, routed_indices)
                routed_probs = routed_raw_probs / routed_raw_probs.sum(dim=-1, keepdim=True).clamp_min(
                    torch.finfo(router_probs.dtype).eps
                )
                routed_probs = routed_probs * (1.0 - required_weight * required_experts.numel())
            else:
                routed_probs = router_probs.new_empty(*batch_shape, 0)
            topk_probs = torch.cat((required_probs, routed_probs), dim=-1)
        else:
            topk_probs = router_probs.gather(-1, topk_indices)
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(topk_probs.dtype).eps)
        return topk_indices, topk_probs

    def _hard_route(
        self,
        hidden_states: torch.Tensor,
        allowed_experts: torch.Tensor,
        required_experts: torch.Tensor,
        routed_allowed_experts: torch.Tensor,
        modality_name: str | None,
        token_mask: torch.Tensor | None,
    ) -> MoERoutingOutput:
        batch_size, seq_len = hidden_states.shape[:2]
        flat_count = batch_size * seq_len
        routed_slots = self.top_k - required_experts.numel()
        if routed_slots > 0:
            offset = self._hard_route_offset(modality_name, routed_allowed_experts.numel())
            slots = torch.arange(flat_count * routed_slots, device=hidden_states.device).view(flat_count, routed_slots)
            routed_indices = routed_allowed_experts[(slots + offset) % routed_allowed_experts.numel()].view(
                batch_size,
                seq_len,
                routed_slots,
            )
        else:
            routed_indices = torch.empty(batch_size, seq_len, 0, device=hidden_states.device, dtype=torch.long)
        if required_experts.numel() > 0:
            required_indices = self._expand_required_indices(required_experts, torch.Size((batch_size, seq_len)))
            topk_indices = torch.cat((required_indices, routed_indices), dim=-1)
        else:
            topk_indices = routed_indices
        topk_probs = self._hard_topk_probs(hidden_states, topk_indices, required_experts)

        router_logits = hidden_states.new_full(
            (batch_size, seq_len, self.num_experts),
            torch.finfo(hidden_states.dtype).min,
        )
        router_probs = torch.zeros_like(router_logits)
        router_probs.scatter_add_(-1, topk_indices, topk_probs)
        router_logits = router_logits.masked_fill(router_probs > 0, 0.0)
        return self._build_output(router_logits, router_probs, topk_indices, topk_probs, modality_name, token_mask)

    def _hard_topk_probs(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        required_experts: torch.Tensor,
    ) -> torch.Tensor:
        topk_probs = torch.full_like(topk_indices, 1.0 / self.top_k, dtype=hidden_states.dtype)
        required_count = required_experts.numel()
        if required_count > 0 and self.moe.required_expert_weight_mode == "fixed":
            required_weight = float(self.moe.required_expert_weight)
            topk_probs[..., :required_count] = required_weight
            routed_slots = self.top_k - required_count
            if routed_slots > 0:
                routed_weight = (1.0 - required_weight * required_count) / routed_slots
                topk_probs[..., required_count:] = routed_weight
        return topk_probs

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
            valid_tokens = token_weight.sum().clamp_min(torch.finfo(router_probs.dtype).eps)
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


def resolve_route_expert_ids(moe_cfg, group_names: t.Sequence[str] | None) -> tuple[int, ...] | None:
    if not group_names:
        return None
    if moe_cfg is None or not getattr(moe_cfg, "enabled", False):
        raise ValueError("--route-expert-groups requires model.backbone.moe.enabled=true.")

    expert_groups = getattr(moe_cfg, "expert_groups", None) or {}
    missing = [group_name for group_name in group_names if group_name not in expert_groups]
    if missing:
        raise ValueError(f"Unknown route expert group(s): {missing}. Available groups: {sorted(expert_groups)}.")

    expert_ids: set[int] = set()
    for group_name in group_names:
        expert_ids.update(int(expert_id) for expert_id in expert_groups[group_name])
    required_expert_ids = set(getattr(moe_cfg, "required_expert_ids", None) or ())
    missing_required = sorted(required_expert_ids - expert_ids)
    if missing_required:
        raise ValueError(f"Route expert groups {list(group_names)} exclude required_expert_ids {missing_required}.")
    return tuple(sorted(expert_ids))


def apply_route_expert_filter(
    module: nn.Module,
    moe_cfg,
    group_names: t.Sequence[str] | None,
) -> tuple[int, ...] | None:
    expert_ids = resolve_route_expert_ids(moe_cfg, group_names)
    for child in module.modules():
        if isinstance(child, TopKRouter):
            child.set_route_filter_expert_ids(expert_ids)
    return expert_ids


def clear_route_expert_filter(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, TopKRouter):
            child.set_route_filter_expert_ids(None)
