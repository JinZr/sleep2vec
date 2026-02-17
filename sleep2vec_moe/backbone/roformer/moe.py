from __future__ import annotations

"""Sparse MoE building blocks for RoFormer FFN replacement."""

import math

import torch
from torch import nn

from .configuration import RoFormerConfig


def _get_activation_fn(name: str):
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


def _cv2(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = x.mean()
    if torch.abs(mean).item() <= eps:
        return torch.zeros((), dtype=torch.float32, device=x.device)
    var = x.var(unbiased=False)
    return var / (mean.square() + eps)


class ExpertFFN(nn.Module):
    """Single expert MLP block used inside sparse MoE."""

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = _get_activation_fn(config.hidden_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TopKRouter(nn.Module):
    """Token router that computes top-k expert assignments."""

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.router_noise_std = config.moe_router_noise_std
        self.use_context = config.moe_router_use_context

        self.token_router = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.context_router = None
        if self.use_context:
            self.context_router = nn.Linear(config.moe_router_ctx_dim, self.num_experts, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.num_experts))

    def forward(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.training and self.router_noise_std > 0.0:
            x_router = x + torch.randn_like(x) * self.router_noise_std
        else:
            x_router = x

        logits = self.token_router(x_router)
        if self.context_router is not None and ctx is not None:
            if ctx.dim() != 2:
                raise ValueError(f"router_ctx must have shape [B, C], got shape {tuple(ctx.shape)}")
            logits = logits + self.context_router(ctx).unsqueeze(1)
        logits = logits + self.bias

        probs = torch.softmax(logits.to(dtype=torch.float32), dim=-1)
        topk_w, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)
        if self.top_k > 1:
            topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        topk_w = topk_w.to(dtype=x.dtype)
        return probs, topk_idx.to(dtype=torch.long), topk_w, logits.to(dtype=torch.float32)


class SparseMoE(nn.Module):
    """Token-level sparse MoE with top-k routing and capacity-based dropping."""

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.capacity_factor_train = config.moe_capacity_factor_train
        self.capacity_factor_eval = config.moe_capacity_factor_eval
        self.aux_loss_coef = config.moe_aux_loss_coef
        self.z_loss_coef = config.moe_z_loss_coef
        self.group_balance_coef = config.moe_group_balance_coef
        self.group_keys = tuple(config.moe_group_keys)
        self.min_capacity = 1

        self.router = TopKRouter(config)
        self.experts = nn.ModuleList([ExpertFFN(config) for _ in range(self.num_experts)])

    def _capacity(self, num_tokens: int) -> int:
        capacity_factor = self.capacity_factor_train if self.training else self.capacity_factor_eval
        capacity = math.ceil(capacity_factor * (num_tokens * self.top_k) / float(self.num_experts))
        return max(capacity, self.min_capacity)

    def _compute_group_balance_loss(
        self,
        router_probs: torch.Tensor,
        group_ids: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        device = router_probs.device
        if self.group_balance_coef <= 0.0:
            return torch.zeros((), dtype=torch.float32, device=device)
        if not group_ids:
            return torch.zeros((), dtype=torch.float32, device=device)
        if router_probs.numel() == 0:
            return torch.zeros((), dtype=torch.float32, device=device)

        batch_size, seq_len, _ = router_probs.shape
        router_probs_flat = router_probs.reshape(batch_size * seq_len, self.num_experts)
        batch_indices = (
            torch.arange(batch_size, dtype=torch.long, device=device)
            .unsqueeze(1)
            .expand(batch_size, seq_len)
            .reshape(-1)
        )
        key_losses: list[torch.Tensor] = []
        for key in self.group_keys:
            values = group_ids.get(key)
            if values is None:
                continue
            if not torch.is_tensor(values):
                values = torch.as_tensor(values, dtype=torch.long, device=device)
            else:
                values = values.to(device=device, dtype=torch.long)
            if values.dim() != 1:
                raise ValueError(f"router_group_ids['{key}'] must have shape [B], got {tuple(values.shape)}")
            if values.numel() == 0:
                continue

            groups = values[batch_indices]
            valid_mask = groups >= 0
            if not valid_mask.any():
                continue
            groups = groups[valid_mask]
            probs_valid = router_probs_flat[valid_mask]
            unique_groups = groups.unique()
            if unique_groups.numel() == 0:
                continue

            per_group_losses: list[torch.Tensor] = []
            for group_value in unique_groups.tolist():
                group_mask = groups == int(group_value)
                if not group_mask.any():
                    continue
                q = probs_valid[group_mask].mean(dim=0)
                per_group_losses.append(self.num_experts * torch.sum(q.square()))

            if per_group_losses:
                key_losses.append(torch.stack(per_group_losses).mean())

        if not key_losses:
            return torch.zeros((), dtype=torch.float32, device=device)
        return torch.stack(key_losses).mean()

    def forward(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor | None = None,
        group_ids: dict[str, torch.Tensor] | None = None,
        collect_stats: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, seq_len, hidden_size = x.shape
        num_tokens = batch_size * seq_len
        capacity = self._capacity(num_tokens)

        router_probs, topk_idx, topk_w, router_logits = self.router(x, ctx=ctx)
        x_flat = x.reshape(num_tokens, hidden_size)
        idx_flat = topk_idx.reshape(num_tokens, self.top_k)
        w_flat = topk_w.reshape(num_tokens, self.top_k)

        y_flat = torch.zeros_like(x_flat)
        dispatched = torch.zeros((num_tokens, self.top_k), dtype=torch.bool, device=x.device)
        dropped = 0

        for expert_idx, expert in enumerate(self.experts):
            positions = (idx_flat == expert_idx).nonzero(as_tuple=False)
            if positions.numel() == 0:
                continue
            token_ids = positions[:, 0]
            kslot_ids = positions[:, 1]
            weights = w_flat[token_ids, kslot_ids]
            num_assignments = token_ids.numel()

            if num_assignments > capacity:
                keep = torch.topk(weights.to(dtype=torch.float32), k=capacity, dim=0).indices
                token_ids = token_ids[keep]
                kslot_ids = kslot_ids[keep]
                weights = weights[keep]
                dropped += num_assignments - capacity

            dispatched[token_ids, kslot_ids] = True
            expert_out = expert(x_flat[token_ids])
            y_flat.index_add_(0, token_ids, expert_out * weights.unsqueeze(-1))

        dispatch_expert_ids = idx_flat[dispatched]
        load_counts = torch.bincount(dispatch_expert_ids, minlength=self.num_experts).to(dtype=torch.float32)
        dispatch_total = load_counts.sum()
        mean_importance = router_probs.mean(dim=(0, 1))
        if dispatch_total.item() <= 0:
            expert_load = torch.zeros(self.num_experts, dtype=torch.float32, device=x.device)
            lb_loss_raw = router_probs.sum() * 0.0
        else:
            expert_load = load_counts / dispatch_total
            lb_loss_raw = self.num_experts * torch.sum(expert_load * mean_importance)

        z_loss_raw = torch.logsumexp(router_logits, dim=-1).square().mean()

        group_loss_raw = self._compute_group_balance_loss(
            router_probs=router_probs,
            group_ids=group_ids,
        )

        aux_loss = self.aux_loss_coef * lb_loss_raw
        aux_loss = aux_loss + self.z_loss_coef * z_loss_raw
        aux_loss = aux_loss + self.group_balance_coef * group_loss_raw

        dropped_fraction = torch.tensor(
            dropped / float(max(1, num_tokens * self.top_k)),
            dtype=torch.float32,
            device=x.device,
        )
        load_cv2 = _cv2(expert_load)
        importance_cv2 = _cv2(mean_importance)

        aux = {
            "aux_loss": aux_loss,
            "lb_loss_raw": lb_loss_raw.detach(),
            "z_loss_raw": z_loss_raw.detach(),
            "group_loss_raw": group_loss_raw.detach(),
            "expert_load": expert_load.detach(),
            "expert_load_count": load_counts.detach(),
            "expert_importance": mean_importance.detach(),
            "dropped_fraction": dropped_fraction.detach(),
            "load_cv2": load_cv2.detach(),
            "importance_cv2": importance_cv2.detach(),
        }
        if collect_stats:
            token_dispatched = dispatched.any(dim=1).reshape(batch_size, seq_len)
            aux["router_logits"] = router_logits
            aux["router_probs"] = router_probs
            aux["expert_indices"] = topk_idx
            aux["dispatch_mask"] = dispatched.reshape(batch_size, seq_len, self.top_k)
            aux["dropped_mask"] = ~token_dispatched
            aux["capacity"] = int(capacity)
            aux["num_experts"] = int(self.num_experts)
        return y_flat.reshape(batch_size, seq_len, hidden_size), aux
