# coding=utf-8

from __future__ import annotations

import typing as t

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .utils import get_activation_fn


class ExpertFFN(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: t.Union[str, t.Callable[[Tensor], Tensor]],
    ) -> None:
        super().__init__()
        self.dense_in = nn.Linear(hidden_size, intermediate_size)
        self.act = get_activation_fn(activation)
        self.dense_out = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dense_in(x)
        x = self.act(x)
        x = self.dense_out(x)
        return x


class TopKRouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        *,
        context_dim: int = 0,
        router_type: str = "linear",
        router_hidden_dim: int = 256,
        router_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be > 0 for MoE routing.")

        self.num_experts = num_experts
        self.context_dim = int(context_dim)
        self.router_type = router_type
        self.router_dropout = float(router_dropout)

        if router_type == "linear":
            self.router = nn.Linear(hidden_size, num_experts, bias=False)
            self.context_router = (
                nn.Linear(self.context_dim, num_experts, bias=True) if self.context_dim > 0 else None
            )
        elif router_type == "mlp":
            input_dim = hidden_size + self.context_dim
            self.router = nn.Sequential(
                nn.Linear(input_dim, router_hidden_dim),
                nn.GELU(),
                nn.Dropout(router_dropout),
                nn.Linear(router_hidden_dim, num_experts),
            )
            self.context_router = None
        else:
            raise ValueError(f"Unsupported router_type '{router_type}'. Use 'linear' or 'mlp'.")

        self.logits_dropout = nn.Dropout(router_dropout) if router_dropout > 0 else None

    def forward(
        self,
        x: Tensor,
        *,
        context: Tensor | None = None,
        token_mask: Tensor | None = None,
        top_k: int = 1,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
        # x: [B, L, H]
        bsz, seq_len, hidden = x.shape
        flat_x = x.reshape(-1, hidden)

        if self.context_dim > 0:
            if context is None:
                context = torch.zeros(bsz, self.context_dim, device=x.device, dtype=x.dtype)
            if context.dim() != 2 or context.shape[0] != bsz or context.shape[1] != self.context_dim:
                raise ValueError(
                    f"router_context must have shape [B, {self.context_dim}], got {tuple(context.shape)}"
                )
            context_exp = context[:, None, :].expand(bsz, seq_len, self.context_dim).reshape(-1, self.context_dim)
        else:
            context_exp = None

        # Route logits in float32 for stability.
        flat_x_float = flat_x.float()
        if self.router_type == "linear":
            logits = self.router(flat_x_float)
            if self.context_router is not None:
                logits = logits + self.context_router(context_exp.float())
        else:
            if context_exp is None:
                router_in = flat_x_float
            else:
                router_in = torch.cat([flat_x_float, context_exp.float()], dim=-1)
            logits = self.router(router_in)

        if self.logits_dropout is not None:
            logits = self.logits_dropout(logits)

        probs = F.softmax(logits, dim=-1)

        active = None
        if token_mask is not None:
            active = token_mask.reshape(-1)
            if active.dtype is not torch.bool:
                active = active > 0
            active_f = active.to(dtype=probs.dtype)
            probs = probs * active_f[:, None]

        k = max(1, int(top_k))
        topk_probs, topk_idx = torch.topk(probs, k=k, dim=-1)
        denom = topk_probs.sum(dim=-1, keepdim=True)
        topk_weight = torch.where(denom > 0, topk_probs / denom, torch.zeros_like(topk_probs))

        if active is not None:
            topk_weight = topk_weight * active.to(dtype=topk_weight.dtype)[:, None]

        return topk_idx, topk_weight, probs, logits, active


class MoEFeedForward(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        activation: t.Union[str, t.Callable[[Tensor], Tensor]],
        dropout: float,
        layer_norm_eps: float,
        num_experts: int,
        top_k: int,
        context_dim: int = 0,
        router_type: str = "linear",
        router_hidden_dim: int = 256,
        router_dropout: float = 0.0,
        use_z_loss: bool = True,
        return_routing: bool = True,
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.use_z_loss = bool(use_z_loss)
        self.return_routing = bool(return_routing)

        self.router = TopKRouter(
            hidden_size,
            self.num_experts,
            context_dim=context_dim,
            router_type=router_type,
            router_hidden_dim=router_hidden_dim,
            router_dropout=router_dropout,
        )
        self.experts = nn.ModuleList(
            [
                ExpertFFN(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    activation=activation,
                )
                for _ in range(self.num_experts)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self,
        x: Tensor,
        *,
        context: Tensor | None = None,
        token_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor | t.Sequence[Tensor] | None]]:
        bsz, seq_len, hidden = x.shape

        topk_idx, topk_weight, probs, logits, active = self.router(
            x,
            context=context,
            token_mask=token_mask,
            top_k=self.top_k,
        )

        flat_x = x.reshape(-1, hidden)
        output = torch.zeros_like(flat_x)

        # Dispatch tokens to experts.
        for expert_idx, expert in enumerate(self.experts):
            mask = topk_idx == expert_idx
            if not mask.any():
                continue
            token_mask_any = mask.any(dim=-1)
            token_indices = token_mask_any.nonzero(as_tuple=True)[0]
            if token_indices.numel() == 0:
                continue
            expert_in = flat_x.index_select(0, token_indices)
            expert_out = expert(expert_in)

            weight = (mask.float() * topk_weight).sum(dim=-1)
            weight = weight.index_select(0, token_indices).unsqueeze(-1)
            output.index_add_(0, token_indices, expert_out * weight.to(dtype=expert_out.dtype))

        output = output.view(bsz, seq_len, hidden)
        output = self.dropout(output)
        output = self.layer_norm(output + x)

        stats: dict[str, Tensor | t.Sequence[Tensor] | None] = {
            "aux_loss": None,
            "z_loss": None,
            "route_mean": None,
        }

        active_count = None
        if active is not None:
            active_count = int(active.sum().item())

        if active is None or active_count > 0:
            probs_used = probs
            if active is not None:
                probs_used = probs * active.to(dtype=probs.dtype)[:, None]

            importance = probs_used.sum(dim=0)
            load = torch.zeros(self.num_experts, device=probs.device, dtype=probs.dtype)

            top1 = topk_idx[:, 0]
            if active is not None:
                top1 = top1[active]
            if top1.numel() > 0:
                load = load.scatter_add(0, top1, torch.ones_like(top1, dtype=probs.dtype))

            denom = float(active_count if active_count is not None else probs.shape[0])
            if denom > 0:
                importance = importance / denom
                load = load / denom
                stats["aux_loss"] = self.num_experts * (importance * load).sum()

            if self.use_z_loss:
                z = torch.logsumexp(logits, dim=-1)
                if active is not None:
                    z = z[active]
                if z.numel() > 0:
                    stats["z_loss"] = (z**2).mean()

        if self.return_routing:
            probs_reshaped = probs.view(bsz, seq_len, self.num_experts)
            if token_mask is not None:
                mask = token_mask.to(dtype=probs_reshaped.dtype).unsqueeze(-1)
                denom = mask.sum(dim=1).clamp_min(1.0)
                stats["route_mean"] = (probs_reshaped * mask).sum(dim=1) / denom
            else:
                stats["route_mean"] = probs_reshaped.mean(dim=1)

        return output, stats


__all__ = ["ExpertFFN", "TopKRouter", "MoEFeedForward"]
