import typing as t

import torch
from transformers.models.switch_transformers.modeling_switch_transformers import (
    load_balancing_loss_func,
)

from .base import AuxiliaryLoss, LossOutput, register_aux_loss


@register_aux_loss("router_load_balancing")
class RouterLoadBalancingLoss(AuxiliaryLoss):
    """Auxiliary loss wrapper around SwitchTransformers' router load balancing."""

    requires_router_outputs = True

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        router_outputs: t.Any,
        batch: t.Mapping[str, torch.Tensor] | None = None,
    ) -> t.Optional[LossOutput]:
        if router_outputs is None:
            return None

        per_view_losses: list[torch.Tensor] = []

        for single_router in router_outputs:
            if single_router is None:
                continue

            layers = list(single_router) if isinstance(single_router, (list, tuple)) else [single_router]
            for layer_out in layers:
                if isinstance(layer_out, (list, tuple)) and layer_out and isinstance(layer_out[0], torch.Tensor):
                    router_logits = layer_out[0]
                elif isinstance(layer_out, torch.Tensor):
                    router_logits = layer_out
                else:
                    continue

                if router_logits.dim() < 2:
                    continue

                router_probs = torch.softmax(router_logits, dim=-1)
                expert_indices = torch.argmax(router_logits, dim=-1)
                per_view_losses.append(load_balancing_loss_func(router_probs, expert_indices))

        if not per_view_losses:
            return None

        stacked = torch.stack(per_view_losses)
        loss = stacked.sum() if self.reduction == "sum" else stacked.mean()

        metrics = {"router_lb_loss": loss.detach()}
        return LossOutput(loss=loss, metrics=metrics)
