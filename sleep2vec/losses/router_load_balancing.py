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

        router_views = list(router_outputs) if isinstance(router_outputs, (list, tuple)) else [router_outputs]

        def _extract_logits_and_probs(candidate):
            logits = None
            probs = None

            if isinstance(candidate, dict):
                logits = candidate.get("router_logits")
                probs = candidate.get("router_probs")
            else:
                logits = getattr(candidate, "router_logits", None)
                probs = getattr(candidate, "router_probs", None)

            return logits, probs

        for single_router in router_views:
            if single_router is None:
                continue

            layers = list(single_router) if isinstance(single_router, (list, tuple)) else [single_router]
            for layer_out in layers:
                router_logits = None
                router_probs = None
                logits_were_int = False

                if isinstance(layer_out, (list, tuple)) and layer_out and isinstance(layer_out[0], torch.Tensor):
                    router_logits = layer_out[0]
                elif isinstance(layer_out, torch.Tensor):
                    router_logits = layer_out
                else:
                    router_logits, router_probs = _extract_logits_and_probs(layer_out)

                if router_logits is not None and not torch.is_floating_point(router_logits):
                    logits_were_int = True
                    router_logits = router_logits.float()

                if router_logits is not None and router_logits.dim() < 2:
                    continue
                if router_logits is None and router_probs is None:
                    continue

                if router_probs is None:
                    router_probs = router_logits if logits_were_int else torch.softmax(router_logits, dim=-1)
                if router_probs.dim() < 2:
                    continue

                expert_indices = torch.argmax(router_probs, dim=-1)
                per_view_losses.append(load_balancing_loss_func(router_probs, expert_indices))

        if not per_view_losses:
            return None

        stacked = torch.stack(per_view_losses)
        loss = stacked.sum() if self.reduction == "sum" else stacked.mean()

        metrics = {"router_lb_loss": loss.detach()}
        return LossOutput(loss=loss, metrics=metrics)
