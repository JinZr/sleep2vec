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

        def _iter_layers(single_router):
            """Yield (logits, probs) pairs for each layer across all router output formats."""
            if single_router is None:
                return

            if isinstance(single_router, (list, tuple)):
                for element in single_router:
                    # Common HF pattern: a tuple of (logits, probs) tensors per layer.
                    if (
                        isinstance(element, (list, tuple))
                        and len(element) == 2
                        and all(torch.is_tensor(e) for e in element)
                    ):
                        yield element[0], element[1]
                    else:
                        yield from _iter_layers(element)
                return

            if isinstance(single_router, dict) or hasattr(single_router, "router_logits") or hasattr(
                single_router, "router_probs"
            ):
                logits, probs = _extract_logits_and_probs(single_router)
                logits_seq = list(logits) if isinstance(logits, (list, tuple)) else ([logits] if logits is not None else [])
                probs_seq = list(probs) if isinstance(probs, (list, tuple)) else ([probs] if probs is not None else [])
                max_len = max(len(logits_seq), len(probs_seq))
                if max_len == 0:
                    return
                logits_seq += [None] * (max_len - len(logits_seq))
                probs_seq += [None] * (max_len - len(probs_seq))
                for lg, pb in zip(logits_seq, probs_seq):
                    yield lg, pb
                return

            # Fallback: treat the object itself as logits.
            yield single_router, None

        def _scatter_selected_probs(router_logits: torch.Tensor, router_probs: torch.Tensor | None):
            """
            Handle dispatch-style outputs where router_logits carries a one-hot mask and
            router_probs only contains the selected expert weight(s).
            """
            if router_probs is None or router_logits is None or router_logits.dim() < 2:
                return router_probs

            # Case 1: probs missing the expert dimension entirely, e.g. [B, top_k] vs [B, top_k, num_experts].
            if router_logits.dim() == router_probs.dim() + 1 and router_logits.shape[:-1] == router_probs.shape:
                expert_indices = torch.argmax(router_logits, dim=-1)
                dense = torch.zeros(router_logits.shape, device=router_probs.device, dtype=router_probs.dtype)
                dense.scatter_(dense.dim() - 1, expert_indices.unsqueeze(-1), router_probs.unsqueeze(-1))
                return dense

            # Case 2: probs has a trailing singleton expert dim, e.g. [B, top_k, 1] vs [B, top_k, num_experts].
            if router_logits.shape[:-1] == router_probs.shape[:-1] and router_probs.shape[-1] == 1:
                expert_indices = torch.argmax(router_logits, dim=-1)
                squeezed = router_probs.squeeze(-1)
                dense = torch.zeros(router_logits.shape, device=router_probs.device, dtype=router_probs.dtype)
                dense.scatter_(dense.dim() - 1, expert_indices.unsqueeze(-1), squeezed.unsqueeze(-1))
                return dense

            return router_probs

        for single_router in router_views:
            for router_logits, router_probs in _iter_layers(single_router):
                if router_logits is not None and not torch.is_tensor(router_logits):
                    continue
                if router_probs is not None and not torch.is_tensor(router_probs):
                    continue

                logits_were_int = False
                if router_logits is not None and not torch.is_floating_point(router_logits):
                    logits_were_int = True
                    router_logits = router_logits.float()

                router_probs = _scatter_selected_probs(router_logits, router_probs)

                if router_probs is None:
                    if router_logits is None or router_logits.dim() < 2:
                        continue
                    router_probs = router_logits if logits_were_int else torch.softmax(router_logits, dim=-1)

                if router_probs is None or router_probs.dim() < 2:
                    continue

                expert_indices = torch.argmax(router_probs, dim=-1)

                # Flatten across any token/top-k dims so the HF helper sees [N, num_experts].
                router_probs_flat = router_probs.view(-1, router_probs.shape[-1])
                expert_indices_flat = (
                    expert_indices.reshape(-1, expert_indices.shape[-1])
                    if expert_indices.dim() > 1
                    else expert_indices.view(-1)
                )

                per_view_losses.append(load_balancing_loss_func(router_probs_flat, expert_indices_flat))

        if not per_view_losses:
            return None

        stacked = torch.stack(per_view_losses)
        loss = stacked.sum() if self.reduction == "sum" else stacked.mean()

        metrics = {"router_lb_loss": loss.detach()}
        return LossOutput(loss=loss, metrics=metrics)
