from __future__ import annotations

"""Output containers for standalone RoFormer encoder."""

from dataclasses import dataclass
import typing as t

import torch


@dataclass
class RoFormerModelOutput:
    """Container for encoder outputs."""

    last_hidden_state: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None
    moe_loss: torch.Tensor | None = None
    moe_metrics: dict[str, t.Any] | None = None
