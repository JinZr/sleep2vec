from __future__ import annotations

import math

import torch


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int | None,
) -> torch.optim.lr_scheduler.LambdaLR:
    if warmup_steps is None:
        warmup = int(0.03 * total_steps)
    else:
        warmup = int(warmup_steps)
    warmup = max(0, min(warmup, total_steps))

    def lr_lambda(step):
        if step < warmup:
            return float(step) / float(max(1, warmup))
        progress = (step - warmup) / float(max(1, total_steps - warmup))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
