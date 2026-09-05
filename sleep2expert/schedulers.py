from __future__ import annotations

import math

import torch


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int | None,
    decay_floor: float = 0.1,
    decay_shape: str = "cosine",
    decay_ratio: float | None = None,
) -> torch.optim.lr_scheduler.LambdaLR:
    floor = float(decay_floor)
    if not 0.0 <= floor <= 1.0:
        raise ValueError("decay_floor must be in [0, 1].")
    if decay_shape not in {"cosine", "linear"}:
        raise ValueError("decay_shape must be 'cosine' or 'linear'.")

    if warmup_steps is None:
        warmup = int(0.03 * total_steps)
    else:
        warmup = int(warmup_steps)
    warmup = max(0, min(warmup, total_steps))

    decay_start = warmup
    if decay_ratio is not None:
        if not 0.0 < decay_ratio <= 1.0:
            raise ValueError("decay_ratio must be in (0, 1].")
        decay_steps = int(total_steps * decay_ratio)
        if decay_steps < 1 or warmup + decay_steps > total_steps:
            raise ValueError("WSD requires at least one decay step and warmup + decay steps <= total_steps.")
        decay_start = total_steps - decay_steps

    def lr_lambda(step):
        if step < warmup:
            return float(step) / float(max(1, warmup))
        if decay_ratio is not None:
            progress = min(1.0, max(0.0, (step - decay_start) / float(total_steps - decay_start)))
        else:
            progress = (step - warmup) / float(max(1, total_steps - warmup))
        if decay_shape == "linear":
            decay = max(0.0, 1.0 - progress)
        else:
            decay = 0.5 * (1 + math.cos(math.pi * progress))
        return floor + (1.0 - floor) * decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
