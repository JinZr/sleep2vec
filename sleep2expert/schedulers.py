from __future__ import annotations

from fractions import Fraction
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
        decay_steps = int(total_steps * Fraction(str(decay_ratio)))
        if decay_steps < 1 or warmup + decay_steps > total_steps:
            raise ValueError("WSD requires at least one decay step and warmup + decay steps <= total_steps.")
        decay_start = total_steps - decay_steps

    def lr_lambda(step):
        if step < warmup:
            return float(step) / float(max(1, warmup))
        if decay_ratio is not None:
            # LambdaLR index zero is used by the first update; total_steps is installed only afterward.
            progress = min(1.0, max(0.0, (step - decay_start + 1) / float(total_steps - decay_start)))
        else:
            progress = (step - warmup) / float(max(1, total_steps - warmup))
        if decay_shape == "linear":
            decay = max(0.0, 1.0 - progress)
        else:
            decay = 0.5 * (1 + math.cos(math.pi * progress))
        return floor + (1.0 - floor) * decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def validate_finetune_scheduler_args(args) -> None:
    scheduler_name = getattr(args, "lr_scheduler", "decay")
    floor = float(getattr(args, "lr_decay_floor", 0.1))
    if not 0.0 <= floor <= 1.0:
        raise ValueError("lr_decay_floor must be in [0, 1].")
    decay_ratio = getattr(args, "lr_decay_ratio", None)
    plateau_factor = getattr(args, "lr_plateau_factor", None)
    plateau_patience = getattr(args, "lr_plateau_patience", None)
    if scheduler_name not in {"decay", "wsd", "plateau"}:
        raise ValueError("lr_scheduler must be 'decay', 'wsd', or 'plateau'.")
    if scheduler_name == "wsd" and decay_ratio is None:
        raise ValueError("WSD requires lr_decay_ratio.")
    if decay_ratio is not None and not 0.0 < decay_ratio <= 1.0:
        raise ValueError("lr_decay_ratio must be in (0, 1].")
    if scheduler_name != "wsd" and decay_ratio is not None:
        raise ValueError("lr_decay_ratio is only supported by WSD.")
    if scheduler_name != "plateau" and (plateau_factor is not None or plateau_patience is not None):
        raise ValueError("lr_plateau_factor and lr_plateau_patience are only supported by plateau.")
    if scheduler_name == "plateau":
        if getattr(args, "print_diagnostics", False):
            raise ValueError("Plateau does not support diagnostics without validation.")
        if getattr(args, "warmup_steps", None) is not None:
            raise ValueError("Plateau does not support warmup_steps.")
        if getattr(args, "lr_decay_shape", "cosine") != "cosine":
            raise ValueError("Plateau does not support lr_decay_shape.")
        if not args.monitor.startswith("val_"):
            raise ValueError("Plateau requires a validation monitor starting with 'val_'.")
        factor = 0.1 if plateau_factor is None else plateau_factor
        patience = 10 if plateau_patience is None else plateau_patience
        if not 0.0 < factor < 1.0:
            raise ValueError("lr_plateau_factor must be in (0, 1).")
        if patience < 0:
            raise ValueError("lr_plateau_patience must be nonnegative.")
