from __future__ import annotations

import copy
import math
import typing as t

import torch
import torch.nn as nn

from wrist2vec.config import ModelAveragingConfig
from wrist2vec.registry import register_model_averager

from .base import BaseModelAverager


@register_model_averager("ema")
class EmaModelAverager(BaseModelAverager):
    """Exponential moving average with cosine momentum schedule."""

    def __init__(self, cfg: ModelAveragingConfig, student: nn.Module):
        params = dict(cfg.params or {})
        enabled = bool(params.get("enabled", False))
        use_for_eval = bool(params.get("use_for_eval", True))
        state_prefix = params.get("state_prefix") or "ema_model"
        super().__init__(name="ema", student=student, use_for_eval=use_for_eval, state_prefix=state_prefix)
        self.base_momentum = float(params.get("base_momentum", 0.996))
        self.final_momentum = float(params.get("final_momentum", 1.0))
        self._total_steps: int | None = None
        self._enabled_flag = enabled

        if enabled:
            self.averaged_model = clone_ema_model(student)

    @property
    def enabled(self) -> bool:
        return self._enabled_flag and self.averaged_model is not None

    def on_fit_start(self, trainer) -> None:
        if not self.enabled:
            return
        if hasattr(trainer, "estimated_stepping_batches"):
            self._total_steps = int(trainer.estimated_stepping_batches)
        else:
            self._total_steps = None

    def on_load_checkpoint(self, checkpoint: t.Dict[str, t.Any]) -> None:
        if not self.enabled:
            return

        state_dict = checkpoint.get("state_dict", {})
        prefix = f"{self.state_prefix}."
        has_ema = any(k.startswith(prefix) for k in state_dict)
        if has_ema:
            return

        if self.averaged_model is None:
            self.averaged_model = clone_ema_model(self.student)

        student_prefix = "model."
        ema_state = {
            f"{prefix}{k[len(student_prefix):]}": v for k, v in state_dict.items() if k.startswith(student_prefix)
        }
        if ema_state:
            state_dict.update(ema_state)
            checkpoint["state_dict"] = state_dict

    def on_train_batch_end(self, *, trainer, global_step: int) -> None:
        if not self.enabled:
            return
        momentum = self._momentum_for_step(global_step, trainer)
        ema_update(self.student, self.averaged_model, momentum=momentum)

    def _momentum_for_step(self, global_step: int, trainer) -> float:
        total_steps = self._total_steps
        if total_steps is None:
            total_steps = int(getattr(trainer, "estimated_stepping_batches", 0))
        return cosine_ema_momentum(
            step=global_step,
            total_steps=total_steps,
            base_momentum=self.base_momentum,
            final_momentum=self.final_momentum,
        )


def clone_ema_model(student: nn.Module) -> nn.Module:
    """Return a frozen copy of the student model for EMA tracking."""
    ema = copy.deepcopy(student)
    for p in ema.parameters():
        p.requires_grad = False
    return ema


def ema_update(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    """In-place EMA update: teacher = m * teacher + (1 - m) * student."""
    if teacher is None or student is None:
        return
    with torch.no_grad():
        student_params = dict(student.named_parameters())
        for name, p_teacher in teacher.named_parameters():
            p_student = student_params.get(name)
            if p_student is None:
                continue
            p_teacher.data.mul_(momentum).add_(p_student.data, alpha=1.0 - momentum)


def cosine_ema_momentum(step: int, total_steps: int, base_momentum: float, final_momentum: float = 1.0) -> float:
    """
    Cosine schedule for EMA momentum (DINOv2 style).
    When step=0 -> base_momentum; when step>=total_steps -> final_momentum.
    """
    if total_steps <= 0 or step >= total_steps:
        return final_momentum
    t = step / float(total_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * t))
    return final_momentum - (final_momentum - base_momentum) * cosine
