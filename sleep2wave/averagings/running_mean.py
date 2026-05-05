from __future__ import annotations

import logging
import typing as t

import torch
import torch.nn as nn

from sleep2wave.config import ModelAveragingConfig
from sleep2wave.registry import register_model_averager

from .base import BaseModelAverager
from .ema import clone_ema_model


def _inplace_weighted_average(
    dst_model: nn.Module,
    src_model: nn.Module,
    *,
    weight_dst: float,
    weight_src: float,
) -> None:
    """In-place weighted average of two state dicts: dst = dst*weight_dst + src*weight_src."""
    with torch.no_grad():
        dst_state = dst_model.state_dict()
        src_state = src_model.state_dict()
        for name, dst_tensor in dst_state.items():
            src_tensor = src_state[name]
            if dst_tensor.is_floating_point():
                dst_tensor.mul_(weight_dst).add_(src_tensor, alpha=weight_src)
            else:
                dst_tensor.copy_(src_tensor)


@register_model_averager("running_mean")
class RunningAverageModelAverager(BaseModelAverager):
    """Arithmetic running average (Icefall-style) updated every `average_period` steps."""

    def __init__(self, cfg: ModelAveragingConfig, student: nn.Module):
        params = dict(cfg.params or {})
        enabled = bool(params.get("enabled", False))
        use_for_eval = bool(params.get("use_for_eval", True))
        state_prefix = params.get("state_prefix")
        super().__init__(name="running_mean", student=student, use_for_eval=use_for_eval, state_prefix=state_prefix)
        self.average_period = max(1, int(params.get("average_period", 200)))
        self.start_step = max(1, int(params.get("start_step", self.average_period)))
        self._enabled_flag = enabled
        self._avg_origin_step: int | None = 0
        self._missing_avg_on_resume = False

        if enabled:
            self.averaged_model = clone_ema_model(student)

    @property
    def enabled(self) -> bool:
        return self._enabled_flag and self.averaged_model is not None

    def on_load_checkpoint(self, checkpoint: t.Dict[str, t.Any]) -> None:
        if not self.enabled:
            return

        state_dict = checkpoint.get("state_dict", {})
        prefix = f"{self.state_prefix}."
        has_avg = any(k.startswith(prefix) for k in state_dict)
        if has_avg:
            return

        if self.averaged_model is None:
            self.averaged_model = clone_ema_model(self.student)

        student_prefix = "model."
        avg_state = {
            f"{prefix}{k[len(student_prefix):]}": v for k, v in state_dict.items() if k.startswith(student_prefix)
        }
        if avg_state:
            state_dict.update(avg_state)
            checkpoint["state_dict"] = state_dict
        else:
            # Missing averaged weights; we'll restart averaging from the resume step.
            self._avg_origin_step = None
            self._missing_avg_on_resume = True
            logging.warning(
                "Running-mean averager: averaged weights missing in checkpoint; restarting average from resume step."
            )

    def on_train_batch_end(self, *, trainer, global_step: int) -> None:
        if not self.enabled or self.averaged_model is None:
            return
        step = global_step + 1  # Lightning global_step is zero-based
        update_due = step >= self.start_step and (step - self.start_step) % self.average_period == 0
        if not update_due:
            return

        if self._avg_origin_step is None:
            # Restart averaging from current step when no historical average exists.
            self._avg_origin_step = step - self.average_period

        effective_step = max(self.average_period, step - self._avg_origin_step)
        weight_cur = self.average_period / float(effective_step)
        weight_avg = 1.0 - weight_cur
        _inplace_weighted_average(self.averaged_model, self.student, weight_dst=weight_avg, weight_src=weight_cur)
