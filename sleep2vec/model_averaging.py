from __future__ import annotations

import logging
import typing as t

import torch
import torch.nn as nn

from sleep2vec.config import ModelAveragingConfig
from sleep2vec.pretrain.ema import clone_ema_model, cosine_ema_momentum, ema_update
from sleep2vec.registry import available_model_averagers, get_model_averager_builder, register_model_averager


class BaseModelAverager:
    """Hooks that manage a tracked copy of the student model."""

    def __init__(
        self,
        *,
        name: str,
        student: nn.Module,
        use_for_eval: bool = True,
        state_prefix: str | None = None,
    ):
        self.name = name
        self.student = student
        self.use_for_eval = use_for_eval
        self.state_prefix = state_prefix or f"{name}_model"
        self.averaged_model: nn.Module | None = None

    @property
    def enabled(self) -> bool:
        return self.averaged_model is not None

    def attach_to_module(self, lightning_module: nn.Module) -> None:
        """Registers the averaged model on the LightningModule for checkpointing."""
        if self.averaged_model is not None:
            setattr(lightning_module, self.state_prefix, self.averaged_model)

    def eval_model(self) -> nn.Module:
        if self.use_for_eval and self.averaged_model is not None:
            return self.averaged_model
        return self.student

    # Lifecycle hooks -----------------------------------------------------
    def on_fit_start(self, trainer) -> None:
        return None

    def on_load_checkpoint(self, checkpoint: t.Dict[str, t.Any]) -> None:
        return None

    def on_train_batch_end(self, *, trainer, global_step: int) -> None:
        return None


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


def build_model_averager(cfg: ModelAveragingConfig | None, student: nn.Module) -> BaseModelAverager | None:
    if cfg is None or not cfg.name:
        return None
    builder = get_model_averager_builder(cfg.name)
    averager = builder(cfg, student)
    if averager is None:
        return None
    if hasattr(averager, "enabled") and not averager.enabled:
        return averager
    return averager


__all__ = [
    "BaseModelAverager",
    "EmaModelAverager",
    "RunningAverageModelAverager",
    "available_model_averagers",
    "build_model_averager",
    "register_model_averager",
]
