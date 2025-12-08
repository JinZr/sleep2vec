from __future__ import annotations

import typing as t

import torch.nn as nn

from sleep2vec.config import ModelAveragingConfig
from sleep2vec.registry import get_model_averager_builder


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
        return None
        return None
        return None
