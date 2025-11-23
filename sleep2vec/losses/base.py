import typing as t

import torch
import torch.nn as nn


class LossOutput(t.NamedTuple):
    loss: torch.Tensor
    metrics: t.Dict[str, torch.Tensor]
    extras: t.Optional[t.Dict[str, torch.Tensor]] = None


class ContrastiveLoss(nn.Module):
    """Base class for contrastive objectives."""

    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        first_hidden: torch.Tensor,
        second_hidden: torch.Tensor,
        batch: t.Mapping[str, torch.Tensor],
    ) -> LossOutput:
        raise NotImplementedError


LOSS_REGISTRY: t.Dict[str, t.Type[ContrastiveLoss]] = {}


def register_loss(name: str):
    def decorator(cls: t.Type[ContrastiveLoss]):
        if name in LOSS_REGISTRY:
            raise ValueError(f"Loss '{name}' is already registered.")
        LOSS_REGISTRY[name] = cls
        cls.registry_name = name
        return cls

    return decorator


def create_loss(name: str, **kwargs) -> ContrastiveLoss:
    if name not in LOSS_REGISTRY:
        raise KeyError(f"Unknown loss '{name}'. Available losses: {sorted(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[name](**kwargs)


def available_losses() -> t.List[str]:
    return sorted(LOSS_REGISTRY.keys())
