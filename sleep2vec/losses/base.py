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


class AuxiliaryLoss(nn.Module):
    """Base class for auxiliary objectives (e.g., router load-balancing)."""

    requires_router_outputs: bool = False

    def forward(self, model_outputs: t.Any, batch: t.Mapping[str, torch.Tensor]) -> t.Optional[LossOutput]:
        raise NotImplementedError


LOSS_REGISTRY: t.Dict[str, t.Type[ContrastiveLoss]] = {}
AUX_LOSS_REGISTRY: t.Dict[str, t.Type[AuxiliaryLoss]] = {}


def register_loss(name: str):
    def decorator(cls: t.Type[ContrastiveLoss]):
        if name in LOSS_REGISTRY:
            raise ValueError(f"Loss '{name}' is already registered.")
        LOSS_REGISTRY[name] = cls
        cls.registry_name = name
        return cls

    return decorator


def register_aux_loss(name: str):
    def decorator(cls: t.Type[AuxiliaryLoss]):
        if name in AUX_LOSS_REGISTRY:
            raise ValueError(f"Aux loss '{name}' is already registered.")
        AUX_LOSS_REGISTRY[name] = cls
        cls.registry_name = name
        return cls

    return decorator


def create_loss(name: str, **kwargs) -> ContrastiveLoss:
    if name not in LOSS_REGISTRY:
        raise KeyError(f"Unknown loss '{name}'. Available losses: {sorted(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[name](**kwargs)


def create_aux_loss(name: str, **kwargs) -> AuxiliaryLoss:
    if name not in AUX_LOSS_REGISTRY:
        raise KeyError(
            f"Unknown aux loss '{name}'. Available aux losses: {sorted(AUX_LOSS_REGISTRY.keys())}"
        )
    return AUX_LOSS_REGISTRY[name](**kwargs)


def available_losses() -> t.List[str]:
    return sorted(LOSS_REGISTRY.keys())


def available_aux_losses() -> t.List[str]:
    return sorted(AUX_LOSS_REGISTRY.keys())
