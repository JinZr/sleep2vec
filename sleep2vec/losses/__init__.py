# Import modules that register implementations.
from . import info_nce  # noqa: F401
from . import weighted_info_nce  # noqa: F401
from .base import (
    LOSS_REGISTRY,
    ContrastiveLoss,
    LossOutput,
    available_losses,
    create_loss,
    register_loss,
)

__all__ = [
    "ContrastiveLoss",
    "LossOutput",
    "available_losses",
    "create_loss",
    "register_loss",
    "LOSS_REGISTRY",
]
