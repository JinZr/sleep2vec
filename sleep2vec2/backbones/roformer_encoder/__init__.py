"""Standalone RoFormer encoder (Icefall Zipformer-style layout).

Recipe-facing import surface:

    from roformer_encoder import RoFormerEncoderModel, RoFormerEncoderConfig

The actual architecture is defined in `roformer.py` (analogous to Icefall's
`zipformer.py`). Helper utilities live alongside this package.
"""

from .config import RoFormerEncoderConfig
from .roformer import (
    RoFormerEmbeddings,
    RoFormerEncoder,
    RoFormerEncoderModel,
    RoFormerEncoderOutput,
    RoFormerLayer,
    RoFormerModelOutput,
    RoFormerSelfAttention,
)

__all__ = [
    "RoFormerEncoderConfig",
    "RoFormerEmbeddings",
    "RoFormerSelfAttention",
    "RoFormerLayer",
    "RoFormerEncoder",
    "RoFormerEncoderModel",
    "RoFormerEncoderOutput",
    "RoFormerModelOutput",
]
