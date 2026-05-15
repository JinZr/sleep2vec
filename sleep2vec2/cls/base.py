import torch
import torch.nn as nn


class ClsEmbedding(nn.Module):
    """Base interface for CLS embedding handling."""

    name: str = "none"

    def __init__(self):
        super().__init__()

    @property
    def has_cls(self) -> bool:
        return False

    def add_cls_and_mask(self, tokens: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def split_hidden(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Return (token_hidden_without_cls, cls_hidden or None, token_mask_without_cls or None)."""
        raise NotImplementedError


__all__ = ["ClsEmbedding"]
