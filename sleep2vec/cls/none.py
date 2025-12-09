import torch

from .base import ClsEmbedding


class NoClsEmbedding(ClsEmbedding):
    """Legacy behavior: no CLS token is added."""

    name = "none"

    def add_cls_and_mask(self, tokens: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = tokens.shape
        device = tokens.device
        padding_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        for i in range(B):
            valid_len = int(lengths[i].item())
            padding_mask[i, :valid_len] = True
        return tokens.float(), padding_mask

    def split_hidden(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        return hidden, None, attention_mask


__all__ = ["NoClsEmbedding"]
