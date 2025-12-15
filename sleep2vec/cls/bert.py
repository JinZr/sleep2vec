import torch

from .base import ClsEmbedding


class BertClsEmbedding(ClsEmbedding):
    """
    BIOT / BERT-style: prepend a learnable CLS token, mark it valid in the attention mask,
    and expose both CLS and token-level representations.
    """

    name = "bert"

    def __init__(self, hidden_size: int):
        super().__init__()
        self.cls_token = torch.nn.Parameter(torch.randn(hidden_size))

    @property
    def has_cls(self) -> bool:
        return True

    def add_cls_and_mask(self, tokens: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, D = tokens.shape
        device = tokens.device

        cls_token_expanded = self.cls_token.view(1, 1, D).expand(B, 1, D)
        tokens_with_cls = torch.cat([cls_token_expanded, tokens], dim=1)  # [B, L+1, D]

        padding_mask = torch.zeros(B, L + 1, dtype=torch.bool, device=device)
        for i in range(B):
            valid_len = int(lengths[i].item())
            padding_mask[i, : valid_len + 1] = True  # +1 for CLS

        return tokens_with_cls.float(), padding_mask

    def split_hidden(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        cls_hidden = hidden[:, 0]
        token_hidden = hidden[:, 1:]
        token_mask = None
        if attention_mask is not None:
            token_mask = attention_mask[:, 1:]
        return token_hidden, cls_hidden, token_mask


__all__ = ["BertClsEmbedding"]
