import typing as t

import torch
import torch.nn.functional as F

from .base import ContrastiveLoss, LossOutput, register_loss


def _contrastive_accuracy(
    logits_12: torch.Tensor, logits_21: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad():
        pred12 = logits_12.argmax(dim=-1)
        pred21 = logits_21.argmax(dim=-1)
        acc = 0.5 * (
            (pred12 == labels).float().mean() + (pred21 == labels).float().mean()
        )
    return acc


@register_loss("info_nce")
class InfoNCELoss(ContrastiveLoss):
    """Vanilla symmetric InfoNCE objective."""

    def forward(
        self,
        first_hidden: torch.Tensor,
        second_hidden: torch.Tensor,
        batch: t.Mapping[str, torch.Tensor],
    ) -> LossOutput:
        T = self.temperature
        B, L, _ = first_hidden.shape

        first_norm = F.normalize(first_hidden, dim=-1).transpose(0, 1)  # [L,B,D]
        second_norm = F.normalize(second_hidden, dim=-1).transpose(0, 1)

        labels = torch.arange(B, device=first_hidden.device).expand(L, B)
        labels_flat = labels.reshape(L * B)

        logits_12 = torch.einsum("lbd,lmd->lbm", first_norm, second_norm) / T
        logits_21 = torch.einsum("lbd,lmd->lbm", second_norm, first_norm) / T

        logits_12_flat = logits_12.reshape(L * B, B)
        logits_21_flat = logits_21.reshape(L * B, B)

        loss_12 = F.cross_entropy(logits_12_flat, labels_flat)
        loss_21 = F.cross_entropy(logits_21_flat, labels_flat)
        loss = 0.5 * (loss_12 + loss_21)

        acc = _contrastive_accuracy(logits_12_flat, logits_21_flat, labels_flat)
        metrics = {
            "contrastive_loss": loss.detach(),
            "contrastive_acc": acc,
        }
        return LossOutput(loss=loss, metrics=metrics)
