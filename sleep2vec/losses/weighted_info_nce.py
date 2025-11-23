import typing as t

import torch
import torch.nn.functional as F

from .base import ContrastiveLoss, LossOutput, register_loss


def _contrastive_accuracy(logits_12: torch.Tensor, logits_21: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        pred12 = logits_12.argmax(dim=-1)
        pred21 = logits_21.argmax(dim=-1)
        acc = 0.5 * ((pred12 == labels).float().mean() + (pred21 == labels).float().mean())
    return acc


@register_loss("weighted_info_nce")
class WeightedInfoNCELoss(ContrastiveLoss):
    """InfoNCE variant that re-weights negatives + applies hardness margins."""

    def __init__(
        self,
        temperature: float,
        hard_scale: float = 0.10,
        pos_margin: float = 0.0,
    ):
        super().__init__(temperature)
        self.hard_scale = hard_scale
        self.pos_margin = pos_margin

    def forward(
        self,
        first_hidden: torch.Tensor,
        second_hidden: torch.Tensor,
        batch: t.Mapping[str, torch.Tensor],
    ) -> LossOutput:
        if "w" not in batch or "h" not in batch:
            raise KeyError("Batch missing 'w' or 'h' tensors required for weighted InfoNCE.")
        T = self.temperature
        B, L, _ = first_hidden.shape

        first_norm = F.normalize(first_hidden, dim=-1).transpose(0, 1)
        second_norm = F.normalize(second_hidden, dim=-1).transpose(0, 1)

        base_12 = torch.einsum("lbd,lmd->lbm", first_norm, second_norm) / T
        base_21 = torch.einsum("lbd,lmd->lbm", second_norm, first_norm) / T

        w = batch["w"].to(first_hidden.device)
        h = batch["h"].to(first_hidden.device)
        log_w = torch.log(w.clamp_min(1e-6)).clone()
        log_w.fill_diagonal_(0.0)
        log_w = log_w.unsqueeze(0)

        neg_margin = (self.hard_scale * h / T).clone()
        neg_margin.fill_diagonal_(0.0)
        neg_margin = neg_margin.unsqueeze(0)

        logits_12 = base_12 + log_w - neg_margin
        logits_21 = base_21 + log_w - neg_margin

        labels = torch.arange(B, device=first_hidden.device).expand(L, B)
        labels_flat = labels.reshape(L * B)

        if self.pos_margin != 0.0:
            diag = torch.arange(B, device=first_hidden.device)
            logits_12[:, diag, diag] += self.pos_margin / T
            logits_21[:, diag, diag] += self.pos_margin / T

        logits_12_flat = logits_12.reshape(L * B, B)
        logits_21_flat = logits_21.reshape(L * B, B)

        loss_12 = F.cross_entropy(logits_12_flat, labels_flat)
        loss_21 = F.cross_entropy(logits_21_flat, labels_flat)
        loss = 0.5 * (loss_12 + loss_21)

        acc = _contrastive_accuracy(base_12.reshape(L * B, B), base_21.reshape(L * B, B), labels_flat)
        metrics = {
            "contrastive_loss": loss.detach(),
            "contrastive_acc": acc,
        }
        return LossOutput(loss=loss, metrics=metrics)
