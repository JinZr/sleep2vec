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


def _coalesce_route_mean(value: t.Any) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return torch.stack(list(value), dim=0).mean(dim=0)
    return value


def _sum_optional(values: t.Sequence[torch.Tensor | None]) -> torch.Tensor | None:
    total = None
    for v in values:
        if v is None:
            continue
        total = v if total is None else total + v
    return total


@register_loss("dash_moe_info_nce")
class DashMoeInfoNCELoss(ContrastiveLoss):
    """Weighted InfoNCE with MoE routing alignment + auxiliary losses."""

    def __init__(
        self,
        temperature: float,
        hard_scale: float = 0.10,
        pos_margin: float = 0.0,
        lambda_align: float = 0.05,
        lambda_aux: float = 0.01,
        lambda_z: float = 1e-3,
        align_type: str = "sym_kl",
        eps: float = 1e-6,
    ):
        super().__init__(temperature)
        self.hard_scale = hard_scale
        self.pos_margin = pos_margin
        self.lambda_align = lambda_align
        self.lambda_aux = lambda_aux
        self.lambda_z = lambda_z
        self.align_type = align_type
        self.eps = eps

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
        log_w = torch.log(w.clamp_min(self.eps)).clone()
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
        contrastive_loss = 0.5 * (loss_12 + loss_21)

        acc = _contrastive_accuracy(base_12.reshape(L * B, B), base_21.reshape(L * B, B), labels_flat)

        route_mean_first = _coalesce_route_mean(batch.get("moe_route_mean_first"))
        route_mean_second = _coalesce_route_mean(batch.get("moe_route_mean_second"))
        route_align = None
        if route_mean_first is not None and route_mean_second is not None:
            if self.align_type == "mse":
                route_align = ((route_mean_first - route_mean_second) ** 2).mean()
            else:
                p = route_mean_first.clamp_min(self.eps)
                q = route_mean_second.clamp_min(self.eps)
                p = p / p.sum(dim=-1, keepdim=True).clamp_min(self.eps)
                q = q / q.sum(dim=-1, keepdim=True).clamp_min(self.eps)
                kl_pq = (p * (p.log() - q.log())).sum(dim=-1)
                kl_qp = (q * (q.log() - p.log())).sum(dim=-1)
                route_align = (kl_pq + kl_qp).mean()

        aux_loss = batch.get("moe_aux_loss")
        if aux_loss is None:
            aux_loss = _sum_optional(
                [batch.get("moe_aux_loss_first"), batch.get("moe_aux_loss_second")]
            )
        z_loss = batch.get("moe_z_loss")
        if z_loss is None:
            z_loss = _sum_optional([batch.get("moe_z_loss_first"), batch.get("moe_z_loss_second")])

        total = contrastive_loss
        if route_align is not None:
            total = total + self.lambda_align * route_align
        if aux_loss is not None:
            total = total + self.lambda_aux * aux_loss
        if z_loss is not None:
            total = total + self.lambda_z * z_loss

        metrics = {
            "contrastive_loss": contrastive_loss.detach(),
            "contrastive_acc": acc,
        }
        if route_align is not None:
            metrics["route_align"] = route_align.detach()
        if aux_loss is not None:
            metrics["moe_aux"] = aux_loss.detach()
        if z_loss is not None:
            metrics["moe_z"] = z_loss.detach()

        return LossOutput(loss=total, metrics=metrics)
