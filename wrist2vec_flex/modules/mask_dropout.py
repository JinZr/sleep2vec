from __future__ import annotations

import torch


def apply_mask_dropout(
    mask: torch.Tensor,
    dropout_rate: float,
    min_keep: int,
    *,
    protected_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if dropout_rate <= 0.0:
        return mask
    if min_keep <= 0:
        raise ValueError("min_keep must be a positive integer.")

    mask = mask.to(dtype=torch.bool)
    if protected_mask is None:
        protected = torch.zeros_like(mask)
    else:
        protected = protected_mask.to(device=mask.device, dtype=torch.bool)
        if protected.shape != mask.shape:
            raise ValueError(
                f"protected_mask shape must match mask, got {tuple(protected.shape)} vs {tuple(mask.shape)}."
            )
        protected = protected & mask

    keep = torch.rand(mask.shape, device=mask.device) >= float(dropout_rate)
    dropped = mask & (keep | protected)

    available_counts = mask.sum(dim=-1)
    target_counts = torch.minimum(available_counts, torch.full_like(available_counts, int(min_keep)))
    need = target_counts - dropped.sum(dim=-1)
    if not (need > 0).any():
        return dropped

    flat_mask = mask.reshape(-1, mask.shape[-1])
    flat_dropped = dropped.reshape(-1, mask.shape[-1])
    flat_need = need.reshape(-1)
    for row_idx in torch.nonzero(flat_need > 0, as_tuple=False).flatten():
        candidates = torch.nonzero(flat_mask[row_idx] & ~flat_dropped[row_idx], as_tuple=False).flatten()
        if candidates.numel() == 0:
            continue
        chosen = candidates[torch.randperm(candidates.numel(), device=mask.device)[: int(flat_need[row_idx].item())]]
        flat_dropped[row_idx, chosen] = True

    return flat_dropped.reshape_as(mask)


__all__ = ["apply_mask_dropout"]
