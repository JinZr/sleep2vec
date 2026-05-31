import typing as t

import torch
import torch.nn as nn


class ChannelAggregator(nn.Module):
    """Base class for fusing modality features."""

    name: str = "base"

    def __init__(self, feature_dim: int, n_mods: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_mods = n_mods

    def forward(
        self,
        feats: t.List[torch.Tensor],
        *,
        channel_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, bool]:
        """
        Args:
            feats: list of modality tensors, each [B, D] or [B, L, D]
            channel_mask: optional bool tensor [B, n_mods] marking available modalities
        Returns:
            fused: [B, D] or [B, L, D]
            has_L: bool, whether sequence dimension is present
        """
        raise NotImplementedError

    def _validate_shapes(self, feats: t.List[torch.Tensor]) -> tuple[bool, t.List[torch.Tensor]]:
        if len(feats) != self.n_mods:
            raise ValueError(f"Expect {self.n_mods} modality features, got {len(feats)}")
        x0 = feats[0]
        has_L = x0.dim() == 3
        processed = []
        for f in feats:
            if f.dim() == 2:
                f = f.unsqueeze(1)
                f_has_L = False
            elif f.dim() == 3:
                f_has_L = True
            else:
                raise ValueError(f"Each modality feature must be rank-2 or rank-3, got {f.shape}.")
            if f_has_L != has_L:
                raise ValueError("Mixing sequential and non-sequential features is not supported.")
            if has_L and f.size(1) != x0.size(1):
                raise ValueError("All modalities must have matching sequence length.")
            if f.shape[-1] != self.feature_dim:
                raise ValueError(f"feature_dim mismatch: expect {self.feature_dim}, got {f.shape[-1]}")
            processed.append(f)
        return has_L, processed

    def _normalize_channel_mask(
        self,
        channel_mask: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if channel_mask is None:
            return torch.ones(batch_size, self.n_mods, dtype=torch.bool, device=device)

        mask = channel_mask.to(device=device, dtype=torch.bool)
        if mask.dim() != 2 or mask.shape != (batch_size, self.n_mods):
            raise ValueError(
                f"channel_mask must have shape [B, {self.n_mods}], got {tuple(mask.shape)} " f"for B={batch_size}."
            )
        if not mask.any(dim=1).all():
            raise ValueError("channel_mask has no available channel for at least one sample.")
        return mask

    def _prepare_inputs(
        self,
        feats: t.List[torch.Tensor],
        channel_mask: torch.Tensor | None,
    ) -> tuple[bool, torch.Tensor, torch.Tensor]:
        has_L, processed = self._validate_shapes(feats)
        stack = torch.stack(processed, dim=1)  # [B, M, L, D]
        mask = self._normalize_channel_mask(
            channel_mask,
            batch_size=stack.size(0),
            device=stack.device,
        )
        return has_L, stack, mask

    @staticmethod
    def _restore_rank(fused: torch.Tensor, has_L: bool) -> torch.Tensor:
        if not has_L:
            fused = fused.squeeze(1)
        return fused


__all__ = ["ChannelAggregator"]
