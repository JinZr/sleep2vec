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

    def forward(self, feats: t.List[torch.Tensor]) -> tuple[torch.Tensor, bool]:
        """
        Args:
            feats: list of modality tensors, each [B, D] or [B, L, D]
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


__all__ = ["ChannelAggregator"]
