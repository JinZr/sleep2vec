import typing as t

import torch
import torch.nn as nn

from sleep2expert.downstreams.head_registry import register_head

from .base import FeatureFusion


class RegressionHead(nn.Module):
    """Two-layer regression head that reuses the shared fusion logic."""

    def __init__(
        self,
        target: str,
        feature_dim: int,
        n_mods: int,
        out_dim: int = 1,
        *,
        agg: str = "gated_scalar",
        hidden_dim: t.Optional[int] = None,
        dropout: float = 0.1,
        act: t.Type[nn.Module] = nn.ELU,
        extra_feature_dim: int = 0,
    ):
        super().__init__()
        self.target = target
        self.fusion = FeatureFusion(feature_dim, n_mods, agg)
        self.out_dim = out_dim
        self.extra_feature_dim = extra_feature_dim
        in_dim = self.fusion.output_dim + extra_feature_dim
        hidden_dim = hidden_dim or in_dim

        layers: t.List[nn.Module] = [nn.Linear(in_dim, hidden_dim), act()]
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.regressor = nn.Sequential(*layers)

    def forward(self, feature_of_different_mods: t.List[torch.Tensor], extra_features: torch.Tensor | None = None):
        fused, has_L = self.fusion.aggregator(feature_of_different_mods)
        if extra_features is not None:
            if self.extra_feature_dim < 1:
                raise ValueError("extra_features were provided but the regression head has no extra_feature_dim.")
            if has_L:
                raise ValueError("extra_features are only supported for non-sequence regression heads.")
            if extra_features.dim() != 2:
                raise ValueError(f"extra_features must be rank-2, got {extra_features.shape}.")
            if extra_features.size(-1) != self.extra_feature_dim:
                raise ValueError(f"extra_features dim must be {self.extra_feature_dim}, got {extra_features.size(-1)}.")
            fused = torch.cat([fused, extra_features.to(device=fused.device, dtype=fused.dtype)], dim=-1)
        elif self.extra_feature_dim:
            raise ValueError("extra_features are required for this regression head.")
        if has_L:
            B, L, E = fused.shape
            fused = fused.view(B * L, E)
            out = self.regressor(fused).view(B, L, self.out_dim)
        else:
            out = self.regressor(fused)

        return out


@register_head("regression")
def build_regression_head(
    *,
    target,
    feature_dim,
    n_mods,
    output_dim,
    agg: str = "mean",
    hidden_dim: t.Optional[int] = None,
    dropout: float = 0.1,
    act: t.Type[nn.Module] = nn.ELU,
    extra_feature_dim: int = 0,
    **_,
) -> nn.Module:
    return RegressionHead(
        target,
        feature_dim,
        n_mods,
        out_dim=output_dim,
        agg=agg,
        hidden_dim=hidden_dim,
        dropout=dropout,
        act=act,
        extra_feature_dim=extra_feature_dim,
    )


__all__ = ["RegressionHead", "build_regression_head"]
