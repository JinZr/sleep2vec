import typing as t

import torch
import torch.nn as nn

from sleep2wave.downstreams.head_registry import register_head

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
    ):
        super().__init__()
        self.target = target
        self.fusion = FeatureFusion(feature_dim, n_mods, agg)
        self.out_dim = out_dim
        in_dim = self.fusion.output_dim
        hidden_dim = hidden_dim or in_dim

        layers: t.List[nn.Module] = [nn.Linear(in_dim, hidden_dim), act()]
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.regressor = nn.Sequential(*layers)

    def forward(self, feature_of_different_mods: t.List[torch.Tensor]):
        fused, has_L = self.fusion.aggregator(feature_of_different_mods)
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
    )


__all__ = ["RegressionHead", "build_regression_head"]
