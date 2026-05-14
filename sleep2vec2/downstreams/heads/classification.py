import typing as t

import torch
import torch.nn as nn

from sleep2vec2.downstreams.head_registry import register_head

from .base import FeatureFusion


class ClassificationHead(nn.Module):
    """
    Multi-modal classification head.
    Inputs: list of modality tensors, each [B, D] or [B, L, D]
    Outputs: logits [B, C] or [B, L, C]
    """

    def __init__(
        self,
        feature_dim: int,
        n_mods: int,
        n_classes: int,
        *,
        agg: str = "gated_scalar",
        hidden_dim: t.Optional[int] = None,
        dropout: float = 0.1,
        act: t.Type[nn.Module] = nn.ELU,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_mods = n_mods
        self.n_classes = n_classes
        self.dropout = dropout
        self.act = act

        self.fusion = FeatureFusion(feature_dim, n_mods, agg)
        in_dim = self.fusion.output_dim
        self.mlp = self._build_two_layer_mlp(
            in_dim=in_dim,
            hidden_dim=hidden_dim or in_dim,
            out_dim=n_classes,
            dropout=dropout,
            act=act,
        )

    @staticmethod
    def _build_single_layer_mlp(
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float,
        act: t.Type[nn.Module],
    ) -> nn.Sequential:
        layers: t.List[nn.Module] = [act(), nn.Linear(in_dim, out_dim)]
        return nn.Sequential(*layers)

    @staticmethod
    def _build_two_layer_mlp(
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float,
        act: t.Type[nn.Module],
    ) -> nn.Sequential:
        layers: t.List[nn.Module] = [act(), nn.Linear(in_dim, hidden_dim)]
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [act(), nn.Linear(hidden_dim, out_dim)]
        return nn.Sequential(*layers)

    @staticmethod
    def _build_three_layer_mlp(
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float,
        act: t.Type[nn.Module],
    ) -> nn.Sequential:
        layers: t.List[nn.Module] = [act(), nn.Linear(in_dim, hidden_dim)]
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [act(), nn.Linear(hidden_dim, hidden_dim)]
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [act(), nn.Linear(hidden_dim, out_dim)]
        return nn.Sequential(*layers)

    def forward(self, feature_of_different_mods: t.List[torch.Tensor]) -> torch.Tensor:
        fused, has_L = self.fusion.aggregator(feature_of_different_mods)
        if has_L:
            B, L, E = fused.shape
            fused = fused.view(B * L, E)
            out = self.mlp(fused).view(B, L, self.n_classes)
        else:
            out = self.mlp(fused)
        return out


@register_head("classification")
def build_classification_head(
    *,
    target,
    feature_dim,
    n_mods,
    output_dim,
    agg: str = "gated_scalar",
    hidden_dim: t.Optional[int] = None,
    dropout: float = 0.1,
    act: t.Type[nn.Module] = nn.ELU,
    **_,
) -> nn.Module:
    return ClassificationHead(
        feature_dim,
        n_mods,
        output_dim,
        agg=agg,
        hidden_dim=hidden_dim,
        dropout=dropout,
        act=act,
    )


__all__ = ["ClassificationHead", "build_classification_head"]
