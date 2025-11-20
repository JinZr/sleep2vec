# sleep2vec/model/heads.py
import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F

from .head_registry import register_head


class FeatureFusion(nn.Module):
    """Shared multi-modal fusion block used by both classification and regression."""

    def __init__(self, feature_dim: int, n_mods: int, agg: str):
        super().__init__()
        if n_mods < 1:
            raise ValueError("n_mods must be >= 1.")
        assert agg in {"mean", "concat", "gated_scalar"}
        if n_mods == 1 and agg != "concat":
            agg = "concat"  # fall back to concatenation for single modality
        self.feature_dim = feature_dim
        self.n_mods = n_mods
        self.agg = agg
        self.output_dim = feature_dim * n_mods if agg == "concat" else feature_dim
        if agg == "gated_scalar":
            # 每个模态一个可学习标量 → softmax 归一化
            self.gates = nn.Parameter(torch.zeros(n_mods))

    def forward(
        self, feature_of_different_mods: t.List[torch.Tensor]
    ) -> tuple[torch.Tensor, bool]:
        if len(feature_of_different_mods) != self.n_mods:
            raise ValueError(
                f"Expect {self.n_mods} modality features, got {len(feature_of_different_mods)}"
            )

        x0 = feature_of_different_mods[0]
        has_L = x0.dim() == 3  # [B, L, D] or [B, D]

        feats = []
        for feat in feature_of_different_mods:
            if feat.dim() == 2:  # [B, D] -> [B, 1, D]
                feat_has_L = False
                feat = feat.unsqueeze(1)
            elif feat.dim() == 3:
                feat_has_L = True
            else:
                raise ValueError(
                    "Each modality feature must be rank-2 or rank-3, "
                    f"got shape {feat.shape}."
                )
            if feat_has_L != has_L:
                raise ValueError(
                    "Mixing sequential and non-sequential features is not supported."
                )
            if has_L and feat.size(1) != x0.size(1):
                raise ValueError("All modalities must have matching sequence length.")
            if feat.shape[-1] != self.feature_dim:
                raise ValueError(
                    f"feature_dim mismatch: expect {self.feature_dim}, got {feat.shape[-1]}"
                )
            feats.append(feat)

        if self.agg == "concat":
            fused = torch.cat(feats, dim=-1)
        elif self.agg == "mean":
            fused = torch.stack(feats, dim=0).mean(dim=0)
        else:  # gated_scalar
            weights = F.softmax(self.gates, dim=0)  # [n_mods]
            stack = torch.stack(feats, dim=0)
            fused = (weights[:, None, None, None] * stack).sum(dim=0)

        if not has_L:
            fused = fused.squeeze(1)

        return fused, has_L


class ClassificationHead(nn.Module):
    """
    三种聚合：
      - 'mean'         : Mean Pool (跨模态求均值)      -> 两层 MLP
      - 'concat'       : 拼接 (沿特征维拼接)           -> 两层 MLP
      - 'gated_scalar' : 学习到的标量权重(softmax)加权 -> 单层 Linear
    输入：t.List[Tensor]，长度 = n_mods
         每个张量形状为 [B, D] 或 [B, L, D]（D = feature_dim）
    输出：logits [B, n_classes] 或 [B, L, n_classes]（若 L>1 则逐位置分类）
    """

    def __init__(
        self,
        feature_dim: int,
        n_mods: int,
        n_classes: int,
        *,
        agg: str = "gated_scalar",  # "mean" | "concat" | "gated_scalar"
        hidden_dim: t.Optional[int] = None,  # 两层 MLP 的隐层宽度（mean/concat）
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
        fused, has_L = self.fusion(feature_of_different_mods)
        if has_L:
            B, L, E = fused.shape
            fused = fused.view(B * L, E)
            out = self.mlp(fused).view(B, L, self.n_classes)
        else:
            out = self.mlp(fused)
        return out


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
        fused, has_L = self.fusion(feature_of_different_mods)
        if has_L:
            B, L, E = fused.shape
            fused = fused.view(B * L, E)
            out = self.regressor(fused).view(B, L, self.out_dim)
        else:
            out = self.regressor(fused)

        return out

 
class AttnPooling(nn.Module):
    def __init__(self, d, heads=1, temp=1.0, dropout=0.0):
        super().__init__()
        self.W = nn.Linear(d, d, bias=True)
        self.q = nn.Parameter(torch.randn(heads, d) * 0.02)
        self.heads = heads
        self.temp = temp
        self.drop = nn.Dropout(dropout)

    def forward(self, H, mask):
        Ht = torch.tanh(self.W(H))
        scores = torch.einsum("bld,hd->blh", Ht, self.q) / self.temp
        scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        A = torch.softmax(scores, dim=1)
        A = self.drop(A)
        Z = torch.einsum("blh,bld->bhd", A, H)
        Z = Z.reshape(H.size(0), -1)
        return Z, A


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
