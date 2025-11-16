# sleep2vec/model/heads.py
import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F

from .head_registry import register_head


class ClassificationHead(nn.Module):
    """
    三种聚合：
      - 'mean'         : Mean Pool (跨模态求均值)      -> 两层 MLP
      - 'concat'       : 拼接 (沿特征维拼接)           -> 两层 MLP
      - 'gated_scalar' : 学习到的标量权重(softmax)加权 -> 单层 Linear
    输入：list[Tensor]，长度 = n_mods
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
        assert agg in {"mean", "concat", "gated_scalar"}
        if n_mods == 1:
            agg == "concat"
        self.feature_dim = feature_dim
        self.n_mods = n_mods
        self.n_classes = n_classes
        self.agg = agg
        self.dropout = dropout
        self.act = act

        if agg == "concat":
            in_dim = feature_dim * n_mods
        elif agg == "mean":
            in_dim = feature_dim
        else:  # gated_scalar
            # 每个模态一个可学习标量 → softmax 归一化
            self.gates = nn.Parameter(torch.zeros(n_mods))
            in_dim = feature_dim
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
        assert (
            len(feature_of_different_mods) == self.n_mods
        ), f"Expect {self.n_mods} modality features, got {len(feature_of_different_mods)}"

        x0 = feature_of_different_mods[0]
        has_L = x0.dim() == 3  # [B, L, D] or [B, D]

        # 统一成 [B, L, D]
        feats = []
        for f in feature_of_different_mods:
            if f.dim() == 2:  # [B, D] -> [B, 1, D]
                f = f.unsqueeze(1)
            # 断言特征维一致
            assert (
                f.shape[-1] == self.feature_dim
            ), f"feature_dim mismatch: expect {self.feature_dim}, got {f.shape[-1]}"
            feats.append(f)  # [B, L, D]

        if self.agg == "concat":
            # [B, L, n_mods*D]
            x = torch.cat(feats, dim=-1)
            if not has_L:
                x = x.squeeze(1)  # [B, n_mods*D]
                out = self.mlp(x)  # [B, C]
            else:
                B, L, E = x.shape
                x = x.view(B * L, E)
                out = self.mlp(x).view(B, L, self.n_classes)
            return out

        elif self.agg == "mean":
            # [B, L, D], 先 stack 再沿模态均值
            stack = torch.stack(feats, dim=0)  # [n_mods, B, L, D]
            x = stack.mean(dim=0)  # [B, L, D]
            if not has_L:
                x = x.squeeze(1)  # [B, D]
                out = self.mlp(x)  # [B, C]
            else:
                B, L, E = x.shape
                x = x.view(B * L, E)
                out = self.mlp(x).view(B, L, self.n_classes)
            return out

        else:  # gated_scalar
            w = F.softmax(self.gates, dim=0)  # [n_mods]
            stack = torch.stack(feats, dim=0)  # [n_mods, B, L, D]
            x = (w[:, None, None, None] * stack).sum(dim=0)  # [B, L, D]
            if not has_L:
                x = x.squeeze(1)
                out = self.mlp(x)
            else:
                B, L, E = x.shape
                x = x.view(B * L, E)
                out = self.mlp(x).view(B, L, self.n_classes)
            return out


class RegressionHead(nn.Module):
    def __init__(self, target, feature_dim, n_mods, out_dim=1):
        super().__init__()
        self.target = target
        self.reghead = ClassificationHead(feature_dim, n_mods, out_dim)

    def forward(self, x):
        x = self.reghead(x)
        if self.target == "age":
            x = torch.sigmoid(x) * 100
        return x


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
    **_,
) -> nn.Module:
    return RegressionHead(target, feature_dim, n_mods, out_dim=output_dim)
