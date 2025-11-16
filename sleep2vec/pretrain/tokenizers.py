import torch.nn as nn


class LinearTokenizer(nn.Module):
    def __init__(
        self,
        in_feature_dim: int,
        out_feature_dim: int,
        device: str = "cuda",
        norm_layer: bool = True,
    ):
        super().__init__()
        self.device = device
        self.feature_dim = out_feature_dim

        self.proj = nn.Linear(in_feature_dim, out_feature_dim)
        self.norm = nn.LayerNorm(out_feature_dim) if norm_layer else nn.Identity()

        self.total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {self.total_params}")
        self.trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        print(f"Trainable parameters: {self.trainable_params}")

    def forward(self, x):
        x = x.to(self.device)
        x = self.proj(x)
        x = self.norm(x)
        return x


class SundialTokenizer(nn.Module):
    def __init__(
        self,
        in_feature_dim: int,
        out_feature_dim: int,
        device: str = "cuda",
        norm_layer: bool = True,
    ):
        super().__init__()
        self.device = device
        self.feature_dim = out_feature_dim

        inter = 2 * out_feature_dim
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(0.1)

        self.hidden_layer = nn.Linear(in_feature_dim, inter, bias=True)
        self.output_layer = nn.Linear(inter, out_feature_dim, bias=True)
        self.residual_layer = nn.Linear(in_feature_dim, out_feature_dim, bias=True)

        self.norm = nn.LayerNorm(out_feature_dim) if norm_layer else nn.Identity()

        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

    def forward(self, x):
        x = x.to(self.device)
        y = self.hidden_layer(x)
        y = self.act(y)
        y = self.output_layer(y)
        y = self.dropout(y)

        res = self.residual_layer(x)
        out = y + res
        out = self.norm(out)
        return out
