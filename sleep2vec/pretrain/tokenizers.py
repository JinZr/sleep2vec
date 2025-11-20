import typing as t

import torch.nn as nn

from sleep2vec.config import ChannelConfig
from sleep2vec.registry import get_tokenizer_builder, register_tokenizer


@register_tokenizer("linear")
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


@register_tokenizer("sundial")
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


def build_tokenizer_from_channel(
    channel: ChannelConfig, *, device: str = "cuda"
) -> nn.Module:
    """Instantiate a tokenizer for a specific channel config."""
    builder = get_tokenizer_builder(channel.tokenizer)
    kwargs = dict(channel.tokenizer_kwargs or {})
    kwargs.setdefault("in_feature_dim", channel.input_dim)
    kwargs.setdefault("out_feature_dim", channel.out_dim)
    kwargs.setdefault("device", device)
    return builder(**kwargs)


def build_tokenizer_mapping(
    channels: t.List[ChannelConfig], *, device: str = "cuda"
) -> t.Dict[str, nn.Module]:
    return {
        channel.name: build_tokenizer_from_channel(channel, device=device)
        for channel in channels
    }


__all__ = [
    "LinearTokenizer",
    "SundialTokenizer",
    "build_tokenizer_from_channel",
    "build_tokenizer_mapping",
]
