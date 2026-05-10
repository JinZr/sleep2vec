from __future__ import annotations

import abc
import typing as t

import torch
import torch.nn as nn

from sleep2vec2.config import ChannelConfig
from sleep2vec2.registry import get_tokenizer_builder, register_tokenizer


class BaseTokenizer(nn.Module, metaclass=abc.ABCMeta):
    """Shared utilities and interface for all tokenizers.

    Subclasses should set up layers in ``__init__`` and rely on
    ``_record_parameter_stats`` for consistent parameter accounting.
    """

    feature_dim: int
    device: str
    total_params: int
    trainable_params: int

    def __init__(self, *, out_feature_dim: int, device: str = "cuda"):
        super().__init__()
        self.feature_dim = out_feature_dim
        self.device = device

    @staticmethod
    def _make_norm(dim: int, enabled: bool) -> nn.Module:
        """Returns a LayerNorm when enabled, otherwise identity."""

        return nn.LayerNorm(dim) if enabled else nn.Identity()

    def _record_parameter_stats(self, verbose: bool = False) -> None:
        """Compute parameter counts once layers are built."""

        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if verbose:
            print(f"Total parameters: {self.total_params}")
            print(f"Trainable parameters: {self.trainable_params}")

    @abc.abstractmethod
    def forward(self, x):  # pragma: no cover - interface only
        raise NotImplementedError


@register_tokenizer("linear")
class LinearTokenizer(BaseTokenizer):
    def __init__(
        self,
        in_feature_dim: int,
        out_feature_dim: int,
        device: str = "cuda",
        norm_layer: bool = True,
    ):
        super().__init__(out_feature_dim=out_feature_dim, device=device)

        self.proj = nn.Linear(in_feature_dim, out_feature_dim)
        self.norm = self._make_norm(out_feature_dim, norm_layer)

        self._record_parameter_stats(verbose=True)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


@register_tokenizer("sundial")
class SundialTokenizer(BaseTokenizer):
    def __init__(
        self,
        in_feature_dim: int,
        out_feature_dim: int,
        device: str = "cuda",
        norm_layer: bool = True,
    ):
        super().__init__(out_feature_dim=out_feature_dim, device=device)

        inter = 2 * out_feature_dim
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(0.1)

        self.hidden_layer = nn.Linear(in_feature_dim, inter, bias=True)
        self.output_layer = nn.Linear(inter, out_feature_dim, bias=True)
        self.residual_layer = nn.Linear(in_feature_dim, out_feature_dim, bias=True)

        self.norm = self._make_norm(out_feature_dim, norm_layer)

        self._record_parameter_stats(verbose=False)

    def forward(self, x):
        y = self.hidden_layer(x)
        y = self.act(y)
        y = self.output_layer(y)
        y = self.dropout(y)

        res = self.residual_layer(x)
        out = y + res
        out = self.norm(out)
        return out


@register_tokenizer("sundial2")
class SundialTokenizer2(BaseTokenizer):
    def __init__(
        self,
        in_feature_dim: int,
        out_feature_dim: int,
        device: str = "cuda",
        norm_layer: bool = True,
        # here we try to solve the spikiness problem
        pre_norm: bool = True,
        residual_scale: float = 0.5,
        ff_scale: float = 1.0,
        clamp_value: float | None = None,
        modality_scale: float = 1.0,
    ):
        super().__init__(out_feature_dim=out_feature_dim, device=device)

        inter = 2 * out_feature_dim  # keep the same expansion factor

        self.act = nn.SiLU()
        self.dropout = nn.Dropout(0.1)

        # FFN branch
        self.hidden_layer = nn.Linear(in_feature_dim, inter, bias=True)
        self.output_layer = nn.Linear(inter, out_feature_dim, bias=True)

        # Residual projection branch
        self.residual_layer = nn.Linear(in_feature_dim, out_feature_dim, bias=True)

        # Pre-norm to control the scale of everything entering the tokenizer
        self.pre_norm = nn.LayerNorm(in_feature_dim) if pre_norm else nn.Identity()

        # Post-norm stays conceptually the same as before
        self.norm = self._make_norm(out_feature_dim, norm_layer)

        # New scaling factors
        self.residual_scale = residual_scale
        self.ff_scale = ff_scale
        self.clamp_value = clamp_value

        # Per-modality global scale. This does not change dim, only scales output & gradients.
        self.modality_scale = nn.Parameter(torch.tensor(modality_scale, dtype=torch.float32))

        self._record_parameter_stats(verbose=False)

    def forward(self, x):
        # we use pre-norm setup to bring problematic modalities onto
        # a similar scale before FFN and residual
        x_norm = self.pre_norm(x)

        # FFN branch (the potentially ''spiky'' path in diagnostics)
        y = self.hidden_layer(x_norm)
        y = self.act(y)
        y = self.output_layer(y)

        # Optional soft clamp to avoid extreme tails
        # for SpO2 and nasal etc.
        if self.clamp_value is not None:
            # soft saturation instead of hard clip, which should
            # preserve gradients but limits magnitude
            c = self.clamp_value
            y = c * torch.tanh(y / c)

        y = self.dropout(y)
        y = y * self.ff_scale

        # Residual branch stays linear but now shares the
        # same normalized input
        res = self.residual_layer(x_norm)

        # Residual mixing with explicit scale on FFN branch
        out = res + self.residual_scale * y

        # Post-norm as before, then apply a per-modality scale
        out = self.norm(out)
        out = out * self.modality_scale
        return out


def build_tokenizer_from_channel(channel: ChannelConfig, *, device: str = "cuda") -> nn.Module:
    """Instantiate a tokenizer for a specific channel config."""
    tokenizer_cfg = channel.tokenizer
    if tokenizer_cfg.out_dim is None:
        raise ValueError(f"channel '{channel.name}' is missing tokenizer.out_dim.")
    builder = get_tokenizer_builder(tokenizer_cfg.name)
    kwargs = dict(tokenizer_cfg.kwargs or {})
    kwargs.setdefault("in_feature_dim", channel.input_dim)
    kwargs.setdefault("out_feature_dim", tokenizer_cfg.out_dim)
    kwargs.setdefault("device", device)
    return builder(**kwargs)


def build_tokenizer_mapping(channels: t.List[ChannelConfig], *, device: str = "cuda") -> t.Dict[str, nn.Module]:
    return {channel.name: build_tokenizer_from_channel(channel, device=device) for channel in channels}


__all__ = [
    "BaseTokenizer",
    "LinearTokenizer",
    "SundialTokenizer",
    "SundialTokenizer2",
    "build_tokenizer_from_channel",
    "build_tokenizer_mapping",
]
