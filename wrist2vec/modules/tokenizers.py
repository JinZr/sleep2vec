from __future__ import annotations

import abc
import typing as t

import torch
import torch.nn as nn

from wrist2vec.config import ChannelConfig
from wrist2vec.registry import get_tokenizer_builder, register_tokenizer


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


class ChannelwiseLayerNorm1d(nn.Module):
    """Apply LayerNorm over the channel axis of a `[N, C, T]` tensor."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"ChannelwiseLayerNorm1d expects 3D input [N, C, T], got shape {tuple(x.shape)}")
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


def _resolve_activation(name: str) -> t.Type[nn.Module]:
    registry = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    key = str(name).lower()
    if key not in registry:
        raise ValueError(f"Unsupported activation '{name}'. Expected one of {sorted(registry)}.")
    return registry[key]


def _resolve_group_count(channels: int, requested_groups: int) -> int:
    groups = max(1, min(int(requested_groups), int(channels)))
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


def _build_conv_norm(norm: str, channels: int, *, norm_groups: int) -> nn.Module:
    key = str(norm).lower()
    if key == "group":
        return nn.GroupNorm(_resolve_group_count(channels, norm_groups), channels)
    if key == "batch":
        return nn.BatchNorm1d(channels)
    if key == "layer":
        return ChannelwiseLayerNorm1d(channels)
    raise ValueError(f"Unsupported norm '{norm}'. Expected one of ['group', 'batch', 'layer'].")


def _build_front_pool(name: str, *, stride: int) -> nn.Module:
    key = str(name).lower()
    if key == "none":
        return nn.Identity()
    if key == "max":
        return nn.MaxPool1d(kernel_size=3, stride=stride, padding=1)
    if key == "avg":
        return nn.AvgPool1d(kernel_size=3, stride=stride, padding=1)
    raise ValueError(f"Unsupported front_pool '{name}'. Expected one of ['max', 'avg', 'none'].")


def _validate_schedule(
    values: t.Sequence[int] | None,
    *,
    name: str,
    expected_len: int | None = None,
) -> list[int]:
    if values is None:
        raise ValueError(f"{name} must be provided after default resolution.")
    resolved = [int(v) for v in values]
    if not resolved:
        raise ValueError(f"{name} must be a non-empty list.")
    if any(v <= 0 for v in resolved):
        raise ValueError(f"{name} must contain positive integers. Got {resolved}.")
    if expected_len is not None and len(resolved) != expected_len:
        raise ValueError(f"{name} must have length {expected_len}, got {len(resolved)}.")
    return resolved


def _default_channel_schedule(out_feature_dim: int, num_groups: int) -> list[int]:
    if num_groups == 1:
        return [max(8, int(out_feature_dim))]

    schedule: list[int] = []
    for idx in range(num_groups):
        remaining = num_groups - idx - 1
        if remaining == 0:
            schedule.append(int(out_feature_dim))
        else:
            divisor = 2**remaining
            schedule.append(max(8, int(out_feature_dim) // divisor))
    return schedule


class ResidualConvBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        norm: str,
        norm_groups: int,
        act: t.Type[nn.Module],
        dropout: float,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = _build_conv_norm(norm, out_channels, norm_groups=norm_groups)
        self.act1 = act()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _build_conv_norm(norm, out_channels, norm_groups=norm_groups)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = self._build_shortcut(
            in_channels,
            out_channels,
            stride=stride,
            norm=norm,
            norm_groups=norm_groups,
        )
        self.out_act = act()

    @staticmethod
    def _build_shortcut(
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        norm: str,
        norm_groups: int,
    ) -> nn.Module:
        if stride == 1 and in_channels == out_channels:
            return nn.Identity()
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            _build_conv_norm(norm, out_channels, norm_groups=norm_groups),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.drop(y)
        return self.out_act(y + residual)


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


@register_tokenizer("resnet1d")
class ResNet1dTokenizer(BaseTokenizer):
    """Residual 1D tokenizer for already-chunked single-channel wrist windows.

    This tokenizer only encodes the window content it receives. Window boundaries
    still come from the current data-side non-overlap token chunking, so
    overlap-aware tokenization is intentionally out of scope here.
    """

    def __init__(
        self,
        in_feature_dim: int,
        out_feature_dim: int,
        device: str = "cuda",
        block_counts: t.Sequence[int] | None = None,
        channel_schedule: t.Sequence[int] | None = None,
        stride_schedule: t.Sequence[int] | None = None,
        front_kernel_size: int = 7,
        front_stride: int = 2,
        front_pool: str = "max",
        front_pool_stride: int = 2,
        norm: str = "group",
        norm_groups: int = 8,
        act: str = "relu",
        dropout: float = 0.1,
    ):
        super().__init__(out_feature_dim=out_feature_dim, device=device)
        if int(in_feature_dim) <= 0:
            raise ValueError(f"in_feature_dim must be > 0, got {in_feature_dim}.")
        if int(out_feature_dim) <= 0:
            raise ValueError(f"out_feature_dim must be > 0, got {out_feature_dim}.")
        if front_kernel_size % 2 == 0:
            raise ValueError("front_kernel_size must be odd for length-preserving padding.")
        if front_stride <= 0:
            raise ValueError("front_stride must be > 0.")
        if front_pool_stride <= 0:
            raise ValueError("front_pool_stride must be > 0.")
        if dropout < 0:
            raise ValueError("dropout must be >= 0.")

        block_counts = _validate_schedule(block_counts or [2, 2, 2], name="block_counts")
        channel_schedule = _validate_schedule(
            channel_schedule or _default_channel_schedule(out_feature_dim, len(block_counts)),
            name="channel_schedule",
            expected_len=len(block_counts),
        )
        stride_schedule = _validate_schedule(
            stride_schedule or [2, 2, 2],
            name="stride_schedule",
            expected_len=len(block_counts),
        )

        act_cls = _resolve_activation(act)
        first_width = channel_schedule[0]
        self.front_end = nn.Sequential(
            nn.Conv1d(
                1,
                first_width,
                kernel_size=front_kernel_size,
                stride=front_stride,
                padding=front_kernel_size // 2,
                bias=False,
            ),
            _build_conv_norm(norm, first_width, norm_groups=norm_groups),
            act_cls(),
            _build_front_pool(front_pool, stride=front_pool_stride),
        )

        residual_groups: list[nn.Module] = []
        current_channels = first_width
        for group_blocks, next_channels, stride in zip(block_counts, channel_schedule, stride_schedule):
            blocks: list[nn.Module] = []
            for block_index in range(group_blocks):
                block_stride = stride if block_index == 0 else 1
                block_in_channels = current_channels if block_index == 0 else next_channels
                blocks.append(
                    ResidualConvBlock1d(
                        block_in_channels,
                        next_channels,
                        stride=block_stride,
                        norm=norm,
                        norm_groups=norm_groups,
                        act=act_cls,
                        dropout=float(dropout),
                    )
                )
            residual_groups.append(nn.Sequential(*blocks))
            current_channels = next_channels
        self.residual_stack = nn.ModuleList(residual_groups)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_projection = nn.Linear(current_channels, out_feature_dim)
        self.output_norm = nn.LayerNorm(out_feature_dim)

        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        self._record_parameter_stats(verbose=False)

    def _flatten_single_channel_windows(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if x.dim() == 2:
            return x.unsqueeze(1), (x.shape[0],)
        if x.dim() == 3:
            batch_size, num_tokens, token_width = x.shape
            return x.reshape(batch_size * num_tokens, 1, token_width), (batch_size, num_tokens)
        if x.dim() == 4:
            raise ValueError(
                "resnet1d tokenizer only supports single-channel windows passed as [N, T] or [B, L, T]. "
                f"Got shape {tuple(x.shape)}; multi-channel window inputs [B, L, C, T] are not supported yet."
            )
        raise ValueError(f"resnet1d tokenizer expects 2D or 3D input, got shape {tuple(x.shape)}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flattened, leading_shape = self._flatten_single_channel_windows(x)
        y = self.front_end(flattened)
        for group in self.residual_stack:
            y = group(y)
        y = self.global_pool(y).squeeze(-1)
        y = self.output_projection(y)
        y = self.output_norm(y)

        if len(leading_shape) == 1:
            return y
        return y.view(*leading_shape, self.feature_dim)


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
    "ChannelwiseLayerNorm1d",
    "LinearTokenizer",
    "ResNet1dTokenizer",
    "ResidualConvBlock1d",
    "SundialTokenizer",
    "SundialTokenizer2",
    "build_tokenizer_from_channel",
    "build_tokenizer_mapping",
]
