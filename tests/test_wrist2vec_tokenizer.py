from __future__ import annotations

import pytest
import torch

import wrist2vec.registry as wrist_registry
from wrist2vec.config import ChannelConfig, TokenizerConfig
from wrist2vec.modules.tokenizers import ResNet1dTokenizer, build_tokenizer_from_channel


def _build_resnet1d_channel(**kwargs) -> ChannelConfig:
    return ChannelConfig(
        name="ppg",
        input_dim=128,
        tokenizer=TokenizerConfig(
            name="resnet1d",
            out_dim=32,
            kwargs=kwargs,
        ),
    )


def test_resnet1d_tokenizer_is_registered():
    assert "resnet1d" in wrist_registry.available_tokenizers()


def test_build_resnet1d_tokenizer_from_channel():
    tokenizer = build_tokenizer_from_channel(_build_resnet1d_channel(), device="cpu")

    assert isinstance(tokenizer, ResNet1dTokenizer)
    assert tokenizer.feature_dim == 32


def test_resnet1d_tokenizer_encodes_batched_token_sequences():
    tokenizer = build_tokenizer_from_channel(_build_resnet1d_channel(), device="cpu")

    x = torch.randn(2, 5, 128)
    y = tokenizer(x)

    assert y.shape == (2, 5, 32)


def test_resnet1d_tokenizer_encodes_flat_windows():
    tokenizer = build_tokenizer_from_channel(_build_resnet1d_channel(), device="cpu")

    x = torch.randn(7, 128)
    y = tokenizer(x)

    assert y.shape == (7, 32)


def test_resnet1d_tokenizer_rejects_four_dimensional_inputs():
    tokenizer = build_tokenizer_from_channel(_build_resnet1d_channel(), device="cpu")

    x = torch.randn(2, 5, 1, 128)
    with pytest.raises(ValueError, match="multi-channel window inputs"):
        tokenizer(x)


@pytest.mark.parametrize(
    ("norm", "act", "channel_schedule", "stride_schedule"),
    [
        ("group", "relu", [16, 24, 32], [1, 2, 2]),
        ("batch", "silu", [12, 24], [1, 2]),
        ("layer", "gelu", [32], [1]),
    ],
)
def test_resnet1d_tokenizer_supports_configurable_schedules(
    norm: str,
    act: str,
    channel_schedule: list[int],
    stride_schedule: list[int],
):
    tokenizer = build_tokenizer_from_channel(
        _build_resnet1d_channel(
            block_counts=[1] * len(channel_schedule),
            channel_schedule=channel_schedule,
            stride_schedule=stride_schedule,
            norm=norm,
            act=act,
            front_pool="none",
        ),
        device="cpu",
    )

    x = torch.randn(3, 128)
    y = tokenizer(x)

    assert y.shape == (3, 32)
