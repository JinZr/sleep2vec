from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from sleep2vec2.config import ChannelConfig, TokenizerConfig
from sleep2vec2.modules.tokenizers import build_tokenizer_from_channel


def test_sundial2_tokenizer_accepts_deeper_kwargs():
    channel = ChannelConfig(
        name="eeg",
        input_dim=4,
        tokenizer=TokenizerConfig(name="sundial2", out_dim=6, kwargs={"norm_layer": False, "num_mlp_layers": 3}),
    )

    tokenizer = build_tokenizer_from_channel(channel, device="cpu")
    output = tokenizer(torch.randn(3, 4))

    assert len(tokenizer.extra_output_layers) == 1
    assert isinstance(tokenizer.norm, nn.Identity)
    assert output.shape == (3, 6)
