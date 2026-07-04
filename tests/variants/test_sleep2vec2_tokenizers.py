from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from sleep2expert.modules.tokenizers import SundialTokenizer2 as ExpertSundialTokenizer2
from sleep2vec2.config import ChannelConfig, TokenizerConfig
from sleep2vec2.modules.tokenizers import SundialTokenizer2 as VariantSundialTokenizer2, build_tokenizer_from_channel
from sleep2vec.modules.tokenizers import SundialTokenizer2 as RootSundialTokenizer2


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


@pytest.mark.parametrize(
    "tokenizer_cls",
    [RootSundialTokenizer2, VariantSundialTokenizer2, ExpertSundialTokenizer2],
)
def test_sundial2_scales_residual_and_ffn_branches_independently(tokenizer_cls):
    def make_tokenizer(*, residual_scale, ff_scale):
        tokenizer = tokenizer_cls(
            in_feature_dim=2,
            out_feature_dim=2,
            device="cpu",
            norm_layer=False,
            pre_norm=False,
            residual_scale=residual_scale,
            ff_scale=ff_scale,
            clamp_value=None,
            modality_scale=1.0,
        )
        tokenizer.eval()
        with torch.no_grad():
            tokenizer.hidden_layer.weight.zero_()
            tokenizer.hidden_layer.bias.zero_()
            tokenizer.output_layer.weight.zero_()
            tokenizer.output_layer.bias.copy_(torch.tensor([3.0, 5.0]))
            tokenizer.residual_layer.weight.zero_()
            tokenizer.residual_layer.bias.copy_(torch.tensor([2.0, -4.0]))
        return tokenizer

    x = torch.zeros(1, 2)

    low_residual = make_tokenizer(residual_scale=0.25, ff_scale=1.0)(x)
    high_residual = make_tokenizer(residual_scale=1.5, ff_scale=1.0)(x)
    assert torch.allclose(low_residual, torch.tensor([[3.5, 4.0]]))
    assert torch.allclose(high_residual - low_residual, torch.tensor([[2.5, -5.0]]))

    low_ffn = make_tokenizer(residual_scale=0.5, ff_scale=0.25)(x)
    high_ffn = make_tokenizer(residual_scale=0.5, ff_scale=2.0)(x)
    assert torch.allclose(low_ffn, torch.tensor([[1.75, -0.75]]))
    assert torch.allclose(high_ffn - low_ffn, torch.tensor([[5.25, 8.75]]))
