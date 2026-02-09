from __future__ import annotations

import pytest
import torch.nn as nn

from sleep2vec import registry
from sleep2vec.config import (
    BackboneConfig,
    ChannelConfig,
    ClsConfig,
    ModelConfig,
    ProjectionConfig,
    TokenizerConfig,
    validate_model_config,
)
from sleep2vec.modules.projection import SimCLRProjectionHead, build_projection_head
from sleep2vec.modules.tokenizers import build_tokenizer_from_channel, build_tokenizer_mapping


def _minimal_model_config() -> ModelConfig:
    return ModelConfig(
        channels=[
            ChannelConfig(name="eeg", input_dim=4, tokenizer=TokenizerConfig(name="linear", out_dim=6)),
            ChannelConfig(name="ecg", input_dim=4, tokenizer=TokenizerConfig(name="linear", out_dim=6)),
        ],
        backbone=BackboneConfig(
            name="roformer",
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            vocab_size=1,
        ),
        projection=ProjectionConfig(name="simclr", enabled=True, hidden_dim=8, out_dim=4),
        cls=ClsConfig(downstream="tokens", embedding_type=None),
        head=None,
    )


def test_registry_register_lookup_and_available_lists(monkeypatch):
    monkeypatch.setattr(registry, "BACKBONE_REGISTRY", {})
    monkeypatch.setattr(registry, "TOKENIZER_REGISTRY", {})
    monkeypatch.setattr(registry, "PROJECTION_REGISTRY", {})
    monkeypatch.setattr(registry, "MODEL_AVERAGING_REGISTRY", {})

    @registry.register_backbone("tiny_backbone")
    def backbone_builder(cfg):
        return cfg.name

    @registry.register_tokenizer("tiny_tokenizer")
    def tokenizer_builder(**kwargs):
        return kwargs

    @registry.register_projection("tiny_projection")
    def projection_builder(cfg):
        return cfg

    @registry.register_model_averager("tiny_averager")
    def averager_builder(cfg, student):
        return (cfg, student)

    assert registry.get_backbone_builder("tiny_backbone") is backbone_builder
    assert registry.get_tokenizer_builder("tiny_tokenizer") is tokenizer_builder
    assert registry.get_projection_builder("tiny_projection") is projection_builder
    assert registry.get_model_averager_builder("tiny_averager") is averager_builder

    assert registry.available_backbones() == ["tiny_backbone"]
    assert registry.available_tokenizers() == ["tiny_tokenizer"]
    assert registry.available_projections() == ["tiny_projection"]
    assert registry.available_model_averagers() == ["tiny_averager"]

    with pytest.raises(ValueError, match="already registered"):

        @registry.register_backbone("tiny_backbone")
        def _duplicate_backbone(cfg):  # pragma: no cover - never reached
            return cfg

    with pytest.raises(KeyError, match="Unknown backbone"):
        registry.get_backbone_builder("missing")
    with pytest.raises(KeyError, match="Unknown tokenizer"):
        registry.get_tokenizer_builder("missing")
    with pytest.raises(KeyError, match="Unknown projection"):
        registry.get_projection_builder("missing")
    with pytest.raises(KeyError, match="Unknown model averaging"):
        registry.get_model_averager_builder("missing")


def test_validate_model_config_and_build_tokenizer_mapping():
    model_cfg = _minimal_model_config()
    feature_dim = validate_model_config(model_cfg)
    mapping = build_tokenizer_mapping(model_cfg.channels, device="cpu")

    assert feature_dim == 6
    assert set(mapping.keys()) == {"eeg", "ecg"}
    assert isinstance(mapping["eeg"], nn.Module)
    assert mapping["eeg"].feature_dim == 6


def test_build_tokenizer_from_channel_requires_out_dim():
    channel = ChannelConfig(
        name="eeg",
        input_dim=4,
        tokenizer=TokenizerConfig(name="linear", out_dim=None),
    )

    with pytest.raises(ValueError, match="missing tokenizer.out_dim"):
        build_tokenizer_from_channel(channel, device="cpu")


def test_build_projection_head_enabled_disabled_and_unknown_name():
    disabled_cfg = ProjectionConfig(name="simclr", enabled=False, hidden_dim=8, out_dim=4)
    assert build_projection_head(disabled_cfg, in_dim=8) is None

    enabled_cfg = ProjectionConfig(name="simclr", enabled=True, hidden_dim=8, out_dim=4)
    head = build_projection_head(enabled_cfg, in_dim=8)
    assert isinstance(head, SimCLRProjectionHead)

    unknown_cfg = ProjectionConfig(name="missing_projection", enabled=True, hidden_dim=8, out_dim=4)
    with pytest.raises(KeyError, match="Unknown projection"):
        build_projection_head(unknown_cfg, in_dim=8)
