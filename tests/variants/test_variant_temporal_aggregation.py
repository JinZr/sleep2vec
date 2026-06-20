import importlib

import pytest
import torch


@pytest.mark.parametrize("namespace", ["sleep2vec2", "sleep2expert"])
def test_variant_lstm_temporal_aggregator_builds(namespace: str):
    temporal_module = importlib.import_module(f"{namespace}.downstreams.temporal_aggregation")
    aggregator = temporal_module.build_temporal_aggregator("lstm", hidden_size=8)
    hidden = torch.randn(2, 4, 8)
    mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, False],
        ]
    )

    pooled = aggregator(hidden, mask)

    assert pooled.shape == (2, 8)


@pytest.mark.parametrize("namespace", ["sleep2vec2", "sleep2expert"])
def test_variant_lstm_temporal_aggregator_rejects_non_boolean_bidirectional(namespace: str):
    temporal_module = importlib.import_module(f"{namespace}.downstreams.temporal_aggregation")

    with pytest.raises(ValueError, match="bidirectional must be a boolean"):
        temporal_module.build_temporal_aggregator("lstm", hidden_size=8, bidirectional="false")


@pytest.mark.parametrize("namespace", ["sleep2vec2", "sleep2expert"])
def test_variant_lstm_temporal_aggregator_rejects_zero_length_sequences(namespace: str):
    temporal_module = importlib.import_module(f"{namespace}.downstreams.temporal_aggregation")
    aggregator = temporal_module.build_temporal_aggregator("lstm", hidden_size=8)
    hidden = torch.randn(2, 3, 8)
    mask = torch.tensor(
        [
            [True, True, False],
            [False, False, False],
        ]
    )

    with pytest.raises(ValueError, match="at least one valid token"):
        aggregator(hidden, mask)


@pytest.mark.parametrize("namespace", ["sleep2vec2", "sleep2expert"])
def test_variant_config_accepts_lstm_temporal_aggregator(namespace: str):
    config_module = importlib.import_module(f"{namespace}.config")
    model_cfg = config_module.ModelConfig(
        channels=[
            config_module.ChannelConfig(
                name="ppg",
                input_dim=1,
                tokenizer=config_module.TokenizerConfig(name="linear", out_dim=8),
            ),
        ],
        backbone=config_module.BackboneConfig(
            name="roformer",
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            vocab_size=1,
        ),
        projection=config_module.ProjectionConfig(name="simclr", enabled=True, hidden_dim=8, out_dim=4),
        cls=config_module.ClsConfig(downstream="tokens", embedding_type=None),
        head=config_module.HeadConfig(
            channel_agg=config_module.ChannelAggConfig(name="mean"),
            temporal_agg=config_module.TemporalAggConfig(
                name="lstm",
                kwargs={"bidirectional": True, "num_layers": 1, "dropout": 0.0},
            ),
            name="classification",
        ),
    )

    assert config_module.validate_model_config(model_cfg) == 8
