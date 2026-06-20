import pytest
import torch

from sleep2vec.downstreams.temporal_aggregation import LSTMAggregator, build_temporal_aggregator


@pytest.mark.parametrize("bidirectional", [True, False])
def test_lstm_aggregator_preserves_feature_dim(bidirectional: bool):
    aggregator = build_temporal_aggregator("lstm", hidden_size=8, bidirectional=bidirectional)
    hidden = torch.randn(3, 5, 8)
    mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
            [True, True, False, False, False],
        ]
    )

    pooled = aggregator(hidden, mask)

    assert isinstance(aggregator, LSTMAggregator)
    assert pooled.shape == (3, 8)


def test_lstm_aggregator_ignores_masked_padding_values():
    torch.manual_seed(0)
    aggregator = build_temporal_aggregator("lstm", hidden_size=8)
    mask = torch.tensor([[True, True, True, False, False]])
    hidden = torch.randn(1, 5, 8)
    changed_padding = hidden.clone()
    changed_padding[:, 3:] = torch.randn(1, 2, 8) * 100.0

    pooled = aggregator(hidden, mask)
    pooled_changed_padding = aggregator(changed_padding, mask)

    assert torch.allclose(pooled, pooled_changed_padding, atol=1e-6)


def test_lstm_aggregator_rejects_bidirectional_odd_hidden_size():
    with pytest.raises(ValueError, match="even hidden_size"):
        build_temporal_aggregator("lstm", hidden_size=7)


def test_lstm_aggregator_rejects_non_boolean_bidirectional():
    with pytest.raises(ValueError, match="bidirectional must be a boolean"):
        build_temporal_aggregator("lstm", hidden_size=8, bidirectional="false")


def test_lstm_aggregator_rejects_zero_length_sequences():
    aggregator = build_temporal_aggregator("lstm", hidden_size=8)
    hidden = torch.randn(2, 3, 8)
    mask = torch.tensor(
        [
            [True, True, False],
            [False, False, False],
        ]
    )

    with pytest.raises(ValueError, match="at least one valid token"):
        aggregator(hidden, mask)


def test_lstm_aggregator_forces_single_layer_recurrent_dropout_to_zero():
    aggregator = build_temporal_aggregator("lstm", hidden_size=8, num_layers=1, dropout=0.9)

    assert aggregator.lstm.dropout == 0.0
