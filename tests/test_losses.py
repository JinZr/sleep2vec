from __future__ import annotations

import pytest
import torch

from sleep2vec.losses import available_losses, create_loss


def _random_hidden(seed: int, *, batch_size: int = 3, seq_len: int = 2, hidden_dim: int = 5) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(batch_size, seq_len, hidden_dim)


def test_available_losses_contains_registered_defaults():
    losses = available_losses()
    assert "info_nce" in losses
    assert "weighted_info_nce" in losses


def test_create_loss_rejects_unknown_name():
    with pytest.raises(KeyError, match="Unknown loss 'missing_loss'"):
        create_loss("missing_loss", temperature=0.2)


def test_info_nce_forward_returns_scalar_loss_and_metrics():
    first_hidden = _random_hidden(1)
    second_hidden = _random_hidden(2)
    loss_fn = create_loss("info_nce", temperature=0.2)

    output = loss_fn(first_hidden, second_hidden, batch={})

    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)
    assert "contrastive_loss" in output.metrics
    assert "contrastive_acc" in output.metrics
    assert 0.0 <= float(output.metrics["contrastive_acc"]) <= 1.0


def test_weighted_info_nce_requires_weight_and_hardness_matrices():
    first_hidden = _random_hidden(3)
    second_hidden = _random_hidden(4)
    loss_fn = create_loss("weighted_info_nce", temperature=0.2, hard_scale=0.1, pos_margin=0.0)

    with pytest.raises(KeyError, match="Batch missing 'w' or 'h'"):
        loss_fn(first_hidden, second_hidden, batch={})


def test_weighted_info_nce_forward_with_valid_batch_and_pos_margin():
    batch_size = 3
    first_hidden = _random_hidden(5, batch_size=batch_size)
    second_hidden = _random_hidden(6, batch_size=batch_size)
    w = torch.ones(batch_size, batch_size)
    h = torch.zeros(batch_size, batch_size)
    loss_fn = create_loss("weighted_info_nce", temperature=0.2, hard_scale=0.1, pos_margin=0.3)

    output = loss_fn(first_hidden, second_hidden, batch={"w": w, "h": h})

    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)
    assert "contrastive_loss" in output.metrics
    assert "contrastive_acc" in output.metrics
    assert 0.0 <= float(output.metrics["contrastive_acc"]) <= 1.0
