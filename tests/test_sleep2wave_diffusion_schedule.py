from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sleep2wave.diffusion.schedule import build_diffusion_schedule, cosine_beta_schedule


def test_cosine_beta_schedule_has_expected_shape_and_bounds():
    betas = cosine_beta_schedule(10)

    assert betas.shape == (10,)
    assert torch.isfinite(betas).all()
    assert ((betas > 0) & (betas < 1)).all()


def test_diffusion_schedule_precomputes_expected_shapes():
    schedule = build_diffusion_schedule(10)

    assert schedule.betas.shape == (10,)
    assert schedule.alphas.shape == (10,)
    assert schedule.alpha_bars.shape == (10,)
    assert schedule.sqrt_alpha_bars.shape == (10,)
    assert schedule.sqrt_one_minus_alpha_bars.shape == (10,)


def test_alpha_bars_are_monotonically_non_increasing():
    schedule = build_diffusion_schedule(20)

    assert torch.all(schedule.alpha_bars[1:] <= schedule.alpha_bars[:-1])


def test_diffusion_schedule_rejects_unsupported_schedule():
    with pytest.raises(ValueError, match="beta_schedule must be 'cosine'"):
        build_diffusion_schedule(10, beta_schedule="linear")


def test_diffusion_schedule_rejects_invalid_step_count():
    with pytest.raises(ValueError, match="num_steps must be a positive integer"):
        cosine_beta_schedule(0)
