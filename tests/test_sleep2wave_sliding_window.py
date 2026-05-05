from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sleep2wave.inference.sliding_window import fuse_mask_windows, fuse_overlapping_windows, validate_single_night


def test_sliding_window_mean_fusion_covers_contiguous_night():
    windows = torch.zeros(2, 2, 3, 1, 1)
    windows[:, 0, :, 0, 0] = torch.tensor([0.0, 1.0, 2.0])
    windows[:, 1, :, 0, 0] = torch.tensor([2.0, 3.0, 4.0])

    fused = fuse_overlapping_windows(windows, [0, 2], mode="mean")

    assert fused.values.shape == (2, 5, 1, 1)
    assert fused.epoch_index.tolist() == [0, 1, 2, 3, 4]
    assert fused.values[0, :, 0, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_sliding_window_median_fusion_shape():
    windows = torch.randn(3, 2, 2, 1, 4)

    fused = fuse_overlapping_windows(windows, [0, 1], mode="median")

    assert fused.values.shape == (3, 3, 1, 4)


def test_sliding_window_uncertainty_weighted_fusion_shape():
    windows = torch.randn(4, 2, 2, 1, 4)

    fused = fuse_overlapping_windows(windows, [0, 1], mode="uncertainty_weighted")

    assert fused.values.shape == (4, 3, 1, 4)


def test_sliding_window_rejects_non_contiguous_coverage():
    windows = torch.randn(1, 2, 2, 1, 1)

    with pytest.raises(ValueError, match="contiguous epoch range"):
        fuse_overlapping_windows(windows, [0, 3])


def test_mask_window_fusion_supports_any_and_mean():
    bool_windows = torch.tensor([[True, False], [False, True]])
    float_windows = torch.tensor([[1.0, 0.0], [0.5, 0.5]])

    any_fused = fuse_mask_windows(bool_windows, [0, 1], mode="any")
    mean_fused = fuse_mask_windows(float_windows, [0, 1], mode="mean")

    assert any_fused.values.tolist() == [True, False, True]
    assert mean_fused.values.tolist() == [1.0, 0.25, 0.5]


def test_validate_single_night_rejects_multiple_nights():
    with pytest.raises(ValueError, match="one subject/night"):
        validate_single_night(
            [
                {"subject_id": "s1", "night_id": "n1", "path": "a.npz"},
                {"subject_id": "s1", "night_id": "n2", "path": "b.npz"},
            ]
        )
