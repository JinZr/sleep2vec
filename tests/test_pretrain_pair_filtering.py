from types import SimpleNamespace

import pytest

from sleep2vec.utils import _filter_dataset_for_pair_support


def _sample(sample_id: int, channels: list[str]):
    return SimpleNamespace(
        id=sample_id,
        path=f"/tmp/sample_{sample_id}.npz",
        payload={"available_channels": channels},
    )


def test_filter_dataset_for_pair_support_keeps_only_matching_samples() -> None:
    dataset = SimpleNamespace(
        data=[
            _sample(0, ["a", "b", "c"]),
            _sample(1, ["a", "c"]),
            _sample(2, ["a", "b"]),
            _sample(3, ["b", "c"]),
        ]
    )

    _filter_dataset_for_pair_support(dataset, ("a", "b"), ["a", "b", "c"])

    kept_ids = [sample.id for sample in dataset.data]
    assert kept_ids == [0, 2]


def test_filter_dataset_for_pair_support_raises_when_none_match() -> None:
    dataset = SimpleNamespace(
        data=[
            _sample(0, ["a", "c"]),
            _sample(1, ["b", "c"]),
        ]
    )

    with pytest.raises(ValueError, match="No validation samples support scheduled pair"):
        _filter_dataset_for_pair_support(dataset, ("a", "b"), ["a", "b", "c"])
