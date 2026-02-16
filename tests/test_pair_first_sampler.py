from types import SimpleNamespace

import pytest

from data.samplers import PairFirstBatchSampler


def _make_sample(sample_id: int, channels: list[str]):
    return SimpleNamespace(
        id=sample_id,
        path=f"/tmp/sample_{sample_id}.npz",
        payload={"available_channels": channels},
    )


def test_pair_first_sampler_requires_available_channels() -> None:
    bad_sample = SimpleNamespace(id=0, path="/tmp/x.npz", payload={})
    with pytest.raises(ValueError, match="available_channels"):
        PairFirstBatchSampler(
            [bad_sample],
            channel_names=["a", "b"],
            batch_size=1,
            min_channels=2,
            seed=0,
        )


def test_pair_first_sampler_fails_when_configured_pair_pool_is_empty() -> None:
    data = [_make_sample(i, ["a", "b"]) for i in range(16)]
    with pytest.raises(ValueError, match="empty sample pools"):
        PairFirstBatchSampler(
            data,
            channel_names=["a", "b", "c"],
            batch_size=4,
            min_channels=2,
            seed=0,
        )


def test_pair_first_sampler_emits_single_pair_batches() -> None:
    data = [_make_sample(i, ["a", "b", "c"]) for i in range(48)]
    sampler = PairFirstBatchSampler(
        data,
        channel_names=["a", "b", "c"],
        batch_size=4,
        min_channels=2,
        seed=7,
    )

    batches = list(iter(sampler))
    assert len(batches) == len(sampler)
    assert len(batches) > 0

    for batch in batches:
        assert len(batch) == 4
        pairs = {pair for _, pair in batch}
        assert len(pairs) == 1

    counts = sampler.get_last_epoch_counts()
    assert sum(counts.values()) == len(sampler)


def test_pair_first_sampler_reports_unique_samples_not_exceeding_pool() -> None:
    data = [_make_sample(i, ["a", "b", "c"]) for i in range(96)]
    sampler = PairFirstBatchSampler(
        data,
        channel_names=["a", "b", "c"],
        batch_size=8,
        min_channels=2,
        seed=17,
    )
    _ = list(iter(sampler))

    pools = sampler.get_pair_pool_sizes()
    uniques = sampler.get_last_epoch_unique_sample_counts()
    assert pools
    assert uniques
    assert set(pools.keys()) == set(uniques.keys())
    for pair, pool_size in pools.items():
        assert pool_size > 0
        assert 0 <= uniques[pair] <= pool_size


def test_pair_first_sampler_uniform_distribution_is_reasonable() -> None:
    data = [_make_sample(i, ["a", "b", "c"]) for i in range(600)]
    sampler = PairFirstBatchSampler(
        data,
        channel_names=["a", "b", "c"],
        batch_size=2,
        min_channels=2,
        seed=13,
    )
    _ = list(iter(sampler))

    counts = sampler.get_last_epoch_counts()
    target = sampler.get_target_distribution()
    total = float(sum(counts.values()))
    assert total > 0

    for pair, expected in target.items():
        ratio = counts[pair] / total
        assert abs(ratio - expected) < 0.1


def test_pair_first_sampler_shuffle_no_duplicates_when_pool_is_sufficient() -> None:
    data = [_make_sample(i, ["a", "b", "c"]) for i in range(96)]
    sampler = PairFirstBatchSampler(
        data,
        channel_names=["a", "b", "c"],
        batch_size=8,
        min_channels=2,
        seed=23,
        shuffle=True,
    )

    batches = list(iter(sampler))
    assert batches
    for batch in batches:
        indices = [idx for idx, _ in batch]
        assert len(indices) == 8
        assert len(set(indices)) == len(indices)


def test_pair_first_sampler_shuffle_fallback_allows_duplicates_when_pool_is_small() -> None:
    data = [
        _make_sample(0, ["a", "b"]),
        _make_sample(1, ["a", "b"]),
        _make_sample(2, ["a", "c"]),
        _make_sample(3, ["a", "c"]),
        _make_sample(4, ["b", "c"]),
        _make_sample(5, ["b", "c"]),
    ]
    sampler = PairFirstBatchSampler(
        data,
        channel_names=["a", "b", "c"],
        batch_size=4,
        min_channels=2,
        seed=29,
        shuffle=True,
    )

    batches = list(iter(sampler))
    assert batches
    for batch in batches:
        indices = [idx for idx, _ in batch]
        assert len(indices) == 4
        assert len(set(indices)) < len(indices)
