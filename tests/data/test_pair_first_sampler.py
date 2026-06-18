import inspect
from types import SimpleNamespace

import pytest
from torch.utils.data import BatchSampler
from torch.utils.data.distributed import DistributedSampler

from data.samplers import PairFirstBatchSampler, SequentialPairEvalBatchSampler


def _make_sample(sample_id: int, channels: list[str]):
    return SimpleNamespace(
        id=sample_id,
        path=f"/tmp/sample_{sample_id}.npz",
        payload={"available_channels": channels},
    )


class _EvalDataset:
    channel_names = ["a", "b", "c"]
    min_channels = 3

    def __init__(self) -> None:
        self.data = [_make_sample(i, list(self.channel_names)) for i in range(5)]
        self.data[0] = _make_sample(0, ["a", "b"])

    def __len__(self) -> int:
        return len(self.data)


class _UnevenPairDataset:
    channel_names = ["a", "b", "c"]
    min_channels = 2

    def __init__(self) -> None:
        self.data = [
            _make_sample(0, ["a", "b", "c"]),
            _make_sample(1, ["a", "b"]),
            _make_sample(2, ["a", "b", "c"]),
            _make_sample(3, ["a", "b"]),
        ]

    def __len__(self) -> int:
        return len(self.data)


class _SingleSampleDataset:
    channel_names = ["a", "b", "c"]
    min_channels = 2

    def __init__(self) -> None:
        self.data = [_make_sample(0, list(self.channel_names))]

    def __len__(self) -> int:
        return len(self.data)


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
        track_unique_sample_counts=True,
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


def test_pair_first_sampler_disables_unique_tracking_by_default() -> None:
    data = [_make_sample(i, ["a", "b", "c"]) for i in range(32)]
    sampler = PairFirstBatchSampler(
        data,
        channel_names=["a", "b", "c"],
        batch_size=4,
        min_channels=2,
        seed=11,
    )
    _ = list(iter(sampler))

    assert sampler.is_tracking_unique_sample_counts() is False
    assert sampler.get_last_epoch_unique_sample_counts() == {}


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


def test_pair_first_sampler_can_update_pair_probs() -> None:
    data = [_make_sample(i, ["a", "b", "c"]) for i in range(120)]
    sampler = PairFirstBatchSampler(
        data,
        channel_names=["a", "b", "c"],
        batch_size=2,
        min_channels=2,
        seed=31,
    )

    sampler.set_pair_probs({("a", "b"): 1.0, ("a", "c"): 0.0, ("b", "c"): 0.0})
    _ = list(iter(sampler))

    target = sampler.get_target_distribution()
    counts = sampler.get_last_epoch_counts()
    assert target[("a", "b")] == pytest.approx(1.0)
    assert target[("a", "c")] == pytest.approx(0.0)
    assert target[("b", "c")] == pytest.approx(0.0)
    assert counts[("a", "b")] == len(sampler)
    assert counts[("a", "c")] == 0
    assert counts[("b", "c")] == 0


def test_sequential_pair_eval_sampler_preserves_pair_order_and_tail_batches() -> None:
    data = [_make_sample(i, ["a", "b", "c"]) for i in range(5)]
    sampler = SequentialPairEvalBatchSampler(
        data,
        channel_names=["a", "b", "c"],
        batch_size=2,
        min_channels=2,
    )

    batches = list(iter(sampler))
    assert sampler.pairs == [("a", "b"), ("a", "c"), ("b", "c")]
    assert len(batches) == len(sampler) == 9

    observed_pairs = [{pair for _, pair in batch} for batch in batches]
    assert observed_pairs[:3] == [{("a", "b")}] * 3
    assert observed_pairs[3:6] == [{("a", "c")}] * 3
    assert observed_pairs[6:] == [{("b", "c")}] * 3
    assert [len(batch) for batch in batches] == [2, 2, 1, 2, 2, 1, 2, 2, 1]


def test_sequential_pair_eval_sampler_supports_lightning_distributed_rebuild() -> None:
    dataset = _EvalDataset()
    batch_sampler = SequentialPairEvalBatchSampler(
        dataset.data,
        channel_names=dataset.channel_names,
        batch_size=2,
        min_channels=dataset.min_channels,
    )
    distributed_sampler = DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=False)

    rebuilt = type(batch_sampler)(
        distributed_sampler,
        batch_size=batch_sampler.batch_size,
        drop_last=batch_sampler.drop_last,
    )

    assert isinstance(batch_sampler, BatchSampler)
    assert len(rebuilt) == 3
    assert rebuilt.pairs == [("a", "b"), ("a", "c"), ("b", "c")]
    assert list(rebuilt)[0] == [(1, ("a", "b")), (2, ("a", "b"))]


def test_sequential_pair_eval_sampler_exposes_lightning_sampler_argument() -> None:
    parameters = list(inspect.signature(SequentialPairEvalBatchSampler.__init__).parameters)
    assert parameters[1] == "sampler"


def test_sequential_pair_eval_sampler_shards_pair_batches_evenly_after_lightning_rebuild() -> None:
    dataset = _UnevenPairDataset()
    batch_sampler = SequentialPairEvalBatchSampler(
        dataset.data,
        channel_names=dataset.channel_names,
        batch_size=1,
        min_channels=dataset.min_channels,
    )

    rank0 = type(batch_sampler)(
        DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=False),
        batch_size=batch_sampler.batch_size,
        drop_last=batch_sampler.drop_last,
    )
    rank1 = type(batch_sampler)(
        DistributedSampler(dataset, num_replicas=2, rank=1, shuffle=False),
        batch_size=batch_sampler.batch_size,
        drop_last=batch_sampler.drop_last,
    )

    rank0_batches = list(rank0)
    rank1_batches = list(rank1)

    assert len(rank0) == len(rank1) == 4
    assert len(rank0_batches) == len(rank1_batches) == 4
    assert rank1_batches[-2:] == [[(2, ("a", "c"))], [(2, ("b", "c"))]]


def test_sequential_pair_eval_sampler_pads_pair_batches_evenly_after_lightning_rebuild() -> None:
    dataset = _SingleSampleDataset()
    batch_sampler = SequentialPairEvalBatchSampler(
        dataset.data,
        channel_names=dataset.channel_names,
        batch_size=1,
        min_channels=dataset.min_channels,
    )

    rank0 = type(batch_sampler)(
        DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=False),
        batch_size=batch_sampler.batch_size,
        drop_last=batch_sampler.drop_last,
    )
    rank1 = type(batch_sampler)(
        DistributedSampler(dataset, num_replicas=2, rank=1, shuffle=False),
        batch_size=batch_sampler.batch_size,
        drop_last=batch_sampler.drop_last,
    )

    assert len(rank0) == len(rank1) == 2
    assert list(rank0) == [[(0, ("a", "b"))], [(0, ("b", "c"))]]
    assert list(rank1) == [[(0, ("a", "c"))], [(0, ("a", "b"))]]


def test_sequential_pair_eval_sampler_fails_when_configured_pair_pool_is_empty() -> None:
    data = [_make_sample(i, ["a", "b"]) for i in range(8)]
    with pytest.raises(ValueError, match="empty sample pools"):
        SequentialPairEvalBatchSampler(
            data,
            channel_names=["a", "b", "c"],
            batch_size=2,
            min_channels=2,
        )
