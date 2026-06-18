from __future__ import annotations

from types import SimpleNamespace

from torch.utils.data import BatchSampler
from torch.utils.data.distributed import DistributedSampler

from sleep2vec2.data.samplers import SequentialPairEvalBatchSampler


def _make_sample(sample_id: int, channels: list[str]):
    return SimpleNamespace(
        id=sample_id,
        path=f"/tmp/sample_{sample_id}.npz",
        payload={"available_channels": channels},
    )


class _Dataset:
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


def test_sleep2vec2_pair_eval_sampler_supports_lightning_distributed_rebuild() -> None:
    dataset = _Dataset()
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


def test_sleep2vec2_pair_eval_sampler_shards_pair_batches_evenly_after_lightning_rebuild() -> None:
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
