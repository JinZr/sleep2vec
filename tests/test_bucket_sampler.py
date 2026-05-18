from __future__ import annotations

from types import SimpleNamespace

import torch
from torch.utils.data.distributed import DistributedSampler

from data.default_dataset import DefaultDataset, SampleIndex
import data.samplers as samplers
from data.samplers import (
    AvailableChannelsBucketBatchSampler,
    WeightedRandomDistributedSampler,
    handles_distributed_sharding,
)


def _make_sample(sample_id: int, channels: list[str]):
    return SimpleNamespace(
        id=sample_id,
        path=f"/tmp/sample_{sample_id}.npz",
        payload={"available_channels": channels},
    )


def _make_dataset(
    *,
    allow_missing_channels: bool,
    is_train_set: bool,
    weighted_random_sampler: bool = False,
    weighted_random_sampler_target: str | None = None,
) -> DefaultDataset:
    dataset = object.__new__(DefaultDataset)
    dataset.channel_names = ["a", "b"]
    dataset.randomly_select_channels = False
    dataset.generative = False
    dataset.meta_data_names = []
    dataset.meta_data_regression_names = []
    dataset.allow_missing_channels = allow_missing_channels
    dataset.min_channels = 2
    dataset.bucket_by_available_channels = True
    dataset.train_pair_probs = None
    dataset.train_pair_track_unique_samples = False
    dataset.pair_selector = None
    dataset.weighted_random_sampler = weighted_random_sampler
    dataset.weighted_random_sampler_target = weighted_random_sampler_target
    dataset.extractors = {}
    dataset.tokenizers = {}
    dataset.mask_generators = {}
    dataset.dataloader_config = {"batch_size": 4, "shuffle": False, "num_workers": 0}
    dataset.seed = 0
    dataset.is_train_set = is_train_set
    dataset.data = [
        SampleIndex(
            id=i,
            path=f"/tmp/sample_{i}.npz",
            start=0,
            end=1,
            payload={"available_channels": ["a", "b"]},
            metadata={"src_isDep": int(i == 2)},
        )
        for i in range(3)
    ]
    return dataset


def test_bucket_sampler_eval_does_not_shard_small_loader_across_ranks(monkeypatch) -> None:
    monkeypatch.setattr(samplers, "get_rank_world_size", lambda: (0, 2))
    sampler = AvailableChannelsBucketBatchSampler(
        [_make_sample(i, ["a", "b"]) for i in range(3)],
        batch_size=4,
        min_channels=2,
        shuffle=False,
        drop_last=False,
        shard_across_ranks=False,
        seed=0,
    )

    batches = list(iter(sampler))

    assert handles_distributed_sharding(sampler) is False
    assert len(sampler) == 1
    assert batches == [[0, 1, 2]]


def test_default_dataset_eval_bucket_sampler_keeps_partial_batch() -> None:
    dataset = _make_dataset(allow_missing_channels=True, is_train_set=False)

    loader = dataset.dataloader()

    assert isinstance(loader.batch_sampler, AvailableChannelsBucketBatchSampler)
    assert loader.batch_sampler._drop_last is False
    assert handles_distributed_sharding(loader.batch_sampler) is False


def test_default_dataset_eval_plain_loader_does_not_drop_last() -> None:
    dataset = _make_dataset(allow_missing_channels=False, is_train_set=False)

    loader = dataset.dataloader()

    assert loader.drop_last is False


def test_default_dataset_builds_weighted_random_sampler_for_train_binary_metadata() -> None:
    dataset = _make_dataset(
        allow_missing_channels=False,
        is_train_set=True,
        weighted_random_sampler=True,
        weighted_random_sampler_target="src_isDep",
    )

    loader = dataset.dataloader()

    assert isinstance(loader.sampler, WeightedRandomDistributedSampler)
    assert isinstance(loader.sampler, DistributedSampler)
    assert handles_distributed_sharding(loader.sampler) is True
    assert loader.sampler.num_samples == 3
    assert loader.sampler.weights.tolist() == [0.5, 0.5, 1.0]


def test_default_dataset_does_not_build_weighted_random_sampler_for_eval() -> None:
    dataset = _make_dataset(
        allow_missing_channels=False,
        is_train_set=False,
        weighted_random_sampler=True,
        weighted_random_sampler_target="src_isDep",
    )

    loader = dataset.dataloader()

    assert not isinstance(loader.sampler, WeightedRandomDistributedSampler)


def test_weighted_random_distributed_sampler_shards_global_draws(monkeypatch) -> None:
    sampler = WeightedRandomDistributedSampler(torch.ones(5), num_samples=5, seed=11)

    monkeypatch.setattr(samplers, "get_rank_world_size", lambda: (0, 2))
    rank0 = list(iter(sampler))
    monkeypatch.setattr(samplers, "get_rank_world_size", lambda: (1, 2))
    sampler.set_epoch(0)
    rank1 = list(iter(sampler))

    generator = torch.Generator()
    generator.manual_seed(11)
    global_draws = torch.multinomial(
        torch.ones(5, dtype=torch.double), 6, replacement=True, generator=generator
    ).tolist()

    assert rank0 == global_draws[0::2]
    assert rank1 == global_draws[1::2]
    assert len(rank0) == len(rank1) == 3
