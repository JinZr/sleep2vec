import itertools
import math
import random
import typing as t

import torch

from data.utils import load_npz


class PairBatchSampler(torch.utils.data.Sampler[list[int]]):
    """
    Batch sampler that groups samples by shared channel pairs to keep batch size stable
    when allow_missing_channels=True.
    """

    def __init__(self, *args, **kwargs) -> None:
        dataset = None
        sampler = None
        batch_size = kwargs.pop("batch_size", None)
        drop_last = kwargs.pop("drop_last", None)
        channel_names = kwargs.pop("channel_names", None)
        min_channels = kwargs.pop("min_channels", None)
        shuffle = kwargs.pop("shuffle", None)
        seed = kwargs.pop("seed", 42)
        replacement = kwargs.pop("replacement", True)

        if args:
            if len(args) == 1:
                dataset = args[0]
            elif len(args) == 3:
                sampler, batch_size, drop_last = args
            else:
                raise TypeError("PairBatchSampler expects (dataset, ...) or (sampler, batch_size, drop_last).")

        sampler = kwargs.pop("sampler", sampler)
        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected arguments: {unknown}")

        if dataset is None and sampler is not None and hasattr(sampler, "dataset"):
            dataset = sampler.dataset
        if dataset is None:
            raise ValueError("dataset must be provided for PairBatchSampler.")

        if batch_size is None:
            raise ValueError("batch_size must be provided for PairBatchSampler.")
        if drop_last is None:
            drop_last = False

        if channel_names is None:
            channel_names = getattr(dataset, "channel_names", None)
        if channel_names is None:
            raise ValueError("channel_names must be provided or present on dataset.")

        if min_channels is None:
            min_channels = getattr(dataset, "min_channels", 2)

        if shuffle is None:
            shuffle = bool(getattr(sampler, "shuffle", False))

        if hasattr(sampler, "seed"):
            seed = getattr(sampler, "seed")

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.channel_names = list(channel_names)
        self.channel_name_set = set(self.channel_names)
        self.min_channels = int(min_channels)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.replacement = bool(replacement)
        self.epoch = 0

        self._num_replicas, self._rank = self._get_dist(sampler)
        self._pair_to_indices = self._build_pair_index()
        self._pair_list = list(self._pair_to_indices.keys())
        self._pair_weights = [len(self._pair_to_indices[p]) for p in self._pair_list]
        if not self._pair_list:
            raise ValueError("No valid channel pairs found for batch sampling.")

        self._num_batches_total = self._compute_num_batches_total()

    def _get_dist(self, sampler) -> tuple[int, int]:
        if sampler is not None:
            num_replicas = getattr(sampler, "num_replicas", None)
            rank = getattr(sampler, "rank", None)
            if num_replicas is not None and rank is not None:
                return int(num_replicas), int(rank)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size(), torch.distributed.get_rank()
        return 1, 0

    def _available_channels(self, sample) -> list[str]:
        payload = getattr(sample, "payload", None)
        avail = None
        if isinstance(payload, dict):
            avail = payload.get("available_channels")
        if not avail:
            with load_npz(sample.path) as npz:
                avail = [ch for ch in self.channel_names if ch in npz]
            if isinstance(payload, dict):
                payload["available_channels"] = avail
        return sorted(set(avail) & self.channel_name_set)

    def _build_pair_index(self) -> dict[tuple[str, str], list[int]]:
        pair_to_indices: dict[tuple[str, str], list[int]] = {}
        for idx, sample in enumerate(self.dataset.data):
            avail = self._available_channels(sample)
            if len(avail) < self.min_channels:
                continue
            for pair in itertools.combinations(avail, 2):
                pair_to_indices.setdefault(pair, []).append(idx)
        return pair_to_indices

    def _compute_num_batches_total(self) -> int:
        valid_count = len(self.dataset.data)
        if valid_count == 0:
            return 0
        if self.drop_last:
            return valid_count // self.batch_size
        return math.ceil(valid_count / float(self.batch_size))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self._num_batches_total == 0:
            return 0
        base = self._num_batches_total // self._num_replicas
        extra = self._num_batches_total % self._num_replicas
        return base + (1 if self._rank < extra else 0)

    def __iter__(self):
        if self._num_batches_total == 0:
            return iter([])
        seed = self.seed + (self.epoch if self.shuffle else 0)
        rng = random.Random(seed)

        def sample_batch():
            pair = rng.choices(self._pair_list, weights=self._pair_weights, k=1)[0]
            bucket = self._pair_to_indices[pair]
            if len(bucket) >= self.batch_size:
                if self.replacement:
                    return [bucket[rng.randrange(len(bucket))] for _ in range(self.batch_size)]
                return rng.sample(bucket, self.batch_size)
            return [bucket[rng.randrange(len(bucket))] for _ in range(self.batch_size)]

        def iterator():
            for i in range(self._num_batches_total):
                batch = sample_batch()
                if i % self._num_replicas == self._rank:
                    yield batch

        return iterator()
