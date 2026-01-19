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

    def __init__(
        self,
        dataset,
        *,
        batch_size: int,
        channel_names: t.Sequence[str] | None = None,
        min_channels: int | None = None,
        shuffle: bool | None = None,
        drop_last: bool,
        seed: int = 42,
        replacement: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if channel_names is None:
            channel_names = getattr(dataset, "channel_names", None)
        if channel_names is None:
            raise ValueError("channel_names must be provided or available on the dataset.")
        if min_channels is None:
            min_channels = getattr(dataset, "min_channels", None)
        if min_channels is None:
            raise ValueError("min_channels must be provided or available on the dataset.")
        if shuffle is None:
            dataloader_cfg = getattr(dataset, "dataloader_config", {}) or {}
            shuffle = dataloader_cfg.get("shuffle", False)
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.channel_names = list(channel_names)
        self.channel_name_set = set(self.channel_names)
        self.min_channels = int(min_channels)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.replacement = bool(replacement)
        self.epoch = 0

        self._num_replicas, self._rank = self._get_dist()
        self._pair_to_indices = self._build_pair_index()
        self._pair_list = list(self._pair_to_indices.keys())
        self._pair_weights = [len(self._pair_to_indices[p]) for p in self._pair_list]
        if not self._pair_list:
            raise ValueError("No valid channel pairs found for batch sampling.")

        self._num_batches_total = self._compute_num_batches_total()

    def _get_dist(self) -> tuple[int, int]:
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
        valid_count = len({idx for idxs in self._pair_to_indices.values() for idx in idxs})
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
