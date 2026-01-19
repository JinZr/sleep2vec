"""Custom samplers/batch samplers used by Sleep2Vec datasets.

Why this exists


The pretrain recipe enables ``allow_missing_channels=True`` to support datasets
with heterogeneous montages.

With the legacy setup (``shuffle=True``), a random batch often mixes samples
with different available-channel sets. The intersection of channels across that
batch can be empty, and the collate_fn falls back to a single "best pair". This
effectively turns training into a much easier sub-task (a single dominant pair),
causing train contrastive accuracy to saturate quickly while validation stays
low/noisy.

The batch sampler below buckets samples by their exact available-channel
signature, so every batch comes from a homogeneous montage. This keeps the
per-batch channel intersection large, eliminating the "best_pair" collapse.
"""

from __future__ import annotations

from collections import defaultdict
import math
import random
import typing as t

import torch
from torch.utils.data import Sampler


def _get_dist_info() -> tuple[int, int]:
    """Return (rank, world_size) for torch.distributed when initialized."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


class AvailableChannelsBucketBatchSampler(Sampler[list[int]]):
    """Batch-sampler that buckets indices by available-channel signature.

    The dataset elements are expected to be SampleIndex objects that may carry
    ``payload['available_channels']`` (populated by filter_valid_sample_indices).

    Notes
    -----
    - This sampler is batch-level: it yields lists of indices.
    - It is distributed-aware: each rank yields a disjoint subset of batches.
    - It truncates the total number of batches to a multiple of world_size to
      keep steps identical across ranks (drops a small tail).
    """

    def __init__(
        self,
        data: t.Sequence[t.Any],
        *,
        batch_size: int,
        min_channels: int = 2,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        self._data = data
        self._batch_size = int(batch_size)
        self._min_channels = int(min_channels)
        self._shuffle = bool(shuffle)
        self._drop_last = bool(drop_last)
        self._seed = int(seed)
        self._epoch = 0

        buckets: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for i, src in enumerate(data):
            payload = getattr(src, "payload", None)
            avail = None
            if isinstance(payload, dict):
                avail = payload.get("available_channels")
            if not avail:
                continue
            sig = tuple(sorted(str(ch) for ch in avail))
            if len(sig) < self._min_channels:
                continue
            buckets[sig].append(i)

        if not buckets:
            raise ValueError(
                "AvailableChannelsBucketBatchSampler found no buckets. "
                "Ensure preset stores payload['available_channels'] and min_channels is sensible."
            )

        self._buckets = dict(buckets)

    def _compute_total_batches(self, world_size: int) -> int:
        total_batches = 0
        for idxs in self._buckets.values():
            n = len(idxs)
            if self._drop_last:
                total_batches += n // self._batch_size
            else:
                total_batches += math.ceil(n / self._batch_size)

        total_batches = (total_batches // world_size) * world_size
        if total_batches == 0:
            raise ValueError(
                "Bucketed sampler produced 0 batches after truncation. "
                "Try lowering batch_size or disabling drop_last."
            )
        return total_batches

    def __len__(self) -> int:
        _, world_size = _get_dist_info()
        total_batches = self._compute_total_batches(world_size)
        return total_batches // world_size

    def __iter__(self):
        rank, world_size = _get_dist_info()
        total_batches = self._compute_total_batches(world_size)
        rng = random.Random(self._seed + 1000 * self._epoch + rank)
        self._epoch += 1

        bucket_items = list(self._buckets.items())
        if self._shuffle:
            rng.shuffle(bucket_items)

        global_batch_idx = 0

        for _, idxs in bucket_items:
            idxs_local = list(idxs)
            if self._shuffle:
                rng.shuffle(idxs_local)

            n = len(idxs_local)
            end = n - (n % self._batch_size) if self._drop_last else n
            for start in range(0, end, self._batch_size):
                if global_batch_idx >= total_batches:
                    return
                batch = idxs_local[start : start + self._batch_size]
                if len(batch) < self._batch_size and self._drop_last:
                    continue

                if global_batch_idx % world_size == rank:
                    yield batch
                global_batch_idx += 1

        return
