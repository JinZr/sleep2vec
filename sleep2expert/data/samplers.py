"""Custom samplers/batch samplers used by Sleep2Vec datasets.

Why this exists


The pretrain recipe can enable ``allow_missing_channels=True`` (via CLI) to
support datasets with heterogeneous montages.

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

from array import array
from collections import defaultdict
import itertools
import math
import random
import typing as t

import torch
from torch.utils.data import BatchSampler, Sampler
from torch.utils.data.distributed import DistributedSampler

from sleep2expert.distributed import get_rank_world_size

Pair = tuple[str, str]


def handles_distributed_sharding(sampler: t.Any) -> bool:
    """Return True when a sampler already shards across ranks."""
    if sampler is None:
        return False
    flag = getattr(sampler, "handles_distributed_sharding", None)
    if flag is not None:
        return bool(flag)
    return isinstance(sampler, DistributedShardedBatchSampler)


def _resolve_available_channel_set(src: t.Any, *, allowed: set[str], sample_idx: int) -> set[str]:
    payload = getattr(src, "payload", None)
    if not isinstance(payload, dict) or "available_channels" not in payload:
        sample_id = getattr(src, "id", sample_idx)
        sample_path = getattr(src, "path", "?")
        raise ValueError(
            "Pair samplers require payload['available_channels'] for every sample. "
            f"Missing at sample id={sample_id}, path={sample_path}."
        )

    avail = payload.get("available_channels")
    if not isinstance(avail, (list, tuple, set)):
        sample_id = getattr(src, "id", sample_idx)
        raise ValueError(
            "payload['available_channels'] must be a list/tuple/set. "
            f"Got type={type(avail).__name__} for sample id={sample_id}."
        )
    return {str(ch) for ch in avail if str(ch) in allowed}


def _build_pair_index_pools(
    data: t.Sequence[t.Any],
    *,
    channel_names: t.Sequence[str],
    min_channels: int,
) -> tuple[list[Pair], dict[Pair, array], int]:
    all_pairs: list[Pair] = list(itertools.combinations([str(ch) for ch in channel_names], 2))
    max_index = max(0, len(data) - 1)
    index_typecode = "I" if max_index <= 0xFFFFFFFF else "Q"
    pair_to_indices: dict[Pair, array] = {pair: array(index_typecode) for pair in all_pairs}
    eligible_indices: set[int] = set()
    allowed = {str(ch) for ch in channel_names}

    for i, src in enumerate(data):
        avail_set = _resolve_available_channel_set(src, allowed=allowed, sample_idx=i)
        if len(avail_set) < min_channels:
            continue

        matched = False
        for pair in all_pairs:
            if pair[0] in avail_set and pair[1] in avail_set:
                pair_to_indices[pair].append(i)
                matched = True
        if matched:
            eligible_indices.add(i)

    empty_pairs = [pair for pair in all_pairs if not pair_to_indices[pair]]
    if empty_pairs:
        preview = ", ".join(f"{a}__{b}" for a, b in empty_pairs[:8])
        suffix = " ..." if len(empty_pairs) > 8 else ""
        raise ValueError(
            "Configured pairs have empty sample pools. "
            f"empty_pairs={len(empty_pairs)}/{len(all_pairs)} [{preview}{suffix}]. "
            "Check channel_names, payload['available_channels'], and min_channels consistency."
        )

    eligible_size = len(eligible_indices)
    if eligible_size == 0:
        raise ValueError("Pair samplers found 0 eligible samples.")

    return all_pairs, pair_to_indices, eligible_size


class DistributedShardedBatchSampler(Sampler[list[int]]):
    """Marker base class for batch samplers that already shard by rank."""

    handles_distributed_sharding = True


class WeightedRandomDistributedSampler(DistributedSampler):
    handles_distributed_sharding = True

    def __init__(self, weights: torch.Tensor, num_samples: int, *, seed: int = 0) -> None:
        weights = torch.as_tensor(weights, dtype=torch.double)
        if weights.ndim != 1:
            raise ValueError("weights must be a 1D tensor.")
        if len(weights) == 0:
            raise ValueError("weights must not be empty.")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")
        if float(weights.sum().item()) <= 0.0:
            raise ValueError("weights must contain at least one positive value.")

        super().__init__(range(len(weights)), num_replicas=1, rank=0, shuffle=False, seed=int(seed))
        self.weights = weights
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0
        self._manual_epoch = False

    def __len__(self) -> int:
        _, world_size = get_rank_world_size()
        return int(math.ceil(self.num_samples / float(world_size)))

    def __iter__(self):
        rank, world_size = get_rank_world_size()
        local_samples = int(math.ceil(self.num_samples / float(world_size)))
        total_samples = local_samples * world_size
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        if not self._manual_epoch:
            self.epoch += 1

        indices = torch.multinomial(self.weights, total_samples, replacement=True, generator=generator).tolist()
        return iter(indices[rank:total_samples:world_size])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._manual_epoch = True


class AvailableChannelsBucketBatchSampler(DistributedShardedBatchSampler):
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
        shard_across_ranks: bool = True,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        self._data = data
        self._batch_size = int(batch_size)
        self._min_channels = int(min_channels)
        self._shuffle = bool(shuffle)
        self._drop_last = bool(drop_last)
        self._shard_across_ranks = bool(shard_across_ranks)
        self._seed = int(seed)
        self._epoch = 0
        self._manual_epoch = False
        self.handles_distributed_sharding = self._shard_across_ranks

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

        if self._shard_across_ranks:
            total_batches = (total_batches // world_size) * world_size
        if total_batches == 0:
            raise ValueError(
                "Bucketed sampler produced 0 batches after truncation. "
                "Try lowering batch_size or disabling drop_last."
            )
        return total_batches

    def __len__(self) -> int:
        if self._shard_across_ranks:
            _, world_size = get_rank_world_size()
        else:
            world_size = 1
        total_batches = self._compute_total_batches(world_size)
        return total_batches // world_size

    def __iter__(self):
        if self._shard_across_ranks:
            rank, world_size = get_rank_world_size()
        else:
            rank, world_size = 0, 1
        total_batches = self._compute_total_batches(world_size)
        # Keep shuffle order identical across ranks; sharding happens via modulo.
        rng = random.Random(self._seed + 1000 * self._epoch)
        if not self._manual_epoch:
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

    def set_epoch(self, epoch: int) -> None:
        """Optionally set epoch for deterministic shuffling across ranks."""
        self._epoch = int(epoch)
        self._manual_epoch = True


class PairFirstBatchSampler(DistributedShardedBatchSampler):
    """Batch sampler that first draws a channel pair, then samples indices from its pool.

    The dataset elements must provide ``payload['available_channels']``. When this
    field is missing, initialization fails immediately to enforce config/data
    strictness.
    """

    def __init__(
        self,
        data: t.Sequence[t.Any],
        *,
        channel_names: t.Sequence[str],
        batch_size: int,
        min_channels: int = 2,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
        pair_sampling: str = "uniform",
        pair_probs: dict[Pair, float] | None = None,
        track_unique_sample_counts: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if min_channels < 2:
            raise ValueError(f"min_channels must be >= 2, got {min_channels}")
        if len(channel_names) < 2:
            raise ValueError(f"Need at least 2 channels to build pairs, got {len(channel_names)}")
        if pair_sampling != "uniform":
            raise ValueError(f"Unsupported pair_sampling={pair_sampling!r}. Supported: 'uniform'")

        self._data = data
        self._channel_names = [str(ch) for ch in channel_names]
        self._batch_size = int(batch_size)
        self._min_channels = int(min_channels)
        self._shuffle = bool(shuffle)
        self._drop_last = bool(drop_last)
        self._seed = int(seed)
        self._pair_sampling = pair_sampling
        self._track_unique_sample_counts = bool(track_unique_sample_counts)
        self._epoch = 0
        self._manual_epoch = False
        self._last_epoch_pair_counter: dict[Pair, int] = {}
        self._last_epoch_unique_sample_counts: dict[Pair, int] = {}
        self._pair_pool_sizes: dict[Pair, int] = {}

        all_pairs, pair_to_indices, eligible_size = _build_pair_index_pools(
            data,
            channel_names=self._channel_names,
            min_channels=self._min_channels,
        )
        self._pair_to_indices = pair_to_indices
        self._pairs = list(all_pairs)
        self._pair_pool_sizes = {pair: len(pair_to_indices[pair]) for pair in self._pairs}
        self._eligible_size = eligible_size
        self._pair_probs = self._resolve_pair_probs(pair_probs)

    def _resolve_pair_probs(self, pair_probs: dict[Pair, float] | None) -> list[float]:
        if pair_probs is None:
            uniform = 1.0 / float(len(self._pairs))
            return [uniform for _ in self._pairs]

        raw: list[float] = []
        for pair in self._pairs:
            prob = pair_probs.get(pair, pair_probs.get((pair[1], pair[0]), 0.0))
            raw.append(float(prob))
        if any(p < 0.0 for p in raw):
            raise ValueError("pair_probs cannot contain negative values.")
        total = float(sum(raw))
        if total <= 0.0:
            raise ValueError("pair_probs sum must be > 0 for active pair pools.")
        return [p / total for p in raw]

    def _raw_total_batches(self) -> int:
        full_batches = self._eligible_size // self._batch_size
        if self._drop_last:
            return full_batches
        remainder = self._eligible_size % self._batch_size
        return full_batches + (1 if remainder else 0)

    def _compute_total_batches(self, world_size: int) -> int:
        total_batches = self._raw_total_batches()
        total_batches = (total_batches // world_size) * world_size
        if total_batches == 0:
            raise ValueError(
                "PairFirstBatchSampler produced 0 batches after truncation. "
                "Try lowering batch_size or world_size, or disabling drop_last."
            )
        return total_batches

    def __len__(self) -> int:
        _, world_size = get_rank_world_size()
        total_batches = self._compute_total_batches(world_size)
        return total_batches // world_size

    def __iter__(self):
        rank, world_size = get_rank_world_size()
        total_batches = self._compute_total_batches(world_size)
        raw_total_batches = self._raw_total_batches()
        remainder = self._eligible_size % self._batch_size
        remainder_idx = raw_total_batches - 1 if (not self._drop_last and remainder > 0) else -1

        rng = random.Random(self._seed + 1000 * self._epoch)
        if not self._manual_epoch:
            self._epoch += 1

        local_counts = {pair: 0 for pair in self._pairs}
        self._last_epoch_pair_counter = {pair: 0 for pair in self._pairs}
        if self._track_unique_sample_counts:
            local_unique: dict[Pair, set[int]] = {pair: set() for pair in self._pairs}
            self._last_epoch_unique_sample_counts = {pair: 0 for pair in self._pairs}
        else:
            local_unique = {}
            self._last_epoch_unique_sample_counts = {}
        for global_batch_idx in range(total_batches):
            pair = rng.choices(self._pairs, weights=self._pair_probs, k=1)[0]
            pool = self._pair_to_indices[pair]
            pool_size = len(pool)
            current_bs = self._batch_size
            if global_batch_idx == remainder_idx:
                current_bs = remainder
            if self._shuffle:
                if pool_size >= current_bs:
                    # Prefer without-replacement sampling to avoid duplicate indices
                    # within one batch when the pair pool can satisfy batch_size.
                    positions = rng.sample(range(pool_size), k=current_bs)
                else:
                    # Fallback for undersized pools: keep replacement sampling so
                    # batches stay full-sized.
                    positions = rng.choices(range(pool_size), k=current_bs)
                batch_indices = [int(pool[pos]) for pos in positions]
            else:
                start = (global_batch_idx * current_bs) % pool_size
                batch_indices = [int(pool[(start + j) % pool_size]) for j in range(current_bs)]
            batch = [(idx, pair) for idx in batch_indices]

            if global_batch_idx % world_size == rank:
                local_counts[pair] += 1
                if self._track_unique_sample_counts:
                    local_unique[pair].update(int(x) for x in batch_indices)
                self._last_epoch_pair_counter[pair] = local_counts[pair]
                if self._track_unique_sample_counts:
                    self._last_epoch_unique_sample_counts[pair] = len(local_unique[pair])
                yield batch

        self._last_epoch_pair_counter = local_counts
        if self._track_unique_sample_counts:
            self._last_epoch_unique_sample_counts = {pair: len(local_unique[pair]) for pair in self._pairs}
        else:
            self._last_epoch_unique_sample_counts = {}
        return

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self._manual_epoch = True

    @property
    def pairs(self) -> list[Pair]:
        return list(self._pairs)

    def get_target_distribution(self) -> dict[Pair, float]:
        return {pair: prob for pair, prob in zip(self._pairs, self._pair_probs)}

    def set_pair_probs(self, pair_probs: dict[Pair, float] | None) -> None:
        self._pair_probs = self._resolve_pair_probs(pair_probs)

    def get_last_epoch_counts(self) -> dict[Pair, int]:
        return dict(self._last_epoch_pair_counter)

    def get_last_epoch_unique_sample_counts(self) -> dict[Pair, int]:
        return dict(self._last_epoch_unique_sample_counts)

    def get_pair_pool_sizes(self) -> dict[Pair, int]:
        return dict(self._pair_pool_sizes)

    def is_tracking_unique_sample_counts(self) -> bool:
        return self._track_unique_sample_counts


class SequentialPairEvalBatchSampler(BatchSampler):
    handles_distributed_sharding = False

    def __init__(
        self,
        data: t.Sequence[t.Any] | Sampler[int],
        *,
        channel_names: t.Sequence[str] | None = None,
        batch_size: int,
        min_channels: int = 2,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        dataset = getattr(data, "dataset", None)
        if dataset is None:
            resolved_data = data
            index_sampler = range(len(resolved_data))
        else:
            resolved_data = getattr(dataset, "data", None)
            if resolved_data is None:
                raise ValueError("Injected sampler dataset must expose a 'data' attribute.")
            index_sampler = data

        if channel_names is None:
            channel_names = getattr(dataset, "channel_names", None) if dataset is not None else None
        if channel_names is None:
            raise ValueError("channel_names must be provided when source is not a dataset sampler.")

        if dataset is not None and min_channels == 2:
            min_channels = int(getattr(dataset, "min_channels", min_channels))
        else:
            min_channels = int(min_channels)
        if min_channels < 2:
            raise ValueError(f"min_channels must be >= 2, got {min_channels}")
        if len(channel_names) < 2:
            raise ValueError(f"Need at least 2 channels to build pairs, got {len(channel_names)}")

        super().__init__(index_sampler, int(batch_size), bool(drop_last))
        self._pairs, self._pair_to_indices, _ = _build_pair_index_pools(
            resolved_data,
            channel_names=channel_names,
            min_channels=int(min_channels),
        )
        self._pair_to_index_sets = {pair: set(indices) for pair, indices in self._pair_to_indices.items()}

    def __len__(self) -> int:
        pair_batches = 0
        active_indices = list(self.sampler)
        for pair in self._pairs:
            pair_indices = self._pair_to_index_sets[pair]
            pair_size = sum(1 for idx in active_indices if int(idx) in pair_indices)
            if self.drop_last:
                pair_batches += pair_size // self.batch_size
            else:
                pair_batches += math.ceil(pair_size / self.batch_size)
        return pair_batches

    def __iter__(self):
        active_indices = [int(idx) for idx in self.sampler]
        for pair in self._pairs:
            pair_indices = self._pair_to_index_sets[pair]
            indices = [idx for idx in active_indices if idx in pair_indices]
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield [(idx, pair) for idx in batch]

    @property
    def pairs(self) -> list[Pair]:
        return list(self._pairs)
