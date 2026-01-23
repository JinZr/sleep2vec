import itertools
import random
import typing as t

Pair = tuple[str, str]


def build_all_pairs(channel_names: t.Sequence[str]) -> list[Pair]:
    return list(itertools.combinations(channel_names, 2))


class PairSelector:
    def reset(self) -> None:  # pragma: no cover - default no-op
        return

    def select(self, available: t.Sequence[str]) -> list[str]:
        raise NotImplementedError


class RandomPairSelector(PairSelector):
    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random

    def select(self, available: t.Sequence[str]) -> list[str]:
        if len(available) < 2:
            raise ValueError(f"Need at least 2 channels, got {len(available)}")
        return self._rng.sample(list(available), k=2)


class RoundRobinPairSelector(PairSelector):
    def __init__(self, pairs: t.Sequence[Pair]) -> None:
        if not pairs:
            raise ValueError("RoundRobinPairSelector requires at least one pair.")
        self._pairs = [tuple(p) for p in pairs]
        self._idx = 0

    def reset(self) -> None:
        self._idx = 0

    def select(self, available: t.Sequence[str]) -> list[str]:
        if len(available) < 2:
            raise ValueError(f"Need at least 2 channels, got {len(available)}")
        available_set = set(available)
        start = self._idx
        while True:
            pair = self._pairs[self._idx % len(self._pairs)]
            self._idx += 1
            if pair[0] in available_set and pair[1] in available_set:
                return [pair[0], pair[1]]
            if self._idx - start >= len(self._pairs):
                raise ValueError("No scheduled pair available for current batch.")
