from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ReplayRecord:
    task_type: str
    condition_modalities: tuple[str, ...]
    target_modalities: tuple[str, ...]


class ReplayBuffer:
    def __init__(self, maxlen: int = 1024) -> None:
        if maxlen <= 0:
            raise ValueError("maxlen must be positive.")
        self._items: deque[ReplayRecord] = deque(maxlen=maxlen)

    def append(self, record: ReplayRecord) -> None:
        self._items.append(record)

    def __len__(self) -> int:
        return len(self._items)

    def snapshot(self) -> list[ReplayRecord]:
        return list(self._items)


__all__ = ["ReplayBuffer", "ReplayRecord"]
