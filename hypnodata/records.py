from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RecordTask:
    record_id: str
    center: str
    files: dict[str, Path]
    metadata: dict[str, Any] = field(default_factory=dict)
    source_row: dict[str, Any] = field(default_factory=dict)


def validate_record_id(record_id: str) -> None:
    if record_id in {"", ".", ".."} or "/" in record_id or "\\" in record_id:
        raise ValueError(f"record_id must be a single path-safe segment, got {record_id!r}.")
