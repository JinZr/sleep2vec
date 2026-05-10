from __future__ import annotations

import json
from pathlib import Path
import typing as t


def load_downstream_metrics(path: str | Path | None) -> dict[str, t.Any]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("downstream metrics JSON must contain an object.")
    return payload


__all__ = ["load_downstream_metrics"]
