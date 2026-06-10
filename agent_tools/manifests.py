from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import json_ready


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")


def write_text(path: str | Path, text: str, *, executable: bool = False) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    if executable:
        target.chmod(target.stat().st_mode | 0o111)
