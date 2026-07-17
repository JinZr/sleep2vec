from __future__ import annotations

from pathlib import Path

import numpy as np


def npz_record_path(output_dir: Path, record_id: str) -> Path:
    return output_dir / "backends" / "npz" / "records" / f"{record_id}.npz"


def write_npz_record(output_dir: Path, record_id: str, arrays: dict[str, np.ndarray]) -> Path:
    path = npz_record_path(output_dir, record_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path
