from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EdfSignalInfo:
    path: Path
    raw_label: str
    raw_index: int
    sfreq: float
    unit: str | None
    n_samples: int
    duration: float


@dataclass(frozen=True)
class EdfInventory:
    path: Path
    signals: list[EdfSignalInfo]
    duration: float
    warnings: list[str]


def read_edf_inventory(path: Path) -> EdfInventory:
    try:
        return _read_header_inventory(path)
    except Exception as exc:
        return _read_mne_inventory(path, warning=f"edf_header_parse_failed: {exc}")


def read_edf_signal(
    path: Path,
    raw_label: str,
    raw_unit: str | None = None,
    *,
    raw_index: int | None = None,
) -> np.ndarray:
    mne = _import_mne()
    raw = mne.io.read_raw_edf(path, preload=False, verbose=False, infer_types=False)
    if raw_index is None:
        if raw_label not in raw.ch_names:
            raise KeyError(f"EDF channel {raw_label!r} not found in {path}.")
        index = raw.ch_names.index(raw_label)
    else:
        index = raw_index
    data = raw.get_data(picks=[index])[0]
    raw.close()
    return np.asarray(data * _si_to_unit_scale(raw_unit), dtype=np.float32)


def _read_header_inventory(path: Path) -> EdfInventory:
    with path.open("rb") as file_obj:
        fixed = file_obj.read(256)
        if len(fixed) != 256:
            raise ValueError("EDF header is shorter than 256 bytes.")
        header_bytes = int(_decode(fixed[184:192]) or "0")
        n_records = int(_decode(fixed[236:244]) or "0")
        record_duration = float(_decode(fixed[244:252]) or "0")
        n_signals = int(_decode(fixed[252:256]) or "0")
        if header_bytes < 256 or n_records <= 0 or record_duration <= 0 or n_signals <= 0:
            raise ValueError("EDF header has invalid record or signal counts.")
        signal_header = file_obj.read(header_bytes - 256)
    expected_signal_header = 256 * n_signals
    if len(signal_header) < expected_signal_header:
        raise ValueError("EDF signal header is incomplete.")

    offset = 0
    labels = _read_str_list(signal_header, offset, n_signals, 16)
    offset += 16 * n_signals
    offset += 80 * n_signals
    units = _read_str_list(signal_header, offset, n_signals, 8)
    offset += 8 * n_signals
    offset += 8 * n_signals
    offset += 8 * n_signals
    offset += 8 * n_signals
    offset += 8 * n_signals
    offset += 80 * n_signals
    samples_per_record = [int(value or "0") for value in _read_str_list(signal_header, offset, n_signals, 8)]

    duration = n_records * record_duration
    signals = []
    for idx, (label, unit, per_record) in enumerate(zip(labels, units, samples_per_record)):
        sfreq = per_record / record_duration if record_duration else 0.0
        signals.append(
            EdfSignalInfo(
                path=path,
                raw_label=label,
                raw_index=idx,
                sfreq=sfreq,
                unit=unit or None,
                n_samples=per_record * n_records,
                duration=duration,
            )
        )
    return EdfInventory(path=path, signals=signals, duration=duration, warnings=[])


def _read_mne_inventory(path: Path, *, warning: str) -> EdfInventory:
    mne = _import_mne()
    raw = mne.io.read_raw_edf(path, preload=False, verbose=False, infer_types=False)
    duration = float(raw.n_times / raw.info["sfreq"]) if raw.info["sfreq"] else 0.0
    signals = []
    for idx, label in enumerate(raw.ch_names):
        sfreq = float(raw.info["sfreq"])
        signals.append(
            EdfSignalInfo(
                path=path,
                raw_label=str(label),
                raw_index=idx,
                sfreq=sfreq,
                unit=None,
                n_samples=int(raw.n_times),
                duration=duration,
            )
        )
    raw.close()
    return EdfInventory(path=path, signals=signals, duration=duration, warnings=[warning])


def _read_str_list(data: bytes, offset: int, count: int, width: int) -> list[str]:
    return [_decode(data[offset + idx * width : offset + (idx + 1) * width]) for idx in range(count)]


def _decode(data: bytes) -> str:
    return data.decode("ascii", errors="replace").strip()


def _si_to_unit_scale(unit: str | None) -> float:
    if unit is None:
        return 1.0
    normalized = unit.strip().lower().replace("μ", "u").replace("µ", "u")
    return {
        "v": 1.0,
        "mv": 1e3,
        "uv": 1e6,
        "nv": 1e9,
    }.get(normalized, 1.0)


def _import_mne() -> Any:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    import mne

    return mne
