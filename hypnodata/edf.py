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


@dataclass(frozen=True)
class _EdfHeader:
    header_bytes: int
    n_records: int
    record_duration: float
    labels: list[str]
    units: list[str]
    physical_mins: list[float]
    physical_maxs: list[float]
    digital_mins: list[float]
    digital_maxs: list[float]
    samples_per_record: list[int]


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
    native_error = None
    try:
        return _read_native_edf_signal(path, raw_label, raw_unit, raw_index=raw_index)
    except Exception as exc:
        native_error = exc
    expected_samples = _expected_native_samples(path, raw_label, raw_index=raw_index)
    data = _read_mne_signal(path, raw_label, raw_unit, raw_index=raw_index)
    if expected_samples is not None and len(data) != expected_samples:
        raise ValueError(
            f"EDF channel {raw_label!r} read {len(data)} samples but header declares "
            f"{expected_samples} native samples."
        ) from native_error
    return data


def _read_mne_signal(
    path: Path,
    raw_label: str,
    raw_unit: str | None,
    *,
    raw_index: int | None,
) -> np.ndarray:
    mne = _import_mne()
    raw = mne.io.read_raw_edf(path, preload=False, verbose=False, infer_types=False)
    try:
        if raw_index is None:
            if raw_label not in raw.ch_names:
                raise KeyError(f"EDF channel {raw_label!r} not found in {path}.")
            index = raw.ch_names.index(raw_label)
        else:
            index = raw_index
        data = raw.get_data(picks=[index])[0]
    finally:
        raw.close()
    return np.asarray(data * _si_to_unit_scale(raw_unit), dtype=np.float32)


def _read_header_inventory(path: Path) -> EdfInventory:
    header = _read_edf_header(path)
    duration = header.n_records * header.record_duration
    signals = []
    for idx, (label, unit, per_record) in enumerate(zip(header.labels, header.units, header.samples_per_record)):
        sfreq = per_record / header.record_duration
        signals.append(
            EdfSignalInfo(
                path=path,
                raw_label=label,
                raw_index=idx,
                sfreq=sfreq,
                unit=unit or None,
                n_samples=per_record * header.n_records,
                duration=duration,
            )
        )
    return EdfInventory(path=path, signals=signals, duration=duration, warnings=[])


def _read_edf_header(path: Path) -> _EdfHeader:
    if path.suffix.lower() == ".bdf":
        raise ValueError("Native EDF reader does not support BDF.")
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
    physical_mins = _read_float_list(signal_header, offset, n_signals, 8, "physical_min")
    offset += 8 * n_signals
    physical_maxs = _read_float_list(signal_header, offset, n_signals, 8, "physical_max")
    offset += 8 * n_signals
    digital_mins = _read_float_list(signal_header, offset, n_signals, 8, "digital_min")
    offset += 8 * n_signals
    digital_maxs = _read_float_list(signal_header, offset, n_signals, 8, "digital_max")
    offset += 8 * n_signals
    offset += 80 * n_signals
    samples_per_record = _read_int_list(signal_header, offset, n_signals, 8, "samples_per_record")
    if any(value <= 0 for value in samples_per_record):
        raise ValueError("EDF header has non-positive samples_per_record.")
    for idx, (physical_min, physical_max, digital_min, digital_max) in enumerate(
        zip(physical_mins, physical_maxs, digital_mins, digital_maxs)
    ):
        if physical_max == physical_min:
            raise ValueError(f"EDF channel {idx} has identical physical min/max.")
        if digital_max == digital_min:
            raise ValueError(f"EDF channel {idx} has identical digital min/max.")
    return _EdfHeader(
        header_bytes=header_bytes,
        n_records=n_records,
        record_duration=record_duration,
        labels=labels,
        units=units,
        physical_mins=physical_mins,
        physical_maxs=physical_maxs,
        digital_mins=digital_mins,
        digital_maxs=digital_maxs,
        samples_per_record=samples_per_record,
    )


def _read_native_edf_signal(
    path: Path,
    raw_label: str,
    raw_unit: str | None,
    *,
    raw_index: int | None,
) -> np.ndarray:
    header = _read_edf_header(path)
    index = _channel_index(header.labels, raw_label, raw_index, path)
    per_record = header.samples_per_record[index]
    record_samples = sum(header.samples_per_record)
    start_in_record = sum(header.samples_per_record[:index])
    output = np.empty(header.n_records * per_record, dtype=np.float32)
    gain = (header.physical_maxs[index] - header.physical_mins[index]) / (
        header.digital_maxs[index] - header.digital_mins[index]
    )
    offset = header.physical_mins[index] - header.digital_mins[index] * gain
    unit_scale = _physical_to_si_scale(header.units[index]) * _si_to_unit_scale(raw_unit)
    with path.open("rb") as file_obj:
        for record_idx in range(header.n_records):
            sample_offset = record_idx * record_samples + start_in_record
            file_obj.seek(header.header_bytes + sample_offset * 2)
            raw = file_obj.read(per_record * 2)
            if len(raw) != per_record * 2:
                raise ValueError(f"EDF channel {raw_label!r} has incomplete data record {record_idx}.")
            block = np.frombuffer(raw, dtype="<i2").astype(np.float32)
            left = record_idx * per_record
            output[left : left + per_record] = (block * gain + offset) * unit_scale
    return np.ascontiguousarray(output, dtype=np.float32)


def _expected_native_samples(path: Path, raw_label: str, *, raw_index: int | None) -> int | None:
    try:
        header = _read_edf_header(path)
        index = _channel_index(header.labels, raw_label, raw_index, path)
    except Exception:
        return None
    return header.n_records * header.samples_per_record[index]


def _channel_index(labels: list[str], raw_label: str, raw_index: int | None, path: Path) -> int:
    if raw_index is not None:
        if raw_index < 0 or raw_index >= len(labels):
            raise IndexError(f"EDF channel index {raw_index} is out of range for {path}.")
        return raw_index
    if raw_label not in labels:
        raise KeyError(f"EDF channel {raw_label!r} not found in {path}.")
    return labels.index(raw_label)


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


def _read_float_list(data: bytes, offset: int, count: int, width: int, field: str) -> list[float]:
    values = []
    for raw in _read_str_list(data, offset, count, width):
        value = float(raw)
        if not np.isfinite(value):
            raise ValueError(f"EDF header field {field} is not finite.")
        values.append(value)
    return values


def _read_int_list(data: bytes, offset: int, count: int, width: int, field: str) -> list[int]:
    values = []
    for raw in _read_str_list(data, offset, count, width):
        value = int(raw)
        values.append(value)
    if len(values) != count:
        raise ValueError(f"EDF header field {field} has invalid length.")
    return values


def _decode(data: bytes) -> str:
    return data.decode("ascii", errors="replace").strip()


def _physical_to_si_scale(unit: str | None) -> float:
    if unit is None:
        return 1.0
    normalized = unit.strip().lower().replace("μ", "u").replace("µ", "u")
    return {
        "v": 1.0,
        "mv": 1e-3,
        "uv": 1e-6,
        "nv": 1e-9,
    }.get(normalized, 1.0)


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
