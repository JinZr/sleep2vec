from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

STAGE5_MAPPING = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
SLEEP_STAGE_VALUES = frozenset({1, 2, 3, 4})
ANNOTATION_MATERIALIZATIONS = {"stage", "event_table", "event_dense", "event_anchor", "ahi"}


@dataclass(frozen=True)
class AnnotationSignal:
    canonical_channel: str
    data: np.ndarray
    sfreq: float | None
    raw_file: str
    raw_label: str
    unit: str | None = None
    steps: list[str] = field(default_factory=list)
    materialization: str = "stage"
    output_key: str | None = None
    extra_outputs: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class AnnotationResult:
    signals: list[AnnotationSignal] = field(default_factory=list)


def read_stage_csv(
    path: str | Path,
    *,
    duration_sec: float,
    epoch_sec: float,
    mapping: dict[str, int] | None = None,
    invalid: int = -1,
    label_column: str = "stage",
    start_column: str = "start",
    duration_column: str = "duration",
    canonical_channel: str = "stage5",
) -> AnnotationSignal:
    frame = pd.read_csv(path, low_memory=False)
    required = {label_column, start_column, duration_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Stage CSV missing required column(s): {missing}")
    data = _materialize_stage_epochs(
        labels=frame[label_column].astype(str).tolist(),
        starts=frame[start_column].astype(float).tolist(),
        durations=frame[duration_column].astype(float).tolist(),
        duration_sec=duration_sec,
        epoch_sec=epoch_sec,
        mapping=STAGE5_MAPPING if mapping is None else mapping,
        invalid=invalid,
    )
    return AnnotationSignal(
        canonical_channel=canonical_channel,
        data=data,
        sfreq=1.0 / float(epoch_sec),
        raw_file=str(path),
        raw_label=label_column,
        materialization="stage",
        steps=[f"stage_csv:{epoch_sec:g}s"],
    )


def read_stage_edf_annotations(
    path: str | Path,
    *,
    duration_sec: float,
    epoch_sec: float,
    mapping: dict[str, int] | None = None,
    invalid: int = -1,
    canonical_channel: str = "stage5",
) -> AnnotationSignal:
    mne = _import_mne()
    raw = mne.io.read_raw_edf(path, preload=False, verbose=False, infer_types=False)
    annotations = raw.annotations
    labels = [str(item) for item in annotations.description]
    starts = [float(item) for item in annotations.onset]
    durations = [float(item) for item in annotations.duration]
    raw.close()
    data = _materialize_stage_epochs(
        labels=labels,
        starts=starts,
        durations=durations,
        duration_sec=duration_sec,
        epoch_sec=epoch_sec,
        mapping=STAGE5_MAPPING if mapping is None else mapping,
        invalid=invalid,
    )
    return AnnotationSignal(
        canonical_channel=canonical_channel,
        data=data,
        sfreq=1.0 / float(epoch_sec),
        raw_file=str(path),
        raw_label="edf_annotations",
        materialization="stage",
        steps=[f"stage_edf_annotations:{epoch_sec:g}s"],
    )


def read_event_csv(
    path: str | Path,
    *,
    type_column: str | None = "Type",
    start_column: str = "Start",
    duration_column: str = "Duration",
    mapping: dict[str, int] | None = None,
    default_type: int = 0,
) -> np.ndarray:
    frame = pd.read_csv(path, low_memory=False)
    required = {start_column, duration_column}
    if type_column is not None:
        required.add(type_column)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Event CSV missing required column(s): {missing}")
    starts = frame[start_column].astype(float).to_numpy()
    durations = frame[duration_column].astype(float).to_numpy()
    if type_column is None:
        types = np.full(len(frame), float(default_type), dtype=np.float32)
    elif mapping is None:
        types = frame[type_column].astype(float).to_numpy()
    else:
        mapped = frame[type_column].astype(str).map(mapping)
        if mapped.isna().any():
            unknown = sorted(set(frame.loc[mapped.isna(), type_column].astype(str)))
            raise ValueError(f"Event CSV contains unmapped type(s): {unknown}")
        types = mapped.astype(float).to_numpy()
    return _normalize_event_rows(np.column_stack([types, starts, durations]))


def materialize_event_table(
    events: np.ndarray,
    *,
    canonical_channel: str,
    raw_file: str = "",
    raw_label: str = "events",
    steps: list[str] | None = None,
) -> AnnotationSignal:
    return AnnotationSignal(
        canonical_channel=canonical_channel,
        data=_normalize_event_rows(events),
        sfreq=None,
        raw_file=raw_file,
        raw_label=raw_label,
        materialization="event_table",
        steps=list(steps or ["event_csv"]),
    )


def materialize_dense_events(
    events: np.ndarray,
    *,
    duration_sec: float,
    interval_sec: float,
    canonical_channel: str,
    raw_file: str = "",
    raw_label: str = "events",
    value: float | None = 1.0,
    steps: list[str] | None = None,
) -> AnnotationSignal:
    if interval_sec <= 0:
        raise ValueError("event dense interval_sec must be positive.")
    # Validate source extents before binning; clipped dense arrays would hide mismatched annotation files.
    rows = _normalize_event_rows(events, duration_sec=duration_sec)
    n_samples = int(np.floor(float(duration_sec) / float(interval_sec)))
    data = np.zeros(n_samples, dtype=np.float32)
    for event_type, start, duration in rows:
        stop = float(start + duration)
        if stop <= start:
            continue
        left = max(int(np.floor(float(start) / interval_sec)), 0)
        right = min(int(np.ceil(stop / interval_sec)), n_samples)
        if left < right:
            data[left:right] = float(event_type if value is None else value)
    return AnnotationSignal(
        canonical_channel=canonical_channel,
        data=data,
        sfreq=1.0 / float(interval_sec),
        raw_file=raw_file,
        raw_label=raw_label,
        materialization="event_dense",
        steps=list(steps or ["event_csv"]) + [f"dense:{interval_sec:g}s"],
    )


def materialize_anchor_events(
    events: np.ndarray,
    *,
    duration_sec: float,
    window_sec: float,
    anchor_num: int,
    canonical_channel: str,
    raw_file: str = "",
    raw_label: str = "events",
    steps: list[str] | None = None,
) -> AnnotationSignal:
    if window_sec <= 0:
        raise ValueError("event anchor window_sec must be positive.")
    if anchor_num <= 0:
        raise ValueError("event anchor anchor_num must be positive.")
    # Anchor labels can represent the final partial window, so overlong source rows must fail before filling.
    rows = _normalize_event_rows(events, duration_sec=duration_sec)
    n_windows = int(np.ceil(float(duration_sec) / float(window_sec)))
    data = np.zeros((n_windows, anchor_num * 3), dtype=np.float32)
    if n_windows:
        for _, start, duration in rows:
            _fill_anchor_event(data, float(start), float(duration), float(window_sec), anchor_num)
    return AnnotationSignal(
        canonical_channel=canonical_channel,
        data=data,
        sfreq=1.0 / float(window_sec),
        raw_file=raw_file,
        raw_label=raw_label,
        materialization="event_anchor",
        steps=list(steps or ["event_csv"]) + [f"anchor:{window_sec:g}s:{anchor_num:g}"],
    )


def materialize_ahi_from_events(
    events: np.ndarray,
    stage: np.ndarray,
    *,
    duration_sec: float,
    epoch_sec: float = 30.0,
    interval_sec: float = 1.0,
    canonical_channel: str = "ahi",
    raw_file: str = "",
    raw_label: str = "events",
    steps: list[str] | None = None,
) -> AnnotationSignal:
    if epoch_sec <= 0:
        raise ValueError("AHI epoch_sec must be positive.")
    if interval_sec <= 0:
        raise ValueError("AHI interval_sec must be positive.")
    stage_values = np.asarray(stage).reshape(-1)
    sleep_mask = np.isin(stage_values, list(SLEEP_STAGE_VALUES))
    tst_hours = float(sleep_mask.sum()) * float(epoch_sec) / 3600.0
    if tst_hours <= 0:
        raise ValueError("AHI output requires positive TST from sleep stages.")

    # Validate before sleep-stage filtering so invalid source rows cannot disappear before QC.
    rows = _normalize_event_rows(events, duration_sec=duration_sec)
    # AASM adult respiratory events must last at least 10 seconds.
    rows = rows[rows[:, 2] >= 10.0]
    rows = filter_events_to_sleep_stages(rows, stage_values, epoch_sec=epoch_sec)
    dense = materialize_dense_events(
        rows,
        duration_sec=duration_sec,
        interval_sec=interval_sec,
        canonical_channel=canonical_channel,
        raw_file=raw_file,
        raw_label=raw_label,
        value=1.0,
        steps=list(steps or ["event_csv"]) + ["aasm_min_duration:10s", "stage_sleep_filter"],
    )
    ahi = float(rows.shape[0]) / tst_hours
    return AnnotationSignal(
        canonical_channel=canonical_channel,
        data=dense.data,
        sfreq=dense.sfreq,
        raw_file=raw_file,
        raw_label=raw_label,
        materialization="ahi",
        steps=dense.steps + ["ahi_from_events"],
        output_key="ah_event",
        extra_outputs={
            "ahi": np.asarray(ahi, dtype=np.float32),
            "tst": np.asarray(tst_hours, dtype=np.float32),
        },
    )


def filter_events_to_sleep_stages(
    events: np.ndarray,
    stage: np.ndarray,
    *,
    epoch_sec: float,
    sleep_values: set[int] | frozenset[int] = SLEEP_STAGE_VALUES,
) -> np.ndarray:
    if epoch_sec <= 0:
        raise ValueError("stage filter epoch_sec must be positive.")
    rows = _normalize_event_rows(events)
    if rows.size == 0:
        return rows
    stage_values = np.asarray(stage).reshape(-1)
    kept = []
    for row in rows:
        _, start, duration = row
        stop = start + duration
        left = max(int(np.floor(float(start) / epoch_sec)), 0)
        right = min(int(np.ceil(float(stop) / epoch_sec)), stage_values.shape[0])
        if left < right and any(int(value) in sleep_values for value in stage_values[left:right]):
            kept.append(row)
    if not kept:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(kept, dtype=np.float32)


def _materialize_stage_epochs(
    *,
    labels: list[str],
    starts: list[float],
    durations: list[float],
    duration_sec: float,
    epoch_sec: float,
    mapping: dict[str, int],
    invalid: int,
) -> np.ndarray:
    if epoch_sec <= 0:
        raise ValueError("stage epoch_sec must be positive.")
    n_epochs = int(np.floor(float(duration_sec) / float(epoch_sec)))
    data = np.full(n_epochs, int(invalid), dtype=np.int64)
    filled = np.zeros(n_epochs, dtype=bool)
    tolerance = 1e-6
    for label, start, duration in zip(labels, starts, durations):
        # Validate row extents before epoch clipping; clipped stage rows would hide mismatched hypnograms.
        if not np.isfinite(start) or not np.isfinite(duration) or start < 0 or duration <= 0:
            raise ValueError(f"Stage annotation has invalid extent at start={start:g}, duration={duration:g}.")
        stop = start + duration
        if stop > float(duration_sec) + tolerance:
            raise ValueError(f"Stage annotation stop={stop:g} exceeds duration_sec={duration_sec:g}.")
        first = int(round(start / epoch_sec))
        if abs(first * epoch_sec - start) > tolerance:
            raise ValueError(f"Stage annotation start={start:g} is not aligned to epoch_sec={epoch_sec:g}.")
        last = int(round(stop / epoch_sec))
        # Stage labels are epoch-granular; reject partial rows instead of rounding them into full epochs.
        if abs(last * epoch_sec - stop) > tolerance:
            raise ValueError(f"Stage annotation stop={stop:g} is not aligned to epoch_sec={epoch_sec:g}.")
        count = last - first
        if count < 1:
            continue
        value = int(mapping.get(str(label), invalid))
        left = max(first, 0)
        right = min(first + count, n_epochs)
        if left < right:
            if filled[left:right].any():
                raise ValueError(f"Stage annotation overlaps an existing epoch at start={start:g}.")
            data[left:right] = value
            filled[left:right] = True
    return data


def _normalize_event_rows(events: np.ndarray, *, duration_sec: float | None = None) -> np.ndarray:
    rows = np.asarray(events, dtype=np.float32)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != 3:
        raise ValueError("event rows must have shape (N, 3).")
    starts = rows[:, 1]
    durations = rows[:, 2]
    stops = starts + durations
    invalid_extents = not np.isfinite(rows).all() or (starts < 0).any() or (durations <= 0).any()
    if invalid_extents:
        raise ValueError("event rows contain invalid extents.")
    if duration_sec is not None and (stops > float(duration_sec) + 1e-6).any():
        raise ValueError(f"event rows exceed record duration {float(duration_sec):g}s.")
    return rows.astype(np.float32, copy=False)


def _fill_anchor_event(
    labels: np.ndarray,
    start: float,
    duration: float,
    window_sec: float,
    anchor_num: int,
) -> None:
    stop = start + duration
    if stop <= start:
        return
    stop_for_bin = np.nextafter(stop, -np.inf)
    start_bin = max(int(start // window_sec), 0)
    stop_bin = min(int(stop_for_bin // window_sec), labels.shape[0] - 1)
    for window in range(start_bin, stop_bin + 1):
        left = start if window == start_bin else window * window_sec
        right = stop if window == stop_bin else (window + 1) * window_sec
        _write_anchor(labels, window, left - window * window_sec, right - window * window_sec, window_sec, anchor_num)


def _write_anchor(
    labels: np.ndarray,
    window: int,
    start_offset: float,
    stop_offset: float,
    window_sec: float,
    anchor_num: int,
) -> None:
    for idx in range(anchor_num):
        col = idx * 3
        if labels[window, col] == 0:
            labels[window, col : col + 3] = [
                1.0,
                max(start_offset, 0.0) / window_sec,
                min(stop_offset, window_sec) / window_sec,
            ]
            return


def _import_mne() -> Any:
    import mne

    return mne
