from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AnnotationSignal:
    canonical_channel: str
    data: np.ndarray
    sfreq: float
    raw_file: str
    raw_label: str
    unit: str | None = None
    steps: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AnnotationResult:
    signals: list[AnnotationSignal] = field(default_factory=list)


def read_stage_csv(
    path: str | Path,
    *,
    duration_sec: float,
    epoch_sec: float,
    mapping: dict[str, int],
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
        mapping=mapping,
        invalid=invalid,
    )
    return AnnotationSignal(
        canonical_channel=canonical_channel,
        data=data,
        sfreq=1.0 / float(epoch_sec),
        raw_file=str(path),
        raw_label=label_column,
        steps=[f"stage_csv:{epoch_sec:g}s"],
    )


def read_stage_edf_annotations(
    path: str | Path,
    *,
    duration_sec: float,
    epoch_sec: float,
    mapping: dict[str, int],
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
        mapping=mapping,
        invalid=invalid,
    )
    return AnnotationSignal(
        canonical_channel=canonical_channel,
        data=data,
        sfreq=1.0 / float(epoch_sec),
        raw_file=str(path),
        raw_label="edf_annotations",
        steps=[f"stage_edf_annotations:{epoch_sec:g}s"],
    )


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
    for label, start, duration in zip(labels, starts, durations):
        if start < 0 or duration <= 0:
            continue
        first = int(round(start / epoch_sec))
        count = int(round(duration / epoch_sec))
        if count < 1:
            continue
        if abs(first * epoch_sec - start) > 1e-6:
            raise ValueError(f"Stage annotation start={start:g} is not aligned to epoch_sec={epoch_sec:g}.")
        value = int(mapping.get(str(label), invalid))
        left = max(first, 0)
        right = min(first + count, n_epochs)
        if left < right:
            data[left:right] = value
    return data


def _import_mne() -> Any:
    import mne

    return mne
