from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.signal import resample

from hypnodata.channels import ChannelSelection
from hypnodata.config import SignalSpec


@dataclass(frozen=True)
class ProcessedSignal:
    data: np.ndarray
    sfreq: float
    unit: str | None
    steps: list[str]


def preprocess_signal(raw: np.ndarray, selection: ChannelSelection, spec: SignalSpec) -> ProcessedSignal:
    if selection.raw_sfreq is None or selection.raw_sfreq <= 0 or not math.isfinite(selection.raw_sfreq):
        raise ValueError(f"Signal {selection.canonical_channel!r} has invalid sfreq: {selection.raw_sfreq}")
    data = np.asarray(raw, dtype=np.float32).reshape(-1)
    steps: list[str] = []
    if spec.scale != 1.0:
        data = data * np.float32(spec.scale)
        steps.append(f"scale:{spec.scale:g}")
    if spec.polarity == -1:
        data = -data
        steps.append("polarity_flip")
    sfreq = float(selection.raw_sfreq)
    target_sfreq = float(spec.target_sfreq or sfreq)
    if abs(sfreq - target_sfreq) > 1e-6:
        target_len = int(round(len(data) * target_sfreq / sfreq))
        if target_len <= 0:
            raise ValueError(f"Signal {selection.canonical_channel!r} resampled to empty output.")
        data = resample(data, target_len).astype(np.float32, copy=False)
        sfreq = target_sfreq
        steps.append(f"resample:{selection.raw_sfreq:g}->{target_sfreq:g}")
    for step in spec.preprocess:
        if step in {"filter", "notch"}:
            steps.append(f"{step}:not_implemented")
    if not np.isfinite(data).all():
        raise ValueError(f"Signal {selection.canonical_channel!r} contains NaN or Inf after preprocessing.")
    steps.append("finite_check")
    return ProcessedSignal(
        data=np.ascontiguousarray(data, dtype=np.float32),
        sfreq=sfreq,
        unit=spec.target_unit,
        steps=steps,
    )


def truncate_to_common(signals: dict[str, ProcessedSignal]) -> tuple[dict[str, ProcessedSignal], float, list[str]]:
    if not signals:
        raise ValueError("No available signals to write.")
    durations = {name: len(signal.data) / signal.sfreq for name, signal in signals.items()}
    common_duration = min(durations.values())
    if common_duration <= 0:
        raise ValueError("Common duration is empty after preprocessing.")
    output: dict[str, ProcessedSignal] = {}
    changed = []
    for name, signal in signals.items():
        target_len = int(math.floor(common_duration * signal.sfreq))
        if target_len <= 0:
            raise ValueError(f"Signal {name!r} is empty after truncate_to_common.")
        steps = list(signal.steps)
        steps.append("truncate_to_common")
        if target_len < len(signal.data):
            changed.append(name)
        output[name] = ProcessedSignal(
            data=np.ascontiguousarray(signal.data[:target_len], dtype=np.float32),
            sfreq=signal.sfreq,
            unit=signal.unit,
            steps=steps,
        )
    return output, common_duration, changed
