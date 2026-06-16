from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.signal import filtfilt, iirnotch, resample

from hypnodata.channels import ChannelSelection
from hypnodata.config import FilterStep, NotchStep, SignalSpec


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
    unit_scale = _unit_scale(selection.raw_unit, spec.target_unit, selection.canonical_channel)
    if unit_scale != 1.0:
        data = data * np.float32(unit_scale)
        steps.append(f"unit:{_unit_label(selection.raw_unit)}->{_unit_label(spec.target_unit)}")
    if spec.scale != 1.0:
        data = data * np.float32(spec.scale)
        steps.append(f"scale:{spec.scale:g}")
    if spec.polarity == -1:
        data = -data
        steps.append("polarity_flip")
    sfreq = float(selection.raw_sfreq)
    for step in spec.preprocess:
        if isinstance(step, NotchStep):
            data = _apply_notch(data, sfreq, step, selection.canonical_channel)
            steps.append(f"notch:{step.freq:g}Hz:q={step.q:g}")
        elif isinstance(step, FilterStep):
            data = _apply_filter(data, sfreq, step, selection.canonical_channel)
            steps.append(_filter_step_label(step))
    target_sfreq = float(spec.target_sfreq or sfreq)
    if abs(sfreq - target_sfreq) > 1e-6:
        target_len = int(round(len(data) * target_sfreq / sfreq))
        if target_len <= 0:
            raise ValueError(f"Signal {selection.canonical_channel!r} resampled to empty output.")
        data = resample(data, target_len).astype(np.float32, copy=False)
        sfreq = target_sfreq
        steps.append(f"resample:{selection.raw_sfreq:g}->{target_sfreq:g}")
    if not np.isfinite(data).all():
        raise ValueError(f"Signal {selection.canonical_channel!r} contains NaN or Inf after preprocessing.")
    steps.append("finite_check")
    return ProcessedSignal(
        data=np.ascontiguousarray(data, dtype=np.float32),
        sfreq=sfreq,
        unit=spec.target_unit,
        steps=steps,
    )


def _apply_filter(data: np.ndarray, sfreq: float, step: FilterStep, channel: str) -> np.ndarray:
    _validate_below_nyquist(step.lowcut, sfreq, channel, "filter lowcut")
    _validate_below_nyquist(step.highcut, sfreq, channel, "filter highcut")
    import neurokit2 as nk

    filtered = nk.signal_filter(
        data,
        sampling_rate=sfreq,
        lowcut=step.lowcut,
        highcut=step.highcut,
        method=step.method,
        order=step.order,
    )
    return np.asarray(filtered, dtype=np.float32)


def _apply_notch(data: np.ndarray, sfreq: float, step: NotchStep, channel: str) -> np.ndarray:
    _validate_below_nyquist(step.freq, sfreq, channel, "notch freq")
    b, a = iirnotch(w0=step.freq, Q=step.q, fs=sfreq)
    return filtfilt(b, a, data).astype(np.float32, copy=False)


def _validate_below_nyquist(value: float | None, sfreq: float, channel: str, name: str) -> None:
    if value is not None and value >= sfreq / 2:
        raise ValueError(f"Signal {channel!r} {name} must be below Nyquist ({sfreq / 2:g} Hz).")


def _unit_scale(raw_unit: str | None, target_unit: str | None, channel: str) -> float:
    target = _normalize_unit(target_unit)
    if target is None:
        return 1.0
    raw = _normalize_unit(raw_unit)
    if raw is None:
        raise ValueError(f"Signal {channel!r} target_unit={target_unit!r} requires a raw unit.")
    if raw == target:
        return 1.0
    voltage_scales = {"v": 1.0, "mv": 1e-3, "uv": 1e-6, "nv": 1e-9}
    if raw in voltage_scales and target in voltage_scales:
        return voltage_scales[raw] / voltage_scales[target]
    raise ValueError(f"Signal {channel!r} cannot convert raw unit {raw_unit!r} to target_unit {target_unit!r}.")


def _normalize_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    normalized = unit.strip().lower().replace("μ", "u").replace("µ", "u")
    return normalized or None


def _unit_label(unit: str | None) -> str:
    return "" if unit is None else unit.strip()


def _filter_step_label(step: FilterStep) -> str:
    if step.lowcut is not None and step.highcut is not None:
        mode = "bandpass"
        cutoff = f"{step.lowcut:g}-{step.highcut:g}Hz"
    elif step.lowcut is not None:
        mode = "highpass"
        cutoff = f"{step.lowcut:g}Hz"
    else:
        mode = "lowpass"
        cutoff = f"{step.highcut:g}Hz"
    return f"filter:{step.method}:{mode}:{cutoff}:order={step.order:g}"


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
