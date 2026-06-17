from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import yaml

TOP_LEVEL_KEYS = {"center", "record_discovery", "signals", "backend", "adapter_options"}
DISCOVERY_KEYS = {
    "type",
    "root",
    "pattern",
    "index",
    "record_id_column",
    "file_columns",
    "source_column",
    "split_column",
    "subject_id_column",
    "session_id_column",
    "metadata",
    "metadata_columns",
    "adapter",
}
BACKEND_KEYS = {"type", "min_duration"}
SIGNAL_KEYS = {
    "kind",
    "required",
    "target_sfreq",
    "target_unit",
    "epoch_sec",
    "interval_sec",
    "window_sec",
    "candidates",
    "scale",
    "polarity",
    "preprocess",
}
FILTER_STEP_KEYS = {"type", "method", "order", "lowcut", "highcut"}
NOTCH_STEP_KEYS = {"type", "freq", "q"}
# YAML preprocess only contains user-configurable signal transforms.
# Fixed steps such as resampling, finite checks, and common truncation run in the pipeline.
PREPROCESS_TYPES = {"filter", "notch"}
ANNOTATION_ONLY_KINDS = {"stage", "event_table", "event_dense", "event_anchor", "ahi"}


@dataclass(frozen=True)
class FilterStep:
    method: str
    order: int
    lowcut: float | None = None
    highcut: float | None = None


@dataclass(frozen=True)
class NotchStep:
    freq: float
    q: float


PreprocessStep = FilterStep | NotchStep


@dataclass(frozen=True)
class SignalSpec:
    name: str
    kind: str
    required: bool
    target_sfreq: float | None
    target_unit: str | None
    candidates: list[str]
    epoch_sec: float | None = None
    interval_sec: float | None = None
    window_sec: float | None = None
    scale: float = 1.0
    polarity: int = 1
    preprocess: list[PreprocessStep] = field(default_factory=list)


@dataclass(frozen=True)
class DiscoveryConfig:
    type: str
    root: Path | None = None
    pattern: str = "*.edf"
    index: Path | None = None
    record_id_column: str | None = None
    file_columns: dict[str, str] = field(default_factory=dict)
    source_column: str | None = "source"
    split_column: str | None = "split"
    subject_id_column: str | None = "subject_id"
    session_id_column: str | None = "session_id"
    metadata: dict[str, Any] = field(default_factory=dict)
    metadata_columns: list[str] = field(default_factory=list)
    adapter: str | None = None


@dataclass(frozen=True)
class BackendConfig:
    type: str
    min_duration: float | None = None


@dataclass(frozen=True)
class HypnodataConfig:
    path: Path
    center: str
    record_discovery: DiscoveryConfig
    signals: dict[str, SignalSpec]
    backend: BackendConfig
    adapter_options: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> HypnodataConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Config {config_path} must contain a YAML mapping.")
    _reject_schema_version(raw, "config")
    _reject_unknown_keys(raw, TOP_LEVEL_KEYS, "top level")

    missing = sorted(key for key in ("center", "record_discovery", "signals", "backend") if key not in raw)
    if missing:
        raise ValueError(f"Missing required hypnodata config field(s): {missing}")

    center = str(raw["center"])
    if not center:
        raise ValueError("center must not be empty.")
    discovery = _build_discovery(raw["record_discovery"])
    signals = _build_signals(raw["signals"])
    backend = _build_backend(raw["backend"])
    return HypnodataConfig(
        path=config_path,
        center=center,
        record_discovery=discovery,
        signals=signals,
        backend=backend,
        adapter_options=_optional_mapping(raw.get("adapter_options"), "adapter_options"),
    )


def _build_discovery(raw: Any) -> DiscoveryConfig:
    data = _require_mapping(raw, "record_discovery")
    _reject_schema_version(data, "record_discovery")
    _reject_unknown_keys(data, DISCOVERY_KEYS, "record_discovery")
    discovery_type = str(data.get("type", ""))
    if discovery_type not in {"glob", "csv", "custom"}:
        raise ValueError("record_discovery.type must be one of: glob, csv, custom.")
    if discovery_type == "glob" and not data.get("root"):
        raise ValueError("record_discovery.root is required for type=glob.")
    if discovery_type == "csv" and not data.get("index"):
        raise ValueError("record_discovery.index is required for type=csv.")
    if discovery_type == "custom" and not data.get("adapter"):
        raise ValueError("record_discovery.adapter is required for type=custom.")
    file_columns = data.get("file_columns") or {}
    if not isinstance(file_columns, dict):
        raise ValueError("record_discovery.file_columns must be a mapping.")
    if discovery_type == "csv" and not file_columns:
        raise ValueError("record_discovery.file_columns is required for type=csv.")
    if file_columns and not data.get("record_id_column"):
        raise ValueError("record_discovery.file_columns requires record_id_column.")
    return DiscoveryConfig(
        type=discovery_type,
        root=None if data.get("root") is None else Path(data["root"]),
        pattern=str(data.get("pattern", "*.edf")),
        index=None if data.get("index") is None else Path(data["index"]),
        record_id_column=None if data.get("record_id_column") is None else str(data["record_id_column"]),
        file_columns={str(key): str(value) for key, value in file_columns.items()},
        source_column=_optional_column(data, "source_column", "source"),
        split_column=_optional_column(data, "split_column", "split"),
        subject_id_column=(_optional_column(data, "subject_id_column", "subject_id")),
        session_id_column=(_optional_column(data, "session_id_column", "session_id")),
        metadata=_optional_mapping(data.get("metadata"), "record_discovery.metadata"),
        metadata_columns=_string_list(data.get("metadata_columns", []), "record_discovery.metadata_columns"),
        adapter=None if data.get("adapter") is None else str(data["adapter"]),
    )


def _build_backend(raw: Any) -> BackendConfig:
    data = _require_mapping(raw, "backend")
    _reject_schema_version(data, "backend")
    _reject_unknown_keys(data, BACKEND_KEYS, "backend")
    backend_type = str(data.get("type", ""))
    if backend_type != "npz":
        raise ValueError("backend.type must be 'npz' for hypnodata v1.")
    min_duration = None if data.get("min_duration") is None else float(data["min_duration"])
    if min_duration is not None and min_duration <= 0:
        raise ValueError("backend.min_duration must be positive when set.")
    return BackendConfig(type=backend_type, min_duration=min_duration)


def _build_signals(raw: Any) -> dict[str, SignalSpec]:
    data = _require_mapping(raw, "signals")
    if not data:
        raise ValueError("signals must not be empty.")
    signals: dict[str, SignalSpec] = {}
    for name, raw_spec in data.items():
        canonical = str(name)
        spec = _require_mapping(raw_spec, f"signals.{canonical}")
        _reject_schema_version(spec, f"signals.{canonical}")
        _reject_unknown_keys(spec, SIGNAL_KEYS, f"signals.{canonical}")
        if not spec.get("kind"):
            raise ValueError(f"signals.{canonical}.kind is required.")
        candidates = _build_candidates(spec.get("candidates", []), canonical)
        kind = str(spec["kind"])
        annotation_only = kind in ANNOTATION_ONLY_KINDS
        if canonical == "ahi" and kind != "ahi":
            raise ValueError("signals.ahi must use kind=ahi.")
        if annotation_only and candidates:
            raise ValueError(f"signals.{canonical}.candidates must be empty for kind={kind}.")
        if not annotation_only and not candidates:
            allowed = ", ".join(sorted(ANNOTATION_ONLY_KINDS))
            raise ValueError(f"signals.{canonical}.candidates must not be empty unless kind is one of: {allowed}.")
        epoch_sec, interval_sec, window_sec = _build_annotation_timing(spec, canonical, kind, annotation_only)
        target_sfreq = None if spec.get("target_sfreq") is None else float(spec["target_sfreq"])
        if target_sfreq is not None and target_sfreq <= 0:
            raise ValueError(f"signals.{canonical}.target_sfreq must be positive when set.")
        target_unit = None if spec.get("target_unit") is None else str(spec["target_unit"])
        if annotation_only and target_sfreq is not None:
            raise ValueError(
                f"signals.{canonical}.target_sfreq is not used for annotation-only signals; "
                "use epoch_sec, interval_sec, or window_sec."
            )
        # Annotation outputs are adapter-provided labels, not raw waveforms to be converted or preprocessed.
        raw_only_fields = ["target_unit", "scale", "polarity", "preprocess"]
        for field in raw_only_fields:
            if annotation_only and spec.get(field) is not None:
                raise ValueError(f"signals.{canonical}.{field} is only valid for raw signals.")
        required = spec.get("required", True)
        if not isinstance(required, bool):
            raise ValueError(f"signals.{canonical}.required must be a boolean.")
        preprocess_steps = _build_preprocess_steps(spec.get("preprocess", []), canonical)
        signals[canonical] = SignalSpec(
            name=canonical,
            kind=kind,
            required=required,
            target_sfreq=target_sfreq,
            target_unit=target_unit,
            candidates=candidates,
            epoch_sec=epoch_sec,
            interval_sec=interval_sec,
            window_sec=window_sec,
            scale=float(spec.get("scale", 1.0)),
            polarity=_parse_polarity(spec.get("polarity", 1), canonical),
            preprocess=preprocess_steps,
        )
    _validate_builtin_ahi_contract(signals)
    return signals


def declared_target_sfreq(spec: SignalSpec) -> float | None:
    if spec.epoch_sec is not None:
        return 1.0 / spec.epoch_sec
    if spec.interval_sec is not None:
        return 1.0 / spec.interval_sec
    if spec.window_sec is not None:
        return 1.0 / spec.window_sec
    return spec.target_sfreq


def _build_annotation_timing(
    spec: dict[str, Any],
    canonical: str,
    kind: str,
    annotation_only: bool,
) -> tuple[float | None, float | None, float | None]:
    timing_keys = {"epoch_sec", "interval_sec", "window_sec"}
    if not annotation_only:
        unexpected = sorted(key for key in timing_keys if spec.get(key) is not None)
        if unexpected:
            raise ValueError(f"signals.{canonical}.{unexpected[0]} is only valid for annotation-only signals.")
        return None, None, None

    allowed_by_kind = {
        "stage": "epoch_sec",
        "event_dense": "interval_sec",
        "event_anchor": "window_sec",
        "event_table": None,
        "ahi": "interval_sec",
    }
    expected = allowed_by_kind[kind]
    unexpected = sorted(key for key in timing_keys if key != expected and spec.get(key) is not None)
    if unexpected:
        raise ValueError(f"signals.{canonical}.{unexpected[0]} is not valid for kind={kind}.")
    if expected is None:
        return None, None, None
    value = _optional_positive_float(spec, expected, f"signals.{canonical}")
    if value is None:
        raise ValueError(f"signals.{canonical}.{expected} is required for kind={kind}.")
    if expected == "epoch_sec":
        return value, None, None
    if expected == "interval_sec":
        return None, value, None
    return None, None, value


def _validate_builtin_ahi_contract(signals: dict[str, SignalSpec]) -> None:
    ahi_names = [name for name, spec in signals.items() if spec.kind == "ahi"]
    if not ahi_names:
        return
    if ahi_names != ["ahi"]:
        raise ValueError("kind=ahi must be declared as signals.ahi.")
    ahi = signals["ahi"]
    if ahi.interval_sec is None or not math.isclose(float(ahi.interval_sec), 1.0):
        raise ValueError("signals.ahi.interval_sec must be 1 for built-in AHI output.")
    if "ah_event" in signals:
        raise ValueError("signals.ah_event cannot be declared with signals.ahi because AHI writes NPZ key 'ah_event'.")
    stage = signals.get("stage5")
    if stage is None or stage.kind != "stage":
        raise ValueError("signals.ahi requires signals.stage5 with kind=stage.")
    if stage.epoch_sec is None or not math.isclose(float(stage.epoch_sec), 30.0):
        raise ValueError("signals.ahi requires signals.stage5.epoch_sec to be 30.")


def _build_preprocess_steps(raw: Any, canonical: str) -> list[PreprocessStep]:
    name = f"signals.{canonical}.preprocess"
    if raw is None:
        return []
    items = _require_list(raw, name)
    steps: list[PreprocessStep] = []
    for idx, item_raw in enumerate(items):
        item_name = f"{name}[{idx}]"
        item = _require_mapping(item_raw, item_name)
        step_type = str(item.get("type", ""))
        if step_type == "filter":
            steps.append(_build_filter_step(item, item_name))
        elif step_type == "notch":
            steps.append(_build_notch_step(item, item_name))
        else:
            raise ValueError(f"{item_name}.type must be one of: {', '.join(sorted(PREPROCESS_TYPES))}.")
    return steps


def _build_filter_step(item: dict[str, Any], name: str) -> FilterStep:
    _reject_unknown_keys(item, FILTER_STEP_KEYS, name)
    method = _required_string(item, "method", name)
    if method not in {"bessel", "butterworth"}:
        raise ValueError(f"{name}.method must be one of: bessel, butterworth.")
    order = _required_positive_int(item, "order", name)
    lowcut = _optional_positive_float(item, "lowcut", name)
    highcut = _optional_positive_float(item, "highcut", name)
    if lowcut is None and highcut is None:
        raise ValueError(f"{name} must set at least one of lowcut or highcut.")
    if lowcut is not None and highcut is not None and lowcut >= highcut:
        raise ValueError(f"{name}.lowcut must be smaller than highcut.")
    return FilterStep(method=method, order=order, lowcut=lowcut, highcut=highcut)


def _build_notch_step(item: dict[str, Any], name: str) -> NotchStep:
    _reject_unknown_keys(item, NOTCH_STEP_KEYS, name)
    return NotchStep(
        freq=_required_positive_float(item, "freq", name),
        q=_required_positive_float(item, "q", name),
    )


def _build_candidates(raw: Any, canonical: str) -> list[str]:
    items = _require_list(raw, f"signals.{canonical}.candidates")
    candidates: list[str] = []
    for idx, item_raw in enumerate(items):
        if not isinstance(item_raw, str) or item_raw == "":
            raise ValueError(f"signals.{canonical}.candidates[{idx}] must be a non-empty string.")
        candidates.append(item_raw)
    return candidates


def _parse_polarity(value: Any, canonical: str) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"normal", "positive", "1", "+1"}:
            return 1
        if normalized in {"invert", "inverted", "flip", "-1", "negative"}:
            return -1
    if value in {1, 1.0, True}:
        return 1
    if value in {-1, -1.0}:
        return -1
    raise ValueError(f"signals.{canonical}.polarity must be normal/1 or invert/-1.")


def _optional_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    return _require_mapping(value, name)


def _optional_column(data: dict[str, Any], key: str, default: str) -> str | None:
    if key in data and data[key] is None:
        return None
    return str(data.get(key, default))


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list.")
    return value


def _reject_unknown_keys(data: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown hypnodata config field(s) under {name}: {unknown}")


def _reject_schema_version(data: dict[str, Any], name: str) -> None:
    forbidden = sorted(key for key in data if str(key) in {"schema_version", "version_schema"})
    if forbidden:
        raise ValueError(f"{name} must not define schema/version field(s): {forbidden}")


def _required_string(data: dict[str, Any], key: str, name: str) -> str:
    if key not in data or data[key] is None:
        raise ValueError(f"{name}.{key} is required.")
    value = str(data[key])
    if not value:
        raise ValueError(f"{name}.{key} must not be empty.")
    return value


def _required_positive_int(data: dict[str, Any], key: str, name: str) -> int:
    if key not in data or data[key] is None:
        raise ValueError(f"{name}.{key} is required.")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name}.{key} must be a positive integer.")
    return value


def _required_positive_float(data: dict[str, Any], key: str, name: str) -> float:
    if key not in data or data[key] is None:
        raise ValueError(f"{name}.{key} is required.")
    return _positive_float(data[key], f"{name}.{key}")


def _optional_positive_float(data: dict[str, Any], key: str, name: str) -> float | None:
    if key not in data or data[key] is None:
        return None
    return _positive_float(data[key], f"{name}.{key}")


def _positive_float(value: Any, name: str) -> float:
    parsed = float(value)
    if parsed <= 0 or not math.isfinite(parsed):
        raise ValueError(f"{name} must be a positive number.")
    return parsed


def _string_list(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a string or list of strings.")
    return [str(item) for item in value]
