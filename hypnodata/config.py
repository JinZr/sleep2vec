from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

TOP_LEVEL_KEYS = {"center", "record_discovery", "signals", "backend", "custom", "adapter_options"}
DISCOVERY_KEYS = {
    "type",
    "root",
    "pattern",
    "index",
    "record_id_column",
    "file_column",
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
    "candidates",
    "scale",
    "polarity",
    "preprocess",
}
CANDIDATE_KEYS = {"label", "regex", "priority"}
PREPROCESS_STEPS = {
    "scale",
    "polarity_flip",
    "resample",
    "finite_check",
    "truncate_to_common",
    "filter",
    "notch",
}


@dataclass(frozen=True)
class CandidateSpec:
    label: str | None = None
    regex: str | None = None
    priority: int = 0


@dataclass(frozen=True)
class SignalSpec:
    name: str
    kind: str
    required: bool
    target_sfreq: float | None
    target_unit: str | None
    candidates: list[CandidateSpec]
    scale: float = 1.0
    polarity: int = 1
    preprocess: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DiscoveryConfig:
    type: str
    root: Path | None = None
    pattern: str = "*.edf"
    index: Path | None = None
    record_id_column: str | None = None
    file_column: str = "path"
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
    custom: dict[str, Any] = field(default_factory=dict)
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
        custom=_optional_mapping(raw.get("custom"), "custom"),
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
    if file_columns and not data.get("record_id_column"):
        raise ValueError("record_discovery.file_columns requires record_id_column.")
    return DiscoveryConfig(
        type=discovery_type,
        root=None if data.get("root") is None else Path(data["root"]),
        pattern=str(data.get("pattern", "*.edf")),
        index=None if data.get("index") is None else Path(data["index"]),
        record_id_column=None if data.get("record_id_column") is None else str(data["record_id_column"]),
        file_column=str(data.get("file_column", "path")),
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
        if not candidates and kind != "stage":
            raise ValueError(f"signals.{canonical}.candidates must not be empty.")
        target_sfreq = None if spec.get("target_sfreq") is None else float(spec["target_sfreq"])
        if target_sfreq is not None and target_sfreq <= 0:
            raise ValueError(f"signals.{canonical}.target_sfreq must be positive when set.")
        preprocess_steps = _string_list(spec.get("preprocess", []), f"signals.{canonical}.preprocess")
        unknown_steps = sorted(set(preprocess_steps) - PREPROCESS_STEPS)
        if unknown_steps:
            raise ValueError(f"signals.{canonical}.preprocess has unsupported step(s): {unknown_steps}")
        signals[canonical] = SignalSpec(
            name=canonical,
            kind=kind,
            required=bool(spec.get("required", True)),
            target_sfreq=target_sfreq,
            target_unit=None if spec.get("target_unit") is None else str(spec["target_unit"]),
            candidates=candidates,
            scale=float(spec.get("scale", 1.0)),
            polarity=_parse_polarity(spec.get("polarity", 1), canonical),
            preprocess=preprocess_steps,
        )
    return signals


def _build_candidates(raw: Any, canonical: str) -> list[CandidateSpec]:
    items = _require_list(raw, f"signals.{canonical}.candidates")
    candidates: list[CandidateSpec] = []
    for idx, item_raw in enumerate(items):
        item = _require_mapping(item_raw, f"signals.{canonical}.candidates[{idx}]")
        _reject_unknown_keys(item, CANDIDATE_KEYS, f"signals.{canonical}.candidates[{idx}]")
        has_label = item.get("label") not in (None, "")
        has_regex = item.get("regex") not in (None, "")
        if has_label == has_regex:
            raise ValueError(f"signals.{canonical}.candidates[{idx}] must set exactly one of label or regex.")
        candidates.append(
            CandidateSpec(
                label=None if not has_label else str(item["label"]),
                regex=None if not has_regex else str(item["regex"]),
                priority=int(item.get("priority", 0)),
            )
        )
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


def _string_list(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a string or list of strings.")
    return [str(item) for item in value]
