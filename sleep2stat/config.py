from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_ANALYZER_TYPES = {
    "sleep2vec_downstream",
    "npz_stage_reference",
    "yasa_stage",
    "yasa_bandpower",
    "yasa_spindles",
    "yasa_slowwaves",
    "yasa_rem",
    "yasa_hrv_stage",
    "spo2_summary",
    "spo2_desaturation",
    "event_related_hypoxic_burden",
}
SUPPORTED_REDUCER_TYPES = {
    "hypnogram_stats",
    "transition_stats",
    "stage_agreement",
    "respiratory_stats",
    "demographic_consistency",
    "event_density",
    "stage_specific_summary",
}
YASA_STAGE_FILTER_LABELS = {"W", "N1", "N2", "N3", "REM"}
TOP_LEVEL_KEYS = {"run", "data", "signals", "analyzers", "reducers", "outputs"}
RUN_KEYS = {"name", "output_dir", "skip_existing", "seed"}
DATA_KEYS = {
    "backend",
    "index",
    "split",
    "path_column",
    "duration_column",
    "split_column",
    "source_column",
    "record_id_columns",
    "metadata_columns",
    "kaldi_data_root",
    "kaldi_manifest",
    "token_sec",
    "max_tokens",
}
SIGNALS_KEYS = {"channels"}
CHANNEL_KEYS = {"source", "sfreq", "kind", "input_dim", "unit", "scale", "mne_name"}
ANALYZER_KEYS = {
    "name",
    "type",
    "enabled",
    "namespace",
    "label_name",
    "config",
    "ckpt_path",
    "input_channels",
    "batch_size",
    "stage_key",
    "threshold",
    "postprocess",
    "stage_source",
    "stages",
    "artifact",
    "drop_thresholds",
    "min_duration_sec",
    "max_duration_sec",
    "event_source",
    "outputs",
}
REDUCER_KEYS = {
    "name",
    "type",
    "enabled",
    "source",
    "left",
    "right",
    "age_prediction",
    "sex_prediction",
    "metadata_age_column",
    "metadata_sex_column",
    "options",
}
OUTPUTS_KEYS = {
    "write_global_tables",
    "write_per_record",
    "include_probabilities",
    "include_raw_logits",
    "compression",
    "global_tables",
}
GLOBAL_TABLE_KEYS = {"epoch_alignment", "second_alignment", "event_alignment", "night_stats"}


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: Path
    skip_existing: bool = True
    seed: int = 4523


@dataclass(frozen=True)
class DataConfig:
    backend: str
    index: Path | None
    split: list[str]
    path_column: str
    duration_column: str
    split_column: str
    token_sec: int
    max_tokens: int
    source_column: str | None = "source"
    record_id_columns: list[str] = field(default_factory=list)
    metadata_columns: list[str] = field(default_factory=list)
    kaldi_data_root: Path | None = None
    kaldi_manifest: Path | None = None


@dataclass(frozen=True)
class ChannelSpec:
    source: str
    sfreq: float
    kind: str
    input_dim: int
    unit: str | None = None
    scale: float = 1.0
    mne_name: str | None = None


@dataclass(frozen=True)
class SignalsConfig:
    channels: dict[str, ChannelSpec]


@dataclass(frozen=True)
class AnalyzerConfig:
    name: str
    type: str
    enabled: bool = True
    namespace: str | None = None
    label_name: str | None = None
    config: Path | None = None
    ckpt_path: Path | None = None
    input_channels: list[str] = field(default_factory=list)
    batch_size: int | None = None
    stage_key: str | None = None
    threshold: Any = None
    postprocess: dict[str, Any] = field(default_factory=dict)
    stage_source: str | None = None
    stages: list[str] = field(default_factory=list)
    artifact: dict[str, Any] = field(default_factory=dict)
    drop_thresholds: list[float] = field(default_factory=list)
    min_duration_sec: float | None = None
    max_duration_sec: float | None = None
    event_source: str | None = None
    outputs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReducerConfig:
    name: str
    type: str
    enabled: bool = True
    source: str | None = None
    left: str | None = None
    right: str | None = None
    age_prediction: str | None = None
    sex_prediction: str | None = None
    metadata_age_column: str = "age"
    metadata_sex_column: str = "sex"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutputsConfig:
    write_global_tables: bool = True
    write_per_record: bool = True
    include_probabilities: bool = True
    include_raw_logits: bool = False
    compression: str = "gzip"
    global_tables: dict[str, bool] = field(
        default_factory=lambda: {
            "epoch_alignment": False,
            "second_alignment": False,
            "event_alignment": True,
            "night_stats": True,
        }
    )


@dataclass(frozen=True)
class Sleep2statConfig:
    path: Path
    run: RunConfig
    data: DataConfig
    signals: SignalsConfig
    analyzers: list[AnalyzerConfig]
    reducers: list[ReducerConfig]
    outputs: OutputsConfig


def load_config(path: str | Path) -> Sleep2statConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Config {config_path} must contain a YAML mapping.")

    unknown = sorted(set(raw) - TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(f"Unknown sleep2stat top-level config field(s): {unknown}")

    missing = sorted(TOP_LEVEL_KEYS - set(raw))
    if missing:
        raise ValueError(f"Missing required sleep2stat config block(s): {missing}")

    run_cfg = _build_run_config(raw["run"])
    data_cfg = _build_data_config(raw["data"])
    signals_cfg = _build_signals_config(raw["signals"])
    analyzers = _build_analyzers(raw["analyzers"], signals_cfg)
    _validate_backend_analyzer_support(data_cfg, analyzers)
    reducers = _build_reducers(raw["reducers"])
    _validate_reducer_references(analyzers, reducers)
    outputs = _build_outputs(raw["outputs"])
    return Sleep2statConfig(config_path, run_cfg, data_cfg, signals_cfg, analyzers, reducers, outputs)


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
        raise ValueError(f"Unknown sleep2stat config field(s) under {name}: {unknown}")


def _string_list(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a string or list of strings.")
    return [str(item) for item in value]


def _build_run_config(raw: Any) -> RunConfig:
    data = _require_mapping(raw, "run")
    _reject_unknown_keys(data, RUN_KEYS, "run")
    if not data.get("name"):
        raise ValueError("run.name is required.")
    if not data.get("output_dir"):
        raise ValueError("run.output_dir is required.")
    return RunConfig(
        name=str(data["name"]),
        output_dir=Path(data["output_dir"]),
        skip_existing=bool(data.get("skip_existing", True)),
        seed=int(data.get("seed", 4523)),
    )


def _build_data_config(raw: Any) -> DataConfig:
    data = _require_mapping(raw, "data")
    _reject_unknown_keys(data, DATA_KEYS, "data")
    required_fields = ("backend", "path_column", "duration_column", "split_column", "token_sec", "max_tokens")
    missing_fields = [key for key in required_fields if key not in data or data[key] in (None, "")]
    if missing_fields:
        raise ValueError(f"data missing required field(s): {missing_fields}")
    backend = str(data["backend"])
    if backend not in {"npz", "kaldi"}:
        raise ValueError(f"Unsupported sleep2stat data.backend: {backend!r}. Expected 'npz' or 'kaldi'.")
    if backend == "npz" and not data.get("index"):
        raise ValueError("data.index is required for data.backend=npz.")
    if backend == "kaldi":
        missing = [key for key in ("kaldi_data_root", "kaldi_manifest") if not data.get(key)]
        if missing:
            raise ValueError(f"data.backend=kaldi requires field(s): {missing}")
    token_sec = int(data["token_sec"])
    max_tokens = int(data["max_tokens"])
    if token_sec <= 0:
        raise ValueError("data.token_sec must be positive.")
    if max_tokens <= 0:
        raise ValueError("data.max_tokens must be positive.")
    split = _string_list(data.get("split", []), "data.split")
    if not split:
        raise ValueError("data.split is required and must not be empty for sleep2stat.")
    return DataConfig(
        backend=backend,
        index=None if data.get("index") is None else Path(data["index"]),
        split=split,
        path_column=str(data["path_column"]),
        duration_column=str(data["duration_column"]),
        split_column=str(data["split_column"]),
        source_column=None if data.get("source_column") is None else str(data.get("source_column", "source")),
        record_id_columns=_string_list(data.get("record_id_columns", []), "data.record_id_columns"),
        metadata_columns=_string_list(data.get("metadata_columns", []), "data.metadata_columns"),
        kaldi_data_root=None if data.get("kaldi_data_root") is None else Path(data["kaldi_data_root"]),
        kaldi_manifest=None if data.get("kaldi_manifest") is None else Path(data["kaldi_manifest"]),
        token_sec=token_sec,
        max_tokens=max_tokens,
    )


def _build_signals_config(raw: Any) -> SignalsConfig:
    data = _require_mapping(raw, "signals")
    _reject_unknown_keys(data, SIGNALS_KEYS, "signals")
    raw_channels = _require_mapping(data.get("channels"), "signals.channels")
    if not raw_channels:
        raise ValueError("signals.channels must not be empty.")
    channels: dict[str, ChannelSpec] = {}
    for name, raw_spec in raw_channels.items():
        spec = _require_mapping(raw_spec, f"signals.channels.{name}")
        _reject_unknown_keys(spec, CHANNEL_KEYS, f"signals.channels.{name}")
        missing = [key for key in ("source", "sfreq", "kind", "input_dim") if key not in spec]
        if missing:
            raise ValueError(f"signals.channels.{name} missing required field(s): {missing}")
        channels[str(name)] = ChannelSpec(
            source=str(spec["source"]),
            sfreq=float(spec["sfreq"]),
            kind=str(spec["kind"]),
            input_dim=int(spec["input_dim"]),
            unit=None if spec.get("unit") is None else str(spec["unit"]),
            scale=float(spec.get("scale", 1.0)),
            mne_name=None if spec.get("mne_name") is None else str(spec["mne_name"]),
        )
        if channels[str(name)].input_dim <= 0:
            raise ValueError(f"signals.channels.{name}.input_dim must be positive.")
    return SignalsConfig(channels)


def _build_analyzers(raw: Any, signals: SignalsConfig) -> list[AnalyzerConfig]:
    analyzers = []
    for idx, raw_item in enumerate(_require_list(raw, "analyzers")):
        item = _require_mapping(raw_item, f"analyzers[{idx}]")
        _reject_unknown_keys(item, ANALYZER_KEYS, f"analyzers[{idx}]")
        name = str(item.get("name", ""))
        analyzer_type = str(item.get("type", ""))
        if not name:
            raise ValueError(f"analyzers[{idx}].name is required.")
        if analyzer_type not in SUPPORTED_ANALYZER_TYPES:
            raise ValueError(f"Unknown sleep2stat analyzer type: {analyzer_type!r}.")
        input_channels = _string_list(item.get("input_channels", []), f"analyzers[{idx}].input_channels")
        missing_channels = sorted(channel for channel in input_channels if channel not in signals.channels)
        if missing_channels:
            raise ValueError(f"Analyzer {name!r} references unknown signal channel(s): {missing_channels}")
        postprocess = _require_mapping(item.get("postprocess") or {}, f"analyzers[{idx}].postprocess")
        if "threshold" in postprocess:
            raise ValueError(f"Analyzer {name!r} uses legacy postprocess.threshold; set threshold instead.")
        if analyzer_type == "sleep2vec_downstream":
            required = ["namespace", "label_name", "config", "ckpt_path", "input_channels"]
            missing = [key for key in required if not item.get(key)]
            if missing:
                raise ValueError(f"Analyzer {name!r} missing required field(s): {missing}")
            if str(item.get("label_name", "")).lower() == "ahi":
                required_postprocess = (
                    "min_event_duration_sec",
                    "merge_tolerance_sec",
                    "output_second_alignment",
                    "output_event_alignment",
                )
                missing = [
                    key for key in required_postprocess if key not in postprocess or postprocess[key] in (None, "")
                ]
                if missing:
                    raise ValueError(f"Analyzer {name!r} AHI postprocess missing required field(s): {missing}")
        if analyzer_type == "npz_stage_reference":
            if not item.get("stage_key"):
                raise ValueError(f"Analyzer {name!r} requires stage_key.")
        if analyzer_type.startswith("yasa_") and not input_channels:
            raise ValueError(f"Analyzer {name!r} requires input_channels.")
        if analyzer_type == "yasa_stage":
            eeg_channels = [channel for channel in input_channels if signals.channels[channel].kind.lower() == "eeg"]
            if not eeg_channels:
                raise ValueError(f"Analyzer {name!r} requires at least one EEG input channel.")
        if analyzer_type == "yasa_rem":
            # YASA REM detection requires LOC/ROC EOG; do not accept single-EOG or EMG substitutes.
            kinds = [signals.channels[channel].kind.lower() for channel in input_channels]
            if len(input_channels) != 2 or any(kind != "eog" for kind in kinds):
                raise ValueError(f"Analyzer {name!r} requires exactly two EOG input channels.")
        stages = _string_list(item.get("stages", []), f"analyzers[{idx}].stages")
        if analyzer_type.startswith("yasa_"):
            unknown_stages = sorted(
                {stage for stage in stages if stage.strip().upper() not in YASA_STAGE_FILTER_LABELS}
            )
            if unknown_stages:
                raise ValueError(
                    f"Analyzer {name!r} has unsupported YASA stage filter(s): {unknown_stages}. "
                    f"Expected one of {sorted(YASA_STAGE_FILTER_LABELS)}."
                )
        if analyzer_type.startswith("spo2_") or analyzer_type == "event_related_hypoxic_burden":
            if not input_channels:
                raise ValueError(f"Analyzer {name!r} requires input_channels.")
        if analyzer_type == "spo2_desaturation":
            missing = [
                key for key in ("drop_thresholds", "min_duration_sec") if key not in item or item[key] in (None, "")
            ]
            if missing:
                raise ValueError(f"Analyzer {name!r} missing required field(s): {missing}")
        artifact = dict(item.get("artifact") or {})
        legacy_artifact_keys = sorted({"min_value", "max_value", "max_drop_per_sec"} & set(artifact))
        if legacy_artifact_keys:
            raise ValueError(f"Analyzer {name!r} uses legacy artifact field(s): {legacy_artifact_keys}")
        outputs = dict(item.get("outputs") or {})
        if analyzer_type == "yasa_bandpower" and "stage_source" in outputs:
            raise ValueError(f"Analyzer {name!r} uses legacy outputs.stage_source; set stage_source instead.")
        if analyzer_type == "yasa_bandpower":
            required_outputs = ("by_epoch", "by_stage", "by_night", "relative")
            missing = [key for key in required_outputs if key not in outputs or outputs[key] is None]
            if missing:
                raise ValueError(f"Analyzer {name!r} outputs missing required field(s): {missing}")
            if bool(outputs["by_stage"]) and not item.get("stage_source"):
                raise ValueError(f"Analyzer {name!r} requires stage_source when outputs.by_stage=true.")
        raw_drop_thresholds = _string_list(item.get("drop_thresholds", []), f"analyzers[{idx}].drop_thresholds")
        if analyzer_type == "spo2_desaturation" and not raw_drop_thresholds:
            raise ValueError(f"Analyzer {name!r} requires drop_thresholds.")
        analyzers.append(
            AnalyzerConfig(
                name=name,
                type=analyzer_type,
                enabled=bool(item.get("enabled", True)),
                namespace=None if item.get("namespace") is None else str(item["namespace"]),
                label_name=None if item.get("label_name") is None else str(item["label_name"]),
                config=None if item.get("config") is None else Path(item["config"]),
                ckpt_path=None if item.get("ckpt_path") is None else Path(item["ckpt_path"]),
                input_channels=input_channels,
                batch_size=None if item.get("batch_size") is None else int(item["batch_size"]),
                stage_key=None if item.get("stage_key") is None else str(item["stage_key"]),
                threshold=item.get("threshold"),
                postprocess=postprocess,
                stage_source=None if item.get("stage_source") is None else str(item["stage_source"]),
                stages=stages,
                artifact=artifact,
                drop_thresholds=[float(value) for value in raw_drop_thresholds],
                min_duration_sec=(None if item.get("min_duration_sec") is None else float(item["min_duration_sec"])),
                max_duration_sec=(None if item.get("max_duration_sec") is None else float(item["max_duration_sec"])),
                event_source=None if item.get("event_source") is None else str(item["event_source"]),
                outputs=outputs,
            )
        )
    if not analyzers:
        raise ValueError("analyzers must not be empty.")
    analyzer_names = [analyzer.name for analyzer in analyzers]
    duplicates = sorted({name for name in analyzer_names if analyzer_names.count(name) > 1})
    if duplicates:
        raise ValueError(f"sleep2stat analyzer names must be unique; duplicate analyzer name(s): {duplicates}")
    produced_analyzer_names: set[str] = set()
    for analyzer in analyzers:
        if not analyzer.enabled:
            continue
        stage_sources = []
        if analyzer.stage_source is not None:
            stage_sources.append(("stage_source", analyzer.stage_source))
        if analyzer.type == "sleep2vec_downstream" and (analyzer.label_name or "").lower() == "ahi":
            denominator_stage_source = analyzer.postprocess.get("denominator_stage_source")
            if denominator_stage_source not in (None, ""):
                stage_sources.append(("postprocess.denominator_stage_source", str(denominator_stage_source)))
        for field_name, stage_source in stage_sources:
            if stage_source not in produced_analyzer_names:
                raise ValueError(
                    f"Analyzer {analyzer.name!r} {field_name} must reference an enabled earlier analyzer: "
                    f"{stage_source!r}."
                )
        produced_analyzer_names.add(analyzer.name)
    return analyzers


def _build_reducers(raw: Any) -> list[ReducerConfig]:
    reducers = []
    for idx, raw_item in enumerate(_require_list(raw, "reducers")):
        item = _require_mapping(raw_item, f"reducers[{idx}]")
        _reject_unknown_keys(item, REDUCER_KEYS, f"reducers[{idx}]")
        reducer_type = str(item.get("type", ""))
        name = str(item.get("name", ""))
        if not name:
            raise ValueError(f"reducers[{idx}].name is required.")
        if reducer_type not in SUPPORTED_REDUCER_TYPES:
            raise ValueError(f"Unknown sleep2stat reducer type: {reducer_type!r}.")
        if reducer_type in {
            "hypnogram_stats",
            "transition_stats",
            "respiratory_stats",
            "event_density",
            "stage_specific_summary",
        } and not item.get("source"):
            raise ValueError(f"Reducer {name!r} requires source.")
        if reducer_type == "stage_agreement" and (not item.get("left") or not item.get("right")):
            raise ValueError(f"Reducer {name!r} requires left and right.")
        reducers.append(
            ReducerConfig(
                name=name,
                type=reducer_type,
                enabled=bool(item.get("enabled", True)),
                source=None if item.get("source") is None else str(item["source"]),
                left=None if item.get("left") is None else str(item["left"]),
                right=None if item.get("right") is None else str(item["right"]),
                age_prediction=None if item.get("age_prediction") is None else str(item["age_prediction"]),
                sex_prediction=None if item.get("sex_prediction") is None else str(item["sex_prediction"]),
                metadata_age_column=str(item.get("metadata_age_column", "age")),
                metadata_sex_column=str(item.get("metadata_sex_column", "sex")),
                options=dict(item.get("options") or {}),
            )
        )
    reducer_names = [reducer.name for reducer in reducers]
    duplicates = sorted({name for name in reducer_names if reducer_names.count(name) > 1})
    if duplicates:
        raise ValueError(f"sleep2stat reducer names must be unique; duplicate reducer name(s): {duplicates}")
    return reducers


def _validate_reducer_references(analyzers: list[AnalyzerConfig], reducers: list[ReducerConfig]) -> None:
    analyzer_by_name = {analyzer.name: analyzer for analyzer in analyzers}
    enabled_analyzer_names = {analyzer.name for analyzer in analyzers if analyzer.enabled}
    for reducer in reducers:
        if not reducer.enabled:
            continue
        references = []
        if reducer.source is not None:
            references.append(("source", reducer.source))
        if reducer.left is not None:
            references.append(("left", reducer.left))
        if reducer.right is not None:
            references.append(("right", reducer.right))
        if reducer.age_prediction is not None:
            references.append(("age_prediction", reducer.age_prediction))
        if reducer.sex_prediction is not None:
            references.append(("sex_prediction", reducer.sex_prediction))
        for field_name, reference in references:
            if reference not in analyzer_by_name:
                raise ValueError(
                    f"Reducer {reducer.name!r} references unknown analyzer in {field_name}: {reference!r}."
                )
            if reference not in enabled_analyzer_names:
                raise ValueError(
                    f"Reducer {reducer.name!r} references disabled analyzer in {field_name}: {reference!r}."
                )


def _validate_backend_analyzer_support(data_cfg: DataConfig, analyzers: list[AnalyzerConfig]) -> None:
    if data_cfg.backend != "kaldi":
        return
    yasa_analyzers = [analyzer.name for analyzer in analyzers if analyzer.type.startswith("yasa_")]
    if yasa_analyzers:
        raise ValueError(
            "YASA analyzers require data.backend=npz in sleep2stat v0.1 because Kaldi inputs are token matrices, "
            f"not raw continuous PSG signals: {yasa_analyzers}"
        )


def _build_outputs(raw: Any) -> OutputsConfig:
    data = _require_mapping(raw, "outputs")
    _reject_unknown_keys(data, OUTPUTS_KEYS, "outputs")
    compression = str(data.get("compression", "gzip"))
    if compression not in {"gzip", "none"}:
        raise ValueError("outputs.compression must be either 'gzip' or 'none'.")
    global_tables = _build_global_tables(data.get("global_tables"))
    write_global_tables = bool(data.get("write_global_tables", True))
    write_per_record = bool(data.get("write_per_record", True))
    if write_global_tables and not write_per_record:
        raise ValueError(
            "outputs.write_global_tables=true requires outputs.write_per_record=true because cumulative sleep2stat "
            "summary tables are rebuilt from per-record sidecars."
        )
    return OutputsConfig(
        write_global_tables=write_global_tables,
        write_per_record=write_per_record,
        include_probabilities=bool(data.get("include_probabilities", True)),
        include_raw_logits=bool(data.get("include_raw_logits", False)),
        compression=compression,
        global_tables=global_tables,
    )


def _build_global_tables(raw: Any) -> dict[str, bool]:
    output = {
        "epoch_alignment": False,
        "second_alignment": False,
        "event_alignment": True,
        "night_stats": True,
    }
    if raw is None:
        return output
    data = _require_mapping(raw, "outputs.global_tables")
    _reject_unknown_keys(data, GLOBAL_TABLE_KEYS, "outputs.global_tables")
    for key, value in data.items():
        output[str(key)] = bool(value)
    return output
