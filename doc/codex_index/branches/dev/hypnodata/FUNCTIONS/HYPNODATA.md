# Hypnodata Functions

## `hypnodata.config.load_config`

- File: `hypnodata/config.py`
- Signature: `load_config(path: str | Path) -> HypnodataConfig`
- Purpose and contract: parse a hypnodata YAML file into typed dataclasses,
  reject unknown fields and schema/version fields, and enforce core config
  requirements.
- Important inputs/outputs: YAML path in; `HypnodataConfig` out.
- Side effects: reads one YAML file.
- Reuse guidance: use this for all hypnodata config loading.

## `hypnodata.config._build_preprocess_steps`

- File: `hypnodata/config.py`
- Signature: `_build_preprocess_steps(raw: Any, canonical: str) -> list[PreprocessStep]`
- Purpose and contract: parse `signals.<channel>.preprocess` as an ordered list
  of structured `FilterStep` and `NotchStep` objects.
- Important inputs/outputs: raw YAML value and canonical channel name in;
  typed preprocess steps out.
- Side effects: none.
- Reuse guidance: keep preprocess schema validation here.
- Duplication risk notes: do not accept bare string aliases or fixed internal
  steps here.

## `hypnodata.config.declared_target_sfreq`

- File: `hypnodata/config.py`
- Signature: `declared_target_sfreq(spec: SignalSpec) -> float | None`
- Purpose and contract: return the configured output frequency for raw or
  annotation-only signals.
- Important inputs/outputs: `SignalSpec` in; `target_sfreq` for raw signals,
  `1/epoch_sec`, `1/interval_sec`, or `1/window_sec` for annotation-only
  signals, or `None` when no output frequency applies.
- Side effects: none.
- Reuse guidance: use this in pipeline validation and backend manifest writing
  instead of duplicating frequency derivation.

## `hypnodata.edf.read_edf_signal`

- File: `hypnodata/edf.py`
- Signature: `read_edf_signal(path: Path, raw_label: str, raw_unit: str | None = None, *, raw_index: int | None = None) -> np.ndarray`
- Purpose and contract: read one EDF channel as a 1D float32 array at that
  channel's native header-declared sample count.
- Important inputs/outputs: EDF path, label, optional unit, and optional raw
  channel index in; native per-channel samples out. Mixed-rate EDF channels must
  not be returned on MNE's common/highest-rate time base.
- Side effects: reads one EDF file.
- Reuse guidance: pipeline raw signal loading should call this instead of MNE
  directly.

## `hypnodata.annotations.read_stage_csv`

- File: `hypnodata/annotations.py`
- Signature: `read_stage_csv(path: str | Path, *, duration_sec: float, epoch_sec: float, mapping: dict[str, int] | None = None, invalid: int = -1, label_column: str = "stage", start_column: str = "start", duration_column: str = "duration", canonical_channel: str = "stage5") -> AnnotationSignal`
- Purpose and contract: materialize stage rows into a 1D stage annotation
  array, using the default `Wake/N1/N2/N3/REM -> 0/1/2/3/4` mapping when no
  mapping is supplied.
- Important inputs/outputs: CSV path and epoch seconds in; `stage`
  `AnnotationSignal` out.
- Side effects: reads one CSV file.
- Reuse guidance: adapters should use this for simple stage CSVs before
  building stage-aware event labels.

## `hypnodata.annotations.read_event_csv`

- File: `hypnodata/annotations.py`
- Signature: `read_event_csv(path: str | Path, *, type_column: str | None = "Type", start_column: str = "Start", duration_column: str = "Duration", mapping: dict[str, int] | None = None, default_type: int = 0) -> np.ndarray`
- Purpose and contract: read a standard event CSV into rows shaped `(N, 3)` as
  `[type, start_sec, duration_sec]`.
- Important inputs/outputs: CSV path and column names in; float32 event rows
  out.
- Side effects: reads one CSV file.
- Reuse guidance: adapters should use this instead of re-parsing common
  `Type/Start/Duration` files.

## `hypnodata.annotations.materialize_event_table`

- File: `hypnodata/annotations.py`
- Signature: `materialize_event_table(events: np.ndarray, *, canonical_channel: str, raw_file: str = "", raw_label: str = "events", steps: list[str] | None = None) -> AnnotationSignal`
- Purpose and contract: wrap standard event rows as an `event_table`
  annotation signal.
- Important inputs/outputs: `(N, 3)` rows in; `AnnotationSignal` with
  `sfreq=None` out.
- Side effects: none.
- Reuse guidance: use for `<name>_table` annotation channels.

## `hypnodata.annotations.materialize_dense_events`

- File: `hypnodata/annotations.py`
- Signature: `materialize_dense_events(events: np.ndarray, *, duration_sec: float, interval_sec: float, canonical_channel: str, raw_file: str = "", raw_label: str = "events", value: float | None = 1.0, steps: list[str] | None = None) -> AnnotationSignal`
- Purpose and contract: convert event rows into a 1D dense event timeline.
- Important inputs/outputs: standard event rows and interval seconds in;
  `event_dense` annotation with `sfreq=1/interval_sec` out.
- Side effects: none.
- Reuse guidance: use for `ah_event`, `arousal`, `desaturation`, or
  `snore_event` dense labels.

## `hypnodata.annotations.materialize_anchor_events`

- File: `hypnodata/annotations.py`
- Signature: `materialize_anchor_events(events: np.ndarray, *, duration_sec: float, window_sec: float, anchor_num: int, canonical_channel: str, raw_file: str = "", raw_label: str = "events", steps: list[str] | None = None) -> AnnotationSignal`
- Purpose and contract: convert event rows into window-level anchor labels
  where each anchor stores `[present, start_frac, stop_frac]`.
- Important inputs/outputs: standard event rows, window size, and anchor count
  in; `event_anchor` matrix out.
- Side effects: none.
- Reuse guidance: use for anchor labels inspired by arousal, desaturation, and
  snore workers.

## `hypnodata.annotations.filter_events_to_sleep_stages`

- File: `hypnodata/annotations.py`
- Signature: `filter_events_to_sleep_stages(events: np.ndarray, stage: np.ndarray, *, epoch_sec: float, sleep_values: set[int] | frozenset[int] = SLEEP_STAGE_VALUES) -> np.ndarray`
- Purpose and contract: keep event rows that overlap sleep-stage epochs.
- Important inputs/outputs: event rows and a stage array in; filtered event
  rows out.
- Side effects: none.
- Reuse guidance: use for stage-aware event label generation without writing
  AHI/TST or clinical summaries.

## `hypnodata.config.FilterStep`

- File: `hypnodata/config.py`
- Signature: `FilterStep(method: str, order: int, lowcut: float | None = None, highcut: float | None = None)`
- Purpose and contract: typed representation of a NeuroKit2 filter step.
- Important inputs/outputs: method must be `bessel` or `butterworth`; order is a
  positive integer; at least one cutoff is required.
- Side effects: none.
- Reuse guidance: pass this object to `preprocess_signal`; do not re-parse raw
  YAML in preprocessing code.

## `hypnodata.config.NotchStep`

- File: `hypnodata/config.py`
- Signature: `NotchStep(freq: float, q: float)`
- Purpose and contract: typed representation of an explicit powerline notch
  step.
- Important inputs/outputs: positive `freq` and positive `q`.
- Side effects: none.
- Reuse guidance: pass this object to `preprocess_signal`.

## `hypnodata.preprocess.preprocess_signal`

- File: `hypnodata/preprocess.py`
- Signature: `preprocess_signal(raw: np.ndarray, selection: ChannelSelection, spec: SignalSpec) -> ProcessedSignal`
- Purpose and contract: convert one selected raw signal into a contiguous
  float32 output by applying scale, polarity, structured preprocess steps,
  target resampling, and finite checks.
- Important inputs/outputs: raw signal, selected channel metadata, and
  `SignalSpec` in; `ProcessedSignal` with data, sfreq, unit, and executed step
  names out.
- Side effects: imports NeuroKit2 only when a `FilterStep` executes.
- Reuse guidance: this is the canonical signal preprocessing path.
- Duplication risk notes: do not add filter execution in `pipeline.py` or
  adapters.

## `hypnodata.preprocess.truncate_to_common`

- File: `hypnodata/preprocess.py`
- Signature: `truncate_to_common(signals: dict[str, ProcessedSignal]) -> tuple[dict[str, ProcessedSignal], float, list[str]]`
- Purpose and contract: truncate available processed signals to the shortest
  common duration and record `truncate_to_common` in each signal step list.
- Important inputs/outputs: processed signals in; truncated signals, common
  duration, and changed channel names out.
- Side effects: none.
- Reuse guidance: use after all raw signal preprocessing and before NPZ writing.

## `hypnodata.pipeline.run_pipeline`

- File: `hypnodata/pipeline.py`
- Signature: `run_pipeline(config: HypnodataConfig, *, output_dir: Path, num_workers: int = 1, limit: int | None = None, overwrite: bool = False, resume: bool = False, dry_run: bool = False, crash: bool = False, record_id: str | None = None) -> Path`
- Purpose and contract: execute hypnodata conversion for discovered records,
  handling resume/overwrite/dry-run/crash options and writing final manifests.
- Important inputs/outputs: parsed config and output directory in; output
  directory out. If a record has no available raw signal, `record.metadata`
  must provide a positive finite `duration` for annotation-only materialization.
- Side effects: reads raw records, writes NPZ files, manifests, and progress.
- Reuse guidance: use this as the orchestration entrypoint.

## `hypnodata.manifests.write_manifests`

- File: `hypnodata/manifests.py`
- Signature: `write_manifests(output_dir: Path, config: HypnodataConfig, *, record_rows: list[dict[str, Any]], signal_rows: list[dict[str, Any]], qc_rows: list[dict[str, Any]], failure_rows: list[dict[str, Any]], dry_run: bool) -> None`
- Purpose and contract: write public record/signal/QC/failure CSVs and backend
  manifest JSON.
- Important inputs/outputs: collected rows and config in; manifest files out.
- Side effects: writes files under `manifest/`.
- Reuse guidance: keep public manifest schema changes here and update tests.
