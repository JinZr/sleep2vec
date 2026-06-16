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
  directory out.
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
