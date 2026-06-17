# hypnodata

`hypnodata` is the raw clinical PSG normalization layer for this repo. It reads
clinical EDF-style records, resolves configured channels, and writes a shared
standardized NPZ product plus manifests.

The main output is:

- `manifest/record_manifest.csv`
- `manifest/signal_manifest.csv`
- `manifest/qc_summary.csv`
- `manifest/failures.csv`
- `manifest/backend_manifest.json`
- `status/progress.json`
- `backends/npz/records/<record_id>.npz`

## Runtime Modes

`hypnodata run` is the production conversion path. It writes NPZ records and
terminal manifests only when every selected record succeeds. The output
directory is single-use: a non-empty directory fails before record processing.

`hypnodata run --dry-run` is a lightweight discovery preview. It loads the
config, discovers records, writes `manifest/discovery_preview.csv` plus
`status/progress.json`, and does not read EDF data, materialize annotations, or
write NPZ records.

`hypnodata validate` is the full QC path. It reads records, preprocesses
signals, materializes annotations, writes manifests and `failures.csv`, but does
not write NPZ records. If any record fails validation, the report is still
written and the command exits non-zero.

## Downstream Contract

`record_manifest.csv` is the public NPZ index. Downstream consumers should read
`path`, `duration`, `split`, `source`, `record_id`, and canonical mask columns
such as `eeg_mask`, `spo2_mask`, and `stage_mask`.

`path` points to the standardized NPZ, not the raw EDF. `duration` is the public
duration field. Do not add `duration_sec` aliases to this CSV.

`center` is the normalization/source-center label used by hypnodata. `source` is
the downstream dataset/source field used by loaders, converters, and sampling
filters. `record_id` is the stable record key.

`sleep2stat` should consume this product through `sleep2stat.io.records.load_records`
with `data.backend=npz`, `data.path_column=path`, and `data.duration_column=duration`.

Kaldi output is generated through the existing `preprocess/convert_npz_to_kaldi.py`
converter. hypnodata does not write ark/scp files directly.

Preset generation remains owned by `preprocess/save_dataset_presets.py`. hypnodata
only supplies an index and mask columns; it does not create splits, sample windows,
or preset pickle files.

## CSV Discovery

CSV discovery uses `record_discovery.file_columns` as the canonical file mapping.
Each key becomes a `RecordTask.files` entry, and each value is a CSV column name:

```yaml
record_discovery:
  type: csv
  index: records.csv
  record_id_column: record_id
  file_columns:
    edf: path
```

For multi-file records, add more file keys:

```yaml
file_columns:
  edf: edf_path
  stage: stage_csv
  events: event_csv
```

`file_column` is not accepted; use `file_columns` for single-file and multi-file
CSV discovery.

## Channel Candidates

Raw signal candidates are ordered exact EDF labels. Put the preferred label first:

```yaml
candidates:
  - "EEG C3"
  - "C3-A2"
```

Core hypnodata does not use regex or priority fields for channel resolution.

## Signal Preprocess

`signals.<channel>.preprocess` is an ordered list of structured steps. Core
hypnodata always applies raw-to-target unit conversion, `scale`, `polarity`,
target-rate resampling, finite checks, and common-duration truncation through the
fixed preprocessing path, so do not write those fixed steps in YAML.

Use `type: filter` for NeuroKit2 filtering:

```yaml
preprocess:
  - type: filter
    method: bessel
    order: 4
    lowcut: 0.5
    highcut: 45.0
```

Use `type: notch` when the powerline frequency and Q value must be explicit:

```yaml
preprocess:
  - type: notch
    freq: 50.0
    q: 30.0
```

## Adapter Contract

Use `record_discovery.adapter: module:function` when a center needs custom
collection or metadata/header fixes. The function receives `HypnodataConfig` and
returns an adapter object.

Optional adapter methods:

- `collect_records(config)` for `record_discovery.type=custom`
- `resolve_metadata(record, config)`
- `fix_header(record, inventories, config)`
- `read_annotations(record, config, duration_sec)`

Adapters must return hypnodata structures such as `RecordTask`, `EdfInventory`,
and `AnnotationResult`. Keep center-specific behavior in the adapter; do not add
real center names or hardcoded center rules to core hypnodata.

For records without any available raw signal, adapters must provide a positive
finite `record.metadata["duration"]` so `read_annotations(record, config,
duration_sec)` can materialize annotation-only outputs.

`adapter_options` is the passthrough block for adapters. Core hypnodata keeps
strict unknown-key validation elsewhere and does not use `schema_version`,
`version_schema`, or legacy aliases.

## Annotation Boundary

hypnodata materializes annotation signals only when they are declared under
`signals` and returned by the adapter as `AnnotationResult` entries. Annotation
sources stay adapter-owned: core config does not accept
`signals.<name>.annotation`.

Annotation-only outputs must use empty candidates:

```yaml
signals:
  stage5:
    kind: stage
    required: false
    epoch_sec: 30
    candidates: []
  ah_event_table:
    kind: event_table
    required: false
    candidates: []
  ahi:
    kind: ahi
    required: false
    interval_sec: 1
    candidates: []
  arousal_anchor:
    kind: event_anchor
    required: false
    window_sec: 10
    candidates: []
```

For annotation-only signals, use these second-based output-grid fields instead
of `target_sfreq`: `epoch_sec` for stage arrays, `interval_sec` for dense event
labels and built-in AHI labels, and `window_sec` for anchor labels. Annotation
labels also do not use `target_unit`, `scale`, `polarity`, or `preprocess`;
their values are written from adapter-provided annotation arrays. Built-in
`signals.ahi` requires `stage5.epoch_sec: 30` and `interval_sec: 1`; it writes
the downstream AHI finetune trio `ah_event`, scalar `ahi`, and scalar `tst`.

Adapters can use `hypnodata.annotations` helpers to map CSV or EDF annotations:

- `read_stage_csv` / `read_stage_edf_annotations` produce `stage5`-style stage
  arrays with the default `Wake/N1/N2/N3/REM -> 0/1/2/3/4` mapping.
- `read_event_csv` converts `Type/Start/Duration` tables to standard event rows
  `[type, start_sec, duration_sec]`.
- `materialize_event_table`, `materialize_dense_events`, and
  `materialize_anchor_events` produce event table, dense timeline, and anchor
  label outputs.
- `filter_events_to_sleep_stages` may be used by adapters to keep only events
  overlapping sleep stages.
- `materialize_ahi_from_events` produces the built-in AHI finetune output from
  apnea/hypopnea event rows plus `stage5`: 1 Hz `ah_event`, scalar `ahi`, and
  scalar `tst` in hours.

hypnodata does not parse proprietary XML/event formats here and does not compute
ODI, sleep efficiency, WASO, HRV, or YASA staging. Those remain downstream
analysis tasks.

## Wuji-dl Notes

Historical `wuji-dl` code is useful as implementation background:

- index-first processing
- collector/builder separation
- per-record pipeline execution
- simple factory/import path extension
- physiological signal filtering and stage materialization experience

hypnodata intentionally does not copy:

- mutable free-form worker state
- workers deciding NPZ keys
- runners hardcoding worker or center names
- old worker output shapes as the core contract

The hypnodata contract is the canonical signal name, manifest lineage, QC rows,
and backend manifest.
