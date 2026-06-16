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

## Signal Preprocess

`signals.<channel>.preprocess` is an ordered list of structured steps. Core
hypnodata always applies `scale`, `polarity`, target-rate resampling, finite
checks, and common-duration truncation through the fixed preprocessing path, so
do not write those fixed steps in YAML.

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
- `score_channel_candidate(record, canonical, spec, candidate, signal, config)`
- `read_annotations(record, config, duration_sec)`

Adapters must return hypnodata structures such as `RecordTask`, `EdfInventory`,
and `AnnotationResult`. Keep center-specific behavior in the adapter; do not add
real center names or hardcoded center rules to core hypnodata.

`adapter_options` and `custom` are passthrough blocks for adapters. Core
hypnodata keeps strict unknown-key validation elsewhere and does not use
`schema_version` or legacy aliases.

## Annotation Boundary

hypnodata materializes annotation signals only when they are declared under
`signals` and returned by the adapter as `AnnotationResult` entries. Annotation
sources stay adapter-owned: core config does not accept
`signals.<name>.annotation`.

Annotation-only outputs use empty candidates:

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
  ah_event:
    kind: event_dense
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
labels, and `window_sec` for anchor labels.

Adapters can use `hypnodata.annotations` helpers to map CSV or EDF annotations:

- `read_stage_csv` / `read_stage_edf_annotations` produce `stage5`-style stage
  arrays with the default `Wake/N1/N2/N3/REM -> 0/1/2/3/4` mapping.
- `read_event_csv` converts `Type/Start/Duration` tables to standard event rows
  `[type, start_sec, duration_sec]`.
- `materialize_event_table`, `materialize_dense_events`, and
  `materialize_anchor_events` produce event table, dense timeline, and anchor
  label outputs.
- `filter_events_to_sleep_stages` may be used by adapters to keep only events
  overlapping sleep stages. It does not write TST, AHI, ODI, or other clinical
  summaries.

hypnodata does not parse proprietary XML/event formats here and does not compute
AHI, ODI, sleep efficiency, WASO, HRV, or YASA staging. Those remain downstream
analysis tasks.

## Wuji-dl Notes

Historical `wuji-dl` code is useful as implementation background:

- index-first processing
- collector/builder separation
- per-record pipeline execution
- simple factory/import path extension
- crash/continue failure mode
- physiological signal filtering and stage materialization experience

hypnodata intentionally does not copy:

- mutable free-form worker state
- workers deciding NPZ keys
- runners hardcoding worker or center names
- old worker output shapes as the core contract

The hypnodata contract is the canonical signal name, manifest lineage, QC rows,
and backend manifest.
