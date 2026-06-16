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

v2 only materializes explicitly configured or adapter-provided `stage5`
annotations. Simple CSV stages and EDF annotations can be mapped to `stage5` when
the adapter provides epoch seconds, label mapping, and invalid value.

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
