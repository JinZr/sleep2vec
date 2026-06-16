# System Overview

`hypnodata` is the branch-local raw clinical PSG normalization layer. Its public
contract is raw EDF-style records in, standardized NPZ records plus manifests
out.

## Runtime Flow

1. `hypnodata.config.load_config` parses YAML and builds typed config objects.
2. `hypnodata.discovery.discover_records` creates `RecordTask` objects from
   glob, CSV, or custom adapter discovery.
3. `hypnodata.edf.read_edf_inventory` inspects raw EDF files.
4. `hypnodata.channels.resolve_channels` resolves configured canonical signals
   to raw labels by ordered exact-label candidates or marks optional channels
   missing.
5. `hypnodata.edf.read_edf_signal` reads the selected raw signal.
6. `hypnodata.preprocess.preprocess_signal` applies scale, polarity,
   structured filter/notch steps, resampling, and finite checks.
7. `hypnodata.preprocess.truncate_to_common` aligns available raw signals to a
   common duration.
8. Adapter annotations may materialize configured `stage`, `event_table`,
   `event_dense`, `event_anchor`, or built-in `ahi` outputs.
9. `hypnodata.backends.write_npz_record` writes standardized arrays.
10. `hypnodata.manifests.write_manifests` writes record, signal, QC, failure,
    backend, and progress outputs.

## Preprocess Contract

`signals.<channel>.preprocess` is an ordered list of structured mappings. The
only accepted step types are:

- `type: filter`, implemented with `neurokit2.signal_filter`
- `type: notch`, implemented with `scipy.signal.iirnotch` and `filtfilt`

`scale`, `polarity`, `resample`, `finite_check`, and `truncate_to_common` are
fixed pipeline behavior and are not YAML preprocess step types.

## Annotation Contract

Annotation sources remain adapter-owned. Core YAML declares canonical outputs
under `signals`, but does not accept `signals.<name>.annotation`.

Annotation-only signals use empty `candidates` and one of these `kind` values:

- `stage`: 1D stage arrays such as `stage5`; declare `epoch_sec`
- `event_table`: `(N, 3)` event tables `[type, start_sec, duration_sec]`
- `event_dense`: 1D dense event labels; declare `interval_sec`
- `event_anchor`: 2D anchor labels with three columns per anchor; declare
  `window_sec`
- `ahi`: built-in AHI finetune output; declare `interval_sec: 1`, require
  `stage5.epoch_sec: 30`, and write `ah_event`, scalar `ahi`, and scalar `tst`

Annotation-only signals must not use `target_sfreq` in YAML. Core config
derives the effective output frequency from the second-based fields when
validating adapter output and writing manifests.

Use `hypnodata.annotations` helpers for standard stage/event rows,
stage-aware event filtering, and table/dense/anchor materialization. The
pipeline validates declared canonical names, duplicate annotation names,
raw-signal collisions, shape, and target sampling frequency consistency.

## Output Contract

The public downstream index is `manifest/record_manifest.csv`. It uses `path`
for the standardized NPZ path, `duration` for record duration, and canonical
mask columns such as `eeg_mask`, `spo2_mask`, and `stage_mask`.

`manifest/signal_manifest.csv` records channel lineage and
`preprocess_steps`. Structured filter/notch and annotation materialization
steps must be recorded as real executed steps, never as `not_implemented`
placeholders.
