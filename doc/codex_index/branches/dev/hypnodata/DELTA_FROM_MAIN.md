# Delta From Main

`main` has no branch-specific `hypnodata` package index. The `main` index was
used as baseline guidance for config strictness, preprocessing ownership, and
downstream NPZ/Kaldi/sleep2stat boundaries.

## Branch-Local Additions

- `hypnodata/`: raw clinical EDF-style ingestion and standardized NPZ output.
- `configs/hypnodata/`: example config and boundary documentation.
- `tests/hypnodata/test_hypnodata_*.py`: contract tests for config, preprocessing,
  pipeline outputs, adapters, manifests, conflict/progress, and downstream
  compatibility.

## Current Contract Delta

- `signals.<channel>.preprocess` uses structured mapping steps, not string step
  names.
- `type: filter` is implemented by `neurokit2.signal_filter`.
- `type: notch` is implemented by SciPy with explicit `freq` and `q`.
- `filter:not_implemented` and `notch:not_implemented` are obsolete and must not
  appear in manifests.
- Raw-signal `candidates` are ordered exact-label strings; regex, priority, and
  adapter scoring are not part of the hypnodata config contract.
- Annotation-only signals must use empty `candidates` for `kind: stage`,
  `event_table`, `event_dense`, `event_anchor`, or built-in `ahi`; raw signals
  must declare non-empty candidates.
- Annotation-only signals declare output grids with `epoch_sec`,
  `interval_sec`, or `window_sec` instead of raw-only fields such as
  `target_sfreq`, `target_unit`, `scale`, `polarity`, or `preprocess`.
- Adapter-provided annotation sources can now materialize standard event tables,
  dense event labels, and anchor labels in addition to `stage5`.
- Built-in `signals.ahi` writes the downstream AHI finetune trio `ah_event`,
  scalar `ahi`, and scalar `tst` from stage5 plus apnea/hypopnea event rows.
- Hypnodata still does not write ODI, T90, hypoxic burden, sleep efficiency, or
  other downstream clinical summaries.
- `hypnodata run` is hard-fail and single-use; `run --dry-run` is lightweight
  discovery preview; `hypnodata validate` is the full QC/reporting path without
  NPZ writes.
- `requirements.txt` includes NeuroKit2 and a NumPy version compatible with that
  dependency.

## Main Reuse Boundaries Preserved

- Hypnodata writes NPZ records and manifests only.
- Kaldi conversion remains owned by `preprocess/convert_npz_to_kaldi.py`.
- Preset generation remains owned by `preprocess/save_dataset_presets.py`.
- Sleep2stat consumes the manifest through its existing NPZ record loader.
