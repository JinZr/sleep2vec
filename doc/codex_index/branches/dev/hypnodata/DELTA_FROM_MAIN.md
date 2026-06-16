# Delta From Main

`main` has no branch-specific `hypnodata` package index. The `main` index was
used as baseline guidance for config strictness, preprocessing ownership, and
downstream NPZ/Kaldi/sleep2stat boundaries.

## Branch-Local Additions

- `hypnodata/`: raw clinical EDF-style ingestion and standardized NPZ output.
- `configs/hypnodata/`: example config and boundary documentation.
- `tests/test_hypnodata_*.py`: contract tests for config, preprocessing,
  pipeline outputs, adapters, manifests, resume/progress, and downstream
  compatibility.

## Current Contract Delta

- `signals.<channel>.preprocess` uses structured mapping steps, not string step
  names.
- `type: filter` is implemented by `neurokit2.signal_filter`.
- `type: notch` is implemented by SciPy with explicit `freq` and `q`.
- `filter:not_implemented` and `notch:not_implemented` are obsolete and must not
  appear in manifests.
- `requirements.txt` includes NeuroKit2 and a NumPy version compatible with that
  dependency.

## Main Reuse Boundaries Preserved

- Hypnodata writes NPZ records and manifests only.
- Kaldi conversion remains owned by `preprocess/convert_npz_to_kaldi.py`.
- Preset generation remains owned by `preprocess/save_dataset_presets.py`.
- Sleep2stat consumes the manifest through its existing NPZ record loader.
