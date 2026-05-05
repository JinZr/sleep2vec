# sleep2wave Preprocessing Workflow

## Purpose

Validate sleep2wave indexes and build schema-versioned generative preset pickles.

## Canonical Path

1. Prepare an index CSV with the same core columns used by the base sleep2vec preset path:
   - `path`
   - `duration`
   - `split`
   Optional `subject_id` and `night_id` columns are preserved when present; otherwise sleep2wave uses `path` for both identifiers.
   Optional modality mask columns such as `eeg_mask`, `eog_mask`, and `spo2_mask` are used as row-level availability hints when present.
2. Optionally derive sidecar channels:
   - `sleep2wave.preprocess.derive_sleep2wave_channels`
   This writes deterministic per-record `ibi`, `resp`, and matching epoch-level quality masks.
3. Validate the index:
   - `sleep2wave.preprocess.validate_sleep2wave_index`
4. Build preset windows:
   - `sleep2wave.preprocess.build_sleep2wave_presets`
   Preset building opens each NPZ with a row-level progress bar and stores the true per-window `available_channels` plus `canonical_channel_map`; rows/windows with no usable canonical modalities are skipped. Standard quality masks such as `<modality>_quality_mask` are recorded when present.
5. Train or generate through `Sleep2WaveGenerativeDataset`.

## Data Contract

Each generated `SampleIndex` payload records:

- schema version
- available modalities
- canonical channel map
- availability and quality mask keys
- derived sidecar channel paths when used
- sample rates and frames per epoch
- subject/night metadata
- night epoch count

## Commands

```bash
python -m sleep2wave.preprocess.validate_sleep2wave_index --index index.csv
python -m sleep2wave.preprocess.derive_sleep2wave_channels \
  --index index.csv \
  --output-dir data/sleep2wave_derived \
  --derive ibi resp
python -m sleep2wave.preprocess.build_sleep2wave_presets \
  --index index.csv \
  --output data/sleep2wave_preset.pkl \
  --split train val test \
  --context-epochs 15 \
  --num-workers 8
```

## Edit Hotspots

- Index column and mask semantics: `sleep2wave/data/generative_dataset.py`
- CLI preset writing: `sleep2wave/preprocess/build_sleep2wave_presets.py`
- Index validation: `sleep2wave/preprocess/validate_sleep2wave_index.py`
- Deterministic derived sidecars: `sleep2wave/data/derivations.py`

## Tests

```bash
python3.10 -m pytest -q tests/test_sleep2wave_preprocess_contract.py tests/test_sleep2wave_generative_dataset.py tests/test_sleep2wave_modalities.py
```
