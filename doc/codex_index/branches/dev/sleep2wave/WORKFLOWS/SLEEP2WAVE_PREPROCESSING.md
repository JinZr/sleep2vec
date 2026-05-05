# Sleep2Wave Preprocessing Workflow

## Purpose

Validate Sleep2Wave indexes and build schema-versioned generative preset pickles.

## Canonical Path

1. Prepare an index CSV with required columns:
   - `path`
   - `duration`
   - `split`
   - `subject_id`
   - `night_id`
   - modality mask columns such as `eeg_mask`, `eog_mask`, `spo2_mask`
2. Validate the index:
   - `sleep2wave.preprocess.validate_sleep2wave_index`
3. Build preset windows:
   - `sleep2wave.preprocess.build_sleep2wave_presets`
4. Train or generate through `Sleep2WaveGenerativeDataset`.

## Data Contract

Each generated `SampleIndex` payload records:

- schema version
- available modalities
- canonical channel map
- availability and quality mask keys
- sample rates and frames per epoch
- subject/night metadata
- night epoch count

## Commands

```bash
python -m sleep2wave.preprocess.validate_sleep2wave_index --index index.csv
python -m sleep2wave.preprocess.build_sleep2wave_presets \
  --index index.csv \
  --output data/sleep2wave_preset.pkl \
  --split train val test \
  --context-epochs 15
```

## Edit Hotspots

- Index column and mask semantics: `sleep2wave/data/generative_dataset.py`
- CLI preset writing: `sleep2wave/preprocess/build_sleep2wave_presets.py`
- Index validation: `sleep2wave/preprocess/validate_sleep2wave_index.py`
- Subject split safety: `sleep2wave/data/derivations.py`

## Tests

```bash
python3.10 -m pytest -q tests/test_sleep2wave_preprocess_contract.py tests/test_sleep2wave_generative_dataset.py tests/test_sleep2wave_modalities.py
```
