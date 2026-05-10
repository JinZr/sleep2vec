# sleep2wave Preprocessing Workflow

## Purpose

Validate sleep2wave indexes, build schema-versioned generative preset pickles, or convert the same waveform windows into a package-local Kaldi root.

## Canonical Path

1. Prepare an index CSV with the same core columns used by the base sleep2vec preset path:
   - `path`
   - `duration`
   - `split`
   Optional `subject_id` and `night_id` columns are preserved when present; otherwise sleep2wave uses `path` for both identifiers.
   Optional modality mask columns such as `eeg_mask`, `eog_mask`, and `spo2_mask` are used as row-level availability hints when present.
2. Validate the index:
   - `sleep2wave.preprocess.validate_sleep2wave_index`
3. Build preset windows:
   - `sleep2wave.preprocess.build_sleep2wave_presets`
   Preset building opens each NPZ with a row-level progress bar and stores the true per-window `available_channels` plus `canonical_channel_map`; rows/windows with no usable canonical modalities are skipped. Standard quality masks such as `<modality>_quality_mask` are recorded when present.
4. For Kaldi-backed autoencoder/diffusion/generation runs, convert windows with `sleep2wave.preprocess.convert_npz_to_kaldi` instead of writing a preset pickle.
5. Train or generate through `Sleep2WaveGenerativeDataset`.

## Data Contract

Each generated `SampleIndex` payload records:

- schema version
- available modalities
- canonical channel map
- availability and quality mask keys
- sample rates and frames per epoch
- subject/night metadata
- night epoch count

Kaldi conversion writes `manifest.json` with `format_version: 2`, one `manifests/<split>.csv` per split, and one sorted `channels/<split>/<modality>.ark/.scp` pair per canonical modality. Each stored matrix is `[context_epochs * channel_count, frames_per_epoch]`; the dataset decodes it back to `[context_epochs, channel_count, frames_per_epoch]`. Each split CSV records `sample_key`, `record_key`, `path`, `split`, epoch bounds, `available_channels`, and JSON quality masks when present. `manifest.json` records `backend: kaldi_native_io`, timing metadata, source indexes, and split-specific channel scp metadata.

## Commands

```bash
python -m sleep2wave.preprocess.validate_sleep2wave_index --index index.csv
python -m sleep2wave.preprocess.build_sleep2wave_presets \
  --index index.csv \
  --output data/sleep2wave_preset.pkl \
  --split train val test \
  --context-epochs 15 \
  --num-workers 8

python -m sleep2wave.preprocess.convert_npz_to_kaldi \
  --index index.csv \
  --config configs/sleep2wave/sleep2wave_autoencoder_medium.yaml \
  --output-dir data/sleep2wave_kaldi/medium_15e \
  --split train val test \
  --stride-epochs 15 \
  --path-prefix-map /old/data/root=/new/data/root
```

## Edit Hotspots

- Index column and mask semantics: `sleep2wave/data/generative_dataset.py`
- CLI preset writing: `sleep2wave/preprocess/build_sleep2wave_presets.py`
- Kaldi waveform conversion: `sleep2wave/preprocess/convert_npz_to_kaldi.py`
- Index validation: `sleep2wave/preprocess/validate_sleep2wave_index.py`

## Tests

```bash
python3.10 -m pytest -q tests/test_sleep2wave_preprocess_contract.py tests/test_sleep2wave_generative_dataset.py tests/test_sleep2wave_modalities.py
python3.10 -m pytest -q tests/test_npz_to_kaldi_converter_roundtrip.py tests/test_kaldi_io.py
```
