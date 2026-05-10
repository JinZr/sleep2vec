# Preprocessing Workflow

## Purpose

Prepare CSV splits, inspect channel-mask coverage, generate preset pickles or Kaldi manifests, optionally merge presets, and convert WatchPAT `.zzp` archives when needed.

## Canonical Paths

### CSV To Preset Path

1. `preprocess/split_index_by_dataset.py`
2. optional `preprocess/mask_missing_stats.py`
3. `preprocess/save_dataset_presets.py`
4. optional `preprocess/merge_dataset_presets.py`

### CSV/NPZ To Kaldi Path

1. `preprocess/convert_npz_to_kaldi.py`
2. `manifest.json` consumed by `KaldiPSGDataset`, which selects `manifests/{split}.csv`

### WatchPAT Conversion Path

1. `preprocess/watchpat_zzp_to_edf.py`
2. optional JSON summary output

### Config Validation Path

1. `utils/check_configs.py`
2. config loaders in `sleep2vec/config.py`
3. preset-build helpers reused from `preprocess/save_dataset_presets.py`

## Split Generation

`split_index_by_dataset.py`:

- reads the input CSV
- normalizes mask truthiness through `normalize_mask_frame`
- optionally filters rows by available channel count using `*_mask` columns
- marks external datasets by regex
- assigns `train/val/test` by dataset group
- optionally warns or fails when validation/test miss globally feasible modality pairs
- writes a new CSV with a `split` column

This is the canonical split policy before preset generation.

## Mask Statistics

`mask_missing_stats.py`:

- treats a channel as present only when parsed numeric value equals `1`
- streams large CSVs in chunks
- emits four CSV reports:
  - overall per-channel missing rates
  - per-dataset per-channel missing rates
  - overall row-wise missing-count histogram
  - per-dataset row-wise missing-count histogram

## Preset Generation

`save_dataset_presets.py`:

1. resolves dataset name from the input CSV name or an explicit override
2. loads channels and input dimensions from YAML `model.channels`
3. optionally reads YAML `preset_build.required_channels` and `preset_build.min_channels`
4. auto-injects `stage5` when built-in `ahi` is part of the validation-channel set
5. optionally prefilters the CSV by required mask columns when `allow_missing_channels=False`
6. instantiates `PSGPretrainDataset` for each `(metadata, split)` pair
7. relies on `DefaultDataset` side effects to validate samples and write the preset pickle

Stage/AHI-only test indexes need `path`, `split`, `duration`, task channels/masks, and corresponding NPZ contents, but they do not need `age` or `sex`. Explicit metadata variants requested through `--meta-data-names` still require the matching CSV columns, except built-in AHI `ahi`/`tst` summaries loaded from NPZ.

The preset schema is still implicitly a pickled `list[SampleIndex]`, but the branch now treats `preset_build` as part of the contract for reproducible preset generation.

## Kaldi Conversion

`convert_npz_to_kaldi.py`:

- loads selected channels from YAML `model.channels` and optional built-in channels such as `stage5` and `ahi`
- windows CSV-indexed NPZ recordings into per-sample matrices
- writes one ark/scp pair per channel under `channels/{split}/`
- optionally writes multiple ark shards per split/channel with `--ark-shards`, while keeping one aggregate `{channel}.scp` in `manifest.json`
- writes `manifests/{split}.csv` with `sample_key`, token span, metadata, and `available_channels`
- writes `manifest.json` format v2 with split-specific channel input dimensions and sorted scp paths

Standalone recipes must use their package-local converter, such as `sleep2vec2.preprocess.convert_npz_to_kaldi` or `sleep2expert.preprocess.convert_npz_to_kaldi`, so extractor/tokenizer semantics match the runtime namespace.

## Preset Merge

`merge_dataset_presets.py`:

- loads multiple preset pickles
- verifies each top-level object is a `list`
- concatenates them
- writes a single output pickle

This utility does not normalize schema versions; it trusts all input preset lists to be compatible.

## WatchPAT Conversion

`watchpat_zzp_to_edf.py`:

- locates and reads files inside a `.zzp` archive
- decodes Sleep.dat frame structure
- infers signal layout and channel mapping heuristically
- builds EDF-ready signal descriptions
- writes EDF through either manual or `pyedflib` backend
- optionally writes JSON summary

This path is operationally separate from preset generation and is not used by `PSGPretrainDataset`.

## Config Validation

`utils/check_configs.py`:

- validates that YAML files load successfully through the runtime config loaders
- enforces shared tokenizer-dimension parity through `validate_model_config`
- validates `preset_build` completeness and semantics
- enforces repo-specific policy for `ppg_*finetune*.yaml` recipes

Use this tool when config changes alter loader behavior, built-in task semantics, or preset-build contracts.

## Notebook Status

`preprocess/preprocess_pipeline.ipynb` is manual workflow history. It is useful for human context but should not be treated as the canonical reusable preprocessing implementation.

## Edit Hotspots

- Change split policy: `preprocess/split_index_by_dataset.py`
- Change preset schema or generation behavior: `preprocess/save_dataset_presets.py` plus `data/psg_pretrain_dataset.py`
- Change Kaldi conversion behavior: `preprocess/convert_npz_to_kaldi.py` plus `data/kaldi_psg_dataset.py`
- Change mask semantics: keep `split_index_by_dataset.py` and `mask_missing_stats.py` aligned
- Change config-policy checks: `utils/check_configs.py`
- Change WatchPAT conversion: `preprocess/watchpat_zzp_to_edf.py`
