# Preprocessing Workflow

## Purpose

Prepare CSV splits, inspect channel-mask coverage, generate preset pickles, optionally merge presets, and convert WatchPAT `.zzp` archives when needed.

## Canonical Paths

### CSV To Preset Path

1. `preprocess/split_index_by_dataset.py`
2. optional `preprocess/mask_missing_stats.py`
3. `preprocess/save_dataset_presets.py`
4. optional `preprocess/merge_dataset_presets.py`

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

Stage/AHI-only test indexes may omit `age` and `sex`. Those fields are copied into presets only when present, while explicitly requested metadata such as `--meta-data-names age` or `--meta-data-names sex` still requires the matching CSV column.

The preset schema is still implicitly a pickled `list[SampleIndex]`, but the branch now treats `preset_build` as part of the contract for reproducible preset generation.

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
- Change mask semantics: keep `split_index_by_dataset.py` and `mask_missing_stats.py` aligned
- Change config-policy checks: `utils/check_configs.py`
- Change WatchPAT conversion: `preprocess/watchpat_zzp_to_edf.py`
