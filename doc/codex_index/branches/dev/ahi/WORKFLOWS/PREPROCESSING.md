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

## Split Generation

`split_index_by_dataset.py`:

- reads the input CSV
- optionally filters rows by available channel count using `*_mask` columns
- marks external datasets by regex
- assigns `train/val/test` by dataset group
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

1. requires exactly one index CSV per run
2. loads YAML `model.channels`
3. optionally reads YAML `preset_build.required_channels` and `preset_build.min_channels`
4. resolves the effective validation channels
   - YAML model channels are the default
   - CLI `--channels` can narrow them only when `preset_build.required_channels` is absent
   - built-in validation channels `stage5` and `ahi` are accepted and get fixed input dims / mask columns
5. resolves metadata variants
6. resolves output paths from a template
7. for each `(metadata, split)` pair, instantiates `PSGPretrainDataset`
8. relies on `DefaultDataset` side effects to validate samples and write the preset pickle

### Required-Channel Filtering

- When `--allow-missing-channels` is `true`, the preset builder keeps the original CSV and lets runtime dataset validation decide sample retention, using `min_channels`.
- When `--allow-missing-channels` is `false`, the preset builder first prefilters the CSV by required mask columns before constructing the dataset.
- Built-in mask mapping is part of the contract:
  - `stage5` -> `stage_mask`
  - `ahi` -> `ahi_mask`
  - other channels -> `<channel>_mask`

### Preset Schema

The preset schema is still a pickled `list[SampleIndex]`.

The CLI does not create a second schema for built-in validation channels; it reuses the normal dataset/preset path.

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

## Notebook Status

`preprocess/preprocess_pipeline.ipynb` is manual workflow history. It is useful for human context but should not be treated as the canonical reusable preprocessing implementation.

## Edit Hotspots

- Change split policy: `preprocess/split_index_by_dataset.py`
- Change preset schema or generation behavior: `preprocess/save_dataset_presets.py` plus `data/psg_pretrain_dataset.py`
- Change built-in preset validation-channel policy: keep `_resolve_validation_channels`, `_filter_index_df_for_required_channels`, and `PSGPretrainDataset` aligned
- Change mask semantics: keep `split_index_by_dataset.py`, `mask_missing_stats.py`, and preset required-mask filtering aligned
- Change WatchPAT conversion: `preprocess/watchpat_zzp_to_edf.py`
