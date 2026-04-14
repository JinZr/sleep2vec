# Workflow: Data Prep And Presets

## Primary Path

1. Prepare or merge an index CSV with required columns such as `path`, `split`, `duration`, `age`, and `sex`.
2. If the CSV still needs split labels, run `preprocess/split_index_by_dataset.py`.
3. If channel-availability quality matters, run `preprocess/mask_missing_stats.py` before building presets.
4. Generate preset pickles with `preprocess/save_dataset_presets.py` using a YAML whose `model.channels` declare the runtime channels and `input_dim` widths.
5. Merge several preset pickles with `preprocess/merge_dataset_presets.py` only when the runtime really expects a combined preset.

## Canonical Commands

### Split by dataset

Use `preprocess/split_index_by_dataset.py` when the CSV does not already contain a reliable `split` column.

Important contract:

- external datasets matching `mros|mesa|shhs|hspS0001` become `external`
- internal groups are split with capped val/test sizes

### Inspect missing-channel statistics

Use `preprocess/mask_missing_stats.py` when deciding whether missing-channel pretraining is viable or when picking `min_channels`.

Important contract:

- every `*_mask` column except `stage_mask` is treated as a presence indicator where `1` means present

### Generate presets

Use `preprocess/save_dataset_presets.py`.

Important contracts:

- `--config` is required
- YAML `model.channels` is the source of truth for channel names and `input_dim`
- non-`stage5` channels without `input_dim` fail immediately
- `--channels` must be a subset of YAML-declared channels
- when `--allow-missing-channels` is enabled, preset validation writes `payload["available_channels"]`

### Merge presets

Use `preprocess/merge_dataset_presets.py` only for already-generated preset pickles.

## Reuse Hotspots

- YAML channel resolution: `preprocess.save_dataset_presets._resolve_channels_and_dims`
- Dataset materialization: `data.psg_pretrain_dataset.PSGPretrainDataset`
- Preset validation: `data.utils.filter_valid_sample_indices`

## Failure Modes To Check First

- YAML missing `model.channels`
- YAML channel missing `input_dim`
- requested `--channels` not present in YAML
- preset built without `available_channels` but later used with missing-channel training
- CSV missing `split` when split filtering is requested
