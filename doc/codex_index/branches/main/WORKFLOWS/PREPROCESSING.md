# Preprocessing Workflow

## Purpose

Prepare CSV splits, inspect channel-mask coverage, generate preset pickles, optionally merge presets, convert NPZ windows to Kaldi roots, convert WatchPAT `.zzp` archives, parse UKB metadata, repair Kaldi indexes, and build matched cohorts when needed.

## Canonical Paths

### CSV To Preset Path

1. `preprocess/split_index_by_dataset.py`
2. optional `preprocess/mask_missing_stats.py`
3. `preprocess/save_dataset_presets.py`
4. optional `preprocess/merge_dataset_presets.py`

### CSV/NPZ To Kaldi Path

1. `preprocess/split_index_by_dataset.py`
2. optional `preprocess/mask_missing_stats.py`
3. `preprocess/convert_npz_to_kaldi.py`
4. runtime `data.backend: kaldi` or `--data-backend kaldi`

### WatchPAT Conversion Path

1. `preprocess/watchpat_zzp_to_edf.py`
2. optional JSON summary output

### UKB Asleep Night-Cutting Path

1. `utils/cut_ukb_sleep_with_asleep.py`
2. optional sleep2vec preset or Kaldi conversion later, from the generated NPZ manifest

### UKB Annotation And Demographics Path

1. `utils/parse_ukb_annotations_by_person.py`
2. optional `utils/collect_ukb_demographics.py`

### Kaldi Index Repair Path

1. `utils/fix_kaldi_index.py`
2. `preprocess/convert_npz_to_kaldi.py`

### Case-Control Matching Path

1. `utils/match_case_controls.py`
2. downstream preset or Kaldi prep from the matched cohort CSV

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

## Kaldi Conversion

`convert_npz_to_kaldi.py`:

- reads one or more index CSVs
- resolves channels and input dimensions from config YAML
- honors `preset_build.required_channels` and `preset_build.min_channels`
- expands each source row into fixed token windows
- writes `manifest.json` format v2, split CSV manifests, per-channel `.scp` files, and ark files
- keeps `val`/`test` rows when overlapping windows are requested but converts them with non-overlapping stride unless `--include-overlap-eval-splits` is passed
- defaults to compressed matrix ark storage for non-built-in signal channels in the `train` split
- keeps built-in `stage5`/`ahi` channels and non-train splits as float matrices
- supports shard count, worker count, split filtering, and path-prefix mapping

The package-local mirrors under `sleep2vec2/preprocess/convert_npz_to_kaldi.py` and `sleep2expert/preprocess/convert_npz_to_kaldi.py` should stay behaviorally aligned with the root converter.

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

## UKB Asleep Night Cutting

`cut_ukb_sleep_with_asleep.py`:

- walks a flat or bucketed UKB `.cwa` tree
- imports the standalone pip-installed `asleep` package, not sleep2vec
- reuses asleep parsing, 30 Hz epoching, non-wear handling, and sleep-window detection
- selects only the longest sleep block in each UK local noon-to-noon interval as the nightly segment
- writes per-night compressed NPZ files plus `night_sleep_blocks.csv` per source file and a root `manifest.csv`; dynamic `--time-shift auto` writes local offset-aware `time` and raw `device_time`

This output is an external data-cutting artifact. It does not create `SampleIndex` presets and does not exercise `PSGPretrainDataset`.

## UKB Annotation Parsing And Demographics

`parse_ukb_annotations_by_person.py`:

- accepts a raw UKB root or its `annotations/` subdirectory
- parses `ukb*.tab` files and companion `.html` dictionaries / `.r` codings when present
- writes dataset-level column, field, coding, and missing-coding tables
- writes participant JSON files under `participants/<eid_prefix>/<eid>/<dataset_id>.json`
- normalizes withdrawals into `withdrawals/withdrawn_eids.csv`
- writes a derived `manifest.json` and README

`collect_ukb_demographics.py`:

- scans a participant JSON tree
- extracts primary or fallback sex and age fields
- can compute age from birth year/month and assessment date when explicit age fields are absent
- writes source columns so downstream users know which UKB field supplied each value

These utilities are external metadata-prep tools. They do not load sleep2vec configs or mutate dataset presets.

## Kaldi Index Repair

`fix_kaldi_index.py`:

- reads an index CSV before Kaldi conversion
- requires `path` plus the configured source field
- creates `session_id` when missing
- only changes rows whose source/session-derived Kaldi sample-key prefix would duplicate an earlier row
- overwrites the input with a `.backup` by default, or writes `--output`

Use this when converter preflight reports duplicate sample keys from repeated source/path stems.

## Case-Control Matching

`match_case_controls.py`:

- reads a flat cohort CSV
- filters missing required fields before duplicate suppression
- estimates propensity scores with Patsy/statsmodels design matrices
- supports exact-match columns, numeric calipers, non-reused controls, partial matches, and optional genetic-style weight search
- writes matched rows, unmatched cases, excluded rows, case match counts, and balance diagnostics

This is a cohort-construction utility. It should feed later split/preset/Kaldi prep rather than being imported into runtime dataset classes.

## Config Validation

`utils/check_configs.py`:

- validates that YAML files load successfully through the selected runtime config loader
- routes `configs/sleep2vec2/**` and `configs/sleep2expert/**` through package-local config/preset helpers
- enforces shared tokenizer-dimension parity through `validate_model_config`
- validates `preset_build` completeness and semantics
- enforces repo-specific policy for `ppg_*finetune*.yaml` recipes

Use this tool when config changes alter loader behavior, built-in task semantics, or preset-build contracts.

## Notebook Status

`preprocess/preprocess_pipeline.ipynb` is manual workflow history. It is useful for human context but should not be treated as the canonical reusable preprocessing implementation.

## Edit Hotspots

- Change split policy: `preprocess/split_index_by_dataset.py`
- Change preset schema or generation behavior: `preprocess/save_dataset_presets.py` plus `data/psg_pretrain_dataset.py`
- Change Kaldi conversion behavior: `preprocess/convert_npz_to_kaldi.py`, `data/kaldi_psg_dataset.py`, and package-local variant mirrors when parity is required
- Change mask semantics: keep `split_index_by_dataset.py` and `mask_missing_stats.py` aligned
- Change config-policy checks: `utils/check_configs.py`
- Change WatchPAT conversion: `preprocess/watchpat_zzp_to_edf.py`
- Change standalone UKB/asleep night extraction: `utils/cut_ukb_sleep_with_asleep.py`
- Change UKB annotation parsing or demographics extraction: `utils/parse_ukb_annotations_by_person.py`, `utils/collect_ukb_demographics.py`
- Change duplicate-key repair for Kaldi prep: `utils/fix_kaldi_index.py`
- Change matching/cohort construction: `utils/match_case_controls.py`
