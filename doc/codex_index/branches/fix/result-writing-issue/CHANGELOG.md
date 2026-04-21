# Changelog

## 2026-04-21

- Initialized the `fix/result-writing-issue` branch index from the current branch state using `main` as the starting baseline.
- Updated the finetune runtime docs to record that `sleep2vec.finetune.supervised` now finishes only the W&B run it created and does not let teardown failures mask an already-active primary train/test error.

## 2026-04-20

- Refreshed the `main` branch engineering index for commit `99d22deee69cc3cb9eae9229a8faaa4c33974824`.
- Updated tracked-file coverage counts to reflect the current branch:
  - `sleep2vec/`: 78
  - `configs/`: 32
  - `tests/`: 24
  - `utils/`: 2
- Added workflow coverage for staged adaptation and config validation:
  - `WORKFLOWS/ADAPT.md`
  - `WORKFLOWS/CONFIG_VALIDATION.md`
- Refreshed function catalogs to cover:
  - built-in `stage3` / `stage4` / `ahi` task semantics
  - adaptation config, phase validation, and optimizer grouping
  - result CSV writes from `sleep2vec/results.py`
  - downstream evaluation visualization hooks
  - strict preset-build validation and required-channel prefiltering
  - AHI event metrics and threshold-search flow
- Updated the system overview and module map to reflect:
  - a single validation loader with sequential pair evaluation in pretrain
  - stage-specific adaptation checkpoint layout
  - checkpoint-specific AHI thresholds during test/inference
  - the config-policy tooling path in `utils/check_configs.py`
- Stale entries removed or corrected:
  - old claim that pretrain builds one validation loader per pair
  - old claim that `stage5` was the only built-in sequence task
  - old reuse hotspot pointing at `sleep2vec.metrics.save_result_csv`
  - outdated workflow coverage that omitted `adapt`

## 2026-03-25

- Created the initial `main` branch engineering index.
- Added branch-scoped manual pages:
  - `README.md`
  - `MANIFEST.json`
  - `SYSTEM_OVERVIEW.md`
  - `MODULE_MAP.md`
  - `REUSE_GUIDE.md`
  - `CHANGELOG.md`
  - `FUNCTIONS/`
  - `WORKFLOWS/`
- Indexed tracked code under `sleep2vec/`, `data/`, `preprocess/`, `configs/`, `tests/`, and `utils/`.
- Recorded branch-state notes for `sleep2vec2/`, `sleep2vec_moe/`, and `sleep2vec_hires/` because they contain no tracked source files on `main`.
- Marked notebook coverage as summary-only for `preprocess/preprocess_pipeline.ipynb`.
- Stale entries removed: none, because this was an initialize-main build rather than a refresh.
