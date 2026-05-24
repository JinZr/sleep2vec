# Changelog

## 2026-05-24

- Removed the stale sibling-path prediction CSV helper from the inference results contract and renamed inference-only args fields to `inference_metrics_csv_path` / `inference_prediction_csv_path`.
- Refreshed inference-result guidance for automatic `results/inference/{namespace}/{label}/{prediction_run_id}` run directories, global `overview.csv`, run manifests, and checkpoint-tagged metrics/prediction CSV files.
- Refreshed `dev/save_prob` index entries for package-local inference prediction export across `sleep2vec`, `sleep2vec2`, and `sleep2expert`.
- Added `sleep2vec_inference.py` as the canonical row-extraction and row-aggregation surface for prediction CSV export, with package-local mirrors for variants.
- Updated runtime and model-function guidance so `Sleep2vecFinetuning` owns lifecycle, target extraction, DDP gathering, metrics, and AHI thresholds, while prediction row construction lives outside the Lightning module.
- Updated variant guidance to require root and package-local export surfaces to move together.

## 2026-05-11

- Repaired the `main` branch index for commit `4a80f9bf40ac7e8cb00143b5dc5b5eb5b15710dd`.
- Updated tracked-file coverage counts to reflect the current branch:
  - `sleep2vec/`: 80
  - `data/`: 8
  - `preprocess/`: 7
  - `sleep2vec2/`: 102
  - `sleep2expert/`: 106
  - `configs/`: 100
  - `tests/`: 47
  - `utils/`: 2
- Repaired stale branch-state claims that said `sleep2vec2/` had no tracked source files.
- Added active standalone variant coverage for `sleep2vec2/` and `sleep2expert/`, including package-local data/preprocess mirrors and namespace-parity guidance.
- Added sleep2expert MoE routing, finetune tuning, checkpoint expansion, model-stats, and routing-analysis export guidance.
- Added `WORKFLOWS/VARIANTS_AND_ROUTING.md`.
- Updated config-validation workflow guidance for package-local config and preset helpers.
- Stale entries removed or corrected:
  - old manifest commit/counts from `99d22deee69cc3cb9eae9229a8faaa4c33974824`
  - old claim that active variant directories were placeholders
  - old reuse guidance that treated variant directories as non-active reuse targets
- Added the Kaldi NPZ-to-ark converter contract to the preprocessing function catalog, including default semantic ark compression and package-local mirror parity notes.

## 2026-05-06

- Updated dataset and preprocessing docs for stage/AHI-only preset generation without mandatory `age`/`sex` CSV columns.
- Recorded that built-in `age` and `sex` loader paths now reject presets/indexes without valid labels after split/source filtering.

## 2026-04-21

- Cleaned up stale usage guidance in the `main` branch index:
  - clarified that small, localized fixes only need a quick consult of `README.md` plus one relevant index page
  - kept the full reading pass for broader behavior or contract changes
- Cleaned up stale branch-scope metadata labels:
  - `Commit` -> `Last full refresh commit`
  - `Generated at` -> `Last full refresh at`
  - this avoids implying that every docs-only tweak is a full index rebuild

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
- Marked notebook coverage as summary-only for `preprocess/preprocess_pipeline.ipynb`.
- Stale entries removed: none, because this was an initialize-main build rather than a refresh.
