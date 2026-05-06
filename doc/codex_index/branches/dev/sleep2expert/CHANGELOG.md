# Changelog

## 2026-05-06

- Tightened `sleep2expert` dense-to-MoE checkpoint initialization so missing or shape-incompatible expert/layer-norm tensors fail fast instead of entering fine-tune randomly initialized.
- Updated `sleep2expert.finetune` default base LR from `1e-6` to `1e-4`; downstream MoE semantic LR multipliers still apply on top of the base CLI value.
- Synchronized the optional `age`/`sex` preset metadata contract into `sleep2vec2/` and `sleep2expert/` package-local dataset and finetune loader copies.
- Documented that stage/AHI-only presets may omit `age`/`sex`, while built-in `age`/`sex` tasks fail fast on missing or invalid labels.

## 2026-05-01

- Initialized the `dev/sleep2expert` branch index from the current `main` baseline at commit `21bbd67bc7b69dce4b119141cd201779688f016c`.
- Added branch delta coverage for the `sleep2vec2` standalone recipe:
  - package-local runtime under `sleep2vec2/`
  - package-local data and preprocessing copies under `sleep2vec2/data/` and `sleep2vec2/preprocess/`
  - package-local tracked visualization font assets under `sleep2vec2/visualization/assets/`
  - copied YAML recipes under `configs/sleep2vec2/`
  - standalone RoFormer replacement under `sleep2vec2/backbones/roformer/`
- Updated variant reuse guidance to keep `sleep2vec2` imports local and avoid implicit fallback to base `sleep2vec`, top-level `data`, or top-level `preprocess`.
- Added `DELTA_FROM_MAIN.md` for the branch-specific surface.
- Added branch delta coverage for the `sleep2expert` standalone recipe mirrored from `sleep2vec2`, including package-local runtime, configs, RoFormer, and namespace/parity tests.

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
- Recorded branch-state notes for `sleep2vec2/`, `sleep2vec_moe/`, and `sleep2vec_hires/` because they contain no tracked source files on `main`.
- Marked notebook coverage as summary-only for `preprocess/preprocess_pipeline.ipynb`.
- Stale entries removed: none, because this was an initialize-main build rather than a refresh.
