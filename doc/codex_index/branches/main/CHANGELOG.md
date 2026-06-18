# Changelog

## 2026-06-18

- Updated the `Sleep2vecPretrainModel` constructor guidance to reflect config-only construction and removal of the legacy manual channel/dimension path.
- Updated reuse and module-map notes so future changes do not reintroduce manual backbone-constructor branches.
- Updated loss reuse guidance after moving contrastive accuracy into package-local `losses/utils.py` helpers.

## 2026-06-14

- Repaired the `main` branch index for commit `65a86e37a1d1da2ed7cb02ec21d9457504dcfa0a`.
- Updated README/MANIFEST metadata and tracked-file coverage counts for `configs/`, `tests/`, `skills/`, `recipes/`, and the new tracked `sleep2stat/` package.
- Added `FUNCTIONS/SLEEP2STAT.md` for sleep2stat config parsing, NPZ/Kaldi record loading, analyzer/reducer registries, model/YASA/SpO2 analyzers, stage-denominator helpers, output-bundle writing, and plotting.
- Added `WORKFLOWS/SLEEP2STAT.md` for validation, agent consultation, execution, resume, summarize, plotting, and verification paths.
- Updated system/module/reuse guidance to route sleep2stat changes through `sleep2stat.config.load_config`, `load_records`, `StageSourceResolver`, `Sleep2vecDownstreamAnalyzer`, `AnalysisBundleWriter`, and agent-tooling sleep2stat gates.
- Folded the non-stale guidance from the removed branch-local sleep2stat index into main: reuse `AnalyzerResult`, keep denominator/unit names explicit, keep deprecated aliases narrow, and put plot compatibility fallback in `sleep2stat.plot`.
- Stale entries removed or corrected:
  - old manifest/README commit and timestamp from `28dd58fe944803592b7e9857b3a0f9fbdfffada3`
  - old coverage counts for configs, tests, skills, and recipes
  - missing first-class index coverage for `sleep2stat/`

## 2026-06-10

- Repaired `FUNCTIONS/AGENT_TOOLING.md` signature entries for `agent_tools.index_csv.index_summary` and `agent_tools.progress.write_progress` so reusable-function guidance matches the implementation.
- Updated README/MANIFEST metadata for commit `28dd58fe944803592b7e9857b3a0f9fbdfffada3`.

- Repaired the `main` branch index for commit `dbe6a5e4cf40811138a35870b011a2a6d1bf8b83`.
- Updated README/MANIFEST metadata and tracked-file coverage counts for `sleep2vec/`, `sleep2vec2/`, `sleep2expert/`, `tests/`, `utils/`, `agent_tools/`, `skills/`, and `recipes/`.
- Added `FUNCTIONS/AGENT_TOOLING.md` for the new agent CLI, consultation, plan, hparam, experiment, adaptive-hparam, and progress surfaces.
- Added hparam logit export guidance and edit hotspots for `agent_tools/hparam.py`, `agent_tools/experiments.py`, `agent_tools/adaptive_hparam.py`, and `agent_tools/progress.py`.
- Stale entries removed or corrected:
  - old manifest commit/timestamp from `dbbbe24c24252593f28fdf354cbf3b17bc59e17a`
  - old README commit/timestamp from `0a6d07de56bcc0bbae45fd9fdde6e747cafab238`
  - stale `agent_tools.context` references, replaced with `agent_tools.plans.build_context`
  - old variant coverage counts for `sleep2vec2/` and `sleep2expert/`
- Added agent tooling workflow coverage for `agent_tools/`, `skills/`, `recipes/`, `agent_policies/`, and `doc/agent_contracts/`.
- Documented stop-and-consult gates, `NEEDS_USER_INPUT` exit code `2`, blocked context/plan artifacts, and external-test unlock policy.
- Tightened agent-tooling guidance for variant-aware `sleep2vec` / `sleep2vec2` / `sleep2expert` command generation, explicit final-test checkpoint selection, overwrite guards, and namespaced hparam search keys.
- Updated reuse guidance to route agent-facing workflow support through `agent_tools.plans`, `agent_tools.recipes`, `agent_tools.decisions`, `agent_tools.hparam`, `agent_tools.experiments`, `agent_tools.adaptive_hparam`, and `agent_tools.progress`.
- Corrected PPG AHI config-validation wording to require `[ppg, ahi, stage5]` with `min_channels=3`.

## 2026-05-31

- Added token embedding extraction guidance for `sleep2vec.extract_embeddings` and package-local `sleep2vec2` / `sleep2expert` mirrors, including selected-layer hidden-state export and NPZ/Kaldi manifest outputs.

## 2026-05-25

- Refreshed the `main` branch index for commit `0a6d07de56bcc0bbae45fd9fdde6e747cafab238`.
- Updated tracked-file coverage counts to reflect the current branch:
  - `sleep2vec/`: 81
  - `data/`: 8
  - `preprocess/`: 7
  - `sleep2vec2/`: 103
  - `sleep2expert/`: 107
  - `sleep2vec_moe/`: 0
  - `sleep2vec_hires/`: 0
  - `configs/`: 108
  - `tests/`: 50
  - `utils/`: 7
- Added current LoRA/DoRA config propagation guidance for `r`, `alpha`, `dropout`, `target_modules`, `use_dora`, and separate-adapter trainability.
- Added current downstream metric guidance for binary specificity, macro specificity, and stage alias behavior.
- Added inference W&B artifact guidance for metrics, predictions, manifest, overview, and `prediction_row_count` logging across root and variant inference entrypoints.
- Added example config validation coverage for `configs/examples/**`.
- Stale entries removed or corrected:
  - old manifest/README commit and timestamp from `8c1989dcfb89dc51612656f460d9ebfc8adfb46c`
  - old tracked-file counts for `configs/` and `tests/`
  - old inference side-effect wording that omitted W&B artifact logging

## 2026-05-24

- Refreshed the `main` branch index for commit `8c1989dcfb89dc51612656f460d9ebfc8adfb46c`.
- Updated tracked-file coverage counts to reflect the current branch:
  - `sleep2vec/`: 81
  - `data/`: 8
  - `preprocess/`: 7
  - `sleep2vec2/`: 103
  - `sleep2expert/`: 107
  - `sleep2vec_moe/`: 0
  - `sleep2vec_hires/`: 0
  - `configs/`: 101
  - `tests/`: 49
  - `utils/`: 7
- Added current automatic inference export guidance for run-local metrics CSVs, per-path prediction CSVs, shared `overview.csv`, `run_manifest.json`, `prediction_run_id`, and package-local variant mirrors.
- Added missing standalone utility guidance for UKB annotation parsing, UKB demographic extraction, Kaldi index repair, and case-control matching.
- Stale entries removed or corrected:
  - old manifest/README commit and timestamp from `4a80f9bf40ac7e8cb00143b5dc5b5eb5b15710dd`
  - old tracked-file counts for `sleep2vec/`, `sleep2vec2/`, `sleep2expert/`, `configs/`, `tests/`, and `utils/`
  - old variant coverage counts in `FUNCTIONS/VARIANT_SURFACES.md`

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
