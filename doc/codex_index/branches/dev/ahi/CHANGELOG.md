# Changelog

## 2026-04-18

- Added single-channel PPG built-in `ahi` finetune recipe files for both medium and large backbone sizes, both using `cls.embedding_type: bert` with token-level downstream heads.
- Recorded the local config-policy exception that lets `ppg_ahi_finetune*.yaml` require `preset_build.required_channels: [ppg, ahi]` while the preset builder still auto-expands built-in `ahi` validation to include `stage5`.

## 2026-04-17

- Documented the built-in `ahi` runtime split between detection-style event metrics and NPZ-aligned scalar summary AHI.
- Recorded that scalar summary AHI now counts stage-filtered raw predicted positive runs without merge or min-duration filtering, while event TP/FP/FN semantics stay unchanged.
- Fixed the missing-channel collate fallback so built-in `ahi` availability is recomputed from `ah_event` plus scalar `ahi` / `tst`, matching preset validation for legacy presets without serialized `available_channels`.

## 2026-04-15

- Repaired the `dev/ahi` handbook metadata to the current branch tip `0de463929a695c30ec29fa94cfd1e0c5df9e8d92`.
- Converted `DELTA_FROM_MAIN.md` from working-tree-only wording to the committed two-commit branch delta.
- Refreshed reuse and workflow notes for built-in `ahi` threshold persistence, event metrics, built-in validation channels, and `ah_event_mask` preset prefiltering.
- Clarified runtime documentation that built-in `ahi` event gating on TST-qualified samples is symmetric for ground-truth and predictions.
- Recorded the AHI event-eval contract as inclusive `>=10s` duration filtering and explicit Pearson -> MAE -> higher-threshold tie-break selection.

## 2026-04-14

- Initialized the `dev/ahi` branch handbook from the checked-out branch state.
- Documented built-in `ahi` task semantics in task/config, dataset, runtime, and workflow pages.
- Recorded that built-in sequence labels now include `stage3`, `stage4`, `stage5`, and `ahi`.
- Recorded the new runtime split between sleep-stage remapping and `ahi` multi-label BCE/metric reduction.
- Added `DELTA_FROM_MAIN.md` for the branch-specific working-tree delta.

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
- Recorded branch-state notes for `sleep2vec2/`, `sleep2vec_moe/`, and `sleep2vec_hires/` because they contain no tracked source files on that indexed branch.
- Marked notebook coverage as summary-only for `preprocess/preprocess_pipeline.ipynb`.
- Stale entries removed: none, because this was an initialize-main build rather than a refresh.
