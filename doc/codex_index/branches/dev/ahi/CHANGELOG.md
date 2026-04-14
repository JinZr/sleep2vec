# Changelog

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
- Recorded branch-state notes for `sleep2vec2/`, `sleep2vec_moe/`, and `sleep2vec_hires/` because they contain no tracked source files on `main`.
- Marked notebook coverage as summary-only for `preprocess/preprocess_pipeline.ipynb`.
- Stale entries removed: none, because this was an initialize-main build rather than a refresh.
