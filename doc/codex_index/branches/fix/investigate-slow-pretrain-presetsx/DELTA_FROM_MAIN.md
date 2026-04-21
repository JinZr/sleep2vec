# Delta From `main`

## Baseline Status

- Branch: `fix/investigate-slow-pretrain-presetsx`
- HEAD: `6e687f06f03ac4204da3f5c12107a63b4fb56593`
- `main`: `6e687f06f03ac4204da3f5c12107a63b4fb56593`
- Merge base: `6e687f06f03ac4204da3f5c12107a63b4fb56593`
- Commits ahead of `main`: `0`
- Main handbook availability: available under `doc/codex_index/branches/main/`
- Working tree status: tracked edits pending under indexed product roots

## Committed Branch Delta Relative To `main`

- none

## Current Working Tree Delta

Changed:

- `preprocess/save_dataset_presets.py`
- `tests/test_save_dataset_presets.py`

Effect:

- omitted `--num-workers` now restores automatic validation-thread selection for the common single-preset path instead of forcing `filter_valid_sample_indices(..., max_workers=1)`
- explicit `--num-workers` still behaves as the total preset-generation budget, and multi-job builds continue to run one validation thread per worker process by default
- added a focused regression test that pins the single-job default behavior at the CLI layer

## Areas With No Branch-Local Source Delta

- `sleep2vec2/`: no tracked source delta relative to `main`
- `sleep2vec_moe/`: no tracked source delta relative to `main`
- `sleep2vec_hires/`: no tracked source delta relative to `main`

## Stale Entries Removed

- copied `dev/ahi` branch metadata and committed-delta wording
- copied branch-history notes that did not apply to this fix branch
- stale preset worker-default documentation inherited from the copied baseline

## Unresolved Ambiguities

- Targeted `pytest` execution could not run in the available `python3` environment because `pytest` is not installed.
