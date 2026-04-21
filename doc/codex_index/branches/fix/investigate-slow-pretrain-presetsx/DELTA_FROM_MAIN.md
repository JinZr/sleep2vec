# Delta From `main`

## Baseline Status

- Branch: `fix/investigate-slow-pretrain-presetsx`
- HEAD: `e5d6e4ccb4fcf0d9e15ba5292ae00f7fba2f0955`
- `main`: `6e687f06f03ac4204da3f5c12107a63b4fb56593`
- Merge base: `6e687f06f03ac4204da3f5c12107a63b4fb56593`
- Commits ahead of `main`: `1`
- Main handbook availability: available under `doc/codex_index/branches/main/`
- Working tree status: tracked edits pending under indexed product roots

## Committed Branch Delta Relative To `main`

Changed:

- `doc/codex_index/branches/fix/investigate-slow-pretrain-presetsx/`
- `preprocess/save_dataset_presets.py`
- `tests/test_save_dataset_presets.py`

Effect:

- initialized and checked in the branch-specific engineering index for this fix branch
- omitted `--num-workers` restores automatic validation-thread selection for the single-preset path instead of forcing `filter_valid_sample_indices(..., max_workers=1)`
- added a focused regression test that pins that single-job default behavior at the CLI layer

## Current Working Tree Delta

Changed:

- `preprocess/save_dataset_presets.py`
- `tests/test_save_dataset_presets.py`
- `doc/codex_index/branches/fix/investigate-slow-pretrain-presetsx/`

Effect:

- omitted `--num-workers` now restores automatic validation-thread selection for the multi-preset path too, instead of forcing `filter_valid_sample_indices(..., max_workers=1)` inside worker processes
- explicit `--num-workers` now behaves as both the outer preset-generation budget and each multi-job inner validation worker budget
- added focused regression tests that pin both the multi-job default behavior and the explicit multi-job propagation at the CLI layer

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
