# Branch Index: `exp/wearable`

This directory is the branch-specific Codex engineering handbook for the checked-out branch `exp/wearable`.

## Snapshot

- Branch: `exp/wearable`
- Commit: `c6cee5cef93c4228b2b11cfd79a83becdbe059a0`
- Generated at: `2026-04-10T17:43:50+0800`
- Mode: `refresh`
- Source of truth: current working tree plus tracked Git history on this branch
- Working tree status: dirty

## What Changed On This Branch

This branch is centered on wearable adaptation and missing-channel-aware pretraining:

- adds a dedicated adaptation entrypoint in `sleep2vec/adapt.py`
- adds staged adaptation logic in `sleep2vec/sleep2vec_adaptation.py`
- extends YAML config parsing with `adapt.*` blocks and validation
- switches preset generation and runtime dataset setup to YAML-defined `model.channels`
- adds pair-first and bucketed sampling behavior for heterogeneous channel availability
- adds wearable adaptation configs for `ppg` and `actigraphy_vm`
- adds matching regression tests for adaptation, bucket sampling, generic channel datasets, and preset generation
- the current tracked checkout also broadens built-in downstream sleep-staging tasks to `stage3`, `stage4`, and `stage5`, with `stage3`/`stage4` remapped from raw `stage5` token labels during finetune and inference

## Read Order

1. `SYSTEM_OVERVIEW.md`
2. `MODULE_MAP.md`
3. `REUSE_GUIDE.md`
4. `DELTA_FROM_MAIN.md`
5. `FUNCTIONS/`
6. `WORKFLOWS/`

## Scope Covered

The index covers tracked code under:

- `sleep2vec/`
- `data/`
- `preprocess/`
- `configs/`
- `tests/`
- `utils/`

The branch also contains directories named `sleep2vec2/`, `sleep2vec_moe/`, and `sleep2vec_hires/`, but on this branch they do not contain tracked source files. See `FUNCTIONS/variant_status.md`.

## Notes

- `doc/codex_index/branches/main/` exists but is empty on this checkout, so there is no usable main-branch handbook baseline.
- `DELTA_FROM_MAIN.md` is derived from `git diff main...HEAD`, not from existing main index files.
- This refresh incorporates tracked working-tree edits in `sleep2vec/common.py`, `sleep2vec/utils.py`, `sleep2vec/sleep2vec_finetuning.py`, `sleep2vec/metrics.py`, `sleep2vec/finetune.py`, `sleep2vec/infer.py`, `tests/test_common_finetune_apply.py`, and `tests/test_metadata_task_validation.py`.
- Untracked files under `configs/` and `tests/` were excluded because this handbook indexes tracked code only.
- When details could not be confirmed cheaply from code, this handbook says `unknown` instead of guessing.
