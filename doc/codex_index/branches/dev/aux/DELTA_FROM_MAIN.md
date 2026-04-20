# Delta From `main`

## Baseline Status

- Branch: `dev/aux`
- HEAD: `99d22deee69cc3cb9eae9229a8faaa4c33974824`
- `main`: `99d22deee69cc3cb9eae9229a8faaa4c33974824`
- Merge base: `99d22deee69cc3cb9eae9229a8faaa4c33974824`
- Commits ahead of `main`: `0`
- Main handbook availability: available under `doc/codex_index/branches/main/`
- Working tree status: dirty under indexed product roots

## Working-Tree Delta Relative To `main`

Changed:

- `sleep2vec/common.py`
- `sleep2vec/config.py`
- `sleep2vec/downstream_model.py`
- `sleep2vec/downstreams/heads/__init__.py`
- `sleep2vec/sleep2vec_finetuning.py`
- `sleep2vec/utils.py`
- `tests/test_check_configs.py`
- `tests/test_config_loading.py`
- `tests/test_stage_task_remapping.py`

Added:

- `sleep2vec/downstreams/heads/temporal_unet.py`
- `configs/ppg_ahi_finetune_large_temporal_unet_aux.yaml`
- `tests/test_downstream_model_auxiliary.py`

Effect:

- added a new registered `temporal_unet` sequence head that reuses the existing fusion and temporal-conv building blocks for large-context token prediction
- added a general metadata auxiliary-task config surface under `finetune.auxiliary_task`
- kept the implementation max-reuse by reusing the existing temporal aggregator registry for auxiliary pooling and the existing `classification` / `regression` heads for auxiliary prediction
- extended `Sleep2vecDownstreamModel` to emit optional auxiliary outputs without changing the built-in `ahi` main-task logits path
- extended `Sleep2vecFinetuning` to optimize `main_loss + loss_weight * aux_loss` while preserving the current AHI checkpoint-threshold and final event-metric semantics
- added the first large-backbone built-in `ahi` recipe for `temporal_unet` plus metadata auxiliary regression
- expanded tests around auxiliary config parsing, metadata-target dataloader plumbing, and dual-output downstream forward behavior

## Areas With No Branch-Local Source Delta

- `data/`: no working-tree delta relative to `main`
- `preprocess/`: no working-tree delta relative to `main`
- `sleep2vec2/`: no tracked source files on this branch
- `sleep2vec_moe/`: no tracked source files on this branch
- `sleep2vec_hires/`: no tracked source files on this branch

## Stale Entries Removed

- copied `dev/ahi` branch labels and committed-delta wording from the initialize baseline
- old branch-history notes that implied `dev/aux` inherited the committed `dev/ahi` divergence from `main`

## Unresolved Ambiguities

- pytest was unavailable in the local Python environment, so the working-tree delta was verified with `compileall` and targeted config checks rather than the full pytest suite.
