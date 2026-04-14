# Delta From `main`

## Baseline Status

- Branch: `dev/ahi`
- HEAD: `2ab2875b0f882c0f03a17e66fcc9831b8de96063`
- `main`: `2ab2875b0f882c0f03a17e66fcc9831b8de96063`
- Merge base: `2ab2875b0f882c0f03a17e66fcc9831b8de96063`
- Commits ahead of `main`: `0`
- Main handbook availability: available under `doc/codex_index/branches/main/`
- Working tree status: dirty; this handbook includes the tracked `ahi` implementation edits listed below

## Checkout-Local Tracked Modifications Beyond `HEAD`

Changed:

- `README.md`
- `data/default_dataset.py`
- `data/psg_pretrain_dataset.py`
- `preprocess/save_dataset_presets.py`
- `sleep2vec/common.py`
- `sleep2vec/finetune.py`
- `sleep2vec/infer.py`
- `sleep2vec/metrics.py`
- `sleep2vec/sleep2vec_finetuning.py`
- `sleep2vec/utils.py`
- `tests/test_common_finetune_apply.py`
- `tests/test_generic_channel_dataset.py`
- `tests/test_metadata_task_validation.py`
- `tests/test_save_dataset_presets.py`
- `tests/test_stage_task_remapping.py`

Effect:

- added built-in `ahi` as a sequence task with `label_source_name='ahi'`, `output_dim=30`, `monitor='val_f1'`, and an internal multi-label flag
- updated built-in `ahi` monitoring to `val_ahi_pearson` and coupled it to checkpoint-persisted validation threshold fitting
- split built-in sequence handling into shared seq-label plumbing plus sleep-stage-only remapping from raw `stage5`
- added runtime `ahi` dataset-channel support and strict preset filtering via `ahi_mask`
- kept `ahi` training on BCE-with-logits over flattened valid positions, but changed validation/test/infer to event-based AHI metrics derived from thresholded events plus raw-`stage5` TST
- kept evaluation visualizations off the existing confusion-matrix path for `ahi`

## Areas With No Branch-Local Source Delta

- `sleep2vec2/`: no tracked source delta relative to `main`
- `sleep2vec_moe/`: no tracked source delta relative to `main`
- `sleep2vec_hires/`: no tracked source delta relative to `main`

## Stale Entries Removed

- `stage5`-only wording for built-in sequence-task support

## Unresolved Ambiguities

- `HEAD` itself has no committed delta from `main`; the branch-specific behavior documented here comes from tracked working-tree edits
- Runtime validation is partially blocked in this environment because `python3.10`, `pytest`, `torch`, and `scipy` are unavailable
