# Delta From `main`

## Baseline Status

- Branch: `dev/ahi`
- HEAD: `0de463929a695c30ec29fa94cfd1e0c5df9e8d92`
- `main`: `2ab2875b0f882c0f03a17e66fcc9831b8de96063`
- Merge base: `2ab2875b0f882c0f03a17e66fcc9831b8de96063`
- Commits ahead of `main`: `3`
- Main handbook availability: available under `doc/codex_index/branches/main/`
- Working tree status: clean under indexed product roots

## Committed Branch Delta Relative To `main`

Changed:

- `data/default_dataset.py`
- `data/psg_pretrain_dataset.py`
- `preprocess/save_dataset_presets.py`
- `sleep2vec/common.py`
- `sleep2vec/finetune.py`
- `sleep2vec/infer.py`
- `sleep2vec/metrics.py`
- `sleep2vec/sleep2vec_finetuning.py`
- `sleep2vec/utils.py`
- `tests/test_ahi_event_metrics.py`
- `tests/test_common_finetune_apply.py`
- `tests/test_generic_channel_dataset.py`
- `tests/test_metadata_task_validation.py`
- `tests/test_save_dataset_presets.py`
- `tests/test_stage_task_remapping.py`

Effect:

- added built-in `ahi` as a sequence classification task with `label_source_name='ahi'`, `output_dim=30`, `is_multilabel=True`, and auxiliary raw `stage5` tokens for evaluation
- enforced built-in `ahi` monitoring as `val_ahi_pearson` and persisted the validation-fitted `ahi_eval_threshold` into downstream checkpoints
- split built-in sequence handling between sleep-stage remapping from raw `stage5` and raw `ahi` multi-label targets with no label remap
- extended runtime `ahi` evaluation from pointwise token metrics to event-based AHI summaries with threshold search, TST gating, severity-threshold metrics, ICC, and Pearson
- fixed AHI event-overlap and severity-threshold boundary semantics so the handbook now matches the current `sleep2vec/metrics.py` implementation
- disallowed `--avg-ckpts > 1` for `ahi` inference because averaged checkpoints cannot carry one reusable validation-fitted threshold
- added runtime `ahi` dataset-channel support, `-1.0` ignore-value padding, and strict preset filtering via `ahi_mask` when missing channels are disallowed
- expanded test coverage with `tests/test_ahi_event_metrics.py` and related config/data/preset assertions for the new `ahi` path

## Areas With No Branch-Local Source Delta

- `sleep2vec2/`: no tracked source delta relative to `main`
- `sleep2vec_moe/`: no tracked source delta relative to `main`
- `sleep2vec_hires/`: no tracked source delta relative to `main`

## Stale Entries Removed

- working-tree-only wording that treated `dev/ahi` as a dirty checkout on top of `main`
- `stage5`-only wording for built-in sequence-task support
- stale branch README counts that no longer matched the tracked tree

## Unresolved Ambiguities

- Runtime execution was not rerun during this documentation-only repair; branch claims were checked by source inspection and tracked tests.
