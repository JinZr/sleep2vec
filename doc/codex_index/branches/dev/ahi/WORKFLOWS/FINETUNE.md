# Finetune Workflow

## Purpose

Attach a downstream head to the shared backbone, optionally load pretrained weights, optionally insert LoRA, then train and evaluate on train/val/test splits.

## Entry Command

Canonical entrypoint: `python -m sleep2vec.finetune --config ... --label-name ...`

Primary code path:

1. `sleep2vec.common.apply_finetune_config`
2. `sleep2vec.finetune.build_version_name`
3. `sleep2vec.finetune.supervised`
4. `sleep2vec.utils.get_finetune_dataloaders`
5. `sleep2vec.sleep2vec_finetuning.Sleep2vecFinetuning`
6. `sleep2vec.downstream_model.Sleep2vecDownstreamModel`
7. Lightning `trainer.fit(...)` then `trainer.test(...)`

## Detailed Flow

1. Load and bind finetune YAML.
   - Parse typed config bundle with `load_finetune_config`.
   - Copy channels, data paths, task semantics, and LoRA flags into `args`.
   - Reject mismatched `data.data_channel_names`.
2. Resolve version name.
   - Prefer `--version-name`.
   - Otherwise derive from label, channel selection, few-shot mode, pretrained-vs-scratch, and optional tag.
3. Persist run artifacts.
   - `log-finetune/<version>/config.yaml`
   - `log-finetune/<version>/cli_args.yaml`
4. Build train/val/test loaders.
   - Always uses `allow_missing_channels=False`.
   - For built-in sequence labels, adds the runtime label source as a dataset-only pseudo-channel so token labels exist in the batch.
   - For `ahi`, also preserves scalar summary metadata `ahi` and `tst` through batch tensorization as regression-style metadata for final validation/test metrics.
5. Instantiate `Sleep2vecFinetuning`.
   - Creates `Sleep2vecPretrainModel` backbone.
   - Wraps it in `Sleep2vecDownstreamModel`.
   - Optionally loads pretrained backbone checkpoint.
   - Optionally freezes backbone and inserts LoRA adapters.
   - Optionally freezes tokenizers.
   - Optionally attaches model averager.
6. Fit.
   - Monitor comes from task semantics in `apply_task_flags`.
   - Best checkpoint is copied to `best.ckpt` when available.
7. Test.
   - Uses best model after training, or `--ckpt-path` when `epochs == 0`.
   - Result metrics appended via `save_result_csv`.

## Label Semantics

Built-in labels:

- `stage3`: classification, `output_dim=3`, sequence prediction, raw labels from `stage5`
- `stage4`: classification, `output_dim=4`, sequence prediction, raw labels from `stage5`
- `stage5`: classification, `output_dim=5`, sequence prediction
- `ahi`: seq multi-label classification, `output_dim=30`, raw labels from NPZ `ah_event`, scalar NPZ summaries `ahi` / `tst` for metrics, monitor `val_ahi_pearson`
- `sex`: classification, `output_dim=2`, non-sequence
- `age`: regression, `output_dim=1`, non-sequence

Custom labels require `finetune.task` in YAML.

## Important Runtime Decisions

- Task semantics are enforced before loaders are built.
- `ahi` reuses the normal sequence head path and keeps pointwise BCE training.
- `ahi` validation/test checkpoint selection uses event-based AHI metrics: threshold search on validation, checkpoint-persisted threshold reuse on test/infer, scalar NPZ `tst` for TST gating, scalar NPZ `ahi` as the summary ground truth, and `TST < 2h` exclusion from final AHI summary metrics.
- CLS vs token downstream behavior is defined by `model.cls`, not by folder naming in `configs/`.
- Layer mix is applied inside `Sleep2vecDownstreamModel`, not in the trainer.
- Loss and metric reduction happen inside `Sleep2vecFinetuning`, not in heads.

## Outputs

- Checkpoints under `log-finetune/<version>/checkpoints/`
- Stable `best.ckpt` copy when training ran and a best checkpoint exists
- Optional results CSV row via `save_result_csv`
- W&B run under project `sleep2vec-finetune`

## Edit Hotspots

- Change task semantics: `sleep2vec/common.py`, `sleep2vec/config.py`
- Change head/layer-mix/LoRA behavior: `sleep2vec/downstream_model.py`, `sleep2vec/downstreams/`
- Change per-stage loss/metrics aggregation: `sleep2vec/sleep2vec_finetuning.py`
- Change finetune data loader or built-in seq batching: `sleep2vec/utils.py`, `data/default_dataset.py`
