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
   - For `ahi`, also injects `stage5` as a required auxiliary runtime label channel for final sleep-stage masking, preserves scalar summary metadata `ahi` and `tst` through batch tensorization as regression-style metadata for final validation/test metrics, and carries per-window `second_valid_mask` so partially masked tokens remain aligned during event evaluation. CSV-backed indexes and older presets may omit same-named serialized scalars because collate-time NPZ backfill supplies them, but regenerated AHI presets are expected to validate `stage5` up front.
   - When `finetune.auxiliary_task` is enabled, appends the auxiliary metadata target through the existing metadata path and reuses the same dataloader/collate contract; no second dataloader or alternate sample format is introduced.
5. Instantiate `Sleep2vecFinetuning`.
   - Creates `Sleep2vecPretrainModel` backbone.
   - Wraps it in `Sleep2vecDownstreamModel`.
   - Reuses the existing main head registry for metadata auxiliary prediction after pooled feature extraction; no auxiliary-head registry is introduced.
   - Optionally loads pretrained backbone checkpoint.
   - Optionally freezes backbone and inserts LoRA adapters.
   - Optionally freezes tokenizers.
   - Optionally attaches model averager.
6. Fit.
   - Monitor comes from task semantics in `apply_task_flags`.
   - Best checkpoint is copied to `best.ckpt` when available.
7. Test.
   - Uses best model after training, or `--ckpt-path` when `epochs == 0`.
   - Result metrics appended via `sleep2vec.results.save_result_csv`, which tags the row with the finetune `version` string, serializes concurrent writers, and only writes from global rank zero.

## Label Semantics

Built-in labels:

- `stage3`: classification, `output_dim=3`, sequence prediction, raw labels from `stage5`
- `stage4`: classification, `output_dim=4`, sequence prediction, raw labels from `stage5`
- `stage5`: classification, `output_dim=5`, sequence prediction
- `ahi`: seq multi-label classification, `output_dim=30`, raw labels from NPZ `ah_event`, required auxiliary runtime `stage5` tokens for final masking, per-window `second_valid_mask` alignment for partial-token padding, scalar NPZ summaries `ahi` / `tst` for metrics rather than required CSV columns. Built-in AHI accepts only `val_ahi_pearson + max`, so validation always runs the same full event-eval path and fits a checkpoint-bound threshold.
- `sex`: classification, `output_dim=2`, non-sequence
- `age`: regression, `output_dim=1`, non-sequence

Custom labels require `finetune.task` in YAML.

## Important Runtime Decisions

- Task semantics are enforced before loaders are built.
- `ahi` reuses the normal sequence head path and keeps pointwise BCE training.
- When a metadata auxiliary task is enabled, it contributes an extra supervised loss term during training but does not change the built-in `ahi` threshold-search, event-metric, or checkpoint-monitor semantics.
- `ahi` validation/test checkpoint selection uses event-based AHI metrics with split post-processing. Validation always fits `ahi_eval_threshold` from cached probabilities on the fine `0.01..0.99` grid, runs that search only on global zero, broadcasts the resulting `{metrics, threshold}` to the other ranks, and persists the fitted threshold into checkpoints. Validation/test loss is accumulated locally during step execution and then reduced once at epoch end before logging, rather than relying on Lightning's eval-side `sync_dist=True` step logging. After the rank-zero-only AHI summary scatter finishes, all ranks synchronize through the strategy barrier before leaving the validation/test epoch, so train-epoch metric collectives cannot run ahead of the visualization path. Train AHI pointwise metrics are computed from globally reduced confusion counts (accuracy/precision/recall/F1 only; no train ROC-AUC), avoiding epoch-wide token concatenation. The automatic finetune-end test and any `--epochs 0` finetune evaluation now reuse the best/manual checkpoint's stored `ahi_eval_threshold`; they do not search a new threshold at test time.
- Multi-device built-in `ahi` finetune installs the reusable callback from `sleep2vec.callbacks.progress_bar`, preserving batch-level updates while skipping the default rank-zero-only train-epoch-end UI refresh/postfix work.
- CLS vs token downstream behavior is defined by `model.cls`, not by folder naming in `configs/`.
- Layer mix is applied inside `Sleep2vecDownstreamModel`, not in the trainer.
- Loss and metric reduction happen inside `Sleep2vecFinetuning`, not in heads.

## Outputs

- Checkpoints under `log-finetune/<version>/checkpoints/`
- Stable `best.ckpt` copy when training ran and a best checkpoint exists
- Optional results CSV row via `sleep2vec.results.save_result_csv` with per-row `experiment_version`
- W&B run under project `sleep2vec-finetune`

## Edit Hotspots

- Change task semantics: `sleep2vec/common.py`, `sleep2vec/config.py`
- Change head/layer-mix/LoRA behavior: `sleep2vec/downstream_model.py`, `sleep2vec/downstreams/`
- Change per-stage loss/metrics aggregation: `sleep2vec/sleep2vec_finetuning.py`
- Change finetune data loader or built-in seq batching: `sleep2vec/utils.py`, `data/default_dataset.py`
