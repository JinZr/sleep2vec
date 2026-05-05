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
   - Copy channels, data paths, task semantics, LoRA flags, and eval-visualization config into `args`.
   - Reject mismatched `data.data_channel_names`.
2. Resolve version name.
   - Prefer `--version-name`.
   - Otherwise derive from label, channel selection, few-shot mode, pretrained-vs-scratch, and optional tag.
3. Persist run artifacts.
   - `log-finetune/<version>/config.yaml`
   - `log-finetune/<version>/cli_args.yaml`
4. Build train/val/test loaders.
   - Always uses `allow_missing_channels=False`.
   - Built-in sequence tasks append runtime label channels to `dataset_channel_names`.
   - `stage3` and `stage4` still pull raw labels from `stage5`.
   - `ahi` adds `ahi` as the primary token label source and `stage5` as an auxiliary label source.
5. Instantiate `Sleep2vecFinetuning`.
   - Creates `Sleep2vecPretrainModel` backbone.
   - Wraps it in `Sleep2vecDownstreamModel`.
   - Optionally loads pretrained backbone checkpoint.
   - Optionally freezes backbone and inserts LoRA adapters.
   - Optionally freezes tokenizers.
   - Optionally attaches a model averager.
   - Optionally enables downstream evaluation visualizations.
6. Fit.
   - Monitor comes from task semantics in `apply_task_flags`.
   - Best checkpoint is copied to `best.ckpt` when available.
7. Test.
   - Uses best model after training, or `--ckpt-path` when `epochs == 0`.
   - Result metrics append via `sleep2vec.results.save_result_csv`.

## Label Semantics

Built-in labels:

- `stage3`: classification, `output_dim=3`, sequence prediction, source labels from `stage5`
- `stage4`: classification, `output_dim=4`, sequence prediction, source labels from `stage5`
- `stage5`: classification, `output_dim=5`, sequence prediction
- `ahi`: multilabel token prediction with `output_dim=30`, validation/test reduced through event-level AHI metrics and a fitted threshold
- `sex`: classification, `output_dim=2`, non-sequence
- `age`: regression, `output_dim=1`, non-sequence

Stage/AHI-only presets may omit `age` and `sex`, but built-in `age` and `sex` runs reject loaded presets/indexes that lack valid labels after split/source filtering.

Custom labels require `finetune.task` in YAML.

## Important Runtime Decisions

- Task semantics are enforced before loaders are built.
- CLS vs token downstream behavior is defined by `model.cls`, not by folder naming in `configs/`.
- Layer mix is applied inside `Sleep2vecDownstreamModel`, not in the trainer.
- AHI validation fits and stores an `ahi_eval_threshold` inside the checkpoint; test and inference require that threshold.
- Confusion matrices, ROC curves, and regression scatter plots are logged from `Sleep2vecFinetuning`, not from entrypoints.

## Outputs

- Checkpoints under `log-finetune/<version>/checkpoints/`
- Stable `best.ckpt` copy when training ran and a best checkpoint exists
- Optional results CSV row via `sleep2vec.results.save_result_csv`
- W&B run under project `sleep2vec-finetune`

## Edit Hotspots

- Change task semantics: `sleep2vec/common.py`, `sleep2vec/config.py`
- Change head/layer-mix/LoRA behavior: `sleep2vec/downstream_model.py`, `sleep2vec/downstreams/`
- Change per-stage loss/metrics aggregation or AHI threshold behavior: `sleep2vec/sleep2vec_finetuning.py`, `sleep2vec/metrics.py`
- Change finetune data loader or built-in label-channel wiring: `sleep2vec/utils.py`, `data/default_dataset.py`, `data/utils.py`
